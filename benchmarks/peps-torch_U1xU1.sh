#!/bin/bash

git clone https://github.com/jurajHasik/peps-torch.git
cd peps-torch # benchmark uses relative paths assuming current working directory
rm -r yastn   # remove submodule directory of yastn in peps-torch ...
ln -s ../../ yastn  # ... replace it by symlink to yastn

export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1 #(depends on the implementation of your SVM, works for PyTorch, this pins threads to cores)

python <<< 'import torch; print(torch.__config__.show())'
python <<< 'import torch; print(torch.__config__.parallel_info())'

lscpu
if ! command -v lscpu &> /dev/null
then
    echo "lscpu could not be found"
else
	lscpu
fi
if ! command -v numactl &> /dev/null
then
    echo "numactl could not be found"
else
	numactl --show
fi

# set of U1xU1 test states indexed by their "dense" bond dimension
prof_states=(
[3]="IPESS_AKLT_3b_D3_1x1_abelian-U1xU1_T3T8_state.json"
[4]="IPESS_CSL_D4_1x1_abelian-U1xU1_state.json"
[7]="IPESS_TRIMER_133bar-133bar_D7_1x1_abelian-U1xU1_state.json"
[9]="IPESS_TRIMER3_D9_1x1_abelian-U1xU1_state_extended.json"
[12]="IPESS_AKLT_3b3b6_D12_U1xU1_state.json"
[13]="IPESS_CSL_D13_1x1_abelian-U1xU1_state.json"
)

convtol=6                              # convergence tolerance of CTM as 1e-$convtol - here difference of corner spectra
preltol=12                             # pseudoinverse relative cutoff on smallest singular values as 1e-$preltol
pmethod="4X4"                          # ctmrg projector method, here full 4x4 system
extra="--GLOBALARGS_dtype complex128"
noise="--instate_noise 0"


for ad in 3 #4 7 9 12 13
do
for ch in $(( 1 * ad * ad ))
do
for run_cpu in 16                      # choose set of cores for benchmarking
do

echo "To run D ${ad} CH ${ch} CPU ${run_cpu}"

export OMP_NUM_THREADS=$run_cpu
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS

instate="test-input/abelian/${prof_states[ad]}"
out_prefix="cpu${run_cpu}_ABU1xU1_D${ad}_1x1_CTMe${convtol}.opt${ch}"
seed="$(shuf -i 1-100000 -n 1)"

python -u -m cProfile -o ${ouf_prefix}.prof examples/kagome/abelian/optim_su3_kagome_U1xU1.py \
--phi 0.5 --theta 0.0 --bond_dim $ad --chi $ch \
--seed $seed $noise \
--instate $instate \
--out_prefix $out_prefix \
--GLOBALARGS_device "cpu" \
--CTMARGS_ctm_max_iter 500 --CTMARGS_ctm_conv_tol 1.0e-$convtol \
--CTMARGS_projector_svd_reltol 1.0e-$preltol \
--CTMARGS_projector_method $pmethod \
--opt_max_iter 2 \
$extra \
--CTMARGS_fwd_checkpoint_move \
--OPTARGS_tolerance_grad 1.0e-8 \
--omp_cores $run_cpu 2> ${out_prefix}.err 1> ${out_prefix}.out &

wait
done
done
done
