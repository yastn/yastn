# autotune

Reusable benchmarks for backend kernels and non-Abelian metadata. Keep raw
outputs outside the repository; [RESULTS.md](RESULTS.md) contains a compact
record of measured findings and their limits.

## Tools

| Script | Purpose |
| --- | --- |
| `bench_streamed_kernels.py` | Compare CPU/CUDA mask, merge, unmerge and exact 2x2 measurements across explicit source trees. |
| `bench_svd_drivers.py` | Compare cuSOLVER SVD drivers against known singular spectra. |
| `bench_nonabelian_cutensor.py` | Compare Torch and cuTENSOR contractions, gradients and allocated memory. |
| `bench_su2_u1_cache.py` | Measure coefficient-cache reuse while U1 labels change. |
| `bench_fusion_metadata.py` | Isolate hard-fusion and adjacent-swap metadata construction. |
| `bench_get_blocks.py` | Measure block topology reuse across dimensions, masks and charges. |
| `bench_nonabelian_plans.py` | Ablate coefficient shortcuts, composed transpose plans and weight lookup on CPU/CUDA; also run single-bond NTU. |

Use each script's `--help` for its options. CPU tools need NumPy/SciPy; Torch
and CUDA tools additionally need their corresponding backend environment.
The cuTENSOR comparison also needs the repository's TAPP/cuTENSOR dependencies.

## Running

Check workstation availability before launching. Run comparisons serially on
each host, set thread counts explicitly, and compare identical inputs with
reduced D >= 12 for primitive benchmarks. NTU reports full and reduced D
separately. A microbenchmark speedup is not a full evolution-step speedup.

```bash
export LC_ALL=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p /tmp/yastn-autotune-results

python autotune/bench_svd_drivers.py --series D2 2D2 3D2 \
  --d-min 12 --d-max 12 --out /tmp/yastn-autotune-results/svd.csv
python autotune/bench_nonabelian_cutensor.py --dimensions 12 16
python autotune/bench_fusion_metadata.py --source /path/to/checkout --rounds 5
python autotune/bench_get_blocks.py --source /path/to/checkout --fuse
```

The metadata tools assert the imported source path. Compare archived source
trees explicitly; a borrowed editable installation can otherwise select the
wrong checkout. Never install packages into a borrowed environment.

For the three-change ablation, use the unchanged `ff987cd5` tree as baseline:

```bash
python autotune/bench_nonabelian_plans.py --sym SU2xU1 --backend np \
  --baseline-tree /path/to/baseline > /tmp/yastn-autotune-results/plans.jsonl
python autotune/bench_nonabelian_plans.py --sym SU2xU1 --weights \
  --baseline-tree /path/to/baseline
python autotune/bench_nonabelian_plans.py --sym SU2xU1 --device cuda \
  --baseline-tree /path/to/baseline \
  --prepare-checkpoint /tmp/yastn-autotune-results/SU2xU1_D16.pt
```

Use `--checkpoint` to reuse that input, `--variants` to select controls, and
`--plan-cache-size` to investigate cache capacity. To verify the unmodified
baseline independently, copy this script into that tree and run
`--variants native` against the same checkpoint. Each process handles one
symmetry and one checkpoint. Cold samples follow `clear_cache`; warm samples
reuse the metadata. Timing records are JSON lines on stdout.

Generated CSV/JSONL, checkpoints, profiles and `results*` directories are
ignored by Git. Detailed historical reports remain available in Git history;
transient experiment archives are kept outside the repository.

The streaming comparison checks every output element against a direct reference,
then reports cold and warm timings and CUDA peak allocation above the live input
baseline. Run source trees serially with the same options; the source path is
asserted inside each process. For example:

```bash
python autotune/bench_streamed_kernels.py --source /path/to/checkout \
  --device cuda --operation mask --D 12 --blocks 2048
python autotune/bench_streamed_kernels.py --source /path/to/checkout \
  --device cpu --operation measure --sym SU2xU1 --D 12 --chi 24 --chunk-size 6
python autotune/bench_streamed_kernels.py --source /path/to/checkout \
  --device cuda --operation merge --D 12 --blocks 32768
python autotune/bench_streamed_kernels.py --source /path/to/checkout \
  --device cuda --operation merge --D 12 --blocks 32768 --fuse-tile 1048576
```

The primitive cases are synthetic block layouts; their speedups are not full
PEPS-step speedups. Measurement cases compare against the unchunked contraction.
For fusion, compare the default adaptive policy with `--fuse-tile 1048576` and
`--fuse-tile 134217728`; JSON records include the actual adaptive tile choices.
Use `--backend torch_cutensor --operation measure` in an environment with TAPP
to compare complete measurements through that backend.
