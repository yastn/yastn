# autotune

Reusable benchmarks for backend kernels and non-Abelian metadata. Keep raw
outputs outside the repository; [RESULTS.md](RESULTS.md) contains a compact
record of measured findings and their limits.

## Tools

| Script | Purpose |
| --- | --- |
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
python autotune/bench_nonabelian_cutensor.py --dimensions 4 8 12
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
