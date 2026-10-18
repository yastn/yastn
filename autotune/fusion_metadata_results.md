# Hard-fusion and adjacent-swap metadata optimization

Measured on 2026-10-18 against baseline commit `1bcf7455` (which already
contains the independent U1 and SU2 coefficient caches). These changes
optimize metadata construction without changing cache capacities or keys.

## Changes

- Canonical-prefix hard fusion reads the outgoing irrep from the existing
  source channel. It does not enumerate fusion outcomes and recheck total
  conservation for each already valid source block. Other groupings retain
  the generic selection-rule path.
- Hard fusion indexes decomposition records once per leg, including internal
  channels where required. Duplicate keys still raise an ambiguity error.
- Adjacent swaps permute the charge table in one operation, preconvert block
  shapes, and reuse slice lengths instead of per-block indexing and np.prod.

## Isolated metadata benchmark

Run from the repository root:

```bash
LC_ALL=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python autotune/bench_fusion_metadata.py --rounds 5
```

Use `--source /path/to/baseline` to run the same workload against another
source tree. The script asserts the imported yastn path. Optional `--profile`
writes cProfile data; profiling runs are outside the timed samples.

Reduced bond dimension is 12; full dimensions are 24 (SU2) and 21 (SU2xU1).
SU2xU1 includes different U1 sectors within each leg. Ranks 4 and 6 cover
front, middle and final swaps. Five samples per phase are summarized by
their median. The top-level builder cache is bypassed in both phases:
cold clears caches managed by yastn.clear_cache; dependencies_warm retains
them. This measures metadata cache misses, not top-level cache-hit latency.

| Symmetry | Rank | Operation | Blocks | Cold speedup | Warm dependencies speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| SU2 | 4 | fuse_hard | 46 | 1.23x | 2.05x |
| SU2 | 4 | swap_0 | 46 | 1.15x | 2.14x |
| SU2 | 4 | swap_1 | 46 | 1.01x | 2.09x |
| SU2 | 4 | swap_2 | 46 | 1.14x | 2.48x |
| SU2 | 6 | fuse_hard | 977 | 1.66x | 1.99x |
| SU2 | 6 | swap_0 | 977 | 1.19x | 2.04x |
| SU2 | 6 | swap_1 | 977 | 1.25x | 2.41x |
| SU2 | 6 | swap_4 | 977 | 1.27x | 2.43x |
| SU2xU1 | 4 | fuse_hard | 47 | 1.29x | 2.03x |
| SU2xU1 | 4 | swap_0 | 47 | 1.08x | 1.88x |
| SU2xU1 | 4 | swap_1 | 47 | 1.10x | 2.35x |
| SU2xU1 | 4 | swap_2 | 47 | 1.04x | 2.43x |
| SU2xU1 | 6 | fuse_hard | 955 | 1.57x | 2.14x |
| SU2xU1 | 6 | swap_0 | 955 | 1.06x | 1.65x |
| SU2xU1 | 6 | swap_1 | 955 | 1.05x | 2.15x |
| SU2xU1 | 6 | swap_4 | 955 | 1.08x | 2.20x |

All 16 nonempty metadata cases had matching full metadata and output-block
fingerprints. Each implementation also checked consistency across all 160
timed calls. Dependency work still dominates some cold cases.

## End-to-end operations

The separate `bench_su2_u1_cache.py --rounds 7` workload uses full D=24,
reduced D=12, 32 tensors per sweep and default cache sizes. The varied-U1
case has equal spin content across tensors and different U1 labels.

| Varied-U1 operation | Before cold (ms) | After cold (ms) | Before warm (ms) | After warm (ms) |
| --- | ---: | ---: | ---: | ---: |
| transpose | 43.281 | 39.815 | 8.397 | 8.231 |
| fuse | 91.629 | 78.577 | 8.402 | 8.468 |
| contract | 66.575 | 65.127 | 10.294 | 10.317 |

All 288 saved before/after arrays matched exactly (maximum absolute error
0). Warm top-level cache hits bypass the optimized work and show no material
gain. Timings come from an interactive local CPU session; small differences
should not be treated as a full PEPS or GPU speedup.

## Regression coverage

NumPy: 221 passed, 136 skipped, 2 deselected.
Torch CPU: 225 passed, 132 skipped, 2 deselected.
Skipped cases include GPU-only tests and redundant CPU scatter variants;
the two METTS tests were deselected. Hubbard quickstart ED and temperature
schedule tests were explicitly enabled.

```bash
python -m pytest tests/tensor/test_nonabelian*.py \
  tests/tensor/test_fuse_hard.py tests/tensor/test_fuse_meta.py \
  tests/tensor/test_cache.py tests/quickstart/test_Hubbard_SU2.py \
  --quickstart -k "not METTS" -q
```

Repeat with `--backend torch` for the Torch suite. The 30 new regression
cases cover both symmetries, positive/negative signatures, nonzero total
charge, two/three-leg prefixes, full/masked/trimmed structures, empty inputs,
and masked adjacent-swap roundtrips and norm preservation.

An additional public-transpose test exposed an existing masked F-move error:
`Cannot match SU2 fusion channels after leg permutation.` This reproduces
on the untouched baseline for both SU2 and SU2xU1 when swapping the second
and third legs of a partially masked six-leg tensor. That public route uses
`_meta_f_move_12_to_23`, outside these two optimized builders. It remains
unchanged; direct adjacent-swap tests pass.
