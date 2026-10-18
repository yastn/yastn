# autotune

Micro-benchmarks used to pick backend kernels/parameters on a given machine.
`bench_svd_drivers.py` and `bench_nonabelian_cutensor.py` are self-contained;
the cache benchmarks below import `yastn` from an explicit checkout (`--source`).

## `bench_svd_drivers.py` — cuSOLVER SVD driver crossover

`torch.linalg.svd` on CUDA dispatches to one of three cuSOLVER routines, selected
with `driver=`:

| driver    | algorithm                                     | notes |
|-----------|-----------------------------------------------|-------|
| `gesvd`   | QR iteration                                  | most accurate, `O(L^3)` with a large constant |
| `gesvdj`  | one-sided Jacobi                               | fast for small `L`, iterative — cost grows fast with `L` |
| `gesvda`  | approximate, block Jacobi ("tall-skinny")      | fastest here, but *approximate*: small singular values lose digits |
| `default` | torch's own choice (`None`)                    | tracks `gesvdj` on square matrices |

The script measures all four on square `L x L` matrices with an **exactly known**
singular spectrum, so each timing comes with an error bar on the singular values.

### Matrices

`A = Q diag(s) P^H` with Haar-random `Q, P` (QR of a Gaussian, phase-fixed) and
`s` normalized to `s[0] = 1`:

* `inv_i`   — `s_i = 1/i`
* `inv_i2`  — `s_i = 1/i^2`
* `inv_i3`  — `s_i = 1/i^3`
* `exp_500` — `s_i = e^{-i/500}` (normalized)
* `exp_200` — `s_i = e^{-i/200}` (normalized)

### Sizes

Parametrized by bond dimension `D = 5 ... 12`:

* `D2`, `2D2`, `3D2` — `L = D^2, 2D^2, 3D^2` (corner- / half-sized objects)
* `D4` — `L = D^4` (double-layer objects)

### Running

```bash
python autotune/bench_svd_drivers.py                          # full sweep, float64
python autotune/bench_svd_drivers.py --series D2 2D2 3D2      # cheap subset
python autotune/bench_svd_drivers.py --dtype complex128 --d-max 9
python autotune/bench_svd_drivers.py --spectra inv_i --drivers gesvd gesvdj
```

Useful knobs: `--repeats`, `--warmup` and `--big-L` control the timing loop;
`--out` writes a CSV plus a `*_crossover.json` with the fitted crossovers.

### Cutoffs

`L = D^4` grows fast enough that the top of the requested `D` range is the whole
cost of the sweep: at `D = 12` it is a 20736 x 20736 dense SVD. Four cutoffs
decide where to stop, and all four are reported rather than silently applied:

| cutoff | flag | when it applies |
|--------|------|-----------------|
| explicit `D` cap, per series | `--d-max`, `--d-max-series D4=9` | before anything runs |
| explicit `L` cap | `--l-max 8192` | before anything runs |
| device memory | `--mem-frac` (default 0.75) | before anything runs |
| predicted time | `--max-seconds` (default 20) | during the sweep |

`--plan-only` prints the resulting size plan and exits without touching the GPU —
use it to size a job before submitting it:

```
$ python autotune/bench_svd_drivers.py --plan-only
size plan  (L <= 10987)
  D2   L=D^2    D=5..12  L=25..144
  2D2  L=2D^2   D=5..12  L=50..288
  3D2  L=3D^2   D=5..12  L=75..432
  D4   L=D^4    D=5..10  L=625..10000   | cut D>=11 (cut-mem: ~8.0 GiB > budget 4.5 GiB)
```

The time cutoff is the one that matters on a large-memory node, where memory
stops nothing: a driver is retired once a call exceeds `--max-seconds`, and, more
importantly, a call is *not started* if extrapolating the previous size at
`O(L^3)` predicts it would exceed the budget. Without that, learning that
`gesvdj` is too slow at `L = 12^4` costs about an hour of wall time. Every size
that was cut is still written to the CSV with status `cut-dmax`, `cut-lmax`,
`cut-mem`, `cut-time` or `retired`, so a plot of the results shows where the
sweep stopped and why.

Output: a per-spectrum table of median times, the fitted `gesvdj -> gesvd`
crossover (log-log interpolation of the time ratio through 1), and the worst
singular-value error seen per driver.

## Cache benchmarks (require `yastn`)

These three measure the symmetric-tensor caches. They import `yastn` from the
checkout given by `--source` (default: the repository this file lives in) and
assert the imported path, so a stale editable install cannot silently win.
Always compare two trees by running the same workload twice, once per `--source`.

### `bench_su2_u1_cache.py` — SU(2)/U(1) split of the coefficient caches

Full bond dimension 24, reduced dimension 12, 32 tensors per sweep, default cache
sizes. The inputs share their spin content and differ only in the U(1) labels.
Cold means `yastn.clear_cache()` before the sweep; warm repeats it immediately.

```bash
python autotune/bench_su2_u1_cache.py --rounds 7 --source /path/to/tree
```

Headline result: cold fuse 91.6 -> 78.6 ms (-14%), warm fuse unchanged,
288 saved output arrays identical. The split does not lower the entry count in
this workload (1980 before, 60 spin + 1920 U1 lookups after); it isolates the
cheap U(1) lookups from the expensive coefficients.

### `bench_fusion_metadata.py` — hard-fusion and adjacent-swap metadata

Reduced bond dimension 12, ranks 4 and 6, SU(2) and SU(2xU1), five samples per
phase. Cold clears the caches managed by `yastn.clear_cache()`; dependencies-warm
keeps them, which isolates the metadata builders from the top-level builder cache.

```bash
python autotune/bench_fusion_metadata.py --rounds 5 --source /path/to/tree
```

Headline result: `fuse_hard` 1.2-1.7x on cold metadata, ~2x when only the
dependencies are warm; swaps 1.0-1.3x cold, 1.7-2.5x warm. All 16 nonempty cases
produce matching metadata and output-block fingerprints.

### `bench_get_blocks.py` — block-table reuse across D, masks and charges

Every leg has reduced `D >= 12`. The workload changes one thing at a time:
the leg dimensions, a block mask, or the leg charges of SU(2xU1).

```bash
python autotune/bench_get_blocks.py --steps 12 --rounds 5 --source /path/to/tree
python autotune/bench_get_blocks.py --fuse --rounds 7 --source /path/to/tree
```

Headline result with the topology cache: dimension-only and mask-only sweeps drop
from ~30 ms to ~1 ms (12 hits, 0 misses per round), while changing the leg charges
correctly misses every time. 1200 block fingerprints are identical across rounds.
This is a metadata-only gain: a full NTU step at D=16 changed by about 1%.
