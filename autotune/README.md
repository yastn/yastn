# autotune

Micro-benchmarks used to pick backend kernels/parameters on a given machine.
Self-contained: nothing here imports `yastn`.

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
