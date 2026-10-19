# Benchmark findings

Machine-specific measurements; compare variants within the same workload and
environment. Raw logs, checkpoints, profiles and one-off experiments are kept
outside the repository. Reproduction commands are in [README.md](README.md).

## Coefficients, recoupling and lookup (2026-10-19)

Baseline: `ff987cd5`. Candidate: `74d3f1db`. The changes use CG orthogonality
for canonical prefix fusion, compose consecutive transpose plans, and replace
linear decomposition searches during fusion-weight construction.

CPU: dual Xeon Gold 5320, NumPy 2.3.5 / SciPy 1.16.3, one thread. Primitive
tests use reduced D=12, five rounds, and nonempty output comparisons.

| Isolated operation | SU2 speedup | SU2xU1 speedup |
| --- | ---: | ---: |
| Cold prefix fuse, coefficient shortcut | 3.09x | 1.36x |
| Warm transpose, composed plan | 4.99x | 4.89x |
| Weight construction with warm dependencies | 1.63x | 2.15x |

The weight case has 979 / 5874 blocks. It measures weight-cache misses;
existing cache hits bypass this work.

NTU: t-J checkerboard, full D=16 (reduced D=8), one NN bond update from the
same cloned checkpoint, SVD initialization, one optimization iteration.
Times are warm medians of three samples, with a separate cold sample.

| Symmetry | CPU baseline s | All changes s | Speedup |
| --- | ---: | ---: | ---: |
| SU2 | 8.5297 | 5.8287 | 1.46x |
| SU2xU1 | 6.2519 | 4.7741 | 1.31x |

All 40 ablation updates completed; an independent baseline tree agreed within
1%. Truncation-error spreads were below 3.5e-16. This diagnostic does not prove
complete physical-state equivalence. A separate interrupted Torch CPU run was
excluded because another workload started on the same host.

GPU: RTX 3090, Torch 2.14.0+cu130. The initial 32-entry transpose cache caused
SU2xU1 NTU to slow down: 2.61 s baseline versus 3.05 s with all changes.
Reverse-order ablation reproduced this. A capacity comparison found a working
set of 75 plans: 32 entries caused 75 new misses per update; 128 entries had
no new misses after warmup and reduced the warm median from 2.98 to 2.35 s.
The 128-entry capacity is an explicit runtime experiment; the current default
is still 32. Choosing and validating the final default remains outstanding.

Correctness suites: NumPy CPU 343 passed; Torch CPU with strict host conversion
347 passed; Torch CUDA 476 passed. Hubbard quickstarts were explicitly enabled;
two expensive METTS cases were deselected. Additional gradient checks cover
vectorized composed plans. CI for `74d3f1db` passed.

## Earlier findings (2026-10-17 to 2026-10-18)

| Change | Baseline | Observed result and scope |
| --- | --- | --- |
| Separate SU2/U1 coefficient caches | `ffe5686f` | Two spin networks evaluated twice instead of 64 times under U1 churn; default-capacity cold fuse improved about 5%, warm fuse unchanged. |
| Hard-fusion and swap metadata | `1bcf7455` | Hard-fusion metadata 1.2-1.7x cold and about 2x with warm dependencies; 16 metadata cases and 288 output arrays matched. |
| Block topology reuse | `747b50b7` | Dimension/mask changes reuse topology; complete cold fuse improved 1.14-1.20x, warm fuse unchanged. New charges still miss. |

Earlier RTX 3090 experiments against `ff987cd5` found that cuTENSOR helped
many-small-block contractions, especially mixed-charge SU2xU1. Combining it
with an experimental column-gather scatter builder improved single-bond D=16
NTU by 1.21x (SU2) and 1.95x (SU2xU1), over five warm samples. These are
separate experiments; the scatter change was not applied to production.
`gesvdj` was slower than `gesvd` on the tested shapes.
