# Benchmark findings

Machine-specific measurements; compare variants within the same workload and
environment. Raw logs, checkpoints, profiles and one-off experiments are kept
outside the repository. Reproduction commands are in [README.md](README.md).

## cuTENSOR packing batches and NNN default (2026-10-28)

Workstation 46, RTX 3090 (24 GiB), Torch 2.10.0+cu126, one CPU thread.
Runs were serial, with occupancy checked at each launch. Raw logs and the
benchmark script are in `/home/yintai/cutensor_20261028_codex/`; the isolated
tested tree is `/tmp/yintai_cutensor_20261028_v1/source/`. Seven changed
production/test files matched the local tree by SHA-256 after the runs.

The cuTENSOR/CUDA regression suite passed 379 cases with one non-strict XPASS
and no skips; this includes real TAPP CUDA cases and CPU descriptor oracles.
NNN append/close and SU2 CTM/NTU passed another 41 cases. NNN selection tests
now honor `--backend torch_cutensor`; the startup log confirms the new NNN
input chunk default of 8 GiB (8589934592 bytes).

The first GPU run exposed three gradient failures: the contiguous-slice pack
shortcut returned a strided incoming gradient view, while TAPP's descriptors
address unit-stride storage. Making that shortcut return contiguous storage
fixed the failures. Added cases cover stride-two and expanded scalar views;
the complete GPU suite above was rerun after the correction.

The benchmark uses float64 SU2/SU2xU1 tensors with reduced degeneracy 12 or 16
in each of three sectors. The baseline loads `backend_torch_cutensor.py` from
`4caae981`; both variants use the same current tensor metadata and inputs.
All 12 variant/workload comparisons checked nonempty, nonzero outputs and both
input gradients against ordinary Torch. The 24 timing/memory records have a
positive completion marker. Timings are medians of three synchronized warm
samples, after one warmup. Memory is extra peak live PyTorch allocation above
the input/reference baseline, not total device memory.

| Symmetry / degeneracy | Operation | Baseline ms | Default ms | 8 MiB ms | Baseline peak MiB | Default peak MiB | 8 MiB peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SU2 / 12 | forward | 4.579 | 4.659 | 6.231 | 84.000 | 84.000 | 23.230 |
| SU2 / 12 | forward + backward | 11.010 | 12.346 | 33.101 | 111.320 | 84.000 | 36.203 |
| SU2 / 16 | forward | 13.695 | 14.261 | 22.072 | 196.000 | 196.000 | 39.000 |
| SU2 / 16 | forward + backward | 35.146 | 38.766 | 70.794 | 352.042 | 236.042 | 86.598 |
| SU2xU1 / 12 | forward | 1.633 | 1.716 | 2.361 | 49.719 | 49.719 | 20.383 |
| SU2xU1 / 12 | forward + backward | 4.023 | 6.007 | 16.615 | 57.945 | 53.199 | 26.736 |
| SU2xU1 / 16 | forward | 4.686 | 5.042 | 7.566 | 88.000 | 88.000 | 29.000 |
| SU2xU1 / 16 | forward + backward | 12.092 | 13.596 | 41.804 | 126.017 | 102.000 | 56.598 |

The default packing budget is at most 1 GiB and did not split these small
primitive workloads. Saving original inputs lowers differentiated peaks but
repacking increases backward time. Forcing 8 MiB with
`YASTN_CUTENSOR_PACK_CHUNK_BYTES=8388608` lowers forward peaks by 59-80% and
forward-plus-backward peaks by 54-75%, with forward 1.36-1.61x slower and
forward-plus-backward 2.01-4.13x slower. Keep the adaptive default unless lower
memory is needed. The cuTENSOR packing budget is separate from
`YASTN_NNN_APPEND_CHUNK_BYTES`. After these runs, its default was changed from
8 GiB to 4 GiB (4294967296 bytes); the recorded test logs retain the 8 GiB
setting used during validation. These results do
not constitute a rerun of the H100 D=21 WCSS job.

## CTMRG lifetimes, QR/SVD and streamed recoupling (2026-10-24)

Baseline: `3e7c8fae7d072e9f656dcaf5f4f245201547fda4`. Candidate: working-tree
changes to QR, SVD truncation and the standard 2x2 CTMRG projector path.
Workstation 46, RTX 3090 (24 GiB), Torch 2.10.0+cu126, one CPU thread.
Raw records: `/home/yintai/ctm_linalg_20261024_codex/` on workstation 46.

The standard projector path requests only R. With gradients disabled, the
backend skips Q; with gradients enabled, Torch retains reduced QR for its
derivative but skips assembling and unfusing an output Q tensor. Truncated
SVD still performs one full decomposition and applies the same global mask,
but masks the fused factors before phase fixing and unfusion. Full factors
are released sequentially when autograd does not need them. The exact SVD
backward continues to retain its required full singular vectors.

The initial QR/SVD-only candidate is preserved as `linalg_source/` in the raw
records directory. `bench_ctm_linalg.py` uses `no_grad`, with one cold sample
and three warm samples. The table reports warm medians; memory is PyTorch
allocated memory, excluding external workspaces and allocator reservations.

| Operation | CPU baseline / QR+SVD s | GPU baseline / QR+SVD s | GPU extra peak baseline / QR+SVD MiB |
| --- | ---: | ---: | ---: |
| QR, four 1728 x 576 blocks | 0.23148 / 0.11431 | 0.06262 / 0.04334 | 94.385 / 53.880 |
| SVD, same blocks, retain 12 total | 0.62292 / 0.56381 | 0.54255 / 0.54172 | 172.156 / 123.367 |

The complete D=12, chi=12 checkerboard sweep compares every updated environment
tensor with the independent baseline: 16 tensors, 84,096 elements for SU2 and
56,448 for SU2xU1. Absolute peaks include the benchmark-held input/copies.

| Full sweep | CPU baseline / QR+SVD s | GPU baseline / QR+SVD s | GPU absolute peak baseline / QR+SVD MiB |
| --- | ---: | ---: | ---: |
| SU2 | 17.12826 / 16.60603 | 5.06202 / 4.85479 | 240.215 / 223.991 |
| SU2xU1 | 36.24987 / 34.83942 | 4.17086 / 3.97953 | 176.683 / 176.683 |

Thus QR/SVD savings alone do not lower the peak of every CTMRG workload.

Local validation: new Torch CPU tests 56 passed; new tests plus SU2 CTM/NTU
under NumPy 67 passed, 8 skipped. The workstation 46 selected Torch CPU suite
passed 241 tests, with 14 skipped and 4 expected failures. It covers the new
memory paths, existing QR/SVD/svds, backend memory, non-Abelian operations,
SU2 CTM/NTU, SI derivatives/pipeline, and dense Ising CTMRG. New cases include
complex data, SU2/SU2xU1, a non-prefix custom mask, empty truncation, mode
dispatch, QR gradcheck, and SVD derivative equivalence to the old pipeline.
The same workstation GPU suite passed 241 tests, with 14 skipped, 3 expected
failures and 1 non-strict expected-failure case passing. The actual cuTENSOR
suite passed all 75 cases. Strict host conversion was enabled.

The subsequent candidate additionally consumes ordinary square-lattice
corners in pairs, without recomputing them, and drops projector-local halves
and R factors after use. Dummy-bond patches and implicit SI halves keep the
original path. New lifetime tests verify exactly four corner contractions
per projector anchor and at most one preceding corner still live when the
next is constructed, for all six move directions, with and without QR.

It also streams multi-plan non-Abelian canonicalization when the global
sparse product cannot be retained within its composition/work budget.
The 65,536-entry cap remains in place. Raw steps are indexed by destination;
only one destination block's composed coefficients are built at a time.
Large blocks accumulate through strided views. Torch batches at most 4096
small-block records with element-index tiles of at most 2**20 entries, and
does not cache the device maps of that streamed product. Forward and its
adjoint allocate one full output, without retaining full intermediate
recoupled tensors. Structural indexing has a 16-entry host cache.

Additional validation on workstation 46: Torch CPU 234 passed and GPU 234
passed; each deselected 28 SI fixed-point/reference cases. The preceding
QR/SVD suite collected those cases under its usual skip rules. Actual
cuTENSOR: 49 passed. The final
batched-adjoint tests additionally passed 8 cases each on Torch CUDA and
cuTENSOR CUDA, including gradcheck and gradgradcheck. Locally, NumPy passed
122 cases and Torch passed 53 additional non-Abelian/PEPS/RDM/measurement
regressions. The streamed tests cover the `(0,2,5,1,3,4)` order inside
`append_vec_tr`, complex data, masked channels, uncovered entries, and the
absence of calls allocating raw intermediate recouplings.

## Streamed masks, fusion maps and exact 2x2 measurements (2026-10-24)

Baseline: `afce33263c7b359823c06b3182a083c67aaf8965`. Candidate: working-tree
changes to the Torch kernels and chunked `measure_2x2`. Workstation 46,
RTX 3090 (24 GiB), Torch 2.10.0+cu126, one CPU thread. Source trees were
isolated and their imported paths asserted. Runs were serial on a free host.
Raw records: `/home/yintai/optimization_20261023_codex/` on workstation 46.

`bench_streamed_kernels.py` measures forward under `no_grad`, with a cold
sample and the median of three warm samples. CUDA memory is extra peak live
PyTorch allocation above the input/reference/cache baseline, including the
output; it excludes external cuTENSOR workspaces and is not total VRAM.
Every result is nonempty, nonzero and compared with a direct reference.

The mask case selects 21,233,664 elements from 2048 D=12 blocks. CPU mask
time fell from 1.41727 s to 0.09319 s (15.21x) by applying axis selections
directly to small blocks. Larger blocks and CUDA use bounded index tiles;
autograd retains block/axis metadata and rebuilds indices during backward.
An initial CPU implementation using flat tiled indices took 1.63946 s and
was replaced before final validation.

The merge/unmerge case has 2048 D=12 blocks and one D=48 block, with
3,649,536 output elements. CPU merge changed from 44.28 to 40.75 ms;
CPU unmerge was essentially unchanged (30.18 to 30.24 ms). The unscaled
merge writes permuted views directly, without flattening them into copies.
Hybrid CUDA kernels now stream both source and destination maps, rather
than retaining full maps. CPU and device metadata have separate caches.

CUDA results for the same primitive cases (time in ms, memory in MiB):

| Operation | Baseline ms | Candidate ms | Baseline extra peak | Candidate extra peak |
| --- | ---: | ---: | ---: | ---: |
| Mask | 12.05 | 13.91 | 569.13 | 210.06 |
| Merge | 7.59 | 7.87 | 280.05 | 108.00 |
| Unmerge | 6.89 | 7.52 | 280.05 | 108.00 |

These changes trade some isolated CUDA speed for bounded memory: mask is
15% slower with 63% less extra peak allocation; merge/unmerge are 4%/9%
slower with 61% less. Mask tile tuning at 2**21 and 2**22 gave 13.91 and
13.46 ms, but increased the extra peak to 258 and 354 MiB, so mask retains
2**20 independently of the adaptive fusion policy.

Exact 2x2 measurements use reduced D=12, chi=24, boundary chunk size 6,
and a two-site number observable. Chunked results agree with the full
unchunked contraction at rtol=1e-10, atol=1e-11. CPU warm times:

| Symmetry | Baseline s | Candidate s | Speedup |
| --- | ---: | ---: | ---: |
| SU2 | 32.8428 | 19.1627 | 1.71x |
| SU2xU1 | 127.4841 | 72.8639 | 1.75x |

Final GPU measurements rerun both trees serially with the same environment:

| Backend | Symmetry | Baseline s | Candidate s | Speedup | Extra peak MiB, baseline -> candidate |
| --- | --- | ---: | ---: | ---: | ---: |
| Torch | SU2 | 2.3133 | 1.4150 | 1.63x | 215.48 -> 284.25 |
| Torch | SU2xU1 | 2.9697 | 1.4799 | 2.01x | 190.27 -> 258.24 |
| cuTENSOR | SU2 | 10.9258 | 10.0346 | 1.09x | 279.53 -> 335.93 |
| cuTENSOR | SU2xU1 | 8.0255 | 6.6719 | 1.20x | 189.74 -> 258.46 |

Ordinary Torch remains faster for this particular measurement workload.
The cuTENSOR comparisons use real TAPP and the same symmetry/input seeds;
they do not show a universal advantage for either backend. Final peak
increases are 68-69 MiB for Torch and 56-69 MiB for cuTENSOR.

A separate warm cProfile run observed 268/238 TAPP calls for SU2/SU2xU1.
The Python `tensordot_bs_v2` wrapper accumulated 4.82/3.52 s, while `_pack`
accumulated 0.14/0.11 s. Metadata construction also accumulated substantial
time in blocking `torch.as_tensor` calls (3.91/1.94 s across the run).
These are host-side profiling observations, not GPU kernel timings: queued
CUDA work can be charged to a later upload or synchronization. They point
to native contraction/layout/dispatch and upload synchronization as further
investigation targets; they do not establish that increasing packing tiles
will fix this workload. Both profiled outputs passed the reference comparison.

Numerator and denominator share unchanged corners; boundary alignment and
invariant contractions run once. Known interval masks avoid CUDA nonzero
synchronizations. A local right-corner cache retains at most 64 MiB of
tensor data. This is a speed/memory tradeoff: keeping both observables and
cached corners alive can increase peak memory. Autograd saved tensors are
not included in the cache cap, and inference savings do not imply equal
savings in a differentiated CTMRG calculation.

Fusion tiles are adaptive: target about eight tiles, rounded to powers of
two, with an estimated scratch budget bounded by 1 GiB and one eighth of
available memory after accounting for the output. Small operations avoid
a CUDA memory query. `YASTN_FUSE_SCATTER_CHUNK` remains an explicit override.
The budget is a conservative estimate, not an allocation guarantee.

A larger hybrid comparison uses 32768 D=12 blocks plus one D=48 block,
56,733,696 checked output elements. All three settings run the same final
code; only the tile policy changes. Each output matches the direct reference.
Actual adaptive choices were 2**23 for all cold and warm calls.

| Tile policy | Small-block tiles | Merge ms | Unmerge ms | Extra peak MiB |
| --- | ---: | ---: | ---: | ---: |
| Fixed 2**20 | 54 | 109.10 | 113.18 | 512.85 |
| Adaptive, 2**23 | 7 | 91.82 | 95.77 | 1072.88 |
| Fixed 2**27 | 1 | 89.49 | 94.21 | 4753.09 |

Adaptive uses 15-16% less time than fixed 2**20. It is within 3% of the
single large tile, with 77% less extra peak allocation. This supports growing
tiles for large workloads while retaining a memory limit; it is not evidence
that a fixed 2**20 tile has negligible overhead at every scale.

Validation: Torch CPU 543 passed, 354 skipped, 2 deselected; Torch CUDA
877 passed, 20 skipped, 2 deselected. Strict host conversion was enabled.
The selected suite includes mask/fusion/contraction
paths, first/second derivatives, cache management, non-Abelian plans and
metadata, PEPS measurements, SU2 CTM/NTU/RDM, and Hubbard quickstarts.
The two expensive METTS cases were explicitly deselected. CPU skips include
CUDA-only fusion variants; this is a selected regression suite, not all tests.
The NumPy chunked-measurement suite also passed: 18 passed, 2 skipped.
The cuTENSOR suite passed all 142 cases without skips, including actual
TAPP CUDA contractions, CPU descriptor oracles, mixed-dtype gradients,
packing gradcheck/gradgradcheck, chunked measurements and SU2 multiplets.
Post-run SHA-256 comparisons matched all six changed production, regression
test and benchmark files between the local tree and the tested remote tree.

A prior real-cuTENSOR packing audit on the same host (2026-10-23, baseline
`35dfe951`, outside this change) found that direct slice packing helped
float64 D=16/24 forward by 1.15-1.19x, but regressed D=12 forward and
D=12/16 forward+backward. Heterogeneous dimensions (12,16,24) improved by
1.18-1.20x forward and 1.04-1.06x forward+backward. Its 14 contractions and
42 output/gradient comparisons passed. Those results support investigating
layout- and gradient-aware packing, not an unconditional direct-copy policy.
This change leaves native TAPP packing unchanged; `torch_cutensor` inherits
the shared Torch mask/fusion optimizations. The audit is archived at
`/home/yintai/cutensor_pack_audit_20261023/REPORT.md` on workstation 46.

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
The default is now 128 entries (`fusion_tree_transpose` in `_merging.py`).

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
