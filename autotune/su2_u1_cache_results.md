# Separate U1 lookups from spin coefficient caches

Baseline: `ffe5686ffc4f3cc44f09be0e7129d15e8edd5101`.
Measured locally on CPU on 2026-10-17 using the NumPy backend.

The old fusion LRU stored both full SU2xU1 keys and their SU2 projections.
These entries competed for the same capacity. The old adjacent-swap LRU
recomputed the same spin coefficient for distinct conserved U1 labels.

The implementation now uses independent LRUs:

| Reported cache name | Contents | Default maximum entries |
| --- | --- | ---: |
| `nonabelian_fusion_u1` | Full-charge fusion lookups | 4096 |
| `nonabelian_fusion_coefficient` | Numerical coefficients, spin-only for SU2xU1 | 65536 |
| `adjacent_swap_u1` | Full-charge swap lookups and U1 selection checks | 4096 |
| `adjacent_swap_coefficient` | Numerical swap coefficients, spin-only for SU2xU1 | 16384 |

SU2 and SU2xU1 reuse the same numerical spin entries. Other symmetry classes
retain their symmetry-specific keys. Existing cache names for the numerical
layers are unchanged; all four caches support inspection, clearing and resizing.
For example:

```python
yastn.set_cache_maxsize({
    'nonabelian_fusion_u1': 4096,
    'nonabelian_fusion_coefficient': 65536,
    'adjacent_swap_u1': 4096,
    'adjacent_swap_coefficient': 16384,
})
```

## Correctness and eviction reproduction

With two numerical fusion entries, two distinct spin networks, and 32 U1
labels, the original implementation evaluated 64 numerical networks for 64
calls. The new implementation evaluated only 2. All 64 values matched the
SU2 references. This deliberately small-cache test demonstrates eviction;
it is not evidence that the default cache was saturated in a production run.

Regression tests cover independent resizing/clearing, repeated U1 lookup hits,
U1 churn, distinct spin channels, and 756 valid/invalid adjacent-swap cases
against the original F/R expression (including 108 nonzero cases).
The selected NumPy suite passed 114 tests; the Torch CPU suite passed 115,
including the subsequently added network-evaluation-count test. Both suites
explicitly enabled the Hubbard quickstart ED and temperature-schedule tests;
the two METTS comparisons were deselected.

## Equal-spin-content benchmark

```bash
LC_ALL=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python autotune/bench_su2_u1_cache.py --rounds 7 --save-results results.npz
```

Full bond dimension is 24, reduced dimension is 12. Each input has 46 blocks
and 11776 reduced entries. Each sweep operates on 32 tensors. Their spin
content and multiplicities agree; only the U1 labels vary. Separate SU2 and
all-zero-U1 controls are included. Timings include metadata and numerical work.

Final sampling order: optimized 7 rounds, baseline 14 rounds, optimized 7 rounds.
Numbers below are medians of 14 sweeps per implementation for varied U1 labels.
Cold means calling `yastn.clear_cache()` before the sweep; symmetry-level caches
outside that API are not reset. Warm means immediately repeating the sweep.

| Operation | Before cold (ms) | After cold (ms) | Before warm (ms) | After warm (ms) |
| --- | ---: | ---: | ---: | ---: |
| Transpose | 40.305 | 42.351 | 8.034 | 8.023 |
| Fuse | 89.078 | 84.460 | 7.996 | 7.978 |
| Contract | 59.402 | 61.762 | 10.106 | 9.833 |

The cold fuse sweep populated 60 spin fusion entries plus 1920 independent
U1 lookup entries, versus 1980 entries in the old mixed LRU. Adjacent swaps
populated 14 spin entries plus 448 U1 lookups, versus 448 full-charge numerical
entries before. Total entry count did not decrease: the benefit is isolation
and reuse, not a demonstrated reduction in total memory.

All 288 saved before/after output arrays matched exactly (maximum absolute
difference 0). Default capacities were not exhausted. Cold fuse improved by
about 5%; warm fuse was unchanged. Transpose and contraction showed no gain
and their cold timings were higher in this sample. Treat small timing changes
on an interactive workstation cautiously. This isolates cache behavior and
does not establish the cause or speedup of a complete PEPS calculation with
different sector distributions.
