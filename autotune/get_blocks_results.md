# Dimension-independent get_blocks topology cache

Measured on 2026-10-18 against baseline `747b50b7`, which includes the
earlier coefficient-cache and fusion-metadata improvements.

## Implementation

`get_blocks` retains its full-struct LRU. For non-Abelian symmetries it now
gets channel-expanded charges, sector indices and channels from a separate
LRU keyed by `(sym, leg_charges, leg_signatures, total_charge)`.
The key excludes D, channel filters, masks and isdiag. Channel filtering
still precedes masking; dimensions, slices and size are computed for the
current struct. The Abelian path is unchanged.

Shared topology arrays are read-only. Different D values reuse these arrays
without copying them. Filtering creates separate arrays. The new cache
retains up to 128 complete, unfiltered topologies, independently of the
existing block-table cache. This bounds entry count, not memory in bytes.

```python
yastn.get_cache_info()['get_blocks_topology']
yastn.set_cache_maxsize({'get_blocks_topology': 128})
yastn.clear_cache()
```

## Block-table benchmark

```bash
LC_ALL=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python autotune/bench_get_blocks.py --steps 16 --rounds 7
```

Use `--source /path/to/baseline` for the same workload on another source
tree. The imported module path is asserted. Minimum reduced leg dimension
is 12. No full tensor data is allocated in this mode.

Each round clears managed caches and builds one seed struct, then queries
16 distinct structs. They vary D, mask, or U1 charges. The first sweep has
16 full-struct cache misses. A subsequent identical sweep tests cache hits.
The table shows medians of seven first sweeps, in milliseconds.

| Symmetry | Rank | Varying | Before (ms) | After (ms) | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| SU2 | 4 | dimensions | 2.171 | 0.104 | 20.79x |
| SU2 | 4 | masks | 2.222 | 0.165 | 13.47x |
| SU2 | 6 | dimensions | 30.546 | 0.343 | 89.05x |
| SU2 | 6 | masks | 31.493 | 1.132 | 27.82x |
| SU2xU1 | 4 | dimensions | 2.650 | 0.108 | 24.56x |
| SU2xU1 | 4 | masks | 2.669 | 0.158 | 16.90x |
| SU2xU1 | 4 | charges | 3.171 | 3.134 | 1.01x |
| SU2xU1 | 6 | dimensions | 43.865 | 0.344 | 127.44x |
| SU2xU1 | 6 | masks | 44.452 | 1.119 | 39.72x |
| SU2xU1 | 6 | charges | 48.092 | 47.819 | 1.01x |

Changing D or masks produces 16 topology hits and no new topology misses
after the seed. Changing U1 charges produces 16 misses, as required.
All 160 before/after block-table fingerprints matched. Each implementation
checked 2240 nonempty outputs across timed first/repeated sweeps.
These large speedups describe get_blocks on reusable topology, not a full
PEPS step. New-charge workloads show essentially unchanged latency.

## Complete hard-fusion operations

```bash
LC_ALL=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python autotune/bench_get_blocks.py --fuse --save-results fused.npz
```

This mode allocates rank-four random tensors outside timing and fuses
`((0, 1), (2, 3))`. Each sweep has 16 tensors with equal sector content and
different D, all with reduced leg dimension >= 12. Cold sweeps clear managed
caches; warm sweeps immediately repeat them. Seven sweeps per phase are
summarized by their median. Inputs use fixed random seeds.

| Symmetry | Before cold (ms) | After cold (ms) | Cold speedup | Before warm (ms) | After warm (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SU2 | 37.494 | 32.922 | 1.14x | 10.665 | 10.880 |
| SU2xU1 | 41.312 | 34.540 | 1.20x | 8.617 | 8.444 |

Each cold sweep still has 96 full-struct get_blocks misses, but the
topology layer has only 6 misses and 90 hits. Warm fusion bypasses these
builds and shows no material improvement. All 32 nonempty saved fused
arrays matched the baseline exactly; each run checked 448 outputs.
These are local NumPy CPU measurements, not GPU or complete PEPS timings.

## Correctness

Before the change, four different D values caused 148 signed_fusion_paths
calls for the SU2 fixture and 152 for SU2xU1. Afterward they cause 37 and
38 respectively: each external-charge combination is expanded only once.

The 18 new regression cases cover dimension reuse, read-only sharing,
channel-filter/mask ordering, different total charges and signatures,
empty legs, ranks 0/1/2/4, diagonal sizes, and cache resizing/clearing/disabling.
An explicit fusion-path oracle also covers A4 outer multiplicities.

NumPy: 255 passed, 136 skipped, 2 deselected.
Torch CPU: 259 passed, 132 skipped, 2 deselected.
Hubbard quickstart ED and temperature-schedule tests were explicitly enabled.
The two METTS tests were deselected. Skips include GPU-only cases and
redundant CPU scatter variants.

```bash
python -m pytest tests/tensor/test_blocks_topology.py \
  tests/tensor/test_nonabelian*.py tests/tensor/test_cache.py \
  tests/tensor/test_block.py tests/tensor/test_empty.py \
  tests/tensor/test_initialization.py tests/tensor/test_fuse_hard.py \
  tests/tensor/test_fuse_meta.py tests/quickstart/test_Hubbard_SU2.py \
  --quickstart -k "not METTS" -q
```

Repeat with `--backend torch` for Torch.
