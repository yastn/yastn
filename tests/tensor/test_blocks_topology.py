"""Dimension-independent block topology preserves channels, masks and slices."""
from itertools import product

import numpy as np
import pytest
import yastn
from yastn.sym import sym_SU2, sym_SU2xU1
from yastn.tensor import _auxiliary
from yastn.tensor._auxiliary import _struct
from yastn.tensor._legbasic import LegBasic
from .test_nonabelian_symmetry import A4FusionRules


@pytest.fixture(autouse=True)
def clean_caches():
    yastn.clear_cache()
    yield
    yastn.clear_cache()


def make_struct(sym, rank=4, dimension=2):
    charges = ((0,), (1,), (2,)) if sym is sym_SU2 else (
        ((0,), (1,), (2,), (3,)) if sym is A4FusionRules else
        ((0, -1), (0, 1), (1, 0), (2, 0)))
    legs = tuple(LegBasic(s=(-1) ** i, t=charges, D=(dimension,) * len(charges)) for i in range(rank))
    return _struct(legs=legs, n=sym.zero())


@pytest.mark.parametrize('sym', [sym_SU2, sym_SU2xU1])
def test_topology_reused_when_dimensions_change(sym, monkeypatch):
    calls = []
    original = sym.signed_fusion_paths

    def counted(cls, *args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sym, 'signed_fusion_paths', classmethod(counted))
    first = _auxiliary.get_blocks(sym, make_struct(sym))
    initial_calls = len(calls)
    assert initial_calls > 0 and first.nblocks > 0
    for dimension in (3, 4, 5):
        found = _auxiliary.get_blocks(sym, make_struct(sym, dimension=dimension))
        np.testing.assert_array_equal(found.t, first.t)
        np.testing.assert_array_equal(found.coords, first.coords)
        assert found.channels == first.channels
        assert found.size == first.nblocks * dimension ** 4
        assert np.all(found.D == dimension)
    assert len(calls) == initial_calls, 'changing D must not expand fusion channels again'
    assert not first.t.flags.writeable and not first.coords.flags.writeable
    with pytest.raises(ValueError, match='read-only'):
        first.t.flat[0] = 999
    with pytest.raises(ValueError, match='read-only'):
        first.coords.flat[0] = 999


@pytest.mark.parametrize('sym', [sym_SU2, sym_SU2xU1])
def test_channel_filter_precedes_mask(sym):
    struct = make_struct(sym)
    full = _auxiliary.get_blocks(sym, struct)
    allowed = tuple(dict.fromkeys(full.channels))[::2]
    indices = [i for i, channel in enumerate(full.channels) if channel in allowed]
    assert 1 < len(indices) < full.nblocks
    filtered = struct.replace(channels=allowed)
    selected = filtered.mask_from_ind(len(indices), list(range(0, len(indices), 2)))
    found = _auxiliary.get_blocks(sym, selected)
    expected = indices[::2]
    assert found.nblocks == len(expected) > 0
    np.testing.assert_array_equal(found.t, full.t[expected])
    np.testing.assert_array_equal(found.coords, full.coords[expected])
    assert found.channels == tuple(full.channels[i] for i in expected)
    np.testing.assert_array_equal(found.slc[:, 0], np.arange(len(expected)) * 16)
    assert found.size == len(expected) * 16
    # Filtering must not write into the shared, unfiltered topology.
    different_D = _auxiliary.get_blocks(sym, make_struct(sym, dimension=3))
    np.testing.assert_array_equal(different_D.t, full.t)
    assert different_D.nblocks == full.nblocks
    empty = _auxiliary.get_blocks(sym, filtered.mask_from_ind(len(indices), []))
    assert empty.nblocks == empty.size == 0
    assert empty.t.shape == (0, 4, sym.NSYM) and empty.coords.shape == (0, 4)


@pytest.mark.parametrize('sym', [sym_SU2, sym_SU2xU1, A4FusionRules])
@pytest.mark.parametrize('rank', [0, 1, 2, 4])
def test_topology_against_explicit_paths(sym, rank):
    struct = make_struct(sym, rank=rank)
    if rank == 1:
        struct = struct.replace(n=struct.legs[0].t[0])
    # Changing the total or a signature must never reuse incompatible topology.
    total = (1, 1) if sym is sym_SU2xU1 else (1,)
    cases = [struct, struct.replace(n=total)]
    if rank:
        cases.append(struct.replace(legs=(struct.legs[0].conj(),) + struct.legs[1:]))
        cases.append(struct.replace(legs=(LegBasic(s=1, t=(), D=()),) + struct.legs[1:]))
    for case in cases:
        charges, coords, channels = [], [], []
        for indices in product(*(range(len(leg.t)) for leg in case.legs)):
            ts = tuple(leg.t[i] for leg, i in zip(case.legs, indices))
            for path in sym.signed_fusion_paths(ts, case.n, tuple(leg.s for leg in case.legs)):
                charges.append(ts)
                coords.append(indices)
                channels.append(tuple(x for charge in path for x in
                                      (charge if isinstance(charge, tuple) else (charge,))))
        found = _auxiliary.get_blocks(sym, case)
        assert found.nblocks == len(charges)
        np.testing.assert_array_equal(found.t, np.asarray(charges, dtype=np.int64).reshape(len(charges), rank, sym.NSYM))
        np.testing.assert_array_equal(found.coords, np.asarray(coords, dtype=np.int64).reshape(len(coords), rank))
        assert found.channels == tuple(channels)
        lengths = np.prod(found.D, axis=1, dtype=np.int64)
        np.testing.assert_array_equal(found.slc[:, 1], np.cumsum(lengths))
        np.testing.assert_array_equal(found.slc[:, 0], np.cumsum(lengths) - lengths)
        assert found.size == sum(lengths)
    assert _auxiliary.get_blocks(sym, struct).nblocks > 0


@pytest.mark.parametrize('sym', [sym_SU2, sym_SU2xU1])
def test_diagonal_dimensions_and_topology_cache_control(sym):
    name = 'get_blocks_topology'
    previous = yastn.get_cache_info()[name].maxsize
    try:
        yastn.set_cache_maxsize({name: 2})
        struct = make_struct(sym, rank=2)
        dense = _auxiliary.get_blocks(sym, struct)
        diagonal = _auxiliary.get_blocks(sym, struct.replace(isdiag=True))
        assert dense.nblocks == diagonal.nblocks > 0
        assert dense.size == 2 * diagonal.size
        np.testing.assert_array_equal(diagonal.slc[:, 1] - diagonal.slc[:, 0], diagonal.D[:, 0])
        info = yastn.get_cache_info()[name]
        assert info.currsize == info.misses == 1 and info.hits == 1
        yastn.clear_cache()
        assert yastn.get_cache_info()[name].currsize == 0
        yastn.set_cache_maxsize(0, only=[name])
        for dimension in (3, 4):
            _auxiliary.get_blocks(sym, make_struct(sym, rank=2, dimension=dimension))
        info = yastn.get_cache_info()[name]
        assert info.currsize == 0 and info.misses == 2
    finally:
        yastn.set_cache_maxsize({name: previous})
