"""Optimized metadata must preserve independently enumerated fusion rules."""
from itertools import product

import numpy as np
import pytest
import yastn

from yastn.sym import sym_SU2, sym_SU2xU1
from yastn.tensor._auxiliary import get_blocks_charges_all
from yastn.tensor._nonabelian import _contract_scalar


@pytest.mark.parametrize('sym', [sym_SU2, sym_SU2xU1])
@pytest.mark.parametrize('rank', [0, 1, 4, 5])
def test_nonabelian_sector_enumeration(sym, rank):
    # 7**5 crosses the meet-in-the-middle threshold. Repeated U1 labels
    # correspond to distinct spin sectors and must retain their own indices.
    sectors = ((0,), (1,), (2,), (3,)) if sym is sym_SU2 else (
        (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1), (2, 0))
    taxes = (sectors,) * rank
    signs = tuple((-1) ** i for i in range(rank))
    for spin in (0, 1, 3):
        target = (spin,) if sym is sym_SU2 else (spin, 1 if rank else 0)
        expected_indices, expected_charges = [], []
        for indices in product(range(len(sectors)), repeat=rank):
            charges = tuple(sectors[i] for i in indices)
            if sym.can_fuse(charges, target, signatures=signs):
                expected_indices.append(indices)
                expected_charges.append(charges)
        charges, indices = get_blocks_charges_all(sym, taxes, signs, target)
        assert np.array_equal(indices, np.asarray(expected_indices, dtype=int).reshape(len(expected_indices), rank))
        assert np.array_equal(charges, np.asarray(expected_charges, dtype=int).reshape(len(expected_charges), rank, sym.NSYM))
    if rank:
        charges, indices = get_blocks_charges_all(sym, ((),) + taxes[1:], signs, sym.zero())
        assert charges.shape == (0, rank, sym.NSYM)
        assert indices.shape == (0, rank)


@pytest.mark.parametrize('charge', [(True, 0), (0, False), (-1, 2), (0, 1.5),
                                   (1,), [[1], [2]], np.array(1), np.ones((2, 1))])
def test_su2xu1_charge_validation(charge):
    with pytest.raises(ValueError):
        sym_SU2xU1.canonical_charge(charge)


def test_su2xu1_charge_containers():
    for charge in ((1, -2), [1, -2], np.array([1, -2]), (np.int64(1), np.int32(-2))):
        result = sym_SU2xU1.canonical_charge(charge)
        assert result == (1, -2)
        assert all(type(x) is int for x in result)


@pytest.mark.parametrize('sym', [sym_SU2, sym_SU2xU1])
@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('prefix_size', [2, 3])
@pytest.mark.parametrize('mode', ['full', 'masked', 'trimmed'])
def test_prefix_fusion_metadata_selection_rules(sym, sign, prefix_size, mode):
    """Compare metadata to explicit fusion rules, including charged tensors."""
    from yastn.tensor._auxiliary import _struct, get_blocks, get_trimmed_struct
    from yastn.tensor._legbasic import LegBasic
    from yastn.tensor._merging import _meta_fuse_hard

    charges = ((0,), (1,), (2,)) if sym is sym_SU2 else ((0, -1), (0, 1), (1, 0), (2, 0))
    total = (1,) if sym is sym_SU2 else (1, 1)
    leg = LegBasic(s=sign, t=charges, D=tuple(range(1, len(charges) + 1)))
    struct = _struct(legs=(leg, leg.conj(), leg, leg.conj(), leg), n=total)
    if mode == 'masked':
        blocks = get_blocks(sym, struct)
        struct = struct.mask_from_ind(blocks.nblocks, list(range(0, blocks.nblocks, 3)))
    legs_sub = None
    selected = struct
    if mode == 'trimmed':
        smaller = LegBasic(s=sign, t=leg.t[:-1], D=leg.D[:-1])
        legs_sub = struct.legs[:-1] + (smaller,)
        selected = get_trimmed_struct(sym, struct, legs_sub)
    source = get_blocks(sym, struct)
    selected_blocks = get_blocks(sym, selected)
    axes = (tuple(range(prefix_size)),) + tuple((i,) for i in range(prefix_size, 5))
    meta, size, fused, _ = _meta_fuse_hard(sym, struct, axes, legs_sub=legs_sub, lazy_threshold=1)
    target = get_blocks(sym, fused)
    assert len(meta) == selected_blocks.nblocks > 0
    assert size == target.size > 0
    by_source = {tuple(slc): i for i, slc in enumerate(source.slc)}
    by_target = {tuple(slc): i for i, slc in enumerate(target.slc)}
    seen = set()
    for sln, Dn, slo, Do, subslices, subdims in meta:
        ia, ic = by_source[(slo.start, slo.stop)], by_target[(sln.start, sln.stop)]
        seen.add(ia)
        old_charges = tuple(map(tuple, source.t[ia]))
        new_charges = tuple(map(tuple, target.t[ic]))
        assert tuple(source.D[ia]) == Do and tuple(target.D[ic]) == Dn
        for group, charge, new_leg in zip(axes, new_charges, fused.legs):
            outcomes = sym.signed_fusion_outcomes(
                tuple(old_charges[i] for i in group), tuple(struct.legs[i].s for i in group), new_leg.s)
            assert charge in outcomes
        assert sym.can_fuse(new_charges, total, signatures=tuple(leg.s for leg in fused.legs))
        assert target.channels[ic] == source.channels[ia][(prefix_size - 1) * sym.NSYM:]
        assert all(s.stop - s.start == d for s, d in zip(subslices, subdims))
    assert len(seen) == len(meta)


@pytest.mark.parametrize('sym', [sym_SU2, sym_SU2xU1])
@pytest.mark.parametrize('sign', [-1, 1])
def test_empty_prefix_and_swap_metadata(sym, sign):
    from yastn.tensor._auxiliary import _struct, get_blocks
    from yastn.tensor._legbasic import LegBasic
    from yastn.tensor._merging import _meta_fuse_hard, _meta_swap_adjacent_fusion_leaves

    charge = (1,) if sym is sym_SU2 else (1, 1)
    leg = LegBasic(s=sign, t=(charge,), D=(2,))
    struct = _struct(legs=(leg, leg.conj()) * 2, n=sym.zero())
    struct = struct.mask_from_ind(get_blocks(sym, struct).nblocks, [])
    meta, size, _, _ = _meta_fuse_hard(sym, struct, ((0, 1), (2,), (3,)))
    assert not meta and size == 0
    for position in range(3):
        _, _, meta, size = _meta_swap_adjacent_fusion_leaves(sym, struct, position)
        assert not meta and size == 0


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_masked_adjacent_swap_roundtrip(config_kwargs, sym):
    from yastn.tensor._merging import _swap_adjacent_fusion_leaves

    config = yastn.make_config(sym=sym, **config_kwargs)
    charges = ((0,), (1,), (2,)) if sym == 'SU2' else ((0, -1), (0, 1), (1, 0), (2, 0))
    leg = yastn.Leg(config, s=1, t=charges, D=(1,) * len(charges))
    a = yastn.rand(config, legs=(leg, leg.conj()) * 3, dtype='complex128')
    a = a.remove_random_blocks(number=a.nblocks // 3, keep_legs=True)
    assert a.nblocks > 0 and a.norm() > 0
    for position in range(5):
        b = _swap_adjacent_fusion_leaves(a, position)
        restored = _swap_adjacent_fusion_leaves(b, position)
        assert (restored - a).norm() < 1e-12
        assert abs(b.norm() - a.norm()) < 1e-12


@pytest.mark.parametrize('dimensions', [(2, 3, 4), (3, 4, 2), (1, 2, 1), (1, 1, 1)])
def test_compiled_scalar_network_reuses_structure_only(dimensions):
    rng = np.random.default_rng(6)
    i, j, k = dimensions
    b, c = rng.normal(size=(j, k)), rng.normal(size=(k, i))
    for _ in range(2):
        a = rng.normal(size=(i, j))
        factors = [a, [0, 1], b, [1, 2], c, [2, 0]]
        assert _contract_scalar(factors) == pytest.approx(np.trace(a @ b @ c), abs=1e-13)
    assert _contract_scalar([]) == 1.


@pytest.mark.parametrize('sym, total_q', [('SU2', 0), ('SU2xU1', 0), ('SU2xU1', 2)])
@pytest.mark.parametrize('na, nc, nb', [(2, 2, 2), (3, 1, 3), (3, 3, 1),
                                      (1, 3, 3), (0, 4, 0), (4, 0, 4)])
def test_nonabelian_gluing_against_magnetic_basis(config_kwargs, sym, total_q, na, nc, nb):
    from yastn.tensor._auxiliary import get_blocks
    from yastn.tensor._nonabelian import intertwiner

    config = yastn.make_config(sym=sym, **config_kwargs)
    qs = [(-1) ** i for i in range(na + nc)]
    qs[0] += total_q
    charges = [(1, q) if sym == 'SU2xU1' else (1,) for q in qs]
    legs_a = [yastn.Leg(config, s=1, t=(charge,), D=(1,)) for charge in charges]
    legs_b = [leg.conj() for leg in legs_a[na:]]
    for i in range(nb):
        q = sum(qs[na:]) if i == 0 else 0
        charge = (1, q) if sym == 'SU2xU1' else (1,)
        legs_b.append(yastn.Leg(config, s=1, t=(charge,), D=(1,)))
    total_a = (0, total_q) if sym == 'SU2xU1' else (0,)
    total_b = (0, -total_q) if sym == 'SU2xU1' and nb == 0 else config.sym.zero()
    a = yastn.rand(config, legs=legs_a, n=total_a, dtype='complex128')
    b = yastn.rand(config, legs=legs_b, n=total_b, dtype='complex128')

    def magnetic(tensor):
        tensor = tensor.consume_transpose()
        dense = np.zeros((2,) * tensor.ndim, dtype=complex)
        blocks = get_blocks(config.sym, tensor.struct)
        for charges, channel, slc in zip(blocks.t, blocks.channels, blocks.slc):
            cg = intertwiner(config.sym, tuple(map(tuple, charges)), tensor.s, tensor.n, tuple(channel))
            value = config.backend.to_numpy(tensor._data[slice(*slc)]).item()
            dense += value * cg
        return dense

    axes = (tuple(range(na, na + nc)), tuple(range(nc)))
    actual = magnetic(yastn.tensordot(a, b, axes=axes))
    expected = np.tensordot(magnetic(a), magnetic(b), axes=axes)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_nonabelian_adjacent_swaps_against_magnetic_basis(config_kwargs, sym):
    from yastn.tensor._auxiliary import get_blocks
    from yastn.tensor._nonabelian import intertwiner

    config = yastn.make_config(sym=sym, **config_kwargs)
    charge = (2,) if sym == 'SU2' else (2, 1)
    leg = yastn.Leg(config, s=1, t=(charge,), D=(1,))
    a = yastn.rand(config, legs=(leg, leg.conj()) * 3, dtype='complex128')

    def magnetic(tensor):
        dense = np.zeros((3,) * 6, dtype=complex)
        blocks = get_blocks(config.sym, tensor.struct)
        for charges, channel, slc in zip(blocks.t, blocks.channels, blocks.slc):
            cg = intertwiner(config.sym, tuple(map(tuple, charges)), tensor.s, tensor.n, tuple(channel))
            dense += config.backend.to_numpy(tensor._data[slice(*slc)]).item() * cg
        return dense

    original = magnetic(a)
    for position in range(5):
        order = list(range(6))
        order[position:position + 2] = reversed(order[position:position + 2])
        moved = a.transpose(order).consume_transpose()
        np.testing.assert_allclose(magnetic(moved), original.transpose(order), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_nonabelian_partial_trace_with_sector_trimming(config_kwargs, sym):
    from yastn.tensor._auxiliary import get_blocks
    from yastn.tensor._nonabelian import intertwiner

    config = yastn.make_config(sym=sym, **config_kwargs)
    charges = [(j,) if sym == 'SU2' else (j, 0) for j in (0, 2, 4, 6)]
    wide = yastn.Leg(config, s=1, t=charges, D=(1,) * 4)
    one = yastn.Leg(config, s=1, t=(charges[1],), D=(1,))
    a = yastn.rand(config, legs=(wide, one, one, one.conj()), dtype='complex128')

    def magnetic(tensor):
        tensor = tensor.consume_transpose()
        sectors, offset = {}, 0
        for charge in tensor.get_legs(0).t:
            sectors[charge] = slice(offset, offset + charge[0] + 1)
            offset += charge[0] + 1
        dense = np.zeros((offset,) + (3,) * (tensor.ndim - 1), dtype=complex)
        blocks = get_blocks(config.sym, tensor.struct)
        for charges, channel, slc in zip(blocks.t, blocks.channels, blocks.slc):
            cg = intertwiner(config.sym, tuple(map(tuple, charges)), tensor.s, tensor.n, tuple(channel))
            dense[sectors[tuple(charges[0])]] += config.backend.to_numpy(tensor._data[slice(*slc)]).item() * cg
        return dense

    reduced = a.trace((2, 3))
    assert reduced.get_legs(0).t == (charges[1],)
    # Closing the last two legs leaves only spin one on the first leg.
    expected = np.trace(magnetic(a), axis1=2, axis2=3)[1:4]
    np.testing.assert_allclose(magnetic(reduced), expected, atol=1e-12, rtol=1e-12)
