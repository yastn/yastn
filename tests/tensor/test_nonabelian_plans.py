"""Coefficient orthogonality and composed permutations against existing oracles."""
from itertools import product

import numpy as np
import pytest
import yastn
from yastn.tensor import _merging as merging, _nonabelian as nonabelian


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('signs,prefix',
                         [(signs, prefix) for signs in product((-1, 1), repeat=4) for prefix in (2, 3)]
                         + [((1, -1, 1), 2), ((-1, 1, -1), 2),
                            ((1, -1, 1, -1, 1), 3), ((-1, 1, -1, 1, -1), 3)])
def test_prefix_coefficient_against_network(sym, signs, prefix, monkeypatch):
    config = yastn.make_config(sym=sym)
    charges = ((0,), (1,), (2,)) if sym == 'SU2' else ((0, -1), (0, 1), (1, 0), (2, 0))
    leg = yastn.Leg(config, s=1, t=charges, D=(1,) * len(charges))
    total = config.sym.zero() if len(signs) == 4 else ((1,) if sym == 'SU2' else (1, 1))
    a = yastn.rand(config, legs=tuple(leg if s == 1 else leg.conj() for s in signs), n=total)
    original = nonabelian._prefix_fusion_coefficient
    checked = []

    def compare(*args):
        found = original(*args)
        if found is not None:
            wanted = nonabelian._fusion_coefficient_network(*args)
            assert found == pytest.approx(wanted, abs=2e-13, rel=2e-13)
            checked.append(found)
        return found

    monkeypatch.setattr(nonabelian, '_prefix_fusion_coefficient', compare)
    yastn.clear_cache()
    fused = a.fuse_legs(axes=(tuple(range(prefix)),) + tuple((i,) for i in range(prefix, len(signs))), mode='hard')
    assert checked and fused.size > 0
    restored = fused.unfuse_legs(axes=0)
    assert (restored - a).norm() < 1e-11


def sequential(a, order):
    current = list(range(len(order)))
    for destination, leaf in enumerate(order):
        source = current.index(leaf)
        while source > destination:
            a = merging._swap_adjacent_fusion_leaves(a, source - 1)
            current[source - 1], current[source] = current[source], current[source - 1]
            source -= 1
    return a


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('order', [(3, 2, 1, 0), (1, 0, 3, 2), (2, 0, 3, 1), (4, 0, 2, 1, 3)])
def test_composed_transpose_against_sequence(config_kwargs, sym, masked, order):
    config = yastn.make_config(sym=sym, **config_kwargs)
    charges = ((0,), (1,), (2,)) if sym == 'SU2' else ((0, -1), (0, 1), (1, 0), (2, 0))
    leg = yastn.Leg(config, s=1, t=charges, D=(1, 2, 1) if sym == 'SU2' else (1, 1, 2, 1))
    total = config.sym.zero() if len(order) == 4 else ((1,) if sym == 'SU2' else (1, 1))
    a = yastn.rand(config, legs=tuple(leg if i % 2 else leg.conj() for i in range(len(order))),
                   n=total, dtype='complex128')
    if masked:
        a = a.remove_random_blocks(number=a.nblocks // 3, keep_legs=True)
    assert a.nblocks > 0
    found = merging._apply_fusion_tree_transpose(a, order)
    expected = sequential(a, order)
    assert found.struct == expected.struct and found.hfs == expected.hfs
    np.testing.assert_allclose(config.backend.to_numpy(found.data), config.backend.to_numpy(expected.data),
                               atol=1e-12, rtol=1e-12)
    doubled = merging._apply_fusion_tree_transpose(2 * a, order)
    assert (doubled - 2 * found).norm() < 1e-11


@pytest.mark.parametrize('multiblock', [False, True])
def test_composed_transpose_gradient(config_kwargs, multiblock):
    torch = pytest.importorskip('torch')
    config = yastn.make_config(sym='SU2', backend='torch', default_device=config_kwargs['default_device'])
    charges = ((0,), (1,), (2,)) if multiblock else ((1,),)
    leg = yastn.Leg(config, s=1, t=charges, D=(1,) * len(charges))
    a = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()), dtype='complex128')
    a.requires_grad_()
    order = (3, 2, 1, 0)
    plans = merging._meta_fusion_tree_transpose(config.sym, a.struct, order)
    assert len(plans) == 1 and len(plans[0][2]) > 0
    if multiblock:
        assert len(plans[0][2]) >= 32  # Exercise the vectorized recoupling adjoint.

    def operation(data):
        return merging._apply_fusion_tree_transpose(a._replace(data=data), order).data

    assert torch.autograd.gradcheck(operation, (a.data,))
    weight = torch.randn_like(operation(a.data))
    result = operation(a.data)
    reference = sequential(a, order).data
    grad, = torch.autograd.grad((result.conj() * weight).real.sum(), a.data)
    ref_grad, = torch.autograd.grad((reference.conj() * weight).real.sum(), a.data)
    torch.testing.assert_close(grad, ref_grad, atol=1e-12, rtol=1e-12)


def test_transpose_cache_can_be_disabled(config_kwargs):
    config = yastn.make_config(sym='SU2', **config_kwargs)
    leg = yastn.Leg(config, s=1, t=((1,),), D=(2,))
    a = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()))
    old_size = yastn.get_cache_info()['fusion_tree_transpose'].maxsize
    try:
        yastn.set_cache_maxsize({'fusion_tree_transpose': 0})
        expected = sequential(a, (3, 2, 1, 0))
        for tensor in (a, 2 * a):
            found = merging._apply_fusion_tree_transpose(tensor, (3, 2, 1, 0))
            scale = 1 if tensor is a else 2
            assert (found - scale * expected).norm() < 1e-11
        info = yastn.get_cache_info()['fusion_tree_transpose']
        assert info.currsize == 0 and info.misses == 2
    finally:
        yastn.set_cache_maxsize({'fusion_tree_transpose': old_size})


def test_high_spin_transpose_against_sequence(config_kwargs):
    config = yastn.make_config(sym='SU2', **config_kwargs)
    leg = yastn.Leg(config, s=1, t=((6,),), D=(1,))
    a = yastn.rand(config, legs=(leg,) * 3 + (leg.conj(),) * 3, dtype='complex128')
    order = (5, 4, 3, 2, 1, 0)
    plans = merging._meta_fusion_tree_transpose(config.sym, a.struct, order)
    assert len(plans) > 1 and all(plan[2] for plan in plans)
    found = merging._apply_fusion_tree_transpose(a, order)
    expected = sequential(a, order)
    assert found.size > 0 and found.struct == expected.struct
    np.testing.assert_allclose(config.backend.to_numpy(found.data), config.backend.to_numpy(expected.data),
                               atol=1e-11, rtol=1e-11)
