"""Streamed multi-step recoupling against the sequential basis changes."""
import numpy as np
import pytest
import yastn
from yastn.tensor import _merging as merging


@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
@pytest.mark.parametrize('channels', [2, 20])
def test_recouple_sequence_adjoint(config_kwargs, dtype, channels):
    config = yastn.make_config(**config_kwargs)
    shape = (2, 3, 4)
    steps = []
    for order in ((2, 0, 1), (1, 0, 2), (0, 2, 1)):
        meta = tuple(record for start in range(0, channels * 24, 48)
                     for record in ((start, start, shape, 0.6), (start, start + 24, shape, 0.8),
                                    (start + 24, start, shape, -0.8), (start + 24, start + 24, shape, 0.6)))
        steps.append((order, meta))
        shape = tuple(shape[i] for i in order)
    steps = tuple(steps)
    tensor = yastn.rand(config, s=(1, -1), D=(1, channels * 24 + 1), dtype=dtype)
    data = tensor.data  # Includes an unused input entry.

    def sequential(x):
        for order, meta in steps:
            x = config.backend.recouple(x, order, meta, channels * 24 + 3)
        return x

    def streamed(x):
        return config.backend.recouple_sequence(x, steps, channels * 24 + 3)

    np.testing.assert_allclose(config.backend.to_numpy(streamed(data)),
                               config.backend.to_numpy(sequential(data)), atol=1e-12, rtol=1e-12)
    assert config.backend.to_numpy(streamed(data))[channels * 24:].tolist() == [0, 0, 0]
    if 'torch' in config.backend.BACKEND_ID:
        import torch
        data.requires_grad_()
        assert torch.autograd.gradcheck(streamed, (data,), fast_mode=True)
        assert torch.autograd.gradgradcheck(streamed, (data,), fast_mode=True)
        weight = torch.randn_like(streamed(data))
        actual, = torch.autograd.grad(streamed(data), data, weight)
        expected, = torch.autograd.grad(sequential(data), data, weight)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('masked', [False, True])
def test_raw_plans_use_one_output(config_kwargs, sym, masked, monkeypatch):
    config = yastn.make_config(sym=sym, **config_kwargs)
    charges = (0, 1, 2) if sym == 'SU2' else ((0, 0), (1, 1), (1, -1))
    leg = yastn.Leg(config, s=1, t=charges, D=(1, 2, 1))
    a = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj(), leg, leg.conj()), dtype='complex128')
    if masked:
        a = a.remove_random_blocks(number=a.nblocks // 4, keep_legs=True)
    order = (0, 2, 5, 1, 3, 4)  # append_vec_tr's second tensordot.
    current = list(range(6))
    expected = a
    plans = []
    for destination, leaf in enumerate(order):
        source = current.index(leaf)
        while source > destination:
            plan = merging._meta_swap_adjacent_fusion_leaves(config.sym, expected.struct, source - 1)
            plans.append(plan)
            expected = merging._apply_recoupling(expected, *plan)
            current[source - 1], current[source] = current[source], current[source - 1]
            source -= 1
    assert len(plans) > 1 and all(plan[2] for plan in plans)
    monkeypatch.setattr(merging, '_meta_fusion_tree_transpose', lambda *args: tuple(plans))

    def forbid_raw(*args):
        raise AssertionError('A raw full-size intermediate recoupling was allocated')

    monkeypatch.setattr(config.backend, 'recouple', forbid_raw)
    found = merging._apply_fusion_tree_transpose(a, order)
    assert found.size > 0 and found.struct == expected.struct and found.hfs == expected.hfs
    np.testing.assert_allclose(config.backend.to_numpy(found.data), config.backend.to_numpy(expected.data),
                               atol=1e-11, rtol=1e-11)
