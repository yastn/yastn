#!/usr/bin/env python3
"""Tests for multi-device + checkpoint_loop in _contract_with_sliced_unroll.

Validates both forward correctness and backward (autograd) gradient flow
across devices with checkpoint_loop=True.

Usage via pytest:
    pytest --backend torch --device cuda:1 --devices cuda:1,cuda:2,cuda:3 tests/tensor/test_checkpoint_multi_device.py

Usage standalone:
    python tests/tensor/test_checkpoint_multi_device.py --device cuda:1 --devices cuda:1,cuda:2,cuda:3
"""
import pytest
import yastn
from yastn.tensor.oe_blocksparse import contract_with_unroll_compute_constants

tol = 1e-10
tol_ad = 1e-6

multidev_test = pytest.mark.skipif(
    "'torch' not in config.getoption('--backend') or config.getoption('--devices') is None",
    reason="Requires --backend torch and --devices cuda:X,cuda:Y,..."
)


@pytest.fixture
def devices(request):
    """Parse --devices into a list of device strings."""
    raw = request.config.getoption("--devices")
    return [d.strip() for d in raw.split(",")]


def _split_leg_intra(leg):
    """Split sector (0,) in two halves; keep sector (1,) whole."""
    D0 = leg[(0,)]
    half = D0 // 2
    return [
        yastn.SlicedLeg(t=[(0,)], D=[half],      slices={(0,): slice(0, half)}),
        yastn.SlicedLeg(t=[(0,)], D=[D0 - half], slices={(0,): slice(half, D0)}),
        yastn.SlicedLeg(t=[(1,)], D=[leg[(1,)]]),
    ]


def _check_grad(tensors_with_grad, ref_tensors_with_grad, label=""):
    """Compare gradients of test tensors against reference tensors."""
    for i, (t, ref_t) in enumerate(zip(tensors_with_grad, ref_tensors_with_grad)):
        g = t.grad()
        g_ref = ref_t.grad()
        assert g is not None and g._data is not None, \
            f"{label}: tensor {i} has no gradient"
        assert g_ref is not None and g_ref._data is not None, \
            f"{label}: ref tensor {i} has no gradient"
        diff = float(yastn.norm(g - g_ref))
        assert diff < tol_ad, \
            f"{label}: gradient mismatch on tensor {i}: {diff}"


# ---------------------------------------------------------------------------
# 1. Forward: checkpoint_loop + multi-device, charge-sector slicing
# ---------------------------------------------------------------------------
@multidev_test
def test_checkpoint_multidev_charge_sector(config_kwargs, devices):
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)

    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l')
    )
    expected = yastn.ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]])
    sliced_j = yastn.make_sliced_legs(leg_j)

    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={'j': sliced_j}, optimize=path,
        checkpoint_loop=True, devices=devices,
    )
    assert result.device == cfg.default_device
    assert float(yastn.norm(result - expected)) < tol


# ---------------------------------------------------------------------------
# 2. Forward: checkpoint_loop + multi-device, intra-sector slicing
# ---------------------------------------------------------------------------
@multidev_test
def test_checkpoint_multidev_intra_sector(config_kwargs, devices):
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(4, 4))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)

    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l')
    )
    expected = yastn.ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]])

    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={'j': _split_leg_intra(leg_j)}, optimize=path,
        checkpoint_loop=True, devices=devices,
    )
    assert result.device == cfg.default_device
    assert float(yastn.norm(result - expected)) < tol


# ---------------------------------------------------------------------------
# 3. Forward: checkpoint_loop + multi-device, uniform integer slicing
# ---------------------------------------------------------------------------
@multidev_test
def test_checkpoint_multidev_uniform(config_kwargs, devices):
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(4, 4))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)

    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), ('i', 'k')
    )
    expected = yastn.ncon([A, B], [[-1, 1], [1, -2]])

    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'),
        unroll={'j': 2}, optimize=path,
        checkpoint_loop=True, devices=devices,
    )
    assert result.device == cfg.default_device
    assert float(yastn.norm(result - expected)) < tol


# ---------------------------------------------------------------------------
# 4. Forward: compute_constants + checkpoint + multi-device
# ---------------------------------------------------------------------------
@multidev_test
def test_checkpoint_multidev_compute_constants(config_kwargs, devices):
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_m = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)
    D = yastn.rand(config=cfg, legs=[leg_l, leg_m.conj()], n=0)

    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), D, ('l', 'm'), ('i', 'm')
    )
    expected = yastn.ncon([A, B, C, D], [[-1, 1], [1, 2], [2, 3], [3, -2]])
    sliced_j = yastn.make_sliced_legs(leg_j)

    result = contract_with_unroll_compute_constants(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), D, ('l', 'm'), ('i', 'm'),
        unroll={'j': sliced_j}, optimize=path,
        checkpoint_loop=True, devices=devices,
    )
    assert result.device == cfg.default_device
    assert float(yastn.norm(result - expected)) < tol


# ---------------------------------------------------------------------------
# 5. Forward: multi-device WITHOUT checkpoint (baseline)
# ---------------------------------------------------------------------------
@multidev_test
def test_multidev_no_checkpoint(config_kwargs, devices):
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)

    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l')
    )
    expected = yastn.ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]])
    sliced_j = yastn.make_sliced_legs(leg_j)

    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={'j': sliced_j}, optimize=path,
        checkpoint_loop=False, devices=devices,
    )
    assert result.device == cfg.default_device
    assert float(yastn.norm(result - expected)) < tol


# ---------------------------------------------------------------------------
# 6. Forward: multi-index unroll + checkpoint + multi-device
# ---------------------------------------------------------------------------
@multidev_test
def test_checkpoint_multidev_multi_index(config_kwargs, devices):
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)

    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l')
    )
    expected = yastn.ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]])

    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={
            'j': yastn.make_sliced_legs(leg_j),
            'k': yastn.make_sliced_legs(leg_k),
        },
        optimize=path,
        checkpoint_loop=True, devices=devices,
    )
    assert result.device == cfg.default_device
    assert float(yastn.norm(result - expected)) < tol


# ---------------------------------------------------------------------------
# 7. Forward: output-index unroll + checkpoint + multi-device
# ---------------------------------------------------------------------------
@multidev_test
def test_checkpoint_multidev_output_unroll(config_kwargs, devices):
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)

    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), ('i', 'k')
    )
    expected = yastn.ncon([A, B], [[-1, 1], [1, -2]])

    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'),
        unroll={'i': yastn.make_sliced_legs(leg_i)},
        optimize=path,
        checkpoint_loop=True, devices=devices,
    )
    assert float(yastn.norm(result - expected)) < tol


# ===========================================================================
# BACKWARD (autograd) tests
# ===========================================================================

# ---------------------------------------------------------------------------
# 8. Backward: checkpoint + multi-device vs single-device reference
# ---------------------------------------------------------------------------
@multidev_test
def test_backward_checkpoint_multidev(config_kwargs, devices):
    """Gradients from checkpoint+multi-device match single-device ncon reference."""
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))

    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)

    ref_A, ref_B, ref_C = A.clone(), B.clone(), C.clone()
    ref_A.requires_grad_(True)
    ref_B.requires_grad_(True)
    ref_C.requires_grad_(True)
    ref_result = yastn.ncon([ref_A, ref_B, ref_C], [[-1, 1], [1, 2], [2, -2]])
    ref_result.norm().backward()

    A.requires_grad_(True)
    B.requires_grad_(True)
    C.requires_grad_(True)
    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l')
    )
    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={'j': yastn.make_sliced_legs(leg_j)}, optimize=path,
        checkpoint_loop=True, devices=devices,
    )
    result.norm().backward()

    _check_grad([A, B, C], [ref_A, ref_B, ref_C], label="checkpoint+multidev")


# ---------------------------------------------------------------------------
# 9. Backward: multi-device WITHOUT checkpoint (baseline gradient check)
# ---------------------------------------------------------------------------
@multidev_test
def test_backward_multidev_no_checkpoint(config_kwargs, devices):
    """Gradients from multi-device (no checkpoint) match single-device ncon reference."""
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))

    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)

    ref_A, ref_B, ref_C = A.clone(), B.clone(), C.clone()
    ref_A.requires_grad_(True)
    ref_B.requires_grad_(True)
    ref_C.requires_grad_(True)
    ref_result = yastn.ncon([ref_A, ref_B, ref_C], [[-1, 1], [1, 2], [2, -2]])
    ref_result.norm().backward()

    A.requires_grad_(True)
    B.requires_grad_(True)
    C.requires_grad_(True)
    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l')
    )
    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={'j': yastn.make_sliced_legs(leg_j)}, optimize=path,
        checkpoint_loop=False, devices=devices,
    )
    result.norm().backward()

    _check_grad([A, B, C], [ref_A, ref_B, ref_C], label="multidev-no-ckpt")


# ---------------------------------------------------------------------------
# 10. Backward: checkpoint + multi-device with intra-sector slicing
# ---------------------------------------------------------------------------
@multidev_test
def test_backward_checkpoint_multidev_intra_sector(config_kwargs, devices):
    """Gradients with intra-sector slicing + checkpoint + multi-device."""
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(4, 4))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))

    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)

    ref_A, ref_B, ref_C = A.clone(), B.clone(), C.clone()
    ref_A.requires_grad_(True)
    ref_B.requires_grad_(True)
    ref_C.requires_grad_(True)
    ref_result = yastn.ncon([ref_A, ref_B, ref_C], [[-1, 1], [1, 2], [2, -2]])
    ref_result.norm().backward()

    A.requires_grad_(True)
    B.requires_grad_(True)
    C.requires_grad_(True)
    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l')
    )
    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={'j': _split_leg_intra(leg_j)}, optimize=path,
        checkpoint_loop=True, devices=devices,
    )
    result.norm().backward()

    _check_grad([A, B, C], [ref_A, ref_B, ref_C], label="ckpt+multidev-intra")


# ---------------------------------------------------------------------------
# 11. Backward: compute_constants + checkpoint + multi-device
# ---------------------------------------------------------------------------
@multidev_test
def test_backward_checkpoint_multidev_compute_constants(config_kwargs, devices):
    """Gradients with compute_constants + checkpoint + multi-device."""
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_m = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))

    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)
    D = yastn.rand(config=cfg, legs=[leg_l, leg_m.conj()], n=0)

    ref_A, ref_B, ref_C, ref_D = A.clone(), B.clone(), C.clone(), D.clone()
    ref_A.requires_grad_(True)
    ref_B.requires_grad_(True)
    ref_C.requires_grad_(True)
    ref_D.requires_grad_(True)
    ref_result = yastn.ncon([ref_A, ref_B, ref_C, ref_D],
                            [[-1, 1], [1, 2], [2, 3], [3, -2]])
    ref_result.norm().backward()

    A.requires_grad_(True)
    B.requires_grad_(True)
    C.requires_grad_(True)
    D.requires_grad_(True)
    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), D, ('l', 'm'), ('i', 'm')
    )
    result = contract_with_unroll_compute_constants(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), D, ('l', 'm'), ('i', 'm'),
        unroll={'j': yastn.make_sliced_legs(leg_j)}, optimize=path,
        checkpoint_loop=True, devices=devices,
    )
    result.norm().backward()

    _check_grad([A, B, C, D], [ref_A, ref_B, ref_C, ref_D],
                label="ckpt+multidev-cc")


# ---------------------------------------------------------------------------
# 12. Backward: multi-index unroll + checkpoint + multi-device
# ---------------------------------------------------------------------------
@multidev_test
def test_backward_checkpoint_multidev_multi_index(config_kwargs, devices):
    """Gradients with two unrolled indices + checkpoint + multi-device."""
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))

    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)

    ref_A, ref_B, ref_C = A.clone(), B.clone(), C.clone()
    ref_A.requires_grad_(True)
    ref_B.requires_grad_(True)
    ref_C.requires_grad_(True)
    ref_result = yastn.ncon([ref_A, ref_B, ref_C], [[-1, 1], [1, 2], [2, -2]])
    ref_result.norm().backward()

    A.requires_grad_(True)
    B.requires_grad_(True)
    C.requires_grad_(True)
    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l')
    )
    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={
            'j': yastn.make_sliced_legs(leg_j),
            'k': yastn.make_sliced_legs(leg_k),
        },
        optimize=path,
        checkpoint_loop=True, devices=devices,
    )
    result.norm().backward()

    _check_grad([A, B, C], [ref_A, ref_B, ref_C], label="ckpt+multidev-multi-idx")
