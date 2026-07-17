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
        checkpoint_loop=True, devices=devices, mp_workers_per_device=1,
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
        checkpoint_loop=True, devices=devices, mp_workers_per_device=1,
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
        checkpoint_loop=True, devices=devices, mp_workers_per_device=1,
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
        checkpoint_loop=False, devices=devices, mp_workers_per_device=1,
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
        checkpoint_loop=True, devices=devices, mp_workers_per_device=1,
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
        checkpoint_loop=True, devices=devices, mp_workers_per_device=1,
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
        checkpoint_loop=True, devices=devices, mp_workers_per_device=1,
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
        checkpoint_loop=False, devices=devices, mp_workers_per_device=1,
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
        checkpoint_loop=True, devices=devices, mp_workers_per_device=1,
    )
    result.norm().backward()

    _check_grad([A, B, C], [ref_A, ref_B, ref_C], label="ckpt+multidev-intra")


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
        checkpoint_loop=True, devices=devices, mp_workers_per_device=1,
    )
    result.norm().backward()

    _check_grad([A, B, C], [ref_A, ref_B, ref_C], label="ckpt+multidev-multi-idx")


# ---------------------------------------------------------------------------
# Workers>1 per device: single-device (intra-device) and packed across two
# devices. Two-index unroll -> up to 4 combos.
# ---------------------------------------------------------------------------
@multidev_test
@pytest.mark.parametrize("n_dev,workers", [
    (1, 2),
    (1, 3),
    pytest.param(2, 2, marks=pytest.mark.skipif(
        "config.getoption('--devices') is None or "
        "len([d for d in config.getoption('--devices').split(',') if d.strip()]) < 2",
        reason="needs >=2 devices")),
])
def test_multidev_workers_per_device(config_kwargs, devices, n_dev, workers):
    """>1 worker per device — round-robin combo assignment + shared per-device IPC replica; on n_dev device(s) with a two-index unroll."""
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    dev = devices[:n_dev]
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)

    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'))
    expected = yastn.ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]])

    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={'j': yastn.make_sliced_legs(leg_j),
                'k': yastn.make_sliced_legs(leg_k)},
        optimize=path,
        checkpoint_loop=True, devices=dev, mp_workers_per_device=workers,
    )
    assert result.device == cfg.default_device
    assert float(yastn.norm(result - expected)) < tol


# ---------------------------------------------------------------------------
# 16. Backward: workers>1 per device, grads vs single-device reference
# ---------------------------------------------------------------------------
@multidev_test
def test_backward_multidev_workers_per_device(config_kwargs, devices):
    """Backward with two workers per device: gradients match the single-device ncon reference."""
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
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'))
    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={'j': yastn.make_sliced_legs(leg_j)}, optimize=path,
        checkpoint_loop=True, devices=devices, mp_workers_per_device=2,
    )
    result.norm().backward()

    _check_grad([A, B, C], [ref_A, ref_B, ref_C], label="workers-per-device backward")


# ---------------------------------------------------------------------------
# 17. Fermionic (Z2 + swap gate): scalar contraction + unroll + multi-device
# ---------------------------------------------------------------------------
@multidev_test
def test_multidev_fermionic_swap(config_kwargs, devices):
    """Fermionic Z2 contraction with a swap gate, dispatched multi-device (adapted from the single-device swap-diagram tests)."""
    cfg = yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
    l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 3))
    lc = l.conj()
    a = yastn.rand(config=cfg, legs=[l, lc])
    b = yastn.rand(config=cfg, legs=[l, lc])
    ref = yastn.ncon([a, b], ((1, 2), (2, 1)), swap=[(1, 2)])
    path, _ = yastn.get_contraction_path(a, ('p', 'q'), b, ('q', 'p'), ())
    result = yastn.contract_with_unroll(
        a, ('p', 'q'), b, ('q', 'p'), (),
        optimize=path, swap=[('p', 'q')],
        unroll={'p': yastn.make_sliced_legs(l)},
        checkpoint_loop=True, devices=devices, mp_workers_per_device=1,
    )
    assert float(yastn.norm(result - ref)) < tol


# ---------------------------------------------------------------------------
# 18. Multi-component symmetry (U1xU1): chain + output legs + unroll + multidev
# ---------------------------------------------------------------------------
@multidev_test
def test_multidev_multisym_u1xu1(config_kwargs, devices):
    """Two-component symmetry (U1xU1): derived output structs carry length-2 charges."""
    cfg = yastn.make_config(sym='U1xU1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=[(0, 0), (1, 1)], D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=[(0, 0), (1, 1)], D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=[(0, 0), (1, 1)], D=(2, 2))
    leg_l = yastn.Leg(cfg, s=1, t=[(0, 0), (1, 1)], D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=(0, 0))
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=(0, 0))
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=(0, 0))
    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'))
    expected = yastn.ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]])
    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={'j': yastn.make_sliced_legs(leg_j)}, optimize=path,
        checkpoint_loop=True, devices=devices, mp_workers_per_device=1,
    )
    assert result.device == cfg.default_device
    assert float(yastn.norm(result - expected)) < tol


# ===========================================================================
# ---------------------------------------------------------------------------
# 19. per_combo_path=True + multi-device (dim_overrides through the pool)
# ---------------------------------------------------------------------------
@multidev_test
def test_multidev_per_combo_path(config_kwargs, devices):
    """per_combo_path=True: per-combo path selection + dim_overrides routed through the pool (asymmetric dims make it non-trivial)."""
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 3))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(4, 5))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 3))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)
    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'))
    expected = yastn.ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]])
    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={'j': yastn.make_sliced_legs(leg_j)}, optimize=path,
        per_combo_path=True,
        checkpoint_loop=True, devices=devices, mp_workers_per_device=1,
    )
    assert result.device == cfg.default_device
    assert float(yastn.norm(result - expected)) < tol


# ---------------------------------------------------------------------------
# 20. Fused OUTPUT leg (meta + hard) + unroll + multi-device. The from-legs
#     derivation must gather and rebuild the fused Leg via zeros so it matches
#     the worker partials (hard fusion covers review finding [1]).
# ---------------------------------------------------------------------------
@multidev_test
@pytest.mark.parametrize("mode", ['meta', 'hard'])
def test_multidev_fused_output_leg(config_kwargs, devices, mode):
    """A fused leg survives to the output; the from-legs derivation rebuilds it via zeros to match the worker partials."""
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i1 = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_i2 = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i1, leg_i2, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    Af = A.fuse_legs(axes=((0, 1), 2), mode=mode)  # legs: (i=(i1,i2) fused, j)
    path, _ = yastn.get_contraction_path(Af, ('i', 'j'), B, ('j', 'k'), ('i', 'k'))
    expected = yastn.ncon([Af, B], [[-1, 1], [1, -2]])
    result = yastn.contract_with_unroll(
        Af, ('i', 'j'), B, ('j', 'k'), ('i', 'k'),
        unroll={'j': yastn.make_sliced_legs(leg_j)}, optimize=path,
        checkpoint_loop=True, devices=devices, mp_workers_per_device=1,
    )
    assert float(yastn.norm(result - expected)) < tol
