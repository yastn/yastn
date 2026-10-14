"""Channel-aware cuTENSOR lowering: CPU descriptor oracle and real CUDA coverage.

The CPU oracle evaluates the emitted descriptors as ordinary dense matrices.
CUDA cases exercise the actual TAPP forward/backward when it is installed.
"""
from types import SimpleNamespace

import pytest
import yastn

torch = pytest.importorskip('torch')

from yastn.backend import backend_torch
from yastn.backend._backend_torch_cutensor_nonabelian import contract
from .test_nonabelian_metadata import test_nonabelian_gluing_against_magnetic_basis as _magnetic_check


def _descriptor_matrix(data, descriptor):
    counts, extents, coords, strides, offsets = descriptor
    assert len(counts) == 2
    rows, cols = extents[:counts[0]], extents[counts[0]:]
    coords = coords.reshape(-1, 2).tolist()
    strides = strides.reshape(-1, 2).tolist()
    assert len(set(map(tuple, coords))) == len(coords)
    matrix = data.new_zeros((sum(rows), sum(cols)))
    end = 0
    regions = []
    for (i, j), stride, offset in zip(coords, strides, offsets):
        m, n = rows[i], cols[j]
        assert stride == [n, 1]
        assert offset == end  # disjoint slices are essential for TAPP backward
        end += m * n
        region = (slice(sum(rows[:i]), sum(rows[:i + 1])),
                  slice(sum(cols[:j]), sum(cols[:j + 1])))
        regions.append(region)
        if data.numel():
            matrix[region] = data[offset:end].reshape(m, n)
    assert not data.numel() or end == data.numel()
    return matrix, regions


def _dense_descriptor_oracle(a, b, axes_a, axes_b, *args):
    assert axes_a == [1] and axes_b == [0]
    assert args[-2] == [0, 1] and len(args[-1]) == 24
    desc_a, desc_b, desc_c = args[:5], args[5:10], args[10:15]
    ma, _ = _descriptor_matrix(a, desc_a)
    mb, _ = _descriptor_matrix(b, desc_b)
    _, regions = _descriptor_matrix(a.new_empty(0), desc_c)
    product = ma @ mb
    return torch.cat([product[region].reshape(-1) for region in regions])


@pytest.fixture(params=['oracle', 'cuda'])
def accelerated(request):
    calls = []
    if request.param == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('CUDA is required for the real cuTENSOR test')
        pytest.importorskip('tapp_torch')
        from yastn.backend import backend_torch_cutensor
        backend = SimpleNamespace(**vars(backend_torch_cutensor))
        operation = backend_torch_cutensor.tensordot_bs_v2
        device = 'cuda'
    else:
        backend = SimpleNamespace(**vars(backend_torch))
        backend.BACKEND_ID = 'torch_cutensor'
        # Exercise production dispatch without requiring the external CUDA op.
        backend.is_cpu_device = lambda device: False
        operation = _dense_descriptor_oracle
        device = 'cpu'

    def counted(*args):
        calls.append(1)
        return operation(*args)

    backend.tensordot_nonabelian = lambda a, b, plan: contract(a, b, plan, counted)
    return backend, device, calls


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('geometry', [(2, 2, 2), (3, 1, 3), (3, 3, 1),
                                      (1, 3, 3), (0, 4, 0), (4, 0, 4)])
def test_cutensor_magnetic_basis(accelerated, sym, geometry):
    backend, device, calls = accelerated
    _magnetic_check({'backend': backend, 'default_device': device}, sym, 0, *geometry)
    assert len(calls) == 1


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('dtypes', [('float64', 'float64'), ('complex128', 'complex128'),
                                    ('float64', 'complex128')])
@pytest.mark.parametrize('axes', [((2, 3), (0, 1)), ((0, 2), (3, 1)), ((), ())])
def test_cutensor_multisector_and_gradients(accelerated, sym, dtypes, axes):
    backend, device, calls = accelerated
    config = yastn.make_config(sym=sym, backend=backend, default_device=device)
    reference_config = config._replace(backend=backend_torch)
    charges = ((0,), (1,)) if sym == 'SU2' else ((0, 0), (1, 1))
    leg = yastn.Leg(config, s=1, t=charges, D=(1, 2))
    a = yastn.rand(config, legs=(leg,) * 2 + (leg.conj(),) * 2, dtype=dtypes[0])
    b = yastn.rand(config, legs=(leg,) * 2 + (leg.conj(),) * 2, dtype=dtypes[1])
    # For the arbitrary axes case all contracted signatures must oppose.
    if axes == ((0, 2), (3, 1)):
        a = a.transpose((1, 0, 3, 2))
    a.requires_grad_()
    b.requires_grad_()
    actual = yastn.tensordot(a, b, axes=axes)
    expected = yastn.tensordot(a._replace(config=reference_config),
                              b._replace(config=reference_config), axes=axes)
    assert actual.struct == expected.struct
    torch.testing.assert_close(actual.data, expected.data, atol=1e-11, rtol=1e-11)
    weight = torch.randn_like(actual.data)
    gradients = torch.autograd.grad((actual.data.conj() * weight).real.sum(), (a.data, b.data))
    reference = torch.autograd.grad((expected.data.conj() * weight).real.sum(), (a.data, b.data))
    for found, wanted in zip(gradients, reference):
        torch.testing.assert_close(found, wanted, atol=1e-10, rtol=1e-10)
    assert len(calls) == 1


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('policy', ['fuse_to_matrix', 'fuse_contracted', 'no_fusion'])
def test_cutensor_scalar_irrep_weight(accelerated, sym, policy):
    backend, device, calls = accelerated
    config = yastn.make_config(sym=sym, backend=backend, default_device=device, tensordot_policy=policy)
    charge = (1,) if sym == 'SU2' else (1, 0)
    leg = yastn.Leg(config, s=1, t=(charge,), D=(1,))
    a = yastn.ones(config, legs=(leg, leg.conj()))
    result = yastn.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))
    assert result.item() == pytest.approx(2)
    assert len(calls) == 1


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_cutensor_masked_and_cached(accelerated, sym):
    backend, device, calls = accelerated
    config = yastn.make_config(sym=sym, backend=backend, default_device=device, lazy_threshold=0.8)
    reference_config = config._replace(backend=backend_torch)
    charges = ((0,), (1,), (2,)) if sym == 'SU2' else ((0, 0), (1, 1), (2, 0))
    leg = yastn.Leg(config, s=1, t=charges, D=(1, 2, 1))
    a = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()))
    b = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()))
    a = a.remove_random_blocks(number=3, keep_legs=True)
    b = b.remove_random_blocks(number=3, keep_legs=True)
    yastn.clear_cache()
    for scale in (1, 2):
        x = a * scale
        actual = yastn.tensordot(x, b, axes=((2, 3), (0, 1)))
        expected = yastn.tensordot(x._replace(config=reference_config),
                                  b._replace(config=reference_config), axes=((2, 3), (0, 1)))
        assert actual.struct == expected.struct
        torch.testing.assert_close(actual.data, expected.data)
    assert len(calls) == 2
    assert yastn.get_cache_info()['tensordot_nonabelian_cutensor'].hits > 0
    assert yastn.get_cache_info()['cutensor_nonabelian_packing'].currsize > 0
    yastn.clear_cache()
    assert yastn.get_cache_info()['cutensor_nonabelian_packing'].currsize == 0


@pytest.mark.parametrize('both_scalar', [False, True])
def test_cutensor_rank_zero(accelerated, both_scalar):
    backend, device, calls = accelerated
    config = yastn.make_config(sym='SU2', backend=backend, default_device=device)
    a = yastn.ones(config, legs=()) * 3
    leg = yastn.Leg(config, s=1, t=(1,), D=(2,))
    b = yastn.ones(config, legs=() if both_scalar else (leg, leg.conj())) * 2
    result = yastn.tensordot(a, b, axes=((), ()))
    torch.testing.assert_close(result.data, b.data * 3)
    assert len(calls) == 1


def test_cutensor_empty_contributions(accelerated):
    backend, device, calls = accelerated
    config = yastn.make_config(sym='SU2', backend=backend, default_device=device)
    leg0 = yastn.Leg(config, s=1, t=(0,), D=(1,))
    leg1 = yastn.Leg(config, s=1, t=(1,), D=(1,))
    a = yastn.ones(config, legs=(leg0, leg0.conj()))
    b = yastn.ones(config, legs=(leg1, leg1.conj()))
    a.requires_grad_()
    b.requires_grad_()
    result = a @ b
    assert result.norm().item() == 0
    gradients = torch.autograd.grad(result.data.sum(), (a.data, b.data))
    assert all(torch.count_nonzero(g).item() == 0 for g in gradients)
    assert not calls


def test_cutensor_gradcheck(accelerated):
    backend, device, calls = accelerated
    config = yastn.make_config(sym='SU2', backend=backend, default_device=device)
    leg = yastn.Leg(config, s=1, t=(1,), D=(1,))
    a = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()), dtype='complex128')
    b = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()), dtype='complex128')

    def evaluate(x, y):
        return yastn.tensordot(a._replace(data=x), b._replace(data=y), axes=((2, 3), (0, 1))).data

    assert torch.autograd.gradcheck(evaluate, (a.data.requires_grad_(), b.data.requires_grad_()))
    assert calls


def test_cutensor_cpu_fallback():
    backend = SimpleNamespace(**vars(backend_torch))
    backend.BACKEND_ID = 'torch_cutensor'

    def forbidden(*args):
        pytest.fail('CPU contraction must not call cuTENSOR')

    backend.tensordot_nonabelian = forbidden
    config = yastn.make_config(sym='SU2', backend=backend)
    leg = yastn.Leg(config, s=1, t=(1,), D=(1,))
    a = yastn.ones(config, legs=(leg, leg.conj()))
    assert yastn.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1)).item() == pytest.approx(2)
