"""Channel-aware cuTENSOR lowering: CPU descriptor oracle and real CUDA coverage.

The CPU oracle evaluates the emitted descriptors as ordinary dense matrices.
CUDA cases exercise the actual TAPP forward/backward when it is installed.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yastn

torch = pytest.importorskip('torch')

from yastn.backend import backend_torch
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
    assert (axes_a, axes_b) in (([1], [0]), ([1], [1]), ([0], [0]))
    assert args[-2] == [0, 1] and len(args[-1]) == 24
    desc_a, desc_b, desc_c = args[:5], args[5:10], args[10:15]
    ma, _ = _descriptor_matrix(a, desc_a)
    mb, _ = _descriptor_matrix(b, desc_b)
    _, regions = _descriptor_matrix(a.new_empty(0), desc_c)
    product = torch.tensordot(ma, mb, dims=(axes_a, axes_b))
    return torch.cat([product[region].reshape(-1) for region in regions])


@pytest.fixture(scope='module')
def oracle_backend():
    """Load the production backend with only its external TAPP imports stubbed.

    A separate module name keeps CPU oracle tests from replacing the real CUDA
    backend. Keeping it in sys.modules lets the cache registry resize/clear it.
    """
    if importlib.util.find_spec('tapp_torch') is not None:
        from yastn.backend import backend_torch_cutensor
        yield backend_torch_cutensor
        return

    name = 'yastn.backend._cutensor_test_oracle'
    path = Path(backend_torch.__file__).with_name('backend_torch_cutensor.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as loaded:
        loaded.setitem(sys.modules, name, module)
        with pytest.MonkeyPatch.context() as imports:
            tapp = ModuleType('tapp_torch')
            tapp.ops = ModuleType('tapp_torch.ops')
            tapp.ops.descriptor_cache_stats = lambda: {}
            tapp.ops.plan_cache_stats = lambda: {}
            imports.setitem(sys.modules, 'tapp_torch', tapp)
            imports.setitem(sys.modules, 'tapp_torch.ops', tapp.ops)
            spec.loader.exec_module(module)
        yield module


@pytest.fixture(params=['oracle', 'cuda'])
def accelerated(request, monkeypatch):
    calls = []
    if request.param == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('CUDA is required for the real cuTENSOR test')
        pytest.importorskip('tapp_torch')
        from yastn.backend import backend_torch_cutensor
        implementation = backend_torch_cutensor
        backend = SimpleNamespace(**vars(implementation))
        operation = backend_torch_cutensor.tensordot_bs_v2
        device = 'cuda'
    else:
        implementation = request.getfixturevalue('oracle_backend')
        backend = SimpleNamespace(**vars(implementation))
        # Exercise production dispatch without requiring the external CUDA op.
        backend.is_cpu_device = lambda device: False
        operation = _dense_descriptor_oracle
        device = 'cpu'

    def counted(*args):
        calls.append(1)
        return operation(*args)

    # Patch the external operation at its production lookup site; packing,
    # contraction dispatch and cache handling all run in the merged backend.
    monkeypatch.setattr(implementation, 'tensordot_bs_v2', counted)
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
    assert len(calls) == 3  # Forward and one cuTENSOR call per input gradient.


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


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('promote', [False, True])
def test_cutensor_chunked_pack(oracle_backend, monkeypatch, dtype, promote, request):
    backend = oracle_backend
    monkeypatch.setattr(backend, '_PACK_CHUNK', 3)
    # Repeated, out-of-order and empty blocks, crossing several chunk boundaries.
    slices = ((4, 9), (0, 0), (1, 5), (4, 9))
    coefficients = (0.5, 1., -2., 3.)
    target_dtype = torch.complex128 if promote else dtype
    data = torch.randn(12, dtype=dtype, device=request.config.getoption('--device'), requires_grad=True)

    def reference(x):
        return torch.cat([x[start:stop].to(target_dtype) * c
                          for (start, stop), c in zip(slices, coefficients)])

    def operation(x):
        return backend._pack(x, slices, coefficients, target_dtype)

    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: saved.append(x) or x, lambda x: x):
        actual = operation(data)
    # Packing is linear: neither source values nor element-sized maps are needed.
    assert actual.numel() == 14
    assert saved == []
    torch.testing.assert_close(actual, reference(data))
    assert torch.autograd.gradcheck(operation, (data,))
    assert torch.autograd.gradgradcheck(operation, (data,))
    weight = torch.randn_like(actual)
    found = torch.autograd.grad(actual, data, weight)[0]
    expected = torch.autograd.grad(reference(data), data, weight)[0]
    torch.testing.assert_close(found, expected)


def test_cutensor_chunked_unpack(oracle_backend, monkeypatch, request):
    backend = oracle_backend
    monkeypatch.setattr(backend, '_PACK_CHUNK', 3)
    data = torch.randn(7, dtype=torch.complex128, device=request.config.getoption('--device'), requires_grad=True)
    parameters = backend._packing_parameters(((2, 6), (9, 12)), data.device)

    def operation(x):
        return backend._Unpack.apply(x, parameters, 14)

    actual = operation(data)
    expected = torch.cat((data.new_zeros(2), data[:4], data.new_zeros(3),
                          data[4:], data.new_zeros(2)))
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(operation, (data,))
    assert torch.autograd.gradgradcheck(operation, (data,))


def test_cutensor_pack_contiguous_alias(oracle_backend, request):
    data = torch.randn(10, device=request.config.getoption('--device'), requires_grad=True)
    packed = oracle_backend._pack(data, ((0, 3), (3, 7)))
    assert packed.data_ptr() == data.data_ptr()
    assert packed.numel() == 7
    torch.testing.assert_close(torch.autograd.grad(packed.sum(), data)[0],
                               data.new_tensor([1.] * 7 + [0.] * 3))


def test_cutensor_pack_mixed_precision_accumulation(oracle_backend, monkeypatch, request):
    monkeypatch.setattr(oracle_backend, '_PACK_CHUNK', 1)
    data = torch.ones(1, dtype=torch.float32, device=request.config.getoption('--device'), requires_grad=True)
    out = oracle_backend._pack(data, ((0, 1),) * 3, dtype=torch.float64)
    weight = out.new_tensor([1e8, 1., -1e8])
    torch.testing.assert_close(torch.autograd.grad(out, data, weight)[0], torch.ones_like(data))


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('dtypes', [('float64', 'float64'), ('complex128', 'complex128'),
                                  ('float32', 'complex128'), ('complex128', 'float32')])
@pytest.mark.parametrize('axes', [((0, 3), (1, 2)), ((2, 3), (3, 2)),
                                  ((0, 1, 2, 3), (1, 0, 3, 2))])
def test_cutensor_recoupled_contraction(accelerated, monkeypatch, sym, dtypes, axes):
    from yastn.tensor import _contractions
    backend, device, calls = accelerated
    config = yastn.make_config(sym=sym, backend=backend, default_device=device)
    charges = (0, 1, 2) if sym == 'SU2' else ((0, 0), (1, 1), (2, 2))
    leg = yastn.Leg(config, t=charges, D=(1, 2, 1))
    a, b = (yastn.rand(config, legs=(leg, leg.conj()) * 2, dtype=dtype)
            for dtype in dtypes)
    a = a.remove_random_blocks(number=a.nblocks // 4, keep_legs=True)
    b = b.remove_random_blocks(number=b.nblocks // 3, keep_legs=True)
    a.requires_grad_()
    b.requires_grad_()
    reference_config = config._replace(backend=backend_torch)
    expected = yastn.tensordot(a._replace(config=reference_config),
                              b._replace(config=reference_config), axes)
    assert expected.size > 0 and expected.norm().item() > 0
    monkeypatch.setattr(_contractions, '_NONABELIAN_RECOUPLE_MIN_SIZE', 0)

    def forbidden(*args, **kwargs):
        pytest.fail('cuTENSOR contraction materialized a full recoupled input')

    monkeypatch.setattr(backend, 'recouple', forbidden)
    monkeypatch.setattr(backend, 'recouple_sequence', forbidden)
    monkeypatch.setattr(backend, 'recouple_dot_sum', forbidden)
    actual = yastn.tensordot(a, b, axes)
    assert len(calls) == 1
    assert (actual.struct, actual.hfs, actual.trans) == (expected.struct, expected.hfs, expected.trans)
    tolerance = 2e-5 if 'float32' in dtypes else 1e-11
    torch.testing.assert_close(actual.data, expected.data, atol=tolerance, rtol=tolerance)
    gradient = torch.randn(actual.size * 2, dtype=actual.data.dtype, device=device)[::2]
    found = torch.autograd.grad(actual.data, (a.data, b.data), gradient)
    reference = torch.autograd.grad(expected.data, (a.data, b.data), gradient)
    for x, y in zip(found, reference):
        torch.testing.assert_close(x, y, atol=tolerance, rtol=tolerance)


def test_cutensor_recoupled_fallback(accelerated, monkeypatch):
    from yastn.backend import _recoupling
    from yastn.tensor import _contractions
    backend, device, calls = accelerated
    config = yastn.make_config(sym='SU2', backend=backend, default_device=device)
    leg = yastn.Leg(config, t=(0, 1, 2), D=(1, 2, 1))
    a = yastn.rand(config, legs=(leg, leg.conj()) * 2)
    b = yastn.rand_like(a)
    axes = ((0, 3), (1, 2))
    expected = yastn.tensordot(a, b, axes)
    assert expected.size > 0 and expected.norm().item() > 0
    monkeypatch.setattr(_contractions, '_NONABELIAN_RECOUPLE_MIN_SIZE', 0)
    monkeypatch.setattr(_recoupling, '_UNFUSE_COMPOSITION_MAX_WORK_RATIO', 0)
    original = backend.tensordot_nonabelian
    declined = []

    def tracked(*args):
        result = original(*args)
        if len(args) > 3:
            assert result is None
            declined.append(True)
        return result

    monkeypatch.setattr(backend, 'tensordot_nonabelian', tracked)
    actual = yastn.tensordot(a, b, axes)
    assert declined == [True] and len(calls) == 2
    torch.testing.assert_close(actual.data, expected.data)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_cutensor_recouple_pack_autograd(oracle_backend, request, dtype):
    from yastn.backend._recoupling import recoupling_sequence_plan
    # Two non-orthogonal basis changes, with permutation and repeated packing.
    first = ((1, 0), ((0, 0, (2, 3), 0.5), (0, 6, (2, 3), -2.),
                       (6, 0, (2, 3), 1.), (6, 6, (2, 3), 0.25)))
    second = ((0, 1), ((0, 0, (3, 2), 2.), (0, 6, (3, 2), -0.5),
                        (6, 0, (3, 2), -1.), (6, 6, (3, 2), 3.)))
    plan = recoupling_sequence_plan((first, second))
    data = torch.randn(24, dtype=dtype, device=request.config.getoption('--device'))[::2].requires_grad_()
    slices, shapes, weights = ((6, 12), (0, 6), (6, 12)), ((3, 2),) * 3, (0.5, -2., 3.)

    def operation(x):
        return oracle_backend._RecouplePack.apply(x, slices, shapes, plan, weights, torch.complex128)

    def reference(x):
        recoupled = backend_torch.recouple_sequence(x.to(torch.complex128), (first, second), 12)
        return torch.cat([recoupled[start:stop] * weight for (start, stop), weight in zip(slices, weights)])

    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: saved.append(x) or x, lambda x: x):
        actual = operation(data)
    assert actual.numel() == 18 and saved == []
    torch.testing.assert_close(actual, reference(data))
    gradient = torch.randn(36, dtype=actual.dtype, device=data.device)[::2]
    torch.testing.assert_close(torch.autograd.grad(actual, data, gradient)[0],
                               torch.autograd.grad(reference(data), data, gradient)[0])
    assert torch.autograd.gradcheck(operation, (data,))
    assert torch.autograd.gradgradcheck(operation, (data,))


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('conjugate', [False, True])
@pytest.mark.parametrize('policy', ['fuse_to_matrix', 'fuse_contracted', 'no_fusion'])
def test_cutensor_recoupled_pending_transpose(accelerated, monkeypatch, sym, conjugate, policy):
    from yastn.backend import _recoupling
    from yastn.tensor import _contractions
    backend, device, calls = accelerated
    config = yastn.make_config(sym=sym, backend=backend, default_device=device, tensordot_policy=policy)
    charges = (0, 1, 2) if sym == 'SU2' else ((0, 0), (1, 1), (2, 2))
    leg = yastn.Leg(config, t=charges, D=(1, 2, 1))
    tensors = []
    for side, order in enumerate(((2, 0, 3, 1), (3, 1, 2, 0))):
        charge = (2 * (1 - side),) if sym == 'SU2' else (2 * (1 - side), 0)
        tensor = yastn.rand(config, legs=(leg, leg.conj()) * 2, n=charge, dtype='complex128')
        storage = tensor.data.new_empty(tensor.size * 2)
        storage[::2] = tensor.data
        tensors.append(tensor._replace(data=storage[::2].detach().requires_grad_()).transpose(order))
    a, b = tensors
    assert not a.data.is_contiguous() and not b.data.is_contiguous()
    axes, conj = ((0, 3), (1, 2)), (conjugate, conjugate)
    reference_config = config._replace(backend=backend_torch)
    expected = yastn.tensordot(a._replace(config=reference_config),
                              b._replace(config=reference_config), axes, conj=conj)
    assert expected.size > 0 and expected.norm().item() > 0
    monkeypatch.setattr(_contractions, '_NONABELIAN_RECOUPLE_MIN_SIZE', 0)
    # Charged outputs repeat many packed blocks and normally trip the cost
    # limit. Force composition here to test the layouts; fallback is separate.
    monkeypatch.setattr(_recoupling, '_UNFUSE_COMPOSITION_MAX_WORK_RATIO', 1000)

    def forbidden(*args, **kwargs):
        pytest.fail('cuTENSOR contraction materialized a full recoupled input')

    monkeypatch.setattr(backend, 'recouple', forbidden)
    monkeypatch.setattr(backend, 'recouple_sequence', forbidden)
    actual = yastn.tensordot(a, b, axes, conj=conj)
    assert len(calls) == 1
    assert (actual.struct, actual.hfs, actual.trans) == (expected.struct, expected.hfs, expected.trans)
    torch.testing.assert_close(actual.data, expected.data)
    gradient = torch.randn_like(actual.data)
    found = torch.autograd.grad(actual.data, (a.data, b.data), gradient)
    reference = torch.autograd.grad(expected.data, (a.data, b.data), gradient)
    for x, y in zip(found, reference):
        torch.testing.assert_close(x, y)


def test_cutensor_recoupled_cpu_fallback(oracle_backend, monkeypatch):
    from yastn.tensor import _contractions
    backend = SimpleNamespace(**vars(oracle_backend))
    config = yastn.make_config(sym='SU2', backend=backend)
    leg = yastn.Leg(config, t=(0, 1, 2), D=(1, 2, 1))
    a = yastn.rand(config, legs=(leg, leg.conj()) * 2)
    b = yastn.rand_like(a)
    axes = ((0, 3), (1, 2))
    expected = yastn.tensordot(a, b, axes)
    assert expected.size > 0 and expected.norm().item() > 0
    monkeypatch.setattr(_contractions, '_NONABELIAN_RECOUPLE_MIN_SIZE', 0)

    def forbidden(*args, **kwargs):
        pytest.fail('CPU contraction called cuTENSOR or materialized a full recoupled input')

    monkeypatch.setattr(backend, 'tensordot_nonabelian', forbidden)
    monkeypatch.setattr(backend, 'recouple', forbidden)
    monkeypatch.setattr(backend, 'recouple_sequence', forbidden)
    actual = yastn.tensordot(a, b, axes)
    torch.testing.assert_close(actual.data, expected.data)


def test_cutensor_recouple_pack_allocation(oracle_backend):
    from yastn.backend._recoupling import recoupling_sequence_plan
    from yastn.tensor._merging import _meta_fusion_tree_transpose
    from yastn.tensor._contractions_cutensor import _meta_tensordot_nonabelian_cutensor
    config = yastn.make_config(sym='SU2', backend='torch')
    leg = yastn.Leg(config, t=(0, 1, 2), D=(3, 3, 1))  # D=12.
    a = yastn.rand(config, legs=(leg, leg.conj()) * 2)
    steps = _meta_fusion_tree_transpose(config.sym, a.struct, (0, 3, 1, 2))
    recoupling = recoupling_sequence_plan(tuple((step[1], step[2]) for step in steps))
    plan = _meta_tensordot_nonabelian_cutensor(
        config.sym, steps[-1][0], a.struct, (0, 1), (2, 3), (0, 1), (2, 3), 0.5)
    assert plan.coefficients and plan.slices_a
    with torch.profiler.profile(profile_memory=True) as profile:
        packed = oracle_backend._RecouplePack.apply(
            a.data, plan.slices_a, plan.shapes_a, recoupling, plan.coefficients, a.data.dtype)
    allocations = [event.self_cpu_memory_usage for event in profile.events()
                   if event.self_cpu_memory_usage > 0]
    assert packed.numel() > 0 and allocations
    # The packed output is the only tensor allocation; source blocks are views.
    assert sum(allocations) == packed.numel() * packed.element_size()

def test_pack_budget_divisor_env(monkeypatch):
    """Invalid or nonpositive divisors fall back to the documented default 8."""
    pytest.importorskip('tapp_torch')
    from yastn.backend.backend_torch_cutensor import _pack_budget_divisor
    monkeypatch.delenv('YASTN_CUTENSOR_PACK_DIVISOR', raising=False)
    assert _pack_budget_divisor() == 8
    monkeypatch.setenv('YASTN_CUTENSOR_PACK_DIVISOR', '4')
    assert _pack_budget_divisor() == 4
    monkeypatch.setenv('YASTN_CUTENSOR_PACK_DIVISOR', '0')
    assert _pack_budget_divisor() == 8
