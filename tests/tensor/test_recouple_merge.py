"""Single-group hard fusion without full recoupling intermediates."""
import pytest
import yastn

torch = pytest.importorskip('torch')
from yastn.backend import backend_torch
from yastn.tensor import _merging as merging


@pytest.fixture(autouse=True)
def exercise_streamed_fusion(monkeypatch):
    # Exercise the memory-saving path with small independent numerical oracles.
    monkeypatch.setattr(merging, '_NONABELIAN_FUSION_MIN_SIZE', 0)


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('axes', [(2, (0, 3), 1), ((3, 0), 1, 2),
                                  (3, (0, 1), 2), (0, (3, 2, 1)),
                                  ((0, 2), (1, 3)), ((2, 1), (3, 0))])
def test_single_group_fusion(config_kwargs, sym, dtype, masked, axes, monkeypatch):
    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym=sym)
    charges = (0, 1, 2) if sym == 'SU2' else ((0, 0), (1, 1), (2, 2))
    leg = yastn.Leg(config, s=1, t=charges, D=(1, 2, 1))
    a = yastn.rand(config, legs=(leg, leg.conj(), leg, leg.conj()), dtype=dtype)
    if masked:
        a = a.remove_random_blocks(number=a.nblocks // 3, keep_legs=True)
    a = a.transpose((1, 3, 0, 2))  # Include pending native-axis permutations.
    a.requires_grad_()
    original = a.data.detach().clone()
    with monkeypatch.context() as patch:
        patch.setattr(backend_torch, 'recouple_and_merge', None)
        patch.setattr(backend_torch, 'recouple_and_unfuse', None)
        expected = a.fuse_legs(axes, mode='hard')

    def forbidden(*args, **kwargs):
        raise AssertionError('Single-group fusion allocated a full intermediate')

    monkeypatch.setattr(backend_torch, 'recouple', forbidden)
    monkeypatch.setattr(backend_torch, 'recouple_sequence', forbidden)
    monkeypatch.setattr(backend_torch, 'transpose_and_merge_scaled', forbidden)
    found = a.fuse_legs(axes, mode='hard')
    assert found.size > 0 and found.nblocks > 1
    assert found.struct == expected.struct and found.hfs == expected.hfs
    torch.testing.assert_close(found.data, expected.data, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(a.data, original, atol=0, rtol=0)
    gradient = torch.randn(found.size * 2, dtype=found.data.dtype, device=found.data.device)[::2]
    actual_grad, = torch.autograd.grad(found.data, a.data, gradient)
    reference_grad, = torch.autograd.grad(expected.data, a.data, gradient)
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
@pytest.mark.parametrize('axes', [(2, (0, 3), 1), ((0, 3), (1, 2))])
def test_single_group_higher_derivatives(config_kwargs, dtype, axes):
    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym='SU2')
    leg = yastn.Leg(config, s=1, t=(1,), D=(2,))
    a = yastn.rand(config, legs=(leg, leg, leg.conj(), leg.conj()), dtype=dtype)
    a.requires_grad_()

    def operation(data):
        return a._replace(data=data).fuse_legs(axes, mode='hard').data

    assert torch.autograd.gradcheck(operation, (a.data,), fast_mode=True)
    assert torch.autograd.gradgradcheck(operation, (a.data,), fast_mode=True)


@pytest.mark.parametrize('operation', ['fuse', 'unfuse', 'fuse_multiple'])
def test_single_group_only_allocates_output(operation):
    """D=12 physical legs; profile actual tensor allocations, not an empty plan."""
    config = yastn.make_config(backend='torch', sym='SU2')
    leg = yastn.Leg(config, s=1, t=(0, 1, 2), D=(3, 3, 1))
    a = yastn.rand(config, legs=(leg, leg.conj(), leg, leg.conj()), dtype='float64')
    if operation == 'unfuse':
        a = a.fuse_legs((1, (0, 3), 2), mode='hard')
        run = lambda: a.unfuse_legs(1)
    elif operation == 'fuse_multiple':
        run = lambda: a.fuse_legs(((0, 3), (1, 2)), mode='hard')
    else:
        run = lambda: a.fuse_legs((2, (0, 3), 1), mode='hard')
    # Warm structural caches before measuring tensor allocations.
    warm = run()
    assert warm.size > 0
    del warm
    with torch.profiler.profile(profile_memory=True) as profile:
        found = run()
    allocations = [event for event in profile.events()
                   if event.self_cpu_memory_usage > 0]
    assert allocations
    assert sum(event.self_cpu_memory_usage for event in allocations) == found.size * 8


def test_single_group_raw_recoupling_steps(config_kwargs, monkeypatch):
    """The bounded composition fallback must work before and after the merge."""
    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym='SU2')
    leg = yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 1, 1))
    a = yastn.rand(config, legs=(leg, leg.conj(), leg, leg.conj(), leg, leg.conj()),
                   dtype='complex128')
    axes = (5, (3, 0), 4, 2, 1)
    with monkeypatch.context() as patch:
        patch.setattr(backend_torch, 'recouple_and_merge', None)
        patch.setattr(backend_torch, 'recouple_and_unfuse', None)
        expected = a.fuse_legs(axes, mode='hard')
    calls = []

    def raw_plans(sym, struct, order):
        current, plans = list(range(len(order))), []
        for destination, leaf in enumerate(order):
            source = current.index(leaf)
            while source > destination:
                plan = merging._meta_swap_adjacent_fusion_leaves(sym, struct, source - 1)
                plans.append(plan)
                struct = plan[0]
                current[source - 1], current[source] = current[source], current[source - 1]
                source -= 1
        calls.append(len(plans))
        return tuple(plans)

    monkeypatch.setattr(merging, '_meta_fusion_tree_transpose', raw_plans)
    found = a.fuse_legs(axes, mode='hard')
    assert len(calls) == 2 and all(count > 1 for count in calls)
    assert found.size > 0 and found.struct == expected.struct
    torch.testing.assert_close(found.data, expected.data, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('axes', [(2, (0, 4), 1, 3), (1, (4, 2, 0), 3)])
def test_single_group_charged_fusion(config_kwargs, axes, monkeypatch):
    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym='SU2')
    leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
    a = yastn.rand(config, legs=tuple(leg if i % 2 else leg.conj() for i in range(5)),
                   n=(1,), dtype='complex128')
    a.requires_grad_()
    with monkeypatch.context() as patch:
        patch.setattr(backend_torch, 'recouple_and_merge', None)
        patch.setattr(backend_torch, 'recouple_and_unfuse', None)
        expected = a.fuse_legs(axes, mode='hard')
    found = a.fuse_legs(axes, mode='hard')
    assert found.size > 0 and found.struct == expected.struct and found.hfs == expected.hfs
    torch.testing.assert_close(found.data, expected.data, atol=1e-12, rtol=1e-12)
    gradient = torch.randn(found.size * 2, dtype=found.data.dtype, device=found.data.device)[::2]
    actual_grad, = torch.autograd.grad(found.data, a.data, gradient)
    reference_grad, = torch.autograd.grad(expected.data, a.data, gradient)
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-12, rtol=1e-12)


def test_recouple_merge_empty_metadata(config_kwargs):
    data = torch.randn(3, dtype=torch.float64, device=config_kwargs['default_device'], requires_grad=True)
    result = backend_torch.recouple_and_merge(data, (), (), (), 5, (1.,), (3,))
    torch.testing.assert_close(result, data.new_zeros(5))
    gradient, = torch.autograd.grad(result.sum(), data, create_graph=True)
    torch.testing.assert_close(gradient, data.new_zeros(3))


def test_multiple_unfuse_only_allocates_output():
    config = yastn.make_config(backend='torch', sym='SU2')
    leg = yastn.Leg(config, s=1, t=(0, 1, 2), D=(3, 3, 1))  # D=12.
    a = yastn.rand(config, legs=(leg, leg.conj(), leg, leg.conj()), dtype='complex128')
    a = a.fuse_legs(((0, 3), (1, 2)), mode='hard')
    warm = a.unfuse_legs((0, 1))
    assert warm.size > 0 and warm.nblocks > 1
    del warm
    with torch.profiler.profile(profile_memory=True) as profile:
        found = a.unfuse_legs((0, 1))
    allocations = [event.self_cpu_memory_usage for event in profile.events()
                   if event.self_cpu_memory_usage > 0]
    assert allocations and sum(allocations) == found.size * 16


def test_multiple_unfuse_composition_budget(config_kwargs, monkeypatch):
    from yastn.backend import _recoupling
    # Eight fully connected channels over three stages have 4096 paths,
    # versus 192 contributions when evaluated sequentially.
    records = tuple((destination, source, (1,), 1.)
                    for destination in range(8) for source in range(8))
    steps = (('recouple', (0,), records),) * 3
    outputs = tuple((i, (1,)) for i in range(8))
    assert _recoupling.unfuse_sequence_plan(steps, outputs) is None

    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym='SU2')
    leg = yastn.Leg(config, t=(0, 1), D=(1, 2))
    a = yastn.rand(config, legs=(leg, leg.conj()) * 2, dtype='complex128')
    a = a.fuse_legs(((0, 3), (1, 2)), mode='hard')
    expected = a.unfuse_legs((0, 1))
    monkeypatch.setattr(_recoupling, '_UNFUSE_COMPOSITION_MAX_WORK_RATIO', 0)
    found = a.unfuse_legs((0, 1))
    assert found.size > 0 and found.struct == expected.struct
    torch.testing.assert_close(found.data, expected.data, atol=1e-12, rtol=1e-12)


def test_multiple_unfuse_charged_noncontiguous(config_kwargs, monkeypatch):
    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym='SU2')
    leg = yastn.Leg(config, t=(1,), D=(2,))
    a = yastn.rand(config, legs=(leg, leg.conj(), leg, leg.conj(), leg), n=(1,), dtype='complex128')
    a = a.fuse_legs(((0, 3), 2, (4, 1)), mode='hard')
    storage = torch.empty(a.size * 2, dtype=a.data.dtype, device=a.data.device)
    storage[1::2] = a.data
    a = a._replace(data=storage[1::2].requires_grad_())
    with monkeypatch.context() as patch:
        patch.setattr(backend_torch, 'recouple_and_unfuse', None)
        expected = a.unfuse_legs((0, 1, 2))
    found = a.unfuse_legs((0, 1, 2))
    assert found.size > 0 and found.struct == expected.struct and found.hfs == expected.hfs
    torch.testing.assert_close(found.data, expected.data, atol=1e-12, rtol=1e-12)
    operation = lambda data: a._replace(data=data).unfuse_legs((0, 1, 2)).data
    assert torch.autograd.gradcheck(operation, (a.data,), fast_mode=True)
    assert torch.autograd.gradgradcheck(operation, (a.data,), fast_mode=True)


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
@pytest.mark.parametrize('axes', [1, (0, 2), (0, 1, 2)])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('groups', [((0, 3), (2, 5), (4, 1)), ((0, 3, 5), (2, 4), 1)])
def test_unfuse_without_recoupling_buffers(config_kwargs, sym, dtype, axes, masked, groups, monkeypatch):
    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym=sym)
    charges = (0, 1) if sym == 'SU2' else ((0, 0), (1, 1))
    leg = yastn.Leg(config, s=1, t=charges, D=(1, 2))
    a = yastn.rand(config, legs=(leg, leg.conj()) * 3, dtype=dtype)
    a = a.fuse_legs(groups, mode='hard')
    if masked:
        a = a.remove_random_blocks(number=a.nblocks // 3, keep_legs=True)
    a.requires_grad_()
    original = a.data.detach().clone()
    with monkeypatch.context() as patch:
        patch.setattr(backend_torch, 'recouple_and_merge', None)
        patch.setattr(backend_torch, 'recouple_and_unfuse', None)
        expected = a.unfuse_legs(axes)

    def forbidden(*args, **kwargs):
        raise AssertionError('Unfusion allocated a full recoupling intermediate')

    monkeypatch.setattr(backend_torch, 'recouple', forbidden)
    monkeypatch.setattr(backend_torch, 'recouple_sequence', forbidden)
    found = a.unfuse_legs(axes)
    assert found.size > 0 and found.struct == expected.struct and found.hfs == expected.hfs
    torch.testing.assert_close(found.data, expected.data, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(a.data, original, atol=0, rtol=0)
    gradient = torch.randn(found.size * 2, dtype=found.data.dtype, device=found.data.device)[::2]
    actual_grad, = torch.autograd.grad(found.data, a.data, gradient)
    reference_grad, = torch.autograd.grad(expected.data, a.data, gradient)
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-12, rtol=1e-12)
    operation = lambda data: a._replace(data=data).unfuse_legs(axes).data
    assert torch.autograd.gradgradcheck(operation, (a.data,), fast_mode=True)
