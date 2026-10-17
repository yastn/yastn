"""Streaming retains cross terms and the existing NN/NNN update."""
import pytest
import yastn
from yastn.tn import fpeps
from yastn.tn.fpeps._evolution import symmetrized_svd


def _state(config_kwargs, sym, prepare=False):
    if sym in ('U1', 'SU2xU1'):
        ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs) if sym == 'U1' else (
            yastn.operators.SpinfulFermions(sym=sym, **config_kwargs))
        if sym == 'U1':
            gate = fpeps.gates.gate_nn_hopping(1., (0.13 + 0.05j), ops.I(), ops.c(), ops.cp())
        else:
            from ._multiplet_models import hubbard_hamiltonian
            gate = fpeps.gates.gate_nn_exp((0.13 + 0.05j), ops.I(), hubbard_hamiltonian(ops, 0.7, 3.2, -0.15))
    else:
        ops = yastn.operators.Spin12(sym=sym, **config_kwargs)
        gate = (fpeps.gates.gate_nn_Heisenberg(1., (0.13 + 0.05j), ops.I(), ops.S()) if sym == 'SU2'
                else fpeps.gates.gate_nn_Heisenberg(1., (0.13 + 0.05j), ops.I(), ops.sz(), ops.sp(), ops.sm()))
    psi = fpeps.product_peps(fpeps.SquareLattice((2, 2), boundary='obc'), ops.I())
    gates = fpeps.gates.distribute(psi, gates_nn=gate, symmetrize=False)
    if prepare:
        fpeps.evolution_step_(fpeps.EnvBP(psi), gates, opts_svd={'D_total': 2})
    else:
        for gg in gates:
            psi.apply_gate_(gg)
    return psi, gate


def _qr(psi, vertical):
    s0, s1 = (0, 0), ((1, 0) if vertical else (0, 1))
    if vertical:
        Q0, R0 = psi[s0].qr(axes=((0, 1, 3, 4), 2), sQ=1, Qaxis=2)
        Q1, R1 = psi[s1].qr(axes=((1, 2, 3, 4), 0), sQ=-1, Qaxis=0, Raxis=-1)
    else:
        Q0, R0 = psi[s0].qr(axes=((0, 1, 2, 4), 3), sQ=-1, Qaxis=3)
        Q1, R1 = psi[s1].qr(axes=((0, 2, 3, 4), 1), sQ=1, Qaxis=1, Raxis=-1)
    return Q0, Q1, R0, R1, s0, s1, 'tb' if vertical else 'lr'


@pytest.mark.parametrize('which', ['NN', 'NN+', 'NN++', 'NNN', 'NNN+', 'NNN++', 'Ladder2'])
@pytest.mark.parametrize('vertical', [False, True])
@pytest.mark.parametrize('sym', ['dense', 'U1', 'SU2'])
def test_streamed_metric(config_kwargs, which, vertical, sym, request):
    if sym == 'SU2' and which == 'Ladder2':
        request.node.add_marker(pytest.mark.xfail(
            strict=True, raises=yastn.YastnError,
            reason='Existing full SU2 Ladder2 metric has incompatible QR signatures.'))
    psi, _ = _state(config_kwargs, sym)
    Q0, Q1, R0, R1, s0, s1, dirn = _qr(psi, vertical)
    full = fpeps.EnvNTU(psi, which=which).bond_metric(Q0, Q1, s0, s1, dirn).g
    metric = fpeps.EnvNTU(psi, which=which, metric_chunk_size=1).bond_metric(Q0, Q1, s0, s1, dirn).g
    assert len(metric.projectors) > 1
    g = (full + full.H) / 2
    x = (R0 @ R1).fuse_legs(axes=[(0, 1)])
    assert (g @ x).norm().item() > 0
    assert (metric @ x - g @ x).norm().item() < 1e-10 * (g @ x).norm().item()
    noise = ((full - full.H).norm() / (2 * g.norm())).item()
    assert metric.nonhermitian_part == pytest.approx(noise, abs=1e-12)

    M0, M1 = symmetrized_svd(R0, R1, {'D_total': 2})
    gf = g.unfuse_legs(axes=0)
    g0 = yastn.tensordot(M1.conj(), gf, axes=(1, 1)).fuse_legs(axes=((1, 0), 2))
    g0 = g0.unfuse_legs(axes=1)
    g0 = yastn.tensordot(g0, M1, axes=(2, 1)).fuse_legs(axes=(0, (1, 2)))
    g1 = yastn.tensordot(M0.conj(), gf, axes=(0, 0)).fuse_legs(axes=((0, 1), 2))
    g1 = g1.unfuse_legs(axes=1)
    g1 = yastn.tensordot(g1, M0, axes=(1, 0)).fuse_legs(axes=(0, (2, 1)))
    assert (metric.normal_matrix(M1, 0) - g0).norm().item() < 1e-10 * g0.norm().item()
    assert (metric.normal_matrix(M0, 1) - g1).norm().item() < 1e-10 * g1.norm().item()

    blocks, cross, largest = 0, 0., 0
    for row, col, block in metric.blocks():
        blocks += 1
        largest = max(largest, block.size)
        if row is not col:
            cross += block.norm().item()
    assert blocks == len(metric.projectors) ** 2
    assert cross > 1e-12
    assert largest < full.size


@pytest.mark.parametrize('sym', ['dense', 'U1', 'SU2', 'SU2xU1'])
@pytest.mark.parametrize('initialization', ['SVD', 'EAT', 'EAT_SVD'])
@pytest.mark.parametrize('path', [((0, 0), (0, 1)),
                                  ((0, 0), (0, 1), (1, 1)),
                                  ((1, 1), (0, 1), (0, 0))])
def test_streamed_gate_evolution(config_kwargs, sym, initialization, path):
    psi, gate = _state(config_kwargs, sym, prepare=True)
    other = psi.clone()
    gate = fpeps.Gate(G=gate.G, sites=path)
    kwargs = dict(opts_svd={'D_total': 2}, fix_metric=None,
                  initialization=initialization, max_iter=2, pinv_cutoffs=(1e-10,))
    full = fpeps.evolution_step_(fpeps.EnvNTU(psi), [gate], **kwargs)
    streamed = fpeps.evolution_step_(fpeps.EnvNTU(other, metric_chunk_size=2), [gate], **kwargs)
    assert len(full) == len(streamed) == len(path) - 1
    for a, b in zip(full, streamed):
        assert a.truncation_error == pytest.approx(b.truncation_error, abs=1e-9)
        assert a.truncation_errors.keys() == b.truncation_errors.keys()
        assert b.iterations and all(count > 0 for count in b.iterations.values())
        if 'EAT' in initialization:
            assert b.eat_metric_error == pytest.approx(a.eat_metric_error, abs=1e-7)
            assert 'eat_opt' in b.truncation_errors
        assert b.min_eigenvalue is None
    # Compare the full physical states, insensitive to internal bond gauges.
    dense = psi.to_tensor()
    streamed = other.to_tensor()
    overlap = abs(yastn.vdot(dense, streamed).item()) / (dense.norm().item() * streamed.norm().item())
    assert overlap == pytest.approx(1., abs=1e-8)


def test_streamed_options_do_not_modify_state(config_kwargs):
    psi, gate = _state(config_kwargs, 'dense')
    before = psi.clone()
    env = fpeps.EnvNTU(psi, metric_chunk_size=1)
    for kwargs, message in [({}, 'fix_metric=None'),
                            ({'fix_metric': None, 'initialization': 'ZMT1eat'}, 'initialization='),
                            ({'fix_metric': None, 'initialization': 'SVD', 'max_iter': 0}, 'max_iter')]:
        with pytest.raises(yastn.YastnError, match=message):
            fpeps.evolution_step_(env, [fpeps.Gate(gate.G, ((0, 0), (0, 1)))],
                                  opts_svd={'D_total': 2}, **kwargs)
        assert all((psi[s] - before[s]).norm().item() == 0 for s in psi.sites())
    for size in (0, -1, 1.5, True):
        with pytest.raises(yastn.YastnError, match='positive integer'):
            fpeps.EnvNTU(psi, metric_chunk_size=size)


@pytest.mark.parametrize('size', [3, 100])
def test_streamed_partial_and_single_chunk(config_kwargs, size):
    psi, _ = _state(config_kwargs, 'dense')
    Q0, Q1, R0, R1, s0, s1, dirn = _qr(psi, False)
    full = fpeps.EnvNTU(psi).bond_metric(Q0, Q1, s0, s1, dirn).g
    metric = fpeps.EnvNTU(psi, metric_chunk_size=size).bond_metric(Q0, Q1, s0, s1, dirn).g
    x = (R0 @ R1).fuse_legs(axes=[(0, 1)])
    assert len(metric.projectors) == (2 if size == 3 else 1)
    assert (metric @ x - ((full + full.H) / 2) @ x).norm().item() < 1e-12
    info = fpeps.truncate_(fpeps.EnvNTU(psi, metric_chunk_size=size),
                          opts_svd={'D_total': 2}, bond=(s0, s1),
                          initialization='SVD_ONLY', fix_metric=None, max_iter=0)
    assert info.best_method == 'svd'
    assert info.truncation_error > 0


@pytest.mark.parametrize('sym', ['dense', 'U1', 'SU2', 'SU2xU1'])
@pytest.mark.parametrize('vertical', [False, True])
@pytest.mark.parametrize('which', ['NN', 'NNN'])
def test_streamed_eat_factors(config_kwargs, sym, vertical, which, monkeypatch):
    psi, _ = _state(config_kwargs, sym)
    psi.config.backend.random_seed(seed=14)
    for site in psi.sites():
        psi[site] = yastn.rand_like(psi[site], dtype='complex128')
        psi[site] = psi[site] / psi[site].norm()
    Q0, Q1, _, _, s0, s1, dirn = _qr(psi, vertical)
    full = fpeps.EnvNTU(psi, which=which).bond_metric(Q0, Q1, s0, s1, dirn).g
    metric = fpeps.EnvNTU(psi, which=which, metric_chunk_size=1).bond_metric(Q0, Q1, s0, s1, dirn).g
    assert len(metric.projectors) > 1
    g = ((full + full.H) / 2).unfuse_legs(axes=(0, 1))
    u, s, v = g.svd_with_truncation(axes=((0, 2), (3, 1)), D_total=1)
    u, v = u.remove_leg(2), v.remove_leg(0)
    # Every contraction requested by EAT must remain a proper sub-block.
    raw_block = metric._raw_block
    calls = []
    def checked_block(row, col):
        block = raw_block(row, col)
        calls.append(block.size)
        assert block.size < full.size
        return block
    monkeypatch.setattr(metric, '_raw_block', checked_block)
    left, right, error = metric.eat_factors()
    normalize = lambda t: t / t.trace().to_number()
    assert (normalize(left) - normalize(u)).norm().item() < 1e-8
    assert (normalize(right) - normalize(v)).norm().item() < 1e-8
    expected = max(0., 1 - (s.norm() / g.norm()).item() ** 2) ** .5
    assert error == pytest.approx(expected, abs=1e-8)
    assert metric.metric_norm == pytest.approx(g.norm().item(), rel=1e-12)
    # Check both singular equations against the full reshuffled reference.
    ar = yastn.tensordot(g, right, axes=((1, 3), (1, 0)), conj=(0, 1))
    al = yastn.tensordot(left, g, axes=((0, 1), (0, 2)), conj=(1, 0)).T
    sigma = ar.norm().item()
    assert (ar - sigma * left).norm().item() / sigma < 1e-9
    assert (al - sigma * right).norm().item() / sigma < 1e-9
    assert calls
    count = len(calls)
    assert metric.eat_factors() is metric._eat_factors
    assert len(calls) == count  # Reuse factors for staged bond truncations.


def test_streamed_eat_failure_and_options(config_kwargs):
    psi, _ = _state(config_kwargs, 'dense')
    psi.config.backend.random_seed(seed=14)
    for site in psi.sites():
        psi[site] = yastn.rand_like(psi[site], dtype='complex128')
    Q0, Q1, _, _, s0, s1, dirn = _qr(psi, False)
    env = fpeps.EnvNTU(psi, metric_chunk_size=1, opts_eat={'max_iter': 1, 'tol': 1e-14})
    metric = env.bond_metric(Q0, Q1, s0, s1, dirn).g
    with pytest.raises(yastn.YastnError, match='did not converge.*relative residual='):
        metric.eat_factors()
    assert metric._eat_factors is None
    for opts in ({'tol': 0}, {'tol': float('nan')}, {'tol': float('inf')},
                 {'max_iter': 0}, {'max_iter': True}, {'max_iter': 1.5}, {'unknown': 1}):
        with pytest.raises(yastn.YastnError, match='opts_eat'):
            fpeps.EnvNTU(psi, metric_chunk_size=1, opts_eat=opts)


@pytest.mark.parametrize('initialization', [None, 'SVD_EAT'])
def test_streamed_eat_default_and_alias(config_kwargs, initialization):
    psi, gate = _state(config_kwargs, 'dense', prepare=True)
    env = fpeps.EnvNTU(psi, metric_chunk_size=1)
    kwargs = {} if initialization is None else {'initialization': initialization}
    infos = fpeps.evolution_step_(
        env, [fpeps.Gate(gate.G, ((0, 0), (0, 1)))],
        opts_svd=[{'D_total': 3}, {'D_total': 2}], fix_metric=None,
        max_iter=1, pinv_cutoffs=(1e-10,), **kwargs)
    assert len(infos) == 1
    assert set(infos[0].truncation_errors) == {'eat', 'eat_opt', 'svd', 'svd_opt'}
    assert 0 <= infos[0].eat_metric_error <= 1
