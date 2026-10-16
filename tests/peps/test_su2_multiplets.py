"""Irreducible operator wires, compared with magnetic/Fock-space matrices."""
import numpy as np
import pytest
from scipy.linalg import expm
import yastn
from yastn.tn import mps, fpeps
from yastn.tn.fpeps.envs.rdm import measure_rdm_nn
from .test_su2_models_ed import _expand_magnetic


def _matrix(mpo, I=None):
    tensor = mpo.to_tensor()
    if I is not None:
        tensor = tensor + 0 * mps.product_mpo(I, N=mpo.N).to_tensor()
    a = _expand_magnetic(tensor)
    order = tuple(range(0, a.ndim, 2)) + tuple(range(1, a.ndim, 2))
    d = np.prod(a.shape[::2])
    return a.transpose(order).reshape(d, d)


def _local(model):
    d = 4 if model == 'Hubbard' else 3
    cu, cd = np.zeros((d, d)), np.zeros((d, d))
    cu[0, -1], cd[0, -2] = 1, 1
    if d == 4:
        cu[-2, 1], cd[-1, 1] = 1, -1
    sp, sz = cu.T @ cd, (cu.T @ cu - cd.T @ cd) / 2
    return cu, cd, ((sp + sp.T) / 2, (sp - sp.T) / 2j, sz)


def _embed(op, site, N, odd=False, order=None):
    d = len(op)
    parity = np.diag([1] * (d - 2) + [-1, -1])
    order = list(range(N)) if order is None else order
    result = np.ones((1, 1))
    for k in range(N):
        local = op if k == site else parity if odd and order[k] < order[site] else np.eye(d)
        result = np.kron(result, local)
    return result


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('model', ['Hubbard', 'tJ'])
def test_multiplet_mpo(config_kwargs, sym, model):
    factory = yastn.operators.SpinfulFermions if model == 'Hubbard' else yastn.operators.SpinfulFermions_tJ
    ops = factory(sym=sym, **config_kwargs)
    cu, cd, spin = _local(model)
    assert ops.cp_spinor().s == (1, -1, -1)
    assert ops.c_spinor().s == (1, -1, 1)
    N = 3
    for order in (None, (2, 0, 1)):
        c = [[_embed(x, i, N, True, order) for x in (cu, cd)] for i in range(N)]
        for i, j in ((0, 2), (2, 0), (1, 1)):
            for reverse in (False, True):
                pair = (ops.c_spinor(), ops.cp_spinor()) if reverse else (ops.cp_spinor(), ops.c_spinor())
                H = mps.generate_mpo(ops.I(), [mps.Hterm(1, (i, j), pair)], N=N, f_map=order)
                exact = sum(c[i][s] @ c[j][s].T if reverse else c[i][s].T @ c[j][s] for s in (0, 1))
                assert np.allclose(_matrix(H, ops.I()), exact, atol=2e-12)
        terms = [mps.Hterm(.7, (0, 2), (ops.cp_spinor(), ops.c_spinor())),
                 mps.Hterm(.7, (2, 0), (ops.cp_spinor(), ops.c_spinor())),
                 mps.Hterm(.3, (1,), (ops.n_total(),))]
        H = mps.generate_mpo(ops.I(), terms, N=N, f_map=order)
        exact = .7 * sum(c[0][s].T @ c[2][s] + c[2][s].T @ c[0][s] for s in (0, 1))
        exact += .3 * sum(x.T @ x for x in c[1])
        assert np.allclose(_matrix(H, ops.I()), exact, atol=3e-12)
    SS = mps.generate_mpo(ops.I(), [mps.Hterm(1, (0, 2), (ops.S(), ops.S()))], N=N)
    exact = sum(_embed(s, 0, N) @ _embed(s, 2, N) for s in spin)
    assert np.allclose(_matrix(SS, ops.I()), exact, atol=2e-12)
    with pytest.raises(yastn.YastnError, match='pairs'):
        mps.generate_mpo(ops.I(), [mps.Hterm(1, (0,), (ops.S(),))], N=N)


@pytest.mark.parametrize('spin', [.5, 1])
def test_spin_multiplet_gate(config_kwargs, spin):
    ops = (yastn.operators.Spin12 if spin == .5 else yastn.operators.Spin1)(sym='SU2', **config_kwargs)
    H = mps.generate_mpo(ops.I(), [mps.Hterm(1.2, (0, 1), (ops.S(), ops.S()))], N=2)
    gate = fpeps.gates.gate_nn_exp(.13, ops.I(), H)
    direct = fpeps.gates.gate_nn_Heisenberg(1.2, .13, ops.I(), ops.S())
    def full(g):
        return yastn.ncon(g.G, [(0, -1, 1), (-2, -3, 1)])
    assert (full(gate) - full(direct)).norm() < 2e-12
    magnetic = np.arange(-spin, spin + 1)
    raising = np.diag(np.sqrt(spin * (spin + 1) - magnetic[:-1] * (magnetic[:-1] + 1)), -1)
    components = ((raising + raising.T) / 2, (raising - raising.T) / 2j, np.diag(magnetic))
    expected = expm(-.13 * 1.2 * sum(np.kron(s, s) for s in components))
    actual = _expand_magnetic(full(direct)).transpose(0, 2, 1, 3).reshape(expected.shape)
    assert np.allclose(actual, expected, atol=2e-12, rtol=0)
    # Pairing follows operator order, including overlapping pairs/same sites.
    terms = [mps.Hterm(1, (0, 1, 1, 2), (ops.S(),) * 4)]
    product = mps.generate_mpo(ops.I(), terms, N=3)
    a = mps.generate_mpo(ops.I(), [mps.Hterm(1, (0, 1), (ops.S(),) * 2)], N=3)
    b = mps.generate_mpo(ops.I(), [mps.Hterm(1, (1, 2), (ops.S(),) * 2)], N=3)
    assert np.allclose(_matrix(product), _matrix(a) @ _matrix(b), atol=2e-12)


@pytest.mark.parametrize('spin', [.5, 1])
def test_multiplet_mps_singlet(config_kwargs, spin):
    ops = (yastn.operators.Spin12 if spin == .5 else yastn.operators.Spin1)(sym='SU2', **config_kwargs)
    psi = mps.random_mps(mps.product_mpo(ops.I(), N=2), n=0, D_total=8)
    norm = mps.vdot(psi, psi)
    casimir = spin * (spin + 1)
    pairs = mps.measure_2site(psi, ops.S(), ops.S(), psi, bonds='a')
    for (i, j), value in pairs.items():
        assert (value / norm).item() == pytest.approx(casimir if i == j else -casimir, abs=2e-12)
    assert (mps.measure_2site(psi, {1: ops.S()}, {0: ops.S()}, psi, bonds=(1, 0)) / norm).item() == pytest.approx(-casimir)
    value = mps.measure_nsite(psi, *(ops.S(),) * 4, ket=psi, sites=(0, 1, 1, 0))
    assert (value / norm).item() == pytest.approx(casimir ** 2, abs=2e-12)


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('dims', [(1, 2), (2, 1)])
def test_multiplet_boundary_and_bp(config_kwargs, sym, dims):
    ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
    psi = fpeps.product_peps(fpeps.SquareLattice(dims, boundary='obc'), ops.I())
    sites = psi.sites()
    gate = fpeps.gates.gate_nn_hopping(.7, .13, ops.I(), ops.c_spinor(), ops.cp_spinor())
    psi.apply_gate_(fpeps.Gate(gate.G, sites))
    original = {s: psi[s].copy() for s in sites}
    cu, cd, spin = _local('Hubbard')
    c = [[_embed(x, i, 2, True) for x in (cu, cd)] for i in range(2)]
    hopping = sum(c[0][s].T @ c[1][s] for s in (0, 1))
    rho = expm(2 * .7 * .13 * (hopping + hopping.T))
    rho /= np.trace(rho)
    cases = [((ops.cp_spinor(), ops.c_spinor()), hopping),
             ((ops.c_spinor(), ops.cp_spinor()), -hopping.T),
             ((ops.S(), ops.S()), sum(_embed(s, 0, 2) @ _embed(s, 1, 2) for s in spin))]
    boundary = fpeps.EnvBoundaryMPS(psi, opts_svd={'D_total': 64}, setup='lrbt')
    bp = fpeps.EnvBP(psi)
    bond = next(iter(psi.bonds()))
    for pair, matrix in cases:
        expected = np.trace(rho @ matrix)
        assert boundary.measure_nn(*pair)[bond].item() == pytest.approx(expected, abs=3e-12)
        assert boundary.measure_nn({bond: {('pair',): pair}})[bond + ('pair',)].item() == pytest.approx(expected, abs=3e-12)
        assert bp.measure_nn(*pair)[bond].item() == pytest.approx(expected, abs=3e-12)
    for site in sites:
        assert (psi[site] - original[site]).norm() == 0


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('direction', ['h', 'v'])
def test_multiplet_measure_pair(config_kwargs, monkeypatch, sym, direction):
    ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
    geometry = fpeps.SquareLattice((1, 3) if direction == 'h' else (3, 1), boundary='obc')
    sites = geometry.sites()
    psi = fpeps.product_peps(geometry, ops.I())
    gate = fpeps.gates.gate_nn_hopping(.7, .13, ops.I(), ops.c_spinor(), ops.cp_spinor())
    for a, b in zip(sites, sites[1:]):
        psi.apply_gate_(fpeps.Gate(gate.G, (a, b)))
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 64, 'tol': 1e-13}, max_sweeps=2)
    cu, cd, spin = _local('Hubbard')
    c = [[_embed(x, i, 3, True) for x in (cu, cd)] for i in range(3)]
    U = np.eye(64)
    for i in (0, 1):
        hopping = sum(c[i][s].T @ c[i+1][s] + c[i+1][s].T @ c[i][s] for s in (0, 1))
        U = expm(.7 * .13 * hopping) @ U
    rho = U @ U.T
    rho /= np.trace(rho)
    for i, j in ((0, 1), (0, 2), (2, 0), (1, 1)):
        pair_sites = (sites[i], sites[j])
        cases = [((ops.S(), ops.S()), sum(_embed(s, i, 3) @ _embed(s, j, 3) for s in spin)),
                 ((ops.cp_spinor(), ops.c_spinor()), sum(c[i][s].T @ c[j][s] for s in (0, 1))),
                 ((ops.c_spinor(), ops.cp_spinor()), sum(c[i][s] @ c[j][s].T for s in (0, 1)))]
        for pair, matrix in cases:
            expected = np.trace(rho @ matrix)
            with monkeypatch.context() as guard:
                guard.setattr(mps, 'zipper', _forbid_window_contraction)
                guard.setattr(mps, 'compression_', _forbid_window_contraction)
                assert env.measure_line(*pair, sites=pair_sites).item() == pytest.approx(expected, abs=3e-11)
                assert env.measure_nsite_exact(*pair, sites=pair_sites).item() == pytest.approx(expected, abs=3e-11)
            assert env.measure_nsite(*pair, sites=pair_sites).item() == pytest.approx(expected, abs=3e-11)
            if (i, j) == (0, 1):
                assert env.measure_nn(*pair, bond=pair_sites).item() == pytest.approx(expected, abs=3e-11)
                assert measure_rdm_nn(sites[0], direction, psi, env, pair).item() == pytest.approx(expected, abs=3e-11)
        result = env.measure_2site(ops.cp_spinor(), ops.c_spinor(), pairs=[pair_sites])
        expected = np.trace(rho @ sum(c[i][s].T @ c[j][s] for s in (0, 1)))
        assert result[pair_sites].item() == pytest.approx(expected, abs=3e-11)


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_multiplet_turn_and_multiple_pairs(config_kwargs, monkeypatch, sym):
    ops = yastn.operators.SpinfulFermions_tJ(sym=sym, **config_kwargs)
    sites = ((0, 0), (1, 0), (0, 1), (1, 1))  # fermionic order
    psi = fpeps.product_peps(fpeps.SquareLattice((2, 2), boundary='obc'), ops.I())
    gate = fpeps.gates.gate_nn_hopping(.7, .13, ops.I(), ops.c_spinor(), ops.cp_spinor())
    cu, cd, spin = _local('tJ')
    c = [[_embed(x, i, 4, True) for x in (cu, cd)] for i in range(4)]
    U = np.eye(81)
    for i, j in ((0, 1), (1, 3)):
        psi.apply_gate_(fpeps.Gate(gate.G, (sites[i], sites[j])))
        H = sum(c[i][s].T @ c[j][s] + c[j][s].T @ c[i][s] for s in (0, 1))
        U = expm(.7 * .13 * H) @ U
    rho = U @ U.T
    rho /= np.trace(rho)
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 64, 'tol': 1e-13}, max_sweeps=2)
    cases = [((ops.S(), ops.S()), sum(_embed(s, 0, 4) @ _embed(s, 3, 4) for s in spin)),
             ((ops.cp_spinor(), ops.c_spinor()), sum(c[0][s].T @ c[3][s] for s in (0, 1)))]
    for pair, matrix in cases:
        expected = np.trace(rho @ matrix)
        pair_sites = (sites[0], sites[3])
        assert env.measure_nsite(*pair, sites=pair_sites, opts_svd={'D_total': 256, 'tol': 1e-13}).item() == pytest.approx(expected, abs=4e-11)
        with monkeypatch.context() as guard:
            guard.setattr(mps, 'zipper', _forbid_window_contraction)
            guard.setattr(mps, 'compression_', _forbid_window_contraction)
            assert env.measure_nsite_exact(*pair, sites=pair_sites).item() == pytest.approx(expected, abs=4e-11)
            assert env.measure_2x2(*pair, sites=pair_sites).item() == pytest.approx(expected, abs=4e-11)
        result = env.measure_2site(*pair, pairs=[pair_sites], opts_svd={'D_total': 256, 'tol': 1e-13})
        assert result[pair_sites].item() == pytest.approx(expected, abs=4e-11)
    # A spinor pair times a same-site spinor pair is hopping times density.
    operators = (ops.cp_spinor(), ops.c_spinor(), ops.cp_spinor(), ops.c_spinor())
    product = sum(c[0][s].T @ c[3][s] for s in (0, 1)) @ sum(x.T @ x for x in c[1])
    actual = env.measure_nsite(*operators, sites=(sites[0], sites[3], sites[1], sites[1]),
                              opts_svd={'D_total': 256, 'tol': 1e-13})
    assert actual.item() == pytest.approx(np.trace(rho @ product), abs=4e-11)
    with monkeypatch.context() as guard:
        guard.setattr(mps, 'zipper', _forbid_window_contraction)
        guard.setattr(mps, 'compression_', _forbid_window_contraction)
        for measure in (env.measure_2x2, env.measure_nsite_exact):
            actual = measure(*operators, sites=(sites[0], sites[3], sites[1], sites[1]))
            assert actual.item() == pytest.approx(np.trace(rho @ product), abs=4e-11)

    # Products retain operator order where the two pairs share a site.
    for indices in ((0, 1, 1, 3), (0, 3, 3, 0), (1, 1, 0, 3)):
        i, j, k, l = indices
        product = (sum(c[i][a].T @ c[j][a] for a in (0, 1))
                   @ sum(c[k][a].T @ c[l][a] for a in (0, 1)))
        with monkeypatch.context() as guard:
            guard.setattr(mps, 'generate_mpo', _forbid_window_contraction)
            for measure in (env.measure_2x2, env.measure_nsite_exact):
                actual = measure(*operators, sites=tuple(sites[n] for n in indices))
                assert actual.item() == pytest.approx(np.trace(rho @ product), abs=4e-11)

    # The row/column sweeps must share each open-source boundary between targets.
    from yastn.tn.fpeps.envs import _env_window
    monkeypatch.setattr(_env_window, '_measure_nsite', _forbid_window_contraction)
    boundary = fpeps.EnvBoundaryMPS(psi, opts_svd={'D_total': 256, 'tol': 1e-13}, setup='lrbt')
    original = {s: psi[s].copy() for s in sites}
    cp, annihilate, S = ops.cp_spinor(), ops.c_spinor(), ops.S()
    pairs = [(a, b) for a in sites for b in sites]
    for peps_env in (env, boundary):
        for direction in ('h', 'v'):
            for pair in ((cp, annihilate), (annihilate, cp), (S, S)):
                zipper_calls = []
                zipper = mps.zipper
                def count_zipper(*args, **kwargs):
                    zipper_calls.append(1)
                    return zipper(*args, **kwargs)
                with monkeypatch.context() as count:
                    count.setattr(mps, 'zipper', count_zipper)
                    result = peps_env.measure_2site(*pair, pairs=pairs, dirn=direction,
                                                   opts_svd={'D_total': 256, 'tol': 1e-13})
                assert len(zipper_calls) == 4
                assert set(result) == set(pairs)
                for i in range(4):
                    for j in range(4):
                        if pair[0] is S:
                            matrix = sum(_embed(s, i, 4) @ _embed(s, j, 4) for s in spin)
                        elif pair[0] is cp:
                            matrix = sum(c[i][s].T @ c[j][s] for s in (0, 1))
                        else:
                            matrix = sum(c[i][s] @ c[j][s].T for s in (0, 1))
                        assert result[sites[i], sites[j]].item() == pytest.approx(np.trace(rho @ matrix), abs=4e-11), (type(peps_env).__name__, direction, i, j, pair[0] is cp, pair[0] is S)
            # Labels and subwindows with a nonzero origin use the native indexing.
            selected = [(sites[1], sites[3]), (sites[3], sites[1])]
            labelled = peps_env.measure_2site([cp, 2 * cp], annihilate, pairs=selected, dirn=direction,
                                             xrange=(1, 2), yrange=(0, 2),
                                             opts_svd={'D_total': 256, 'tol': 1e-13})
            for a, b in selected:
                i, j = sites.index(a), sites.index(b)
                expected = np.trace(rho @ sum(c[i][s].T @ c[j][s] for s in (0, 1)))
                assert labelled[a + (0,), b] == pytest.approx(expected, abs=4e-11)
                assert labelled[a + (1,), b] == pytest.approx(2 * expected, abs=4e-11)
    for site in sites:
        assert (psi[site] - original[site]).norm() == 0


def _forbid_window_contraction(*args, **kwargs):
    raise AssertionError('This measurement must use its own contraction path.')


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_multiplet_mps_pair_sweep(config_kwargs, monkeypatch, sym):
    ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
    psi = mps.product_mpo(ops.I(), N=3)
    H = mps.generate_mpo(ops.I(), [mps.Hterm(1, (i, j), (ops.cp_spinor(), ops.c_spinor()))
                                   for i, j in ((0, 1), (1, 0), (1, 2), (2, 1))], N=3)
    psi = psi + .3 * H
    matrix_state = _matrix(psi, ops.I())
    rho = matrix_state @ matrix_state.conj().T
    assert mps.vdot(psi, psi).item() == pytest.approx(np.trace(rho), abs=3e-12)
    cu, cd, spin = _local('Hubbard')
    c = [[_embed(x, i, 3, True) for x in (cu, cd)] for i in range(3)]
    from yastn.tn.mps import _measure
    monkeypatch.setattr(_measure, 'measure_nsite', _forbid_window_contraction)
    cp, annihilate, S = ops.cp_spinor(), ops.c_spinor(), ops.S()
    for pair in ((cp, annihilate), (annihilate, cp), (S, S)):
        result = mps.measure_2site(psi, *pair, psi, bonds='a')
        for (i, j), value in result.items():
            if pair[0] is S:
                matrix = sum(_embed(s, i, 3) @ _embed(s, j, 3) for s in spin)
            elif pair[0] is cp:
                matrix = sum(c[i][s].T @ c[j][s] for s in (0, 1))
            else:
                matrix = sum(c[i][s] @ c[j][s].T for s in (0, 1))
            assert value.item() == pytest.approx(np.trace(rho @ matrix), abs=3e-12), (i, j, pair[0] is cp, pair[0] is S)

    # Site-dependent values must share the sweep without reusing a previous
    # site's endpoint. Include complex coefficients and reversed/site-local
    # bonds; compare against the independently constructed Fock matrices.
    O = {i: (i + 1j) * cp for i in range(3)}
    P = {j: (2 - j) * annihilate for j in range(3)}
    result = mps.measure_2site(psi, O, P, psi, bonds='a')
    for (i, j), value in result.items():
        matrix = (i + 1j) * (2 - j) * sum(c[i][s].T @ c[j][s] for s in (0, 1))
        assert value.item() == pytest.approx(np.trace(rho @ matrix), abs=5e-12)
    # Distinct irreps (and rank-two scalars) can coexist in sparse dictionaries
    # provided the requested bonds form compatible pairs.
    O = {0: cp, 1: S, 2: ops.n_total()}
    P = {0: S, 1: annihilate, 2: ops.n_total()}
    result = mps.measure_2site(psi, O, P, psi, bonds=[(0, 1), (1, 0), (2, 2)])
    density = sum(c[2][s].T @ c[2][s] for s in (0, 1))
    matrices = {(0, 1): sum(c[0][s].T @ c[1][s] for s in (0, 1)),
                (1, 0): sum(_embed(s, 1, 3) @ _embed(s, 0, 3) for s in spin),
                (2, 2): density @ density}
    for bond, value in result.items():
        assert value.item() == pytest.approx(np.trace(rho @ matrices[bond]), abs=5e-12)
    with pytest.raises(yastn.YastnError, match='compatible pairs'):
        mps.measure_2site(psi, cp, ops.n_total(), psi, bonds=(0, 1))


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_set_operator_existing_string(config_kwargs, sym):
    ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
    psi = fpeps.product_peps(fpeps.SquareLattice((1, 1), boundary='obc'), ops.I())
    original = fpeps.DoublePepsTensor(psi[0, 0], psi[0, 0])
    for op in (ops.S(), ops.cp_spinor(), ops.c_spinor()):
        ten = original.copy()
        ten.set_operator_(op, dirn='l')
        assert ten.op.ndim == 3
        assert ten.op.get_legs(2) == op.get_legs(2)
        expected = original.apply_gate_on_ket(op, dirn='l').fuse_layers()
        assert (ten.fuse_layers() - expected).norm() < 2e-12
        for copied in (ten.copy(), ten.clone(), ten.detach(), ten.to(),
                       fpeps.DoublePepsTensor.from_dict(ten.to_dict(), config=ten.config)):
            assert (copied.fuse_layers() - expected).norm() < 2e-12
        assert (ten.conj().fuse_layers() - expected.conj()).norm() < 2e-12
        ten.set_operator_(ops.n_total(), reset=False)
        restored = fpeps.DoublePepsTensor.from_dict(ten.to_dict(), config=ten.config)
        assert (restored.fuse_layers() - ten.fuse_layers()).norm() < 2e-12
        ten.del_operator_()
        assert (ten.fuse_layers() - original.fuse_layers()).norm() == 0
        assert (ten.ket - original.ket).norm() == 0


def _multiple_multiplet_cases(ops, model, N):
    """Explicit magnetic/Fock matrices; independent of MPO construction."""
    cu, cd, spin = _local(model)
    c = [[_embed(x, i, N, True) for x in (cu, cd)] for i in range(N)]
    def hopping(i, j):
        return sum(c[i][a].T @ c[j][a] for a in (0, 1))
    def ss(i, j):
        return sum(_embed(s, i, N) @ _embed(s, j, N) for s in spin)
    S, cp, annihilate, density = ops.S(), ops.cp_spinor(), ops.c_spinor(), ops.n_total()
    last = N - 1
    n1 = sum(x.T @ x for x in c[1])
    mixed_left = ss(0, 1) @ n1 @ hopping(1, last)
    mixed_right = ss(0, 1) @ hopping(1, last) @ n1
    assert np.linalg.norm(mixed_left - mixed_right) > .1
    cases = [
        ('four_spins', (S,) * 4, (0, 1, 1, last), ss(0, 1) @ ss(1, last)),
        ('six_spins', (S,) * 6, (0, 1, 1, last, last, 0),
         ss(0, 1) @ ss(1, last) @ ss(last, 0)),
        ('four_spinors', (cp, annihilate) * 2, (0, 1, 1, last),
         hopping(0, 1) @ hopping(1, last)),
        # Swap the endpoints of one or both off-site spinor pairs. The
        # four_spinors result must acquire respectively one or two minus
        # signs. This catches missing as well as double-counted fermionic
        # exchange signs in nsite/line insertion and MPO construction.
        ('exchange_one_pair', (annihilate, cp, cp, annihilate), (1, 0, 1, last),
         -hopping(0, 1) @ hopping(1, last)),
        ('exchange_both_pairs', (annihilate, cp) * 2, (1, 0, last, 1),
         hopping(0, 1) @ hopping(1, last)),
        ('six_spinors', (cp, annihilate) * 3, (0, 1, 1, last, last, 0),
         hopping(0, 1) @ hopping(1, last) @ hopping(last, 0)),
        ('dual_and_reversed', (annihilate, cp, cp, annihilate), (last, 1, 1, 0),
         sum(c[last][a] @ c[1][a].T for a in (0, 1)) @ hopping(1, 0)),
        ('onsite_pair', (cp, annihilate) * 2, (1, 1, 0, last), n1 @ hopping(0, last)),
        ('mixed_density_left', (S, S, density, cp, annihilate), (0, 1, 1, 1, last),
         mixed_left),
        ('mixed_density_right', (S, S, cp, annihilate, density), (0, 1, 1, last, 1),
         mixed_right),
    ]
    if N == 4:
        cases.append(('disjoint_endpoints', (cp, annihilate) * 2, (0, 2, 1, 3),
                      hopping(0, 2) @ hopping(1, 3)))
    return cases


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('dims', [(1, 3), (3, 1), (2, 2)])
def test_measure_multiple_multiplets_peps(config_kwargs, monkeypatch, sym, dims):
    ops = yastn.operators.SpinfulFermions_tJ(sym=sym, **config_kwargs)
    psi = fpeps.product_peps(fpeps.SquareLattice(dims, boundary='obc'), ops.I())
    sites = tuple(sorted(psi.sites(), key=lambda s: s[::-1]))  # fermionic order
    N = len(sites)
    cu, cd, _ = _local('tJ')
    c = [[_embed(x, i, N, True) for x in (cu, cd)] for i in range(N)]
    U = np.eye(3 ** N)
    bonds = ((0, 1), (1, 3), (3, 2)) if dims == (2, 2) else ((0, 1), (1, 2))
    gate = fpeps.gates.gate_nn_hopping(.7, .21, ops.I(), ops.c_spinor(), ops.cp_spinor())
    for i, j in bonds:
        psi.apply_gate_(fpeps.Gate(gate.G, (sites[i], sites[j])))
        H = sum(c[i][a].T @ c[j][a] + c[j][a].T @ c[i][a] for a in (0, 1))
        U = expm(.7 * .21 * H) @ U
    rho = U @ U.T
    rho /= np.trace(rho)
    original = {site: psi[site].copy() for site in sites}
    env = fpeps.EnvCTM(psi, init='eye')
    opts_svd = {'D_total': 256, 'tol': 1e-13}
    env.ctmrg_(opts_svd=opts_svd, max_sweeps=2)
    local_measure = env.measure_2x2 if dims == (2, 2) else env.measure_line
    cases = _multiple_multiplet_cases(ops, 'tJ', N)
    for name, operators, indices, matrix in cases:
        at = tuple(sites[i] for i in indices)
        expected = np.trace(rho @ matrix)
        # Exact/line/2x2 must keep their native contraction algorithms.
        with monkeypatch.context() as guard:
            guard.setattr(mps, 'generate_mpo', _forbid_window_contraction)
            guard.setattr(mps, 'zipper', _forbid_window_contraction)
            guard.setattr(mps, 'compression_', _forbid_window_contraction)
            for measure in (env.measure_nsite_exact, local_measure):
                assert measure(*operators, sites=at).item() == pytest.approx(expected, abs=5e-11), (name, measure.__name__)
        actual = env.measure_nsite(*operators, sites=at, opts_svd=opts_svd)
        assert actual.item() == pytest.approx(expected, abs=5e-11), name
    assert any(abs(np.trace(rho @ matrix)) > 1e-4 for _, _, _, matrix in cases)
    for site in sites:
        assert (psi[site] - original[site]).norm() == 0


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_measure_multiple_multiplets_mps(config_kwargs, sym):
    ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
    N = 3
    cu, cd, _ = _local('Hubbard')
    c = [[_embed(x, i, N, True) for x in (cu, cd)] for i in range(N)]
    bonds = ((0, 1), (1, 0), (1, 2), (2, 1))
    H = mps.generate_mpo(ops.I(), [mps.Hterm(1, (i, j), (ops.cp_spinor(), ops.c_spinor()))
                                   for i, j in bonds], N=N)
    psi = mps.product_mpo(ops.I(), N=N) + .3 * H
    dense = np.eye(4 ** N) + .3 * sum(c[i][a].T @ c[j][a] for i, j in bonds for a in (0, 1))
    rho = dense @ dense.T
    norm = mps.vdot(psi, psi)
    assert norm.item() == pytest.approx(np.trace(rho), abs=3e-12)
    terms, expected_sum = [], np.zeros_like(dense, dtype=complex)
    cases = _multiple_multiplet_cases(ops, 'Hubbard', N)
    for k, (name, operators, indices, matrix) in enumerate(cases):
        expected = np.trace(rho @ matrix) / np.trace(rho)
        actual = mps.measure_nsite(psi, *operators, ket=psi, sites=indices) / norm
        assert actual.item() == pytest.approx(expected, abs=5e-12), name
        # A multi-pair term, a sum of such terms, and a scalar-only term all
        # enter generate_mpo through the same public interface.
        amplitude = (k + 1) * .1 + .03j
        terms.append(mps.Hterm(amplitude, indices, operators))
        expected_sum += amplitude * matrix
    terms.extend([mps.Hterm(.4, (1,), (ops.n_total(),)), mps.Hterm(.2)])
    expected_sum += .4 * sum(x.T @ x for x in c[1]) + .2 * np.eye(4 ** N)
    mpo = mps.generate_mpo(mps.product_mpo(ops.I(), N=N), terms, opts_svd={'tol': 1e-13})
    assert np.allclose(_matrix(mpo, ops.I()), expected_sum, atol=4e-12, rtol=0)
