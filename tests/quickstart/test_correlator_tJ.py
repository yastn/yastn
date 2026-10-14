"""Finite-temperature correlators of the 2 x 3 open t-J model against ED.

Run with::

    python -m pytest tests/quickstart/test_correlator_tJ.py --quickstart -vs

The grand-canonical Hamiltonian is
H = -t sum_<ij>,s (cp_i,s c_j,s + h.c.)
    + J sum_<ij> (S_i . S_j - n_i n_j / 4) - mu sum_i n_i.
Double occupancy is excluded. Purification evolves I through beta/2, so its
physical density matrix approximates exp(-beta H) / Z. Defaults are beta=0.2,
4 Strang steps, PEPS bond dimension D=9 and environment dimension chi=32.
All 36 ordered pairs are checked in both scan directions, including on-site,
reverse, nearest-neighbor and longer-range correlators. The ED reference uses
independent projected Fock-space matrices, not YASTN gates or MPOs.
"""
from itertools import product
from time import perf_counter

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

import yastn
from yastn.tn import fpeps, mps


T, J, MU, BETA = 1.0, 0.3, 0.2, 0.2
STEPS, D, CHI = 4, 9, 32
# Finite-step evolution and PEPS/environment truncations are approximate;
# this tolerance concerns thermal correlators, not machine-precision algebra.
# D=9 gives a worst error around 4.6e-5 in the longest-range hopping. Halving
# the timestep does not remove it; increasing D improves this truncation
# error. On-site identities and fermionic exchange are checked separately.
ATOL = 5e-5


def _exact_correlators(sites, bonds):
    """Diagonalize the full 3**6 projected Hilbert space with physical fermions."""
    N = len(sites)
    states = [sum(occupation << (2 * i) for i, occupation in enumerate(local))
              for local in product((0, 1, 2), repeat=N)]
    index = {state: i for i, state in enumerate(states)}
    dim = len(states)
    assert dim == 729
    c = []
    for mode in range(2 * N):
        rows, columns, data = [], [], []
        for column, state in enumerate(states):
            if state & (1 << mode):
                rows.append(index[state ^ (1 << mode)])
                columns.append(column)
                data.append((-1.) ** (state & ((1 << mode) - 1)).bit_count())
        c.append(csr_matrix((data, (rows, columns)), shape=(dim, dim)))
    n = [c[2*i].T @ c[2*i] + c[2*i+1].T @ c[2*i+1] for i in range(N)]
    sz = [(c[2*i].T @ c[2*i] - c[2*i+1].T @ c[2*i+1]) / 2 for i in range(N)]
    sp = [c[2*i].T @ c[2*i+1] for i in range(N)]
    observables = {name: {} for name in ('nn', 'SS', 'cp_c', 'c_cp')}
    for i, a in enumerate(sites):
        for j, b in enumerate(sites):
            observables['nn'][a, b] = n[i] @ n[j]
            observables['SS'][a, b] = sz[i] @ sz[j] + (sp[i] @ sp[j].T + sp[i].T @ sp[j]) / 2
            observables['cp_c'][a, b] = sum(c[2*i+s].T @ c[2*j+s] for s in (0, 1))
            observables['c_cp'][a, b] = sum(c[2*i+s] @ c[2*j+s].T for s in (0, 1))
    H = -MU * sum(n)
    for a, b in bonds:
        hopping = observables['cp_c'][a, b]
        H += -T * (hopping + hopping.T) + J * (observables['SS'][a, b] - observables['nn'][a, b] / 4)
    dense = H.toarray()
    assert np.allclose(dense, dense.T, atol=1e-14, rtol=0)
    energies, vectors = eigh(dense)
    weights = np.exp(-BETA * (energies - energies[0]))
    weights /= weights.sum()
    rho = (vectors * weights) @ vectors.T
    # Tr(rho O) without dense matrix multiplication for every observable.
    expectation = lambda op: float(op.multiply(rho.T).sum())
    values = {name: {pair: expectation(op) for pair, op in entries.items()}
              for name, entries in observables.items()}
    values['density'] = {site: expectation(op) for site, op in zip(sites, n)}
    values['energy'] = float(weights @ energies)
    return values


@pytest.fixture(scope='module')
def exact_tj():
    geometry = fpeps.SquareLattice((2, 3), boundary='obc')
    # Column-major order agrees with the PEPS fermionic ordering.
    sites = tuple(sorted(geometry.sites(), key=lambda site: site[::-1]))
    bonds = tuple(geometry.bonds())
    assert len(bonds) == 7
    return _exact_correlators(sites, bonds)


def _thermal_peps(ops):
    geometry = fpeps.SquareLattice((2, 3), boundary='obc')
    psi = fpeps.product_peps(geometry, ops.I())
    # distribute appends the reversed gate list. A forward/backward sweep
    # therefore evolves through beta/(2*STEPS), not beta/STEPS.
    step = BETA / (4 * STEPS)
    if ops.config.sym.SYM_ID == 'SU2xU1':
        cp, c, S, n = ops.cp_spinor(), ops.c_spinor(), ops.S(), ops.n_total()
        h = mps.generate_mpo(ops.I(), [
            mps.Hterm(-T, (0, 1), (cp, c)), mps.Hterm(-T, (1, 0), (cp, c)),
            mps.Hterm(J, (0, 1), (S, S)), mps.Hterm(-J / 4, (0, 1), (n, n)),
        ], N=2)
        bond_gate = fpeps.gates.gate_nn_exp(step, ops.I(), h)
    else:
        bond_gate = fpeps.gates.gate_nn_tJ(J, T, T, 0, 0, 0, 0, step, ops.I(),
                                        ops.c('u'), ops.cp('u'), ops.c('d'), ops.cp('d'))
    # A separate local term gives every site the same mu, including corners
    # and edges with different coordination numbers in the open lattice.
    local_gate = fpeps.gates.gate_local_occupation(MU, step, ops.I(), ops.n_total())
    gates = fpeps.gates.distribute(geometry, gates_nn=bond_gate, gates_local=local_gate)
    environment = fpeps.EnvNTU(psi, which='NN')
    infos = []
    for k in range(STEPS):
        started = perf_counter()
        infos.append(fpeps.evolution_step_(environment, gates,
                     opts_svd={'D_total': D, 'tol': 1e-12}, max_iter=2))
        print(f'{ops.config.sym.SYM_ID} beta={BETA*(k+1)/STEPS:g}: evolution {perf_counter()-started:.2f}s', flush=True)
    print(f'Accumulated truncation error: {fpeps.accumulated_truncation_error(infos):.3e}', flush=True)
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': CHI, 'tol': 1e-12}, max_sweeps=4)
    return env


def _operator_pairs(ops):
    """Identical scalar observables in component and multiplet notation."""
    out = {'nn': [(1, ops.n_total(), ops.n_total())]}
    if ops.config.sym.SYM_ID == 'SU2xU1':
        out.update(SS=[(1, ops.S(), ops.S())],
                   cp_c=[(1, ops.cp_spinor(), ops.c_spinor())],
                   c_cp=[(1, ops.c_spinor(), ops.cp_spinor())])
    else:
        out.update(SS=[(1, ops.Sz(), ops.Sz()), (.5, ops.Sp(), ops.Sm()), (.5, ops.Sm(), ops.Sp())],
                   cp_c=[(1, ops.cp(s), ops.c(s)) for s in ('u', 'd')],
                   c_cp=[(1, ops.c(s), ops.cp(s)) for s in ('u', 'd')])
    return out


def _check(values, expected, label, atol=ATOL):
    assert values.keys() == expected.keys()
    errors = {key: abs(values[key] - expected[key]) for key in values}
    worst = max(errors, key=errors.get)
    print(f'{label}: max error={errors[worst]:.3e}, at {worst}, '
          f'PEPS={values[worst]:.9g}, ED={expected[worst]:.9g}', flush=True)
    assert errors[worst] < atol, (label, worst, values[worst], expected[worst])


@pytest.mark.skipif("not config.getoption('quickstart')", reason='run with --quickstart')
@pytest.mark.parametrize('sym', ['U1xU1', 'SU2xU1'])
def test_correlator_tJ(config_kwargs, exact_tj, sym):
    ops = yastn.operators.SpinfulFermions_tJ(sym=sym, **config_kwargs)
    env = _thermal_peps(ops)
    sites = tuple(sorted(env.psi.sites(), key=lambda site: site[::-1]))
    pairs = [(a, b) for a in sites for b in sites]  # includes reverse and on-site pairs
    opts_svd = {'D_total': CHI, 'tol': 1e-12}
    density = env.measure_1site(ops.n_total())
    _check(density, exact_tj['density'], f'{sym} density')
    correlations = {}
    for name, components in _operator_pairs(ops).items():
        for direction in ('h', 'v'):
            values = dict.fromkeys(pairs, 0.)
            for coefficient, O, P in components:
                measured = env.measure_2site(O, P, pairs=pairs, dirn=direction, opts_svd=opts_svd)
                for pair, value in measured.items():
                    values[pair] += coefficient * value
            _check(values, exact_tj[name], f'{sym} {name} {direction}')
            correlations[name, direction] = values
        # Independently exercise the native line and full-window contractions.
        for method, pair in ((env.measure_line, ((0, 0), (0, 2))),
                             (env.measure_nsite, ((1, 2), (0, 0)))):
            kwargs = {'opts_svd': opts_svd} if method == env.measure_nsite else {}
            value = sum(coefficient * method(O, P, sites=pair, **kwargs) for coefficient, O, P in components)
            _check({pair: value}, {pair: exact_tj[name][pair]}, f'{sym} {name} {method.__name__}')

    # These are exact algebraic identities of projected fermions, so they
    # must hold much more accurately than the thermal/ED error tolerance.
    # In particular sum_s c_s cp_s = 2*(1-n), not 2-n: doublons are excluded.
    for direction in ('h', 'v'):
        for a in sites:
            n = density[a]
            for name, expected in (('nn', n), ('SS', .75 * n), ('cp_c', n), ('c_cp', 2 * (1 - n))):
                assert correlations[name, direction][a, a] == pytest.approx(expected, abs=1e-8, rel=0)
            for b in sites:
                if a != b:
                    assert correlations['c_cp', direction][a, b] == pytest.approx(
                        -correlations['cp_c', direction][b, a], abs=1e-8, rel=0)
    # Reconstruct the extensive energy using all seven open-lattice bonds.
    energy = -MU * sum(density.values())
    for a, b in env.psi.bonds():
        energy -= T * (correlations['cp_c', 'h'][a, b] + correlations['cp_c', 'h'][b, a])
        energy += J * (correlations['SS', 'h'][a, b] - correlations['nn', 'h'][a, b] / 4)
    _check({'E': energy}, {'E': exact_tj['energy']}, f'{sym} energy', atol=len(sites) * ATOL)


@pytest.mark.skipif("not config.getoption('quickstart')", reason='run with --quickstart')
@pytest.mark.parametrize('sym', ['U1xU1', 'SU2xU1'])
@pytest.mark.parametrize('direction, max_calls', [('h', 6), ('v', 12)])
def test_correlator_tJ_reuses_boundary_mps(config_kwargs, monkeypatch, sym, direction, max_calls):
    """All 36 pairs share propagated MPSs, for ordinary and multiplet operators.

    A product purification keeps this structural regression inexpensive; the
    finite-temperature test above separately checks nonzero correlations.
    The two scan orders propagate each source through later rows/columns:
    2 * 3 * (2*1/2) = 6 zipper calls horizontally, or
    2 * 2 * (3*2/2) = 12 vertically. These are upper bounds to allow future
    optimizations; contracting a separate window for every pair exceeds them.
    """
    ops = yastn.operators.SpinfulFermions_tJ(sym=sym, **config_kwargs)
    psi = fpeps.product_peps(fpeps.SquareLattice((2, 3), boundary='obc'), ops.I())
    env = fpeps.EnvCTM(psi, init='eye')
    sites = psi.sites()
    pairs = [(a, b) for a in sites for b in sites]
    if sym == 'SU2xU1':
        O, P, onsite = ops.cp_spinor(), ops.c_spinor(), 2 / 3
    else:
        O, P, onsite = ops.cp('u'), ops.c('u'), 1 / 3
    zipper, calls = mps.zipper, []
    def counted_zipper(*args, **kwargs):
        calls.append(1)
        return zipper(*args, **kwargs)
    monkeypatch.setattr(mps, 'zipper', counted_zipper)
    values = env.measure_2site(O, P, pairs=pairs, dirn=direction,
                              opts_svd={'D_total': 16, 'tol': 1e-12})
    assert set(values) == set(pairs)
    for (a, b), value in values.items():
        assert value == pytest.approx(onsite if a == b else 0, abs=1e-12, rel=0)
    assert len(calls) <= max_calls


if __name__ == '__main__':
    pytest.main([__file__, '-vs', '--durations=0', '--quickstart'])
