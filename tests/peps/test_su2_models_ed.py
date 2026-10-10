"""Independent finite-system references for spin-invariant PEPS evolution."""
import numpy as np
import pytest
from scipy.linalg import expm
import yastn
from yastn.tn import fpeps
from yastn.tn.fpeps.envs.rdm import rdm1x2


def _fock_operators():
    annihilation = []
    for mode in range(4):
        c = np.zeros((16, 16))
        for state in range(16):
            if state & (1 << mode):
                c[state ^ (1 << mode), state] = (-1) ** (state & ((1 << mode) - 1)).bit_count()
        annihilation.append(c)
    return annihilation, [c.T @ c for c in annihilation]


def _thermal_peps(ops, gate, config_kwargs):
    geometry = fpeps.SquareLattice((1, 2), boundary='obc')
    psi = fpeps.product_peps(geometry, ops.I())
    gates = fpeps.gates.distribute(geometry, gates_nn=gate, symmetrize=False)
    infos = fpeps.evolution_step_(fpeps.EnvNTU(psi, which='NN'), gates,
                                 opts_svd={'D_total': 32}, max_iter=2)
    assert all(info.truncation_error < 1e-12 for info in infos)
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 32}, max_sweeps=2)
    return psi, env


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('model', ['Hubbard', 'tJ'])
@pytest.mark.parametrize('step', [0.13, 0.13j])
def test_fermionic_SU2_thermal_against_ED(config_kwargs, sym, model, step):
    """Unequal chemical potentials/doping expose incorrect sector weights."""
    c, n = _fock_operators()
    t, U, J, mu0, mu1 = 0.7, 3.2, 0.6, -0.2, 0.4
    if model == 'Hubbard':
        mu0 = mu1 = 0.55
        H = U * (n[0] @ n[1] + n[2] @ n[3])
        ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
        gate = getattr(fpeps.gates, 'gate_nn_Hubbard_' + sym)(t, U, mu0, step, ops.I())
        keep = np.arange(16)
    else:
        sz0, sz1 = (n[0] - n[1]) / 2, (n[2] - n[3]) / 2
        sp0, sp1 = c[0].T @ c[1], c[2].T @ c[3]
        H = J * (sz0 @ sz1 + (sp0 @ sp1.T + sp0.T @ sp1) / 2
                 - (n[0] + n[1]) @ (n[2] + n[3]) / 4)
        ops = yastn.operators.SpinfulFermions_tJ(sym=sym, **config_kwargs)
        gate = getattr(fpeps.gates, 'gate_nn_tJ_' + sym)(J, t, mu0, mu1, step, ops.I())
        keep = np.array([s for s in range(16) if s & 3 != 3 and (s >> 2) & 3 != 3])
    H -= mu0 * (n[0] + n[1]) + mu1 * (n[2] + n[3])
    for left, right in ((0, 2), (1, 3)):
        hopping = c[left].T @ c[right]
        H -= t * (hopping + hopping.T)
    H = H[np.ix_(keep, keep)]
    propagator = expm(-step * H)
    rho = propagator @ propagator.conj().T
    rho /= np.trace(rho)
    _, env = _thermal_peps(ops, gate, config_kwargs)
    for site, value in env.measure_1site(ops.n_total()).items():
        number = n[2 * site[1]] + n[2 * site[1] + 1]
        assert value == pytest.approx(np.trace(rho @ number[np.ix_(keep, keep)]), abs=2e-11)
    for value in env.measure_nn(ops.n_total(), ops.n_total()).values():
        number_pair = (n[0] + n[1]) @ (n[2] + n[3])
        assert value == pytest.approx(np.trace(rho @ number_pair[np.ix_(keep, keep)]), abs=2e-11)
    if model == 'Hubbard':
        for value in env.measure_1site(ops.d()).values():
            assert value == pytest.approx(np.trace(rho @ n[0] @ n[1]), abs=2e-11)


@pytest.mark.parametrize('step', [0.17, 0.17j])
def test_heisenberg_SU2_thermal_energy(config_kwargs, step):
    ops = yastn.operators.Spin12(sym='SU2', **config_kwargs)
    J = 1.3
    gate = fpeps.gates.gate_nn_Heisenberg_SU2(J, step, ops.I())
    psi, env = _thermal_peps(ops, gate, config_kwargs)
    rho, _ = rdm1x2((0, 0), psi, env)
    rho = rho.fuse_legs(((0, 2), (1, 3)), mode='hard')
    weights = {j: (j + 1) * np.trace(rho.config.backend.to_numpy(rho[(j, j)])) for j in (0, 2)}
    energy = J * (-0.75 * weights[0] + 0.25 * weights[2]) / sum(weights.values())
    # Independent Pauli-matrix Hamiltonian, as in peps-torch groups/su2.py SS().
    sx = np.array([[0., 1.], [1., 0.]]) / 2
    sy = np.array([[0., -1j], [1j, 0.]]) / 2
    sz = np.diag([1., -1.]) / 2
    H = J * sum(np.kron(s, s) for s in (sx, sy, sz))
    propagator = expm(-step * H)
    exact = propagator @ propagator.conj().T
    expected = np.trace(exact @ H) / np.trace(exact)
    assert energy == pytest.approx(expected, abs=2e-11)
