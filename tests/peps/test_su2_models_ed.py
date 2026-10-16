"""Independent finite-system references for spin-invariant PEPS evolution."""
import numpy as np
import pytest
from scipy.linalg import expm
import yastn
from yastn.tn import fpeps
from ._multiplet_models import hubbard_hamiltonian, tj_hamiltonian


def _expand_magnetic(a):
    """Expand a small test tensor; multiplicities precede magnetic indices."""
    from yastn.tensor._auxiliary import get_blocks
    from yastn.tensor._nonabelian import intertwiner

    a = a.consume_transpose()
    sym, maps, shape = a.config.sym, [], []
    for leg in a.get_legs():
        sectors, start = {}, 0
        for charge, dimension in zip(leg.t, leg.D):
            size = dimension * sym.irrep_dimension(charge)
            sectors[charge] = slice(start, start + size)
            start += size
        maps.append(sectors)
        shape.append(start)
    out = np.zeros(shape, dtype=complex)
    blocks = get_blocks(sym, a.struct)
    for charges, path, dimensions, slc in zip(blocks.t, blocks.channels, blocks.D, blocks.slc):
        charges = tuple(map(tuple, charges))
        cg = intertwiner(sym, charges, a.s, a.n, tuple(path))
        data = a.config.backend.to_numpy(a._data[slice(*slc)]).reshape(dimensions)
        order = tuple(i for axis in range(a.ndim) for i in (axis, axis + a.ndim))
        values = np.tensordot(data, cg, axes=0).transpose(order)
        values = values.reshape(tuple(d * sym.irrep_dimension(t) for d, t in zip(dimensions, charges)))
        out[tuple(sectors[t] for sectors, t in zip(maps, charges))] += values
    return out


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
        gate = fpeps.gates.gate_nn_exp(step, ops.I(), hubbard_hamiltonian(ops, t, U, mu0))
        keep = np.arange(16)
    else:
        sz0, sz1 = (n[0] - n[1]) / 2, (n[2] - n[3]) / 2
        sp0, sp1 = c[0].T @ c[1], c[2].T @ c[3]
        H = J * (sz0 @ sz1 + (sp0 @ sp1.T + sp0.T @ sp1) / 2
                 - (n[0] + n[1]) @ (n[2] + n[3]) / 4)
        ops = yastn.operators.SpinfulFermions_tJ(sym=sym, **config_kwargs)
        gate = fpeps.gates.gate_nn_exp(step, ops.I(), tj_hamiltonian(ops, J, t, mu0, mu1))
        keep = np.array([s for s in range(16) if s & 3 != 3 and (s >> 2) & 3 != 3])
    H -= mu0 * (n[0] + n[1]) + mu1 * (n[2] + n[3])
    for left, right in ((0, 2), (1, 3)):
        hopping = c[left].T @ c[right]
        H -= t * (hopping + hopping.T)
    H = H[np.ix_(keep, keep)]
    propagator = expm(-step * H)
    # Compare every complex matrix element, including the three-particle
    # hopping sign which thermal densities and spectra alone cannot detect.
    full = yastn.ncon(gate.G, [(0, -1, 1), (-2, -3, 1)])
    local_states = [0, 3, 2, 1] if model == 'Hubbard' else [0, 2, 1]
    order = [list(keep).index(left + 4 * right) for left in local_states for right in local_states]
    actual = _expand_magnetic(full).transpose(0, 2, 1, 3).reshape(len(keep), len(keep))
    assert np.allclose(actual, propagator[np.ix_(order, order)], atol=2e-12, rtol=0)
    rho = propagator @ propagator.conj().T
    rho /= np.trace(rho)
    _, env = _thermal_peps(ops, gate, config_kwargs)
    hopping = c[0].T @ c[2] + c[1].T @ c[3]
    sz0, sz1 = (n[0] - n[1]) / 2, (n[2] - n[3]) / 2
    sp0, sp1 = c[0].T @ c[1], c[2].T @ c[3]
    spin = sz0 @ sz1 + (sp0 @ sp1.T + sp0.T @ sp1) / 2
    for pair, observable in (
            ((ops.cp_spinor(), ops.c_spinor()), hopping),
            ((ops.c_spinor(), ops.cp_spinor()), -hopping.T),
            ((ops.S(), ops.S()), spin)):
        expected = np.trace(rho @ observable[np.ix_(keep, keep)])
        for value in env.measure_nn(*pair).values():
            assert value.item() == pytest.approx(expected, abs=2e-11)
    for site, value in env.measure_1site(ops.n_total()).items():
        number = n[2 * site[1]] + n[2 * site[1] + 1]
        assert value.item() == pytest.approx(np.trace(rho @ number[np.ix_(keep, keep)]), abs=2e-11)
    for value in env.measure_nn(ops.n_total(), ops.n_total()).values():
        number_pair = (n[0] + n[1]) @ (n[2] + n[3])
        assert value.item() == pytest.approx(np.trace(rho @ number_pair[np.ix_(keep, keep)]), abs=2e-11)
    if model == 'Hubbard':
        for value in env.measure_1site(ops.d()).values():
            assert value.item() == pytest.approx(np.trace(rho @ n[0] @ n[1]), abs=2e-11)


@pytest.mark.parametrize('step', [0.17, 0.17j])
@pytest.mark.parametrize('spin', [0.5, 1])
def test_heisenberg_SU2_thermal_energy(config_kwargs, step, spin):
    factory = yastn.operators.Spin12 if spin == 0.5 else yastn.operators.Spin1
    ops = factory(sym='SU2', **config_kwargs)
    J = 1.3
    gate = fpeps.gates.gate_nn_Heisenberg(J, step, ops.I(), ops.S())
    psi, env = _thermal_peps(ops, gate, config_kwargs)
    energy = J * env.measure_nn(ops.S(), ops.S(), bond=((0, 0), (0, 1)))
    # Independent magnetic spin matrices, as in peps-torch groups/su2.py SS().
    magnetic = np.arange(-spin, spin + 1)
    raising = np.diag(np.sqrt(spin * (spin + 1) - magnetic[:-1] * (magnetic[:-1] + 1)), -1)
    sx, sy, sz = (raising + raising.T) / 2, (raising - raising.T) / (2j), np.diag(magnetic)
    H = J * sum(np.kron(s, s) for s in (sx, sy, sz))
    propagator = expm(-step * H)
    full = yastn.ncon(gate.G, [(0, -1, 1), (-2, -3, 1)])
    dimension = int(2 * spin + 1) ** 2
    actual = _expand_magnetic(full).transpose(0, 2, 1, 3).reshape(dimension, dimension)
    assert np.allclose(actual, propagator, atol=2e-12, rtol=0)
    exact = propagator @ propagator.conj().T
    expected = np.trace(exact @ H) / np.trace(exact)
    assert energy.item() == pytest.approx(expected, abs=2e-11)
