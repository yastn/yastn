"""Finite-patch SU2 RDMs against independent magnetic-basis density matrices."""
import numpy as np
import pytest
from scipy.linalg import expm
import yastn
from yastn.tn import fpeps
from yastn.tn.fpeps.envs import rdm
from .test_su2_models_ed import _expand_magnetic, _fock_operators
from ._multiplet_models import hubbard_hamiltonian, tj_hamiltonian


def _partial_trace(matrix, dimension, keep):
    tensor = matrix.reshape((dimension,) * 8)
    remaining = 4
    for site in reversed(range(4)):
        if site not in keep:
            tensor = np.trace(tensor, axis1=site, axis2=site + remaining)
            remaining -= 1
    return tensor.reshape(dimension ** len(keep), dimension ** len(keep))


@pytest.mark.parametrize('column', [0, 1])
@pytest.mark.parametrize('model, sym', [
    ('spin12', 'SU2'), ('spin1', 'SU2'),
    ('Hubbard', 'SU2'), ('Hubbard', 'SU2xU1'),
    ('tJ', 'SU2'), ('tJ', 'SU2xU1'),
])
def test_su2_patch_density_matrices(config_kwargs, model, sym, column):
    step = 0.07 + 0.04j
    bond = fpeps.Bond((0, column), (1, column))
    sites = ((0, 0), (1, 0), (0, 1), (1, 1))
    if model in ('spin12', 'spin1'):
        spin = 0.5 if model == 'spin12' else 1
        factory = yastn.operators.Spin12 if spin == 0.5 else yastn.operators.Spin1
        ops = factory(sym=sym, **config_kwargs)
        magnetic = np.arange(-spin, spin + 1)
        raising = np.diag(np.sqrt(spin * (spin + 1) - magnetic[:-1] * (magnetic[:-1] + 1)), -1)
        sx, sy, sz = (raising + raising.T) / 2, (raising - raising.T) / (2j), np.diag(magnetic)
        H = sum(np.kron(s, s) for s in (sx, sy, sz))
        dimension = int(2 * spin + 1)
        vectors = {site: ops.I() for site in sites}
        local = [np.eye(dimension) / dimension for _ in sites]
        gate = fpeps.gates.gate_nn_Heisenberg(1., step, ops.I(), ops.S(), bond=bond)
    else:
        c, n = _fock_operators()
        t, U, J, mu = 0.7, 0.6, 0.3, 0.2
        H = -mu * sum(n)
        for left, right in ((0, 2), (1, 3)):
            hop = c[left].T @ c[right]
            H -= t * (hop + hop.T)
        if model == 'Hubbard':
            ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
            H += U * (n[0] @ n[1] + n[2] @ n[3])
            states = [0, 3, 2, 1]
            gate = fpeps.gates.gate_nn_exp(step, ops.I(), hubbard_hamiltonian(ops, t, U, mu), bond=bond)
        else:
            ops = yastn.operators.SpinfulFermions_tJ(sym=sym, **config_kwargs)
            sz0, sz1 = (n[0] - n[1]) / 2, (n[2] - n[3]) / 2
            sp0, sp1 = c[0].T @ c[1], c[2].T @ c[3]
            H += J * (sz0 @ sz1 + (sp0 @ sp1.T + sp0.T @ sp1) / 2
                      - (n[0] + n[1]) @ (n[2] + n[3]) / 4)
            states = [0, 2, 1]
            gate = fpeps.gates.gate_nn_exp(step, ops.I(), tj_hamiltonian(ops, J, t, mu, mu), bond=bond)
        order = [left + 4 * right for left in states for right in states]
        H = H[np.ix_(order, order)]
        dimension = len(states)
        vectors, local = {}, []
        for index, site in enumerate(sites):
            alpha = 0.15 * (index + 1)
            vectors[site] = ops.I() + alpha * ops.n_total()
            weights = np.array([(1 + alpha * state.bit_count()) ** 2 for state in states])
            local.append(np.diag(weights / weights.sum()))

    propagator = expm(-step * H)
    first = 2 * column
    pair = propagator @ np.kron(local[first], local[first + 1]) @ propagator.conj().T
    pair /= np.trace(pair)
    exact = (np.kron(pair, np.kron(local[2], local[3])) if column == 0
             else np.kron(np.kron(local[0], local[1]), pair))
    psi = fpeps.product_peps(fpeps.SquareLattice((2, 2), boundary='obc'), vectors)
    psi.apply_gate_(gate)
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 32, 'tol': 1e-12}, max_sweeps=2)
    patches = ((rdm.rdm1x1, (0,)), (rdm.rdm1x2, (0, 2)),
               (rdm.rdm2x1, (0, 1)), (rdm.rdm2x2, (0, 1, 2, 3)),
               (rdm.rdm2x2_diagonal, (0, 3)), (rdm.rdm2x2_anti_diagonal, (1, 2)))
    for builder, keep in patches:
        reduced, _ = builder((0, 0), psi, env)
        order = tuple(range(0, reduced.ndim, 2)) + tuple(range(1, reduced.ndim, 2))
        actual = _expand_magnetic(reduced).transpose(order).reshape(dimension ** len(keep), -1)
        if builder is rdm.rdm2x2 and model in ('Hubbard', 'tJ'):
            # The 2x2 contraction leaves graded crossings at the bottom-left
            # ket and top-right bra (see measure_rdm_2x2). Remove those local
            # crossings to compare an ordinary Fock-space density matrix.
            parity = np.array([state.bit_count() % 2 for state in states])
            indices = np.indices((dimension,) * 4).reshape(4, -1)
            p1, p2 = parity[indices[1]], parity[indices[2]]
            sign = (-1.) ** (p1[:, None] * (p1[:, None] + p1[None, :])
                            + p2[None, :] * (p2[:, None] + p2[None, :]))
            actual *= sign
        expected = _partial_trace(exact, dimension, keep)
        assert np.allclose(actual, expected, atol=2e-10, rtol=0), builder.__name__
        assert np.allclose(actual, actual.conj().T, atol=2e-10, rtol=0), builder.__name__
        assert np.trace(actual) == pytest.approx(1, abs=2e-10)
        assert np.linalg.eigvalsh(actual).min() > -2e-10
