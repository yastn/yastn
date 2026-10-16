# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Test definitions of two-site gates. """
import pytest
import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
from ._multiplet_models import hubbard_hamiltonian, tj_hamiltonian

tol = 1e-12  #pylint: disable=invalid-name


def check_1site_gate_taylor(gate, I, H, ds, eps):
    """ check gate vs Taylor expansion of the exp(-ds * H). """
    O = gate.G[0]

    O2 = I + (-ds) * H + (ds ** 2 / 2) * H @ H
    O2 = O2 + (-ds ** 3 / 6) * H @ H @ H + (ds ** 4 / 24) * H @ H @ H @ H
    assert ((O - O2).norm()) < abs(eps) ** 5

    G3 = fpeps.gates.gate_local_exp(ds, I, H)
    O3 = G3.G[0]
    assert ((O - O3).norm()) < tol


def check_2site_gate_taylor(gate, I, H, ds, eps):
    """ check gate vs Taylor expansion of the exp(-ds * H). """
    G = fpeps.gates.gate_nn_exp(ds, I, H)
    O = yastn.ncon(gate.G, [(-0, -1, 1) , (-2, -3, 1)])
    O = O.fuse_legs(axes=((0, 2), (1, 3)))

    H = H.fuse_legs(axes=((0, 2), (1, 3)))
    II = yastn.ncon([I, I], [(-0, -1), (-2, -3)])
    II = II.fuse_legs(axes=((0, 2), (1, 3)))

    O2 = II + (-ds) * H + (ds ** 2 / 2) * H @ H
    O2 = O2 + (-ds ** 3 / 6) * H @ H @ H + (ds ** 4 / 24) * H @ H @ H @ H
    assert ((O - O2).norm()) < abs(eps) ** 5

    O3 = yastn.ncon(G.G, [(-0, -1, 1) , (-2, -3, 1)])
    O3 = O3.fuse_legs(axes=((0, 2), (1, 3)))
    assert ((O - O3).norm()) < tol


def test_hopping_gate(config_kwargs):
    """ test fpeps.gates.gate_nn_hopping. """
    def check_hopping_gate(ops, t, ds):
        I, c, cdag = ops.I(), ops.c(), ops.cp()
        gate = fpeps.gates.gate_nn_hopping(t, ds, I, c, cdag)
        H = - t * yastn.fkron(cdag, c, sites=(0, 1)) \
            - t * yastn.fkron(cdag, c, sites=(1, 0))
        check_2site_gate_taylor(gate, I, H, ds, ds * t)

    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    check_hopping_gate(ops, t=0.5, ds=0.02)
    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **config_kwargs)
    check_hopping_gate(ops, t=2, ds=0.005)
    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', **config_kwargs)
    check_hopping_gate(ops, t=1, ds=0.1)


def test_tJ_gate(config_kwargs):
    """ test fpeps.gates.gate_nn_tJ. """
    def check_tJ_gate(ops, J, tu, td, muu0, muu1, mud0, mud1, ds):
        I, cu, cpu, cd, cpd = ops.I(), ops.c(spin='u'), ops.cp(spin='u'), ops.c(spin='d'), ops.cp(spin='d')
        Sz, Sp, Sm = ops.Sz(), ops.Sp(), ops.Sm()
        nu, nd = ops.n(spin='u'), ops.n(spin='d')
        gate = fpeps.gates.gate_nn_tJ(J, tu, td, muu0, muu1, mud0, mud1, ds, I, cu, cpu, cd, cpd)
        H = - tu * yastn.fkron(cpu, cu, sites=(0, 1)) \
            - tu * yastn.fkron(cpu, cu, sites=(1, 0)) \
            - td * yastn.fkron(cpd, cd, sites=(0, 1)) \
            - td * yastn.fkron(cpd, cd, sites=(1, 0)) \
            + (J / 2) * yastn.fkron(Sp, Sm, sites=(0, 1)) \
            + (J / 2) * yastn.fkron(Sm, Sp, sites=(0, 1)) \
            + J * yastn.fkron(Sz, Sz, sites=(0, 1)) \
            - (J / 4) * yastn.fkron((nu + nd), (nu + nd), sites=(0, 1)) \
            - muu0 * yastn.fkron(nu, I, sites=(0, 1)) \
            - muu1 * yastn.fkron(I, nu, sites=(0, 1)) \
            - mud0 * yastn.fkron(nd, I, sites=(0, 1)) \
            - mud1 * yastn.fkron(I, nd, sites=(0, 1))
        check_2site_gate_taylor(gate, I, H, ds, ds * max(map(abs, (J, tu, td, muu0, muu1, mud0, mud1))))

    ops = yastn.operators.SpinfulFermions_tJ(sym='U1', **config_kwargs)
    J, tu, td, muu0, muu1, mud0, mud1 = 1, 0.5, 0.7, 0.1, 0.2, 0.3, 0.4
    check_tJ_gate(ops, J, tu, td, muu0, muu1, mud0, mud1, ds=0.02)
    ops = yastn.operators.SpinfulFermions_tJ(sym='U1xU1', **config_kwargs)
    J, tu, td, muu0, muu1, mud0, mud1 = -0.5, -0.5, 0.5, 0.2, 0.3, 0.1, 0.1
    check_tJ_gate(ops, J, tu, td, muu0, muu1, mud0, mud1, ds=0.01)
    ops = yastn.operators.SpinfulFermions_tJ(sym='U1xU1xZ2', **config_kwargs)
    J, tu, td, muu0, muu1, mud0, mud1 = 0.5, 0.4, 0.2, 0.1, -0.2, -0.3, -0.1
    check_tJ_gate(ops, J, tu, td, muu0, muu1, mud0, mud1, ds=0.1)


def test_Ising_gate(config_kwargs):
    """ test fpeps.gates.gate_nn_Ising. """
    def check_Ising_gate(ops, J, ds):
        I, sp, sm = ops.I(), ops.sp(), ops.sm()
        X = sp + sm
        gate = fpeps.gates.gate_nn_Ising(J, ds, I, X)
        H = J * yastn.fkron(X, X, sites=(0, 1))
        check_2site_gate_taylor(gate, I, H, ds, ds * J)

    ops = yastn.operators.Spin12(sym='Z2', **config_kwargs)
    check_Ising_gate(ops, J=1, ds=0.02)
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    check_Ising_gate(ops, J=-2, ds=0.05)


def test_Heisenberg_gate(config_kwargs):
    """ test fpeps.gates.gate_nn_Heisenberg. """
    def check_Heisenberg_gate(ops, J, ds):
        I, sp, sm = ops.I(), ops.sp(), ops.sm()
        sx, sy, sz = ops.sx(), ops.sy(), ops.sz()
        gate = fpeps.gates.gate_nn_Heisenberg(J, ds, I, sz, sp, sm)
        H = J * yastn.fkron(sx, sx, sites=(0, 1)) \
          + J * yastn.fkron(sy, sy, sites=(0, 1)) \
          + J * yastn.fkron(sz, sz, sites=(0, 1))

        check_2site_gate_taylor(gate, I, H, ds, ds * J)

    ops = yastn.operators.Spin12(sym='Z2', **config_kwargs)
    check_Heisenberg_gate(ops, J=1, ds=0.02)
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    check_Heisenberg_gate(ops, J=-2, ds=0.05)


@pytest.mark.parametrize('step', [0.07, 0.07j])
@pytest.mark.parametrize('factory, energies', [
    (yastn.operators.Spin12, (-0.75, 0.25)),
    (yastn.operators.Spin1, (-2, -1, 1)),
])
def test_Heisenberg_gate_SU2(config_kwargs, step, factory, energies):
    """SU2 gates have the exact Boltzmann factors in each total-spin sector."""
    ops = factory(sym='SU2', **config_kwargs)
    J = 1.3
    gate = fpeps.gates.gate_nn_Heisenberg(J, step, ops.I(), ops.S())
    fused = _reconstruct_coupled_gate(gate, ops.I())
    for spin, energy in enumerate(energies):
        block = fused.config.backend.to_numpy(fused[(2 * spin, 2 * spin)])
        assert np.allclose(block, [[np.exp(-J * step * energy)]])


def _reconstruct_coupled_gate(gate, I=None):
    full = yastn.ncon(gate.G, [(-0, -1, 1), (-2, -3, 1)])
    fused = full.transpose((0, 2, 1, 3)).fuse_legs(
        ((0, 1), (2, 3)), mode='hard')
    return fused


def _matrix_exp(H, step):
    d, u = np.linalg.eigh(np.asarray(H, dtype=float))
    return (u * np.exp(-step * d)) @ u.T


@pytest.mark.parametrize('model', ['Heisenberg', 'Heisenberg_spin1', 'Hubbard', 'tJ'])
def test_SU2_zero_step_gate(config_kwargs, model):
    if model in ('Heisenberg', 'Heisenberg_spin1'):
        factory = yastn.operators.Spin12 if model == 'Heisenberg' else yastn.operators.Spin1
        ops = factory(sym='SU2', **config_kwargs)
        gate = fpeps.gates.gate_nn_Heisenberg(1.3, 0, ops.I(), ops.S())
    elif model == 'Hubbard':
        ops = yastn.operators.SpinfulFermions(sym='SU2xU1', **config_kwargs)
        gate = fpeps.gates.gate_nn_exp(0, ops.I(), hubbard_hamiltonian(ops, 0.7, 3.2, -0.15))
    else:
        ops = yastn.operators.SpinfulFermions_tJ(sym='SU2xU1', **config_kwargs)
        gate = fpeps.gates.gate_nn_exp(0, ops.I(), tj_hamiltonian(ops, 0.4, 0.8, -0.1, 0.2))
    full = yastn.ncon(gate.G, [(0, -1, 1), (-2, -3, 1)])
    assert (full - yastn.fkron(ops.I(), ops.I())).norm() < 1e-12


@pytest.mark.parametrize('step', [0.04, 0.04j])
def test_Hubbard_gate_SU2xU1(config_kwargs, step):
    ops = yastn.operators.SpinfulFermions(sym='SU2xU1', **config_kwargs)
    t, U, mu = 0.7, 3.2, -0.15
    gate = fpeps.gates.gate_nn_exp(step, ops.I(), hubbard_hamiltonian(ops, t, U, mu))
    fused = _reconstruct_coupled_gate(gate, ops.I())
    rt2 = np.sqrt(2.0)
    expected = {
        (0, 0): [[0]],
        (1, 1): [[-mu, -t], [-t, -mu]],
        (0, 2): [[U - 2 * mu, 0, -rt2 * t],
                 [0, U - 2 * mu, -rt2 * t],
                 [-rt2 * t, -rt2 * t, -2 * mu]],
        (2, 2): [[-2 * mu]],
        (1, 3): [[U - 3 * mu, t], [t, U - 3 * mu]],
        (0, 4): [[2 * U - 4 * mu]],
    }
    for charge, H in expected.items():
        assert np.allclose(fused.config.backend.to_numpy(fused[charge + charge]),
                           _matrix_exp(H, step), atol=1e-12)

    # Independent fermionic reference in the explicit
    # |0>, |down>, |up>, |up down> Fock basis.  In particular this checks the
    # N=3 hopping sign, which cannot be detected from gate eigenvalues alone.
    ops_u1 = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', **config_kwargs)
    Iu = ops_u1.I()
    cu, cpu = ops_u1.c('u'), ops_u1.cp('u')
    cd, cpd = ops_u1.c('d'), ops_u1.cp('d')
    n, d = ops_u1.n_total(), ops_u1.d()
    hopping = -t * (yastn.fkron(cpu, cu, sites=(0, 1))
                    + yastn.fkron(cpu, cu, sites=(1, 0))
                    + yastn.fkron(cpd, cd, sites=(0, 1))
                    + yastn.fkron(cpd, cd, sites=(1, 0)))
    Hu = (hopping + U * (yastn.fkron(d, Iu) + yastn.fkron(Iu, d))
          - mu * (yastn.fkron(n, Iu) + yastn.fkron(Iu, n)))
    gate_u1 = fpeps.gates.gate_nn_exp(step, Iu, Hu)
    dense = yastn.ncon(gate_u1.G, [(-0, -1, 1), (-2, -3, 1)])
    dense = dense.to_numpy().transpose(0, 2, 1, 3).reshape(16, 16)
    assert np.allclose(fused.config.backend.to_numpy(fused[(1, 1, 1, 1)]), dense[np.ix_([1, 4], [1, 4])],
                       atol=1e-12)
    # In the canonical Fock basis the CG singlet is
    # (|up,down> - |down,up>) / sqrt(2).
    projector_n2 = np.zeros((16, 3))
    projector_n2[3, 0] = 1       # |0,D>
    projector_n2[12, 1] = 1      # |D,0>
    projector_n2[[6, 9], 2] = np.array([-1, 1]) / np.sqrt(2)
    assert np.allclose(fused.config.backend.to_numpy(fused[(0, 2, 0, 2)]),
                       projector_n2.T @ dense @ projector_n2, atol=1e-12)
    # One fixed spin component spans |D,sigma>, |sigma,D>; SU(2) guarantees
    # the other component has the same reduced matrix.
    assert np.allclose(fused.config.backend.to_numpy(fused[(1, 3, 1, 3)]), dense[np.ix_([13, 7], [13, 7])],
                       atol=1e-12)


@pytest.mark.parametrize('step', [0.01, 0.01j])
def test_local_exp_SU2xU1_is_positive(config_kwargs, step):
    """Exponentiating a scalar Hamiltonian must not add fusion-gauge signs."""
    ops = yastn.operators.SpinfulFermions(sym='SU2xU1', **config_kwargs)
    identity = ops.I()
    hamiltonian = 10 * (ops.d() - ops.n_total() / 2 + identity / 4)
    gate = fpeps.gates.gate_local_exp(step, identity, hamiltonian).G[0]
    expected = {(0, 0): np.exp(-2.5 * step),
                (1, 1): np.exp(2.5 * step),
                (0, 2): np.exp(-2.5 * step)}
    for charge, value in expected.items():
        assert np.allclose(gate.config.backend.to_numpy(gate[charge + charge]), [[value]])


@pytest.mark.parametrize('step', [0, 0.04, 0.04j])
def test_SU2xU1_generic_gate_exp(config_kwargs, step):
    ops = yastn.operators.SpinfulFermions(sym='SU2xU1', **config_kwargs)
    I = ops.I()
    local = 3.2 * ops.d() - 0.15 * ops.n_total()
    H = yastn.fkron(local, I) + yastn.fkron(I, local)
    gate = fpeps.gates.gate_nn_exp(step, I, H)
    full = yastn.ncon(gate.G, [(0, -1, 1), (-2, -3, 1)])
    local_gate = fpeps.gates.gate_local_exp(step, I, local).G[0]
    assert (full - yastn.fkron(local_gate, local_gate)).norm() < 1e-12


def test_Hubbard_SU2xU1_second_order_trotter_against_ED(config_kwargs):
    """Check the particle-hole Hubbard Strang step against fermionic ED."""
    import scipy.linalg

    t, U, step = 0.7, 3.2, 0.04
    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', **config_kwargs)
    I = ops.I()
    cu, cpu = ops.c('u'), ops.cp('u')
    cd, cpd = ops.c('d'), ops.cp('d')
    n, d = ops.n_total(), ops.d()
    hopping = -t * (yastn.fkron(cpu, cu, sites=(0, 1))
                    + yastn.fkron(cpu, cu, sites=(1, 0))
                    + yastn.fkron(cpd, cd, sites=(0, 1))
                    + yastn.fkron(cpd, cd, sites=(1, 0)))
    local = U * (d - n / 2 + I / 4)
    local_two = yastn.fkron(local, I) + yastn.fkron(I, local)

    def dense(tensor):
        return tensor.to_numpy().transpose(0, 2, 1, 3).reshape(16, 16)

    hop_dense, local_dense = dense(hopping), dense(local_two)
    exact = scipy.linalg.expm(-step * (hop_dense + local_dense))
    strang = (scipy.linalg.expm(-step * hop_dense / 2)
              @ scipy.linalg.expm(-step * local_dense)
              @ scipy.linalg.expm(-step * hop_dense / 2))
    half = (scipy.linalg.expm(-step * hop_dense / 4)
            @ scipy.linalg.expm(-step * local_dense / 2)
            @ scipy.linalg.expm(-step * hop_dense / 4))
    strang_half = half @ half
    error = np.linalg.norm(strang - exact)
    error_half = np.linalg.norm(strang_half - exact)
    assert error_half < error / 3.5  # asymptotically 1/4 for a global Strang error

    # Compare the same Strang operator sector by sector with the SU2xU1
    # reduced representation.  The local energies below are those of
    # U(n_up-1/2)(n_dn-1/2) on both sites.
    rt2 = np.sqrt(2.0)
    ops_su2 = yastn.operators.SpinfulFermions(sym='SU2xU1', **config_kwargs)
    I_su2 = ops_su2.I()
    hopping_gate = fpeps.gates.gate_nn_hopping(t, step / 2, I_su2, ops_su2.c_spinor(), ops_su2.cp_spinor())
    hop_reduced = _reconstruct_coupled_gate(hopping_gate, I_su2)
    local_su2 = U * (ops_su2.d() - ops_su2.n_total() / 2 + I_su2 / 4)
    local_gate = fpeps.gates.gate_local_exp(step, I_su2, local_su2).G[0]
    local_two_gate = fpeps.gates.decompose_nn_gate(yastn.fkron(local_gate, local_gate))
    local_reduced = _reconstruct_coupled_gate(local_two_gate, I_su2)
    projectors = {
        (0, 0): np.eye(16)[:, [0]],
        (1, 1): np.eye(16)[:, [1, 4]],
        (0, 2): np.column_stack((np.eye(16)[:, 3], np.eye(16)[:, 12],
                                 (-np.eye(16)[:, 6] + np.eye(16)[:, 9]) / rt2)),
        (2, 2): np.eye(16)[:, [5]],  # one triplet component
        (1, 3): np.eye(16)[:, [13, 7]],
        (0, 4): np.eye(16)[:, [15]],
    }
    for charge, projector in projectors.items():
        backend = ops_su2.config.backend
        hs = backend.to_numpy(hop_reduced[charge + charge])
        hl = backend.to_numpy(local_reduced[charge + charge])
        reduced = hs @ hl @ hs
        assert np.allclose(reduced, projector.T @ strang @ projector, atol=1e-12)


@pytest.mark.parametrize('step', [0.03, 0.03j])
def test_tJ_gate_SU2xU1(config_kwargs, step):
    ops = yastn.operators.SpinfulFermions_tJ(sym='SU2xU1', **config_kwargs)
    J, t, mu0, mu1 = 0.4, 0.8, -0.1, 0.2
    gate = fpeps.gates.gate_nn_exp(step, ops.I(), tj_hamiltonian(ops, J, t, mu0, mu1))
    fused = _reconstruct_coupled_gate(gate, ops.I())
    expected = {
        (0, 0): [[0]],
        (1, 1): [[-mu1, -t], [-t, -mu0]],
        (0, 2): [[-mu0 - mu1 - J]],
        (2, 2): [[-mu0 - mu1]],
    }
    for charge, H in expected.items():
        assert np.allclose(fused.config.backend.to_numpy(fused[charge + charge]),
                           _matrix_exp(H, step), atol=1e-12)


def test_occupation_gate(config_kwargs):
    """ test fpeps.gates.gate_local_occupation. """
    def check_occupation_gate(ops, mu, ds):
        I, n = ops.I(), ops.n()
        gate = fpeps.gates.gate_local_occupation(mu, ds, I, n)
        H = - mu * n
        check_1site_gate_taylor(gate, I, H, ds, ds * abs(mu))

    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **config_kwargs)
    check_occupation_gate(ops, 2, ds=0.02)
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    check_occupation_gate(ops, -0.1, ds=0.05)


def test_field_gate(config_kwargs):
    """ test fpeps.gates.gate_local_field. """
    def check_field_gate(I, X, h, ds):
        gate = fpeps.gates.gate_local_field(h, ds, I, X)
        H = - h * X
        check_1site_gate_taylor(gate, I, H, ds, ds * abs(h))

    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    check_field_gate(ops.I(), ops.x(), 2, ds=0.02)
    ops = yastn.operators.Spin12(sym='Z2', **config_kwargs)
    check_field_gate(ops.I(), ops.z(), -0.5, ds=0.1)


def test_Coulomb_gate(config_kwargs):
    """ test fpeps.gates.gate_local_Coulomb. """
    def check_Coulomb_gate(ops, muu, mud, U, ds):
        I, nu, nd = ops.I(), ops.n(spin='u'), ops.n(spin='d')
        gate = fpeps.gates.gate_local_Coulomb(muu, mud, U, ds, I, nu, nd)
        H = U * (nu - I / 2) @ (nd - I / 2) - muu * nu - mud * nd - (U / 4) * I
        check_1site_gate_taylor(gate, I, H, ds, ds * max(map(abs, (U, muu, mud))))

    ops = yastn.operators.SpinfulFermions(sym='Z2', **config_kwargs)
    check_Coulomb_gate(ops, 0, 0, 2, ds=0.02)
    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **config_kwargs)
    check_Coulomb_gate(ops, 0.3, 0.2, -2, ds=0.2)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
