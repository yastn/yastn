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
""" Test the expectation values of spin-1/2 fermions with analytical values of fermi sea. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from joblib import Parallel

Pool = Parallel(n_jobs=4)
# Pool = None

def mean(xs):
    return sum(xs) / len(xs)


def test_NTU_spinful_finite(config_kwargs):
    """ Simulate purification of spinful fermions in a small finite system """
    print(" Simulating spinful fermions in a small finite system. ")

    Nx, Ny = 3, 2
    geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary='obc')

    mu_up, mu_dn = 0, 0  # chemical potential
    t_up, t_dn = 1, 1  # hopping amplitude

    U = 0   # TODO:  try to change this test to some non-zero U;  can be artifical example so that the test is better
    beta = 0.1

    dbeta = 0.025
    D = 8

    # prepare evolution gates
    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', **config_kwargs)
    I = ops.I()
    c_up, c_dn, cdag_up, cdag_dn = ops.c(spin='u'), ops.c(spin='d'), ops.cp(spin='u'), ops.cp(spin='d')
    n_up, n_dn =  ops.n(spin='u'), ops.n(spin='d')
    n_int = n_up @ n_dn

    g_hop_u = fpeps.gates.gate_nn_hopping(t_up, dbeta / 2, I, c_up, cdag_up)
    g_hop_d = fpeps.gates.gate_nn_hopping(t_dn, dbeta / 2, I, c_dn, cdag_dn)
    g_loc = fpeps.gates.gate_local_Coulomb(mu_up, mu_dn, U, dbeta / 2, I, n_up, n_dn)
    gates = fpeps.gates.distribute(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_loc)

    # initialized infinite temperature purification
    psi = fpeps.product_peps(geometry, I)

    # time-evolve purification
    env = fpeps.EnvNTU(psi, which='NN+')

    # list of dicts for a two-step truncation
    opts_svd = [{"D_total": 2 * D, 'tol': 1e-14},
                {"D_total": D, 'tol': 1e-14}]

    steps = round((beta / 2) / dbeta)
    dbeta = (beta / 2) / steps

    infos = []
    for step in range(1, steps + 1):
        print(f"beta = {step * dbeta:0.3f}" )
        info = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, fix_metric=None)
        infos.append(info)

    # convergence criteria for CTM based on total energy
    energy_old, tol_exp = 0, 1e-8
    opts_svd_ctm = {'D_total': 40, 'tol': 1e-10}

    env = fpeps.EnvCTM(psi)

    for _ in range(50):
        env.update_(opts_svd=opts_svd_ctm)  # single CMTRG sweep

        # calculate expectation values
        d_oc = env.measure_1site(n_int)
        cdagc_up = env.measure_nn(cdag_up, c_up)  # calculate for all unique bonds
        cdagc_dn = env.measure_nn(cdag_dn, c_dn)  # -> {bond: value}

        energy = U * sum(d_oc.values()) - sum(cdagc_up.values()) - sum(cdagc_dn.values())

        print(f"Energy: {energy:0.10}")
        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

    print(f"Accumulated truncation error: {fpeps.accumulated_truncation_error(infos):0.4f}")

    # analytical nn fermionic correlator at beta = 0.1 for 2D finite 2 x 3 lattice
    nn_bond_1_exact = 0.024917101651703362  # bond between (1, 1) and (1, 2)   # this requires checking; bonds exact vs CTM should match
    nn_bond_2_exact = 0.024896433958165112  # bond between (0, 0) and (1, 0)

    # measure <cdag_1 c_2>
    nn_CTM_bond_1_up = env.measure_nn(cdag_up, c_up, bond=((2, 0), (2, 1)))  # horizontal bond
    nn_CTM_bond_2_up = env.measure_nn(cdag_up, c_up, bond=((0, 1), (1, 1)))  # vertical bond
    nn_CTM_bond_1_dn = env.measure_nn(cdag_dn, c_dn, bond=((2, 0), (2, 1)))  # horizontal bond
    nn_CTM_bond_2_dn = env.measure_nn(cdag_dn, c_dn, bond=((0, 1), (1, 1)))  # vertical bond

    # reverse bond order measuring <cdag_2 c_1>
    nn_CTM_bond_1r_up = env.measure_nn(cdag_up, c_up, bond=((2, 1), (2, 0)))  # horizontal bond
    nn_CTM_bond_2r_up = env.measure_nn(cdag_up, c_up, bond=((1, 1), (0, 1)))  # vertical bond
    nn_CTM_bond_1r_dn = env.measure_nn(cdag_dn, c_dn, bond=((2, 1), (2, 0)))  # horizontal bond
    nn_CTM_bond_2r_dn = env.measure_nn(cdag_dn, c_dn, bond=((1, 1), (0, 1)))  # vertical bonds

    print("Relative errors:")
    print(f"{abs(nn_CTM_bond_1_up - nn_bond_1_exact) / nn_bond_1_exact:0.5f}, " +
          f"{abs(nn_CTM_bond_1_dn - nn_bond_1_exact) / nn_bond_1_exact:0.5f}")
    print(f"{abs(nn_CTM_bond_2_up - nn_bond_2_exact) / nn_bond_2_exact:0.5f}, " +
          f"{abs(nn_CTM_bond_2_dn - nn_bond_2_exact) / nn_bond_2_exact:0.5f}")

    assert abs(nn_CTM_bond_1_up - nn_bond_1_exact) < 1e-4
    assert abs(nn_CTM_bond_1_dn - nn_bond_1_exact) < 1e-4
    assert abs(nn_CTM_bond_2_up - nn_bond_2_exact) < 1e-4
    assert abs(nn_CTM_bond_2_dn - nn_bond_2_exact) < 1e-4
    assert abs(nn_CTM_bond_1r_up - nn_bond_1_exact) < 1e-4
    assert abs(nn_CTM_bond_1r_dn - nn_bond_1_exact) < 1e-4
    assert abs(nn_CTM_bond_2r_up - nn_bond_2_exact) < 1e-4
    assert abs(nn_CTM_bond_2r_dn - nn_bond_2_exact) < 1e-4


def test_NTU_spinful_infinite(config_kwargs):
    """ Simulate purification of spinful fermions in an infinite system.s """
    print("Simulating spinful fermions in an infinite system. """)
    geometry = fpeps.CheckerboardLattice()

    mu_up, mu_dn = 0, 0  # chemical potential
    t_up, t_dn = 1, 1  # hopping amplitude
    U = 0
    beta = 0.1

    dbeta = 0.01
    D = 5

    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', **config_kwargs)
    I = ops.I()
    c_up, c_dn = ops.c(spin='u'), ops.c(spin='d')
    cdag_up, cdag_dn = ops.cp(spin='u'), ops.cp(spin='d')
    n_up, n_dn =  ops.n(spin='u'), ops.n(spin='d')

    g_hop_u = fpeps.gates.gate_nn_hopping(t_up, dbeta / 2, I, c_up, cdag_up)
    g_hop_d = fpeps.gates.gate_nn_hopping(t_dn, dbeta / 2, I, c_dn, cdag_dn)
    g_loc = fpeps.gates.gate_local_Coulomb(mu_up, mu_dn, U, dbeta / 2, I, n_up, n_dn)
    gates = fpeps.gates.distribute(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_loc)

    # initialized at infinite temperature
    psi = fpeps.product_peps(geometry, I)

    opts_svd_ctm = {'D_total': D * D, 'tol': 1e-8}
    opts_svd_evol = {"D_total": D, 'tol': 1e-14}

    steps = round((beta / 2) / dbeta)
    dbeta = (beta / 2) / steps

    infos = []
    init_steps = 2
    # first few steps are performed with NTU-NN+ to reach fixed peps bond dimensions.
    print("Evolve with NN+")
    env = fpeps.EnvNTU(psi, which='NN+')
    for step in range(init_steps):
        print(f"beta = {(step + 1) * dbeta:0.3f}" )
        info = fpeps.evolution_step_(env, gates, opts_svd=opts_svd_evol)
        infos.append(info)

    # after that we switch to fast Full Update
    # here it requirs Peps bond dimensions not to change in time
    print("Switching to full update")
    env = fpeps.EnvCTM(psi, init='eye')
    for _ in range(4):  # few CTM iterations to converge
        env.update_(opts_svd=opts_svd_ctm)

    for step in range(init_steps, steps):
        print(f"beta = {(step + 1) * dbeta:0.3f}" )
        info = fpeps.evolution_step_(env, gates, opts_svd=opts_svd_evol)
        infos.append(info)
        env.update_(opts_svd=opts_svd_ctm)  # update CTM tensors after a full evolution step.
        for inf in info:
            print(inf)

    print(f"Delta_mean: {fpeps.accumulated_truncation_error(infos, statistics='mean'):0.4f}")
    print(f"Delta_max : {fpeps.accumulated_truncation_error(infos, statistics='max'):0.4f}")
    with pytest.raises(yastn.YastnError):
        fpeps.accumulated_truncation_error(infos, statistics='other')
    #
    # CTMRG
    # convergence criteria for CTM based on total energy
    energy_old, tol_exp = 0, 1e-7

    # env = fpeps.EnvCTM(psi)
    for _ in range(10):  # we double-check convergence of CTM tensors
        env.update_(opts_svd=opts_svd_ctm)  # method='2site',
        cdagc_up = env.measure_nn(cdag_up, c_up)
        cdagc_dn = env.measure_nn(cdag_dn, c_dn)
        energy = -2 * mean([*cdagc_up.values(), *cdagc_dn.values()])

        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

    # analytical nn fermionic correlator at beta = 0.1 for 2D infinite lattice
    nn_exact = 0.02481459
    nn_CTM = mean([*cdagc_up.values(), *cdagc_dn.values()])
    print(f"{nn_CTM:0.6f} vs {nn_exact:0.6f}")
    print(f"Relativ error: {abs(nn_CTM - nn_exact) / nn_exact:0.6f}")
    assert abs(nn_CTM - nn_exact) < 1e-4


if __name__ == '__main__':
    pytest.main([__file__, "-vv", "--durations=0"])
