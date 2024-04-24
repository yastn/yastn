""" Test the expectation values of spin-1/2 fermions with analytical values of fermi sea """
import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg


def test_NTU_spinful_finite():
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
    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
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
    opts_svd = {"D_total": D, 'tol_block': 1e-15}
    steps = round((beta / 2) / dbeta)
    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta}" )
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="EAT")


    # convergence criteria for CTM based on total energy
    energy_old, tol_exp = 0, 1e-7
    opts_svd_ctm = {'D_total': 40, 'tol': 1e-10}

    env = fpeps.EnvCTM(psi)

    for _ in range(50):
        env.update_(opts_svd=opts_svd_ctm)  # single CMTRG sweep

        # calculate expectation values
        d_oc = env.measure_1site(n_int)
        cdagc_up = env.measure_nn(cdag_up, c_up)  # calculate for all unique bonds
        cdagc_dn = env.measure_nn(cdag_dn, c_dn)  # -> {bond: value}

        energy = U * sum(d_oc.values()) - sum(cdagc_up.values()) - sum(cdagc_dn.values())

        print("Energy: ", energy)
        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

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

    print(nn_CTM_bond_1_up, nn_CTM_bond_1_dn, 'vs', nn_bond_1_exact)
    print(nn_CTM_bond_2_up, nn_CTM_bond_2_dn, 'vs', nn_bond_2_exact)
    assert pytest.approx(nn_CTM_bond_1_up, abs=1e-4) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_1_dn, abs=1e-4) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_2_up, abs=1e-4) == nn_bond_2_exact
    assert pytest.approx(nn_CTM_bond_2_dn, abs=1e-4) == nn_bond_2_exact

    assert pytest.approx(nn_CTM_bond_1r_up, abs=1e-4) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_1r_dn, abs=1e-4) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_2r_up, abs=1e-4) == nn_bond_2_exact
    assert pytest.approx(nn_CTM_bond_2r_dn, abs=1e-4) == nn_bond_2_exact


def test_NTU_spinful_infinite():
    """ Simulate purification of spinful fermions in an infinite system.s """
    print("Simulating spinful fermions in an infinite system. """)
    geometry = fpeps.CheckerboardLattice()

    mu_up, mu_dn = 0, 0  # chemical potential
    t_up, t_dn = 1, 1  # hopping amplitude
    U = 0
    beta = 0.1

    dbeta = 0.01
    D = 8

    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    I = ops.I()
    c_up, c_dn, cdag_up, cdag_dn = ops.c(spin='u'), ops.c(spin='d'), ops.cp(spin='u'), ops.cp(spin='d')
    n_up, n_dn =  ops.n(spin='u'), ops.n(spin='d')

    g_hop_u = fpeps.gates.gate_nn_hopping(t_up, dbeta / 2, I, c_up, cdag_up)
    g_hop_d = fpeps.gates.gate_nn_hopping(t_dn, dbeta / 2, I, c_dn, cdag_dn)
    g_loc = fpeps.gates.gate_local_Coulomb(mu_up, mu_dn, U, dbeta / 2, I, n_up, n_dn)
    gates = fpeps.gates.distribute(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_loc)

    # initialized at infinite temperature
    psi = fpeps.product_peps(geometry, I)

    env = fpeps.EnvNTU(psi, which='NN++')
    opts_svd = {"D_total": D, 'tol_block': 1e-15}
    steps = round((beta / 2) / dbeta)
    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta}" )
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="EAT")


    # CTMRG
    # convergence criteria for CTM based on total energy
    energy_old, tol_exp = 0, 1e-7

    env = fpeps.EnvCTM(psi)
    opts_svd_ctm = {'D_total': 40, 'tol': 1e-10}

    for _ in range(50):
        env.update_(opts_svd=opts_svd_ctm)  # method='2site',
        cdagc_up = env.measure_nn(cdag_up, c_up)
        cdagc_dn = env.measure_nn(cdag_dn, c_dn)
        energy = -2 * np.mean([*cdagc_up.values(), *cdagc_dn.values()])

        print("Energy: ", energy)
        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

    # analytical nn fermionic correlator at beta = 0.1 for 2D infinite lattice
    nn_exact = 0.02481459
    nn_CTM = np.mean([*cdagc_up.values(), *cdagc_dn.values()])
    print(nn_CTM, 'vs', nn_exact)
    assert pytest.approx(nn_CTM, abs=1e-4) == nn_exact


if __name__ == '__main__':
    test_NTU_spinful_finite()
    test_NTU_spinful_infinite()
