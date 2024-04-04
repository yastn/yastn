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
    fid = ops.I()
    fc_up, fc_dn, fcdag_up, fcdag_dn = ops.c(spin='u'), ops.c(spin='d'), ops.cp(spin='u'), ops.cp(spin='d')
    n_up, n_dn =  ops.n(spin='u'), ops.n(spin='d')
    n_int = n_up @ n_dn

    g_hop_u = fpeps.gates.gate_nn_hopping(t_up, dbeta / 2, fid, fc_up, fcdag_up)
    g_hop_d = fpeps.gates.gate_nn_hopping(t_dn, dbeta / 2, fid, fc_dn, fcdag_dn)
    g_loc = fpeps.gates.gate_local_Coulomb(mu_up, mu_dn, U, dbeta / 2, fid, n_up, n_dn)
    gates = fpeps.gates.distribute(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_loc)


    # initialized infinite temperature purification
    psi = fpeps.product_peps(geometry, fid)

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
        cdagc_up = env.measure_nn(fcdag_up, fc_up)  # calculate for all unique bonds
        cdagc_dn = env.measure_nn(fcdag_dn, fc_dn)  # -> {bond: value}

        energy = U * sum(d_oc.values()) - sum(cdagc_up.values()) - sum(cdagc_dn.values())

        print("Energy: ", energy)
        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

    # analytical nn fermionic correlator at beta = 0.1 for 2D finite 2 x 3 lattice
    nn_bond_1_exact = 0.024917101651703362  # bond between (1, 1) and (1, 2)   # this requires checking; bonds exact vs CTM should match
    nn_bond_2_exact = 0.024896433958165112  # bond between (0, 0) and (1, 0)

    nn_CTM_bond_1_up = env.measure_nn(fcdag_up, fc_up, bond=((2, 0), (2, 1)))  # horizontal bond
    nn_CTM_bond_2_up = env.measure_nn(fcdag_up, fc_up, bond=((0, 1), (1, 1)))  # vertical bond
    nn_CTM_bond_1_dn = env.measure_nn(fcdag_dn, fc_dn, bond=((2, 0), (2, 1)))  # horizontal bond
    nn_CTM_bond_2_dn = env.measure_nn(fcdag_dn, fc_dn, bond=((0, 1), (1, 1)))  # vertical bond

    print(nn_CTM_bond_1_up, nn_CTM_bond_1_dn, 'vs', nn_bond_1_exact)
    print(nn_CTM_bond_2_up, nn_CTM_bond_2_dn, 'vs', nn_bond_2_exact)
    assert pytest.approx(nn_CTM_bond_1_up, abs=1e-4) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_1_dn, abs=1e-4) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_2_up, abs=1e-4) == nn_bond_2_exact
    assert pytest.approx(nn_CTM_bond_2_dn, abs=1e-4) == nn_bond_2_exact


def test_NTU_spinful_infinite():
    """ Simulate purification of spinful fermions in an infinite system.s """
    print("Simulating spinful fermions in an infinite system. """)
    geometry = fpeps.CheckerboardLattice()

    mu_up, mu_dn = 0, 0  # chemical potential
    t_up, t_dn = 1, 1  # hopping amplitude
    U = 0
    beta = 0.1

    dbeta = 0.025
    D = 8

    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', backend=cfg.backend, default_device=cfg.default_device)
    fid = ops.I()
    fc_up, fc_dn, fcdag_up, fcdag_dn = ops.c(spin='u'), ops.c(spin='d'), ops.cp(spin='u'), ops.cp(spin='d')
    n_up, n_dn =  ops.n(spin='u'), ops.n(spin='d')

    g_hop_u = fpeps.gates.gate_nn_hopping(t_up, dbeta / 2, fid, fc_up, fcdag_up)
    g_hop_d = fpeps.gates.gate_nn_hopping(t_dn, dbeta / 2, fid, fc_dn, fcdag_dn)
    g_loc = fpeps.gates.gate_local_Coulomb(mu_up, mu_dn, U, dbeta / 2, fid, n_up, n_dn)
    gates = fpeps.gates.distribute(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_loc)

    # initialized at infinite temperature
    psi = fpeps.product_peps(geometry, fid)

    env = fpeps.EnvNTU(psi, which='NNN++')
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
        cdagc_up = env.measure_nn(fcdag_up, fc_up)
        cdagc_dn = env.measure_nn(fcdag_dn, fc_dn)
        energy = -2 * np.mean([*cdagc_up.values(), *cdagc_dn.values()])

        print("Energy: ", energy)
        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

    # analytical nn fermionic correlator at beta = 0.1 for 2D infinite lattice
    nn_exact = 0.02481459
    nn_CTM = np.mean([*cdagc_up.values(), *cdagc_dn.values()])
    print(nn_CTM, 'vs', nn_exact, nn_CTM - nn_exact)
    assert pytest.approx(nn_CTM, abs=1e-4) == nn_exact


if __name__ == '__main__':
    test_NTU_spinful_finite()
    test_NTU_spinful_infinite()
