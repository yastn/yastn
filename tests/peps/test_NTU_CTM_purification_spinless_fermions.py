""" Test the expectation values of spinless fermions with analytical values of fermi sea for finite and infinite lattices """
import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps
try:
    from .configs import config_U1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1_R_fermionic as cfg


def test_NTU_spinless_finite():
    """ Simulate purification of free fermions in a small finite system """
    boundary = 'obc'
    Nx, Ny = 3, 2
    geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary=boundary)

    mu = 0  # chemical potential
    t = 1  # hopping amplitude
    beta = 0.2

    dbeta = 0.01
    D = 6

    ops = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = ops.I(), ops.c(), ops.cp()

    g_hop = fpeps.gates.gate_nn_hopping(t, dbeta / 2, fid, fc, fcdag)  # nn gate for 2D fermi sea
    g_loc = fpeps.gates.gate_local_occupation(mu, dbeta / 2, ops.I(), ops.n())  # local gate for spinless fermi sea
    gates = fpeps.gates_homogeneous(geometry, gates_nn=g_hop, gates_local=g_loc)

    psi = fpeps.product_peps(geometry, fid) # initialized at infinite temperature
    env = fpeps.EnvNTU(psi, which='NNN++')

    opts_svd = {"D_total": D, 'tol_block': 1e-15}
    steps = round((beta / 2) / dbeta)
    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta}" )
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="SVD")

    # convergence criteria for CTM based on total energy
    energy_old, tol_exp = 0, 1e-8

    opts_svd_ctm = {'D_total': 40, 'tol': 1e-10}

    env = fpeps.EnvCTM(psi)
    for step in fpeps.ctmrg_(env, max_sweeps=50, iterator_step=2, fix_signs=False, opts_svd=opts_svd_ctm):
        cdagc = env.measure_nn(fcdag, fc)
        energy = -2 * np.mean([*cdagc.values()])

        print("energy: ", energy)
        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

    nn_CTM_bond_1 = env.measure_nn(fcdag, fc, bond=((2, 0), (2, 1)))
    nn_CTM_bond_2 = env.measure_nn(fcdag, fc, bond=((0, 1), (1, 1)))

    # analytical nn fermionic correlator at beta = 0.2 for 2D finite (2, 3) lattice;
    nn_bond_1_exact = 0.04934701696955436  # bond between (1, 1) and (1, 2)
    nn_bond_2_exact = 0.04918555449042906  # bond between (0, 0) and (1, 0)
    assert pytest.approx(nn_CTM_bond_1, abs=1e-6) == nn_bond_1_exact
    assert pytest.approx(nn_CTM_bond_2, abs=1e-6) == nn_bond_2_exact


def test_NTU_spinless_infinite():
    """ Simulate purification of free fermions in an infinite system.s """
    geometry = fpeps.CheckerboardLattice()

    mu = 0  # chemical potential
    t = 1  # hopping amplitude
    beta = 0.2

    D = 6
    dbeta = 0.01

    ops = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = ops.I(), ops.c(), ops.cp()
    g_hop = fpeps.gates.gate_nn_hopping(t, dbeta / 2, fid, fc, fcdag)  # nn gate for 2D fermi sea
    g_loc = fpeps.gates.gate_local_occupation(mu, dbeta / 2, ops.I(), ops.n())  # local gate for spinless fermi sea
    gates = fpeps.gates_homogeneous(geometry, gates_nn=g_hop, gates_local=g_loc)

    psi = fpeps.product_peps(geometry, fid) # initialized at infinite temperature
    env = fpeps.EnvNTU(psi, which='NN')

    opts_svd = {"D_total": D, 'tol_block': 1e-15}
    steps = round((beta / 2) / dbeta)
    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta}" )
        out = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="SVD")

    # convergence criteria for CTM based on total energy
    energy_old, tol_exp = 0, 1e-7

    env = fpeps.EnvCTM(psi)
    opts_svd_ctm = {'D_total': 40, 'tol': 1e-10}
    for step in fpeps.ctmrg_(env, max_sweeps=50, iterator_step=1, opts_svd=opts_svd_ctm):
        cdagc = env.measure_nn(fcdag, fc)
        energy = -2 * np.mean([*cdagc.values()])

        print("energy: ", energy)
        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

    # analytical nn fermionic correlator at beta = 0.2 for 2D infinite lattice
    energy_exact = -0.09712706
    assert pytest.approx(energy, abs=1e-5) == energy_exact

if __name__ == '__main__':
    test_NTU_spinless_finite()
    test_NTU_spinless_infinite()
