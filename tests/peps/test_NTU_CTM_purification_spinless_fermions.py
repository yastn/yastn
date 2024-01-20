""" Test the expectation values of spinless fermions with analytical values of fermi sea for finite and infinite lattices """
import numpy as np
import pytest
import logging
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.ctm import nn_exp_dict, ctmrg, EV2ptcorr
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

    G_hop = fpeps.gates.gate_hopping(t, dbeta / 2, fid, fc, fcdag)  # nn gate for 2D fermi sea
    g_loc = fpeps.gates.gate_local_fermi_sea(mu, dbeta / 2, fid, fc, fcdag) # local gate for spinless fermi sea
    g_nn = [G_hop]
    gates = fpeps.gates_homogeneous(geometry, g_nn, g_loc)

    psi = fpeps.product_peps(geometry, fid) # initialized at infinite temperature
    env = fpeps.EnvNTU(psi)

    opts_svd = {"D_total": D, 'tol_block': 1e-15}
    steps = np.rint((beta / 2) / dbeta).astype(int)
    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta}" )
        out = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="SVD")

    # convergence criteria for CTM based on total energy
    chi = 40  # environmental bond dimension
    tol = 1e-10  # truncation of singular values of ctm projectors
    max_sweeps = 50
    tol_exp = 1e-7  # difference of some expectation value must be lower than tolerance

    ops = {'cdagc': {'l': fcdag, 'r': fc},
           'ccdag': {'l': fc, 'r': fcdag}}

    cf_energy_old = 0
    opts_svd_ctm = {'D_total': chi, 'tol': tol}
    for step in ctmrg(psi, max_sweeps, iterator_step=2, AAb_mode=0, fix_signs=False, opts_svd=opts_svd_ctm):
        obs_hor, obs_ver =  nn_exp_dict(psi, step.env, ops)

        cdagc = (sum(abs(val) for val in obs_hor['cdagc'].values()) +
                 sum(abs(val) for val in obs_ver['cdagc'].values()))
        ccdag = (sum(abs(val) for val in obs_hor['ccdag'].values()) +
                 sum(abs(val) for val in obs_ver['ccdag'].values()))
        cf_energy = - (cdagc + ccdag) / (Nx * Ny)

        print("energy: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break
        cf_energy_old = cf_energy

    bd_h = fpeps.Bond(site0=(2, 0), site1=(2, 1))
    bd_v = fpeps.Bond(site0=(0, 1), site1=(1, 1))

    nn_CTM_bond_1 = 0.5 * (abs(EV2ptcorr(psi, step.env, ops['cdagc'], bd_h.site0, bd_h.site1)) +
                           abs(EV2ptcorr(psi, step.env, ops['ccdag'], bd_h.site0, bd_h.site1)))
    nn_CTM_bond_2 = 0.5 * (abs(EV2ptcorr(psi, step.env, ops['cdagc'], bd_v.site0, bd_v.site1)) +
                           abs(EV2ptcorr(psi, step.env, ops['ccdag'], bd_v.site0, bd_v.site1)))

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

    D = 8
    dbeta = 0.01

    ops = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = ops.I(), ops.c(), ops.cp()
    GA_nn, GB_nn = fpeps.gates.gate_hopping(t, dbeta / 2, fid, fc, fcdag)  # nn gate for 2D fermi sea
    g_loc = fpeps.gates.gate_local_fermi_sea(mu, dbeta / 2, fid, fc, fcdag) # local gate for spinless fermi sea
    g_nn = [(GA_nn, GB_nn)]
    gates = fpeps.gates_homogeneous(geometry, g_nn, g_loc)

    psi = fpeps.product_peps(geometry, fid) # initialized at infinite temperature
    env = fpeps.EnvNTU(psi)

    opts_svd = {"D_total": D, 'tol_block': 1e-15}
    steps = np.rint((beta / 2) / dbeta).astype(int)
    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta}" )
        out = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="SVD")

    # convergence criteria for CTM based on total energy
    chi = 40  # environmental bond dimension
    tol = 1e-10  # CTM projectors
    max_sweeps = 50
    tol_exp = 1e-7  # difference of some observable must be lower than tolernace

    ops = {'cdagc': {'l': fcdag, 'r': fc},
           'ccdag': {'l': fc, 'r': fcdag}}
    cf_energy_old = 0
    opts_svd_ctm = {'D_total': chi, 'tol': tol}
    for step in ctmrg(psi, max_sweeps, iterator_step=1, AAb_mode=0, opts_svd=opts_svd_ctm):
        obs_hor, obs_ver =  nn_exp_dict(psi, step.env, ops)

        cdagc = (sum(abs(val) for val in obs_hor.get('cdagc').values()) +
                 sum(abs(val) for val in obs_ver.get('cdagc').values()))
        ccdag = (sum(abs(val) for val in obs_hor.get('ccdag').values()) +
                 sum(abs(val) for val in obs_ver.get('ccdag').values()))
        cf_energy = -0.125 * (cdagc + ccdag)
        print("energy: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break
        cf_energy_old = cf_energy

    obs_hor, obs_ver = nn_exp_dict(psi, step.env, ops)
    nn_CTM = 0.125 * (cdagc + ccdag)

    # analytical nn fermionic correlator at beta = 0.2 for 2D infinite lattice with checkerboard ansatz
    nn_exact = 0.04856353
    print(nn_CTM)
    assert pytest.approx(nn_CTM, abs=1e-5) == nn_exact

if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    test_NTU_spinless_finite()
    test_NTU_spinless_infinite()
