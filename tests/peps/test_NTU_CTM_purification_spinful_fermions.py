""" Test the expectation values of spin 1/2 fermions with analytical values of fermi sea """
import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.ctm import nn_exp_dict, ctmrg, one_site_dict, EV2ptcorr

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg


def test_NTU_spinful_finite():
    """ Simulate purification of spinful fermions in a small finite system """
    print(" Simulating spinful fermions in a small finite system. ")
    boundary = 'obc'
    Nx, Ny = 3, 2
    geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary=boundary)

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
    n_int = n_up @ n_dn
    g_hop_u = fpeps.gates.gate_nn_hopping(t_up * dbeta / 2, fid, fc_up, fcdag_up)
    g_hop_d = fpeps.gates.gate_nn_hopping(t_dn * dbeta / 2, fid, fc_dn, fcdag_dn)
    g_loc = fpeps.gates.gate_local_Coulomb(mu_up, mu_dn, U, dbeta / 2, fid, n_up, n_dn)
    gates = fpeps.gates_homogeneous(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_loc)

    psi = fpeps.product_peps(geometry, fid) # initialized at infinite temperature
    env = fpeps.EnvNTU(psi, which='NNN++')

    opts_svd = {"D_total": D, 'tol_block': 1e-15}
    steps = np.rint((beta / 2) / dbeta).astype(int)
    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta}" )
        out = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="EAT")

    # convergence criteria for CTM based on total energy
    chi = 40  # environmental bond dimension
    tol = 1e-10  # truncation of singular values of CTM projectors
    max_sweeps = 50
    tol_exp = 1e-7   # difference of some observable must be lower than tolernace

    ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}
    cf_energy_old = 0
    opts_svd_ctm = {'D_total': chi, 'tol': tol}
    for step in ctmrg(psi, max_sweeps, iterator_step=1, AAb_mode=0, opts_svd=opts_svd_ctm):
        # first entry of the function gives average of one-site observables of the sites
        d_oc = one_site_dict(psi, step.env, n_int)
        obs_hor, obs_ver =  nn_exp_dict(psi, step.env, ops)
        cdagc_up = 0.5 * (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) +
                          sum(abs(val) for val in obs_ver.get('cdagc_up').values()))
        ccdag_up = 0.5 * (sum(abs(val) for val in obs_hor.get('ccdag_up').values()) +
                          sum(abs(val) for val in obs_ver.get('ccdag_up').values()))
        cdagc_dn = 0.5 * (sum(abs(val) for val in obs_hor.get('cdagc_dn').values()) +
                          sum(abs(val) for val in obs_ver.get('cdagc_dn').values()))
        ccdag_dn = 0.5 * (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) +
                          sum(abs(val) for val in obs_ver.get('cdagc_up').values()))

        cf_energy = U * sum(d_oc.values()) - (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn) / (Nx * Ny)

        print("Energy: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break
        cf_energy_old = cf_energy

    bd_h = fpeps.Bond(site0=(2, 0), site1=(2, 1))
    bd_v = fpeps.Bond(site0=(0, 1), site1=(1, 1))

    nn_CTM_bond_1_up = 0.5 * (abs(EV2ptcorr(psi, step.env, ops['cdagc_up'], bd_h.site0, bd_h.site1)) +
                              abs(EV2ptcorr(psi, step.env, ops['ccdag_up'], bd_h.site0, bd_h.site1)))
    nn_CTM_bond_2_up = 0.5 * (abs(EV2ptcorr(psi, step.env, ops['cdagc_up'], bd_v.site0, bd_v.site1)) +
                              abs(EV2ptcorr(psi, step.env, ops['ccdag_up'], bd_v.site0, bd_v.site1)))
    nn_CTM_bond_1_dn = 0.5 * (abs(EV2ptcorr(psi, step.env, ops['cdagc_dn'], bd_h.site0, bd_h.site1)) +
                              abs(EV2ptcorr(psi, step.env, ops['ccdag_dn'], bd_h.site0, bd_h.site1)))
    nn_CTM_bond_2_dn = 0.5 * (abs(EV2ptcorr(psi, step.env, ops['cdagc_dn'], bd_v.site0, bd_v.site1)) +
                              abs(EV2ptcorr(psi, step.env, ops['ccdag_dn'], bd_v.site0, bd_v.site1)))

    # analytical nn fermionic correlator at beta = 0.1 for 2D finite 2 x 3 lattice
    nn_bond_1_exact = 0.024917101651703362  # bond between (1, 1) and (1, 2)
    nn_bond_2_exact = 0.024896433958165112  # bond between (0, 0) and (1, 0)
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

    g_hop_u = fpeps.gates.gate_nn_hopping(t_up * dbeta / 2, fid, fc_up, fcdag_up)
    g_hop_d = fpeps.gates.gate_nn_hopping(t_dn * dbeta / 2, fid, fc_dn, fcdag_dn)
    g_loc = fpeps.gates.gate_local_Coulomb(mu_up, mu_dn, U, dbeta / 2, fid, n_up, n_dn)
    gates = fpeps.gates_homogeneous(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_loc)

    # initialized at infinite temperature
    psi = fpeps.product_peps(geometry, fid)

    env = fpeps.EnvNTU(psi, which='NNN++')

    opts_svd = {"D_total": D, 'tol_block': 1e-15}
    steps = np.rint((beta / 2) / dbeta).astype(int)
    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta}" )
        out = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="EAT")

    # convergence criteria for CTM based on total energy
    chi = 40  # environmental bond dimension
    tol = 1e-10  # truncation of singular values of CTM projectors
    max_sweeps = 50
    tol_exp = 1e-7  # difference of some observable must be lower than tolernace

    ops = {'cdagc_up': {'l': fcdag_up, 'r': fc_up},
           'ccdag_up': {'l': fc_up, 'r': fcdag_up},
           'cdagc_dn': {'l': fcdag_dn, 'r': fc_dn},
           'ccdag_dn': {'l': fc_dn, 'r': fcdag_dn}}

    cf_energy_old = 0
    opts_svd_ctm = {'D_total': chi, 'tol': tol}

    for step in ctmrg(psi, max_sweeps, iterator_step=2, AAb_mode=0, opts_svd=opts_svd_ctm):
        obs_hor, obs_ver =  nn_exp_dict(psi, step.env, ops)
        cdagc_up = (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) +
                    sum(abs(val) for val in obs_ver.get('cdagc_up').values()))
        ccdag_up = (sum(abs(val) for val in obs_hor.get('ccdag_up').values()) +
                    sum(abs(val) for val in obs_ver.get('ccdag_up').values()))
        cdagc_dn = (sum(abs(val) for val in obs_hor.get('cdagc_dn').values()) +
                    sum(abs(val) for val in obs_ver.get('cdagc_dn').values()))
        ccdag_dn = (sum(abs(val) for val in obs_hor.get('cdagc_up').values()) +
                    sum(abs(val) for val in obs_ver.get('cdagc_up').values()))
        cf_energy = -0.125 * (cdagc_up + ccdag_up +cdagc_dn + ccdag_dn)

        print("Energy: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break
        cf_energy_old = cf_energy

    nn_CTM = (cdagc_up + ccdag_up + cdagc_dn + ccdag_dn) / 16

    # analytical nn fermionic correlator at beta = 0.1 for 2D infinite lattice with checkerboard ansatz
    nn_exact = 0.02481459
    print(nn_CTM, 'vs', nn_exact)
    assert pytest.approx(nn_CTM, abs=1e-4) == nn_exact


if __name__ == '__main__':
    test_NTU_spinful_finite()
    test_NTU_spinful_infinite()
