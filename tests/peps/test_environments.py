""" Test the expectation values of spinless fermions with analytical values of fermi sea for finite and infinite lattices """
import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.mps as mps
from yastn.tn.fpeps.ctm import nn_exp_dict, ctmrg, one_site_dict, EV2ptcorr

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg


def test_envs_spinless_finite():

    boundary = 'obc'
    Nx, Ny = 3, 2
    geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary=boundary)

    mu = 0  # chemical potential
    t = 1  # hopping amplitude
    beta = 0.2
    dbeta = 0.025

    D = 6

    opt = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = opt.I(), opt.c(), opt.cp()

    GA_nn, GB_nn = fpeps.gates.gate_hopping(t, dbeta / 2, fid, fc, fcdag)  # nn gate for 2D fermi sea
    g_loc = fpeps.gates.gate_local_fermi_sea(mu, dbeta / 2, fid, fc, fcdag)  # local gate for spinless fermi sea
    g_nn = [(GA_nn, GB_nn)]
    gates = fpeps.gates_homogeneous(geometry, g_nn, g_loc)

    psi = fpeps.product_peps(geometry, fid) # initialized at infinite temperature
    env = fpeps.EnvNTU(psi, which='NNh')

    opts_svd = {'D_total': D, 'tol_block': 1e-15}
    steps = np.rint((beta / 2) / dbeta).astype(int)
    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta}" )
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="EAT")


    # convergence criteria for CTM based on total energy
    chi = 30 # environmental bond dimension
    tol = 1e-10 # truncation of singular values of ctm projectors
    max_sweeps = 50
    tol_exp = 1e-7   # difference of some expectation value must be lower than tolerance

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

        cf_energy = - (cdagc + ccdag) / (Nx * Ny)

        print("energy: ", cf_energy)
        if abs(cf_energy - cf_energy_old) < tol_exp:
            break # here break if the relative differnece is below tolerance
        cf_energy_old = cf_energy

    mpsenv = fpeps.EnvBoundaryMps(psi, opts_svd=opts_svd_ctm, setup='tlbr')

    for ny in range(psi.Ny):
        vR0 = step.env.env2mps(index=ny, index_type='r')
        vR1 = mpsenv.env2mps(index=ny, index_type='r')
        vL0 = step.env.env2mps(index=ny, index_type='l')
        vL1 = mpsenv.env2mps(index=ny, index_type='l')

        print(mps.vdot(vR0, vR1) / (vR0.norm() * vR1.norm()))  # problem with phase in peps?
        print(mps.vdot(vL0, vL1) / (vL0.norm() * vL1.norm()))
        assert abs(abs(mps.vdot(vR0, vR1)) / (vR0.norm() * vR1.norm()) - 1) < 1e-7
        assert abs(abs(mps.vdot(vL0, vL1)) / (vL0.norm() * vL1.norm()) - 1) < 1e-7

    for nx in range(psi.Nx):
        vT0 = step.env.env2mps(index=nx, index_type='t')
        vT1 = mpsenv.env2mps(index=nx, index_type='t')
        vB0 = step.env.env2mps(index=nx, index_type='b')
        vB1 = mpsenv.env2mps(index=nx, index_type='b')

        print(mps.vdot(vT0, vT1) / (vT0.norm() * vT1.norm()))  # problem with phase in peps?
        print(mps.vdot(vB0, vB1) / (vB0.norm() * vB1.norm()))
        assert abs(abs(mps.vdot(vT0, vT1)) / (vT0.norm() * vT1.norm()) - 1) < 1e-7
        assert abs(abs(mps.vdot(vB0, vB1)) / (vB0.norm() * vB1.norm()) - 1) < 1e-7


if __name__ == '__main__':
    test_envs_spinless_finite()
