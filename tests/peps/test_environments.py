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


def test_finite_spinless_boundary_mps_ctmrg():

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
        vR0 = step.env.env2mps(n=ny, dirn='r')
        vR1 = mpsenv.env2mps(n=ny, dirn='r')
        vL0 = step.env.env2mps(n=ny, dirn='l')
        vL1 = mpsenv.env2mps(n=ny, dirn='l')

        print(mps.vdot(vR0, vR1) / (vR0.norm() * vR1.norm()))  # problem with phase in peps?
        print(mps.vdot(vL0, vL1) / (vL0.norm() * vL1.norm()))
        assert abs(abs(mps.vdot(vR0, vR1)) / (vR0.norm() * vR1.norm()) - 1) < 1e-7
        assert abs(abs(mps.vdot(vL0, vL1)) / (vL0.norm() * vL1.norm()) - 1) < 1e-7

    for nx in range(psi.Nx):
        vT0 = step.env.env2mps(n=nx, dirn='t')
        vT1 = mpsenv.env2mps(n=nx, dirn='t')
        vB0 = step.env.env2mps(n=nx, dirn='b')
        vB1 = mpsenv.env2mps(n=nx, dirn='b')

        print(mps.vdot(vT0, vT1) / (vT0.norm() * vT1.norm()))  # problem with phase in peps?
        print(mps.vdot(vB0, vB1) / (vB0.norm() * vB1.norm()))
        assert abs(abs(mps.vdot(vT0, vT1)) / (vT0.norm() * vT1.norm()) - 1) < 1e-7
        assert abs(abs(mps.vdot(vB0, vB1)) / (vB0.norm() * vB1.norm()) - 1) < 1e-7


def test_spinless_infinite_approx():
    """ Simulate purification of free fermions in an infinite system.s """
    geometry = fpeps.SquareLattice(dims=(3, 3), boundary='infinite')

    mu, t, beta = 0, 1, 0.5  # chemical potential
    D = 6
    dbeta = 0.05

    ops = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    fid, fc, fcdag = ops.I(), ops.c(), ops.cp()
    GA_nn, GB_nn = fpeps.gates.gate_hopping(t, dbeta / 2, fid, fc, fcdag)  # nn gate for 2D fermi sea
    g_loc = fpeps.gates.gate_local_fermi_sea(mu, dbeta / 2, fid, fc, fcdag) # local gate for spinless fermi sea
    g_nn = [(GA_nn, GB_nn)]
    gates = fpeps.gates_homogeneous(geometry, g_nn, g_loc)

    psi = fpeps.product_peps(geometry, fid) # initialized at infinite temperature
    env = fpeps.EnvNTU(psi, which='NNh')

    opts_svd = {"D_total": D , 'tol_block': 1e-15}
    steps = np.rint((beta / 2) / dbeta).astype(int)
    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta}" )
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="SVD")

    opts_svd = {"D_total": 2 * D, 'tol_block': 1e-15}

    envs = {}
    envs['NNN']  = fpeps.EnvNTU(psi, which='NNN')
    envs['43']   = fpeps.EnvApproximate(psi, which='43', opts_svd= opts_svd)
    envs['NNNh'] = fpeps.EnvNTU(psi, which='NNNh')
    envs['43h']  = fpeps.EnvApproximate(psi, which='43h', opts_svd= opts_svd)
    envs['65']   = fpeps.EnvApproximate(psi, which='65', opts_svd= opts_svd)
    envs['65h']  = fpeps.EnvApproximate(psi, which='65h', opts_svd= opts_svd)
    envs['87']   = fpeps.EnvApproximate(psi, which='87', opts_svd= opts_svd)
    envs['87h']  = fpeps.EnvApproximate(psi, which='87h', opts_svd= opts_svd)

    for st0, st1 in [[(0, 0), (0, 1)], [(0, 1), (1, 1)]]:
        bd = fpeps.Bond(st0, st1)
        QA, QB = psi[st0], psi[st1]
        Gs = {k: env.bond_metric(bd, QA, QB) for k, env in envs.items()}
        Gs = {k: v / v.norm() for k, v in Gs.items()}
        assert (Gs['NNN'] - Gs['43']).norm() < 1e-6
        assert (Gs['NNNh'] - Gs['43h']).norm() < 1e-6
        assert ((Gs['43'] - Gs['43h']).norm()) < 1e-3
        assert ((Gs['43h'] - Gs['65']).norm()) < 1e-4
        assert ((Gs['65'] - Gs['65h']).norm()) < 1e-5
        assert ((Gs['65h'] - Gs['87']).norm()) < 1e-5
        assert ((Gs['87'] - Gs['87h']).norm()) < 1e-5

if __name__ == '__main__':
    test_finite_spinless_boundary_mps_ctmrg()
    test_spinless_infinite_approx()
