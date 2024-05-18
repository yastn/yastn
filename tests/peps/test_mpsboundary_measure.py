""" Test PEPS measurments with MpsBoundary in a product state. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.mps as mps
try:
    from .configs import config as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config as cfg


tol = 1e-12

def mean(xs):
    return sum(xs) / len(xs)


@pytest.mark.parametrize("boundary", ["obc", "cylinder"])
def test_mpsboundary_measure(boundary):
    """ Initialize a product PEPS and perform a set of measurment. """

    ops = yastn.operators.Spin1(sym='Z3', backend=cfg.backend, default_device=cfg.default_device)

    # initialized PEPS in a product state
    geometry = fpeps.SquareLattice(dims=(4, 3), boundary=boundary)
    sites = geometry.sites()
    vals = [1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1]
    vals = dict(zip(sites, vals))
    occs = {s: ops.vec_z(val=v) for s, v in vals.items()}
    psi = fpeps.product_peps(geometry, occs)

    opts_svd = {'D_total': 2, 'tol': 1e-10}
    env = fpeps.EnvBoundaryMps(psi, opts_svd=opts_svd, setup='lr')

    esz = env.measure_1site(ops.sz())
    assert all(abs(v - esz[s]) < tol for s, v in vals.items())

    eszz = env.measure_2site(ops.sz(), ops.sz(), opts_svd=opts_svd)
    assert all(abs(vals[s1] * vals[s2] - v) < tol for (s1, s2), v in eszz.items())

    vloc = [-1, 0, 1]
    pr = [ops.vec_z(val=v) for v in vloc]
    pr2 = [x.tensordot(x.conj(), axes=((), ())) for x in pr]
    pr2s = {s: pr2[:] for s in sites}

    smpl = env.sample(pr2s)
    assert all(vloc[smpl[s]] == vals[s] for s in sites)


    prs = {s: pr[:] for s in sites}

    proj_psi = psi.copy()
    for k in psi.sites():
        leg = psi[k].get_legs(axes=-1)
        _, leg = leg.unfuse_leg()
        for i, t in enumerate(prs[k]):
            prs[k][i] = t.add_leg(leg=leg).fuse_legs(axes=[(0, 1)]).conj()

        proj_psi[k] = psi[k] @ prs[k][smpl[k]]

    proj_env = fpeps.EnvBoundaryMps(proj_psi, opts_svd=opts_svd)

    smpl1 = {}
    smpl2 = {}

    opts_var = {"max_sweeps": 2}
    for trial in ["uniform", "local"]:
        proj_env.sample_MC_(smpl, smpl1, smpl2, psi, prs, opts_svd, opts_var, trial=trial)
        assert all(vloc[smpl1[s]] == vals[s] for s in sites)
        assert all(vloc[smpl2[s]] == vals[s] for s in sites)

    with pytest.raises(yastn.YastnError):
        proj_env.sample_MC_(smpl, smpl1, smpl2, psi, prs, opts_svd, opts_var, trial="some")
        # trial='some' not supported.


def test_finite_spinless_boundary_mps_ctmrg():
    """ compare boundary Mps with CTM"""
    boundary = 'obc'
    Nx, Ny = 3, 2
    geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary=boundary)

    mu = 0  # chemical potential
    t = 1  # hopping amplitude
    beta = 0.1
    dbeta = 0.025

    D = 5

    ops = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    I, c, cdag = ops.I(), ops.c(), ops.cp()

    g_hop = fpeps.gates.gate_nn_hopping(t, dbeta / 2, I, c, cdag)  # nn gate for 2D fermi sea
    gates = fpeps.gates.distribute(geometry, gates_nn=g_hop)

    psi = fpeps.product_peps(geometry, I) # initialized at infinite temperature
    env = fpeps.EnvNTU(psi, which='NN+')

    opts_svd = {'D_total': D, 'tol_block': 1e-15}

    steps = round((beta / 2) / dbeta)
    dbeta = (beta / 2) / steps

    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta:0.3f}" )
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd)


    # convergence criteria for CTM based on total energy
    energy_old, tol_exp = 0, 1e-7

    env = fpeps.EnvCTM(psi)
    opts_svd_ctm = {'D_total': 30, 'tol': 1e-10}
    for _ in range(50):
        env.update_(opts_svd=opts_svd_ctm)
        cdagc = env.measure_nn(cdag, c)
        energy = -2 * mean([*cdagc.values()])

        print("energy: ", energy)
        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

    mpsenv = fpeps.EnvBoundaryMps(psi, opts_svd=opts_svd_ctm, setup='tlbr')

    for ny in range(psi.Ny):
        vR0 = env.boundary_mps(n=ny, dirn='r')
        vR1 = mpsenv.boundary_mps(n=ny, dirn='r')
        vL0 = env.boundary_mps(n=ny, dirn='l')
        vL1 = mpsenv.boundary_mps(n=ny, dirn='l')

        print(mps.vdot(vR0, vR1) / (vR0.norm() * vR1.norm()))  # problem with phase in peps?
        print(mps.vdot(vL0, vL1) / (vL0.norm() * vL1.norm()))
        assert abs(mps.vdot(vR0, vR1) / (vR0.norm() * vR1.norm()) - 1) < 1e-7
        assert abs(mps.vdot(vL0, vL1) / (vL0.norm() * vL1.norm()) - 1) < 1e-7

    for nx in range(psi.Nx):
        vT0 = env.boundary_mps(n=nx, dirn='t')
        vT1 = mpsenv.boundary_mps(n=nx, dirn='t')
        vB0 = env.boundary_mps(n=nx, dirn='b')
        vB1 = mpsenv.boundary_mps(n=nx, dirn='b')

        print(mps.vdot(vT0, vT1) / (vT0.norm() * vT1.norm()))  # problem with phase in peps?
        print(mps.vdot(vB0, vB1) / (vB0.norm() * vB1.norm()))
        assert abs(mps.vdot(vT0, vT1) / (vT0.norm() * vT1.norm()) - 1) < 1e-7
        assert abs(mps.vdot(vB0, vB1) / (vB0.norm() * vB1.norm()) - 1) < 1e-7



if __name__ == '__main__':
    test_mpsboundary_measure(boundary='obc')
    test_mpsboundary_measure(boundary='cylinder')
    test_finite_spinless_boundary_mps_ctmrg()
