""" Test PEPS measurments with MpsBoundary in a product state. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps

try:
    from .configs import config_dense as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_dense as cfg


tol = 1e-12

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
        proj_env.sample_MC_(proj_psi, smpl, smpl1, smpl2, psi, prs, opts_svd, opts_var, trial=trial)
        assert all(vloc[smpl1[s]] == vals[s] for s in sites)
        assert all(vloc[smpl2[s]] == vals[s] for s in sites)

    with pytest.raises(yastn.YastnError):
        proj_env.sample_MC_(proj_psi, smpl, smpl1, smpl2, psi, prs, opts_svd, opts_var, trial="some")
        # trial='some' not supported.


if __name__ == '__main__':
    test_mpsboundary_measure(boundary='obc')
    test_mpsboundary_measure(boundary='cylinder')
