import numpy as np
import pytest
import scipy.linalg
import yastn
try:
    from .configs import config_dense, config_U1, config_Z2
except ImportError:
    from configs import config_dense, config_U1, config_Z2


def run_expmv(A, v, tau, tol, ncv, hermitian):
    """ tests of yastn.linalg.expmv() for calculating expm(tau * A) * v """
    if hermitian:
        A = (A + A.conj().transpose(axes=(2, 3, 0, 1))) / 2
    f = lambda x: yastn.tensordot(A, x, axes=((2, 3), (0, 1)))
    An = A.to_numpy()
    sA = An.shape
    An = An.reshape((sA[0] * sA[1], sA[2] * sA[3]))
    vn = v.to_numpy().reshape(-1)
    wn = scipy.linalg.expm(tau * An) @ vn

    out, info = yastn.expmv(f, v, tau, tol, ncv, hermitian=hermitian, normalize=False, return_info=True)
    assert out.are_independent(v)
    w = out.to_numpy().reshape(-1)
    normwn = np.linalg.norm(wn)
    err = np.linalg.norm(w - wn) / normwn if normwn > 0 else np.linalg.norm(w - wn)
    print(info, err)
    assert err <= tol

    if scipy.linalg.norm(vn) > 0:
        wn /= scipy.linalg.norm(wn)
        out = yastn.expmv(f, v, tau, tol, ncv, hermitian=hermitian, normalize=True)
        assert out.are_independent(v)
        w = out.to_numpy().reshape(-1)
        assert np.linalg.norm(w - wn) <= tol * np.linalg.norm(wn)
    else:
        with pytest.raises(yastn.YastnError):
            out = yastn.expmv(f, v, tau, tol, ncv, hermitian=hermitian, normalize=True)


@pytest.mark.parametrize("D, ncv, tau, tol", [(8, 5, 2., 1e-10), (8, 5, 2j, 1e-10), (8, 5, -2., 1e-10), (8, 5, -2j, 1e-10), (8, 5, 0, 1e-10), (4, 20, 2, 1e-10)])
def test_expmv(D, ncv, tau, tol):
    """ initialize test of yastn.expmv() for various symmetries. """
    leg_dense = yastn.Leg(config_dense, s=1, D=[D])
    leg_Z2 = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(D//2, D//2))
    leg_U1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(D//4, D//2, D//4))

    for cfg, leg in ((config_dense, leg_dense), (config_Z2, leg_Z2), (config_U1, leg_U1)):
        A = yastn.rand(config=cfg, legs=[leg.conj(), leg, leg, leg.conj()], dtype='complex128')
        v = yastn.rand(config=cfg, legs=[leg.conj(), leg], dtype='complex128')
        run_expmv(A, v, tau, tol, ncv, hermitian=True)
        run_expmv(A, v, tau, tol, ncv, hermitian=False)

    v0 = 0 * v
    A0 = 0 * A
    run_expmv(A0, v, tau, tol, ncv, hermitian=True)
    run_expmv(A0, v, tau, tol, ncv, hermitian=False)
    run_expmv(A, v0, tau, tol, ncv, hermitian=True)
    run_expmv(A, v0, tau, tol, ncv, hermitian=False)



@pytest.mark.parametrize("t, tol", [(2.0j, 1e-10), (2.0j, 1e-4), (-2.0j, 1e-10), (-2.0j, 1e-4), (2.0, 1e-10), (2.0, 1e-4), (-2.0, 1e-10), (-2.0, 1e-4)])
def test_expmv_tm(t, tol):
    """ combining yastn.expmv() with more complicated contraction """
    legs = [yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4))]
    a = yastn.rand(config=config_U1, legs=legs)

    # dense transfer matrix build from a
    tm = yastn.ncon([a, a], [(-1, 1, -3), (-2, 1, -4)], conjs=(0, 1))
    tm = tm.fuse_legs(axes=((0, 1), (2, 3)))
    tmn = tm.to_numpy()

    ## initializing random tensor matching TM, with extra leg carrying charges; this are independent blocks in calculation
    legs = [a.get_legs(2).conj(), a.get_legs(2), yastn.Leg(a.config, s=1, t=(-2, -1, 0, 1, 2), D=(1, 1, 1, 1, 1))]
    v = yastn.rand(config=a.config, legs=legs, dtype='float64')

    vn = np.sum(v.fuse_legs(axes=[(0, 1), 2]).to_numpy(), axis=1)
    wn1 = scipy.linalg.expm(t * tmn) @ vn

    f = lambda x : yastn.ncon([a, a, x], [(-1, 1, 2), (-2, 1, 3), (2, 3, -3)], conjs=(0, 1, 0))
    out, info = yastn.expmv(f, v, t, tol, return_info=True)
    assert out.are_independent(v)

    wn2 = np.sum(out.fuse_legs(axes=[(0, 1), 2]).to_numpy(), axis=1)
    err = np.linalg.norm(wn1 - wn2) / np.linalg.norm(wn1)
    print(info, err)
    assert err < tol


if __name__ == '__main__':
    test_expmv(8, 5, 2.0, 1e-10)
    test_expmv(8, 5, 2.0j, 1e-10)
    test_expmv(8, 5, -2.0, 1e-10)
    test_expmv(8, 5, -2.0j, 1e-10)
    test_expmv(8, 5, 0, 1e-10)
    test_expmv(4, 20, 2.0, 1e-10)
    test_expmv_tm(0.5j, 1e-10)
    test_expmv_tm(0.5, 1e-3)
    test_expmv_tm(0, 1e-3)