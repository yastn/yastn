import numpy as np
import pytest
import scipy.linalg
import yast
if __name__ == '__main__':
    from configs import config_dense, config_Z2, config_U1
else:
    from .configs import config_dense, config_Z2, config_U1


def run_expmv(A, v, tau, tol, ncv, hermitian):
    if hermitian:
        A = (A + A.conj().transpose(axes=(2, 3, 0, 1))) / 2
    f = lambda x: yast.tensordot(A, x, axes=((2, 3), (0, 1)))
    An = A.to_numpy()
    sA = An.shape
    An = An.reshape((sA[0]*sA[1], sA[2]*sA[3]))
    vn = v.to_numpy().reshape(-1)
    wn = scipy.linalg.expm(tau * An) @ vn

    out, info = yast.expmv(f, v, tau, tol, ncv, hermitian=hermitian, normalize=False, return_info=True)
    assert out.are_independent(v)
    w = out.to_numpy().reshape(-1)
    normwn = np.linalg.norm(wn)
    err = np.linalg.norm(w - wn) / normwn if normwn > 0 else np.linalg.norm(w - wn)
    print(info, err)
    assert err <= tol

    if scipy.linalg.norm(vn) > 0:
        wn /= scipy.linalg.norm(wn)
        out = yast.expmv(f, v, tau, tol, ncv, hermitian=hermitian, normalize=True)
        assert out.are_independent(v)
        w = out.to_numpy().reshape(-1)
        assert np.linalg.norm(w - wn) <= tol * np.linalg.norm(wn)
    else:
        with pytest.raises(yast.YastError):
            out = yast.expmv(f, v, tau, tol, ncv, hermitian=hermitian, normalize=True)


@pytest.mark.parametrize("D, ncv, tau, tol", [(8, 5, 2., 1e-10), (8, 5, 2j, 1e-10), (8, 5, -2., 1e-10), (8, 5, -2j, 1e-10), (8, 5, 0, 1e-10), (4, 20, 2, 1e-10)])
def test_expmv(D, ncv, tau, tol):
    # dense
    Ds = D
    A = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=[Ds, Ds, Ds, Ds], dtype='complex128')
    v = yast.rand(config=config_dense, s=(-1, 1), D=[Ds, Ds], dtype='complex128')
    run_expmv(A, v, tau, tol, ncv, hermitian=True)
    run_expmv(A, v, tau, tol, ncv, hermitian=False)

    # Z2
    ts = (0, 1)
    Ds = (D//2, D//2)
    A = yast.rand(config=config_Z2, s=(-1, 1, 1, -1), t=[ts, ts, ts, ts], D=[Ds, Ds, Ds, Ds], dtype='complex128')
    v = yast.rand(config=config_Z2, s=(-1, 1), t=[ts, ts], D=[Ds, Ds], dtype='complex128')
    run_expmv(A, v, tau, tol, ncv, hermitian=True)
    run_expmv(A, v, tau, tol, ncv, hermitian=False)

    # U1
    ts = (-1, 0, 1)
    Ds = (D//4, D//2, D//4)
    A = yast.rand(config=config_U1, s=(-1, 1, 1, -1), t=[ts, ts, ts, ts], D=[Ds, Ds, Ds, Ds], dtype='complex128')
    v = yast.rand(config=config_U1, s=(-1, 1), t=[ts, ts], D=[Ds, Ds], dtype='complex128')
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
    a = yast.rand(config=config_U1, s=(1, 1, -1), n=0,
                  t=[(-1, 0, 1), (0, 1), (-1, 0, 1)],
                  D=[(2, 3, 4), (2, 3), (2, 3, 4)])

    # dense transfer matrix build from a
    tm = yast.ncon([a, a], [(-1, 1, -3), (-2, 1, -4)], conjs=(0, 1))
    tm.fuse_legs(axes=((0, 1), (2, 3)), inplace=True)
    tmn = tm.to_numpy()

    ## initializing random tensor matching TM
    v = yast.randR(config=a.config, legs=[(a, 2, 'flip_s'), (a, 2), {'s': 1, -2: 1, -1: 1, 0: 1, 1: 1, 2: 1}])

    vn = np.sum(v.fuse_legs(axes=[(0, 1), 2]).to_numpy(), axis=1)
    wn1 = scipy.linalg.expm(t * tmn) @ vn

    f = lambda x : yast.ncon([a, a, x], [(-1, 1, 2), (-2, 1, 3), (2, 3, -3)], conjs=(0, 1, 0))
    out, info = yast.expmv(f, v, t, tol, return_info=True)
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