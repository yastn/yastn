from context import yast
from context import config_dense, config_Z2, config_U1
import numpy as np
import scipy.linalg
import pytest

tol = 1e-12

def run_expm_hermitian(A, v, tau, eigs_tol, exp_tol, kp):
    A = (A + A.conj().transpose(axes=(2, 3, 0, 1))) / 2
    A /= A.norm()
    v /= v.norm()
    Av = lambda x: yast.tensordot(A, x, axes=((2, 3), (0, 1)))
    out = yast.expmv(Av=Av, init=[v], Bv=None, dt=tau, eigs_tol=eigs_tol,
                     exp_tol=exp_tol, k=kp, hermitian=True, cost_estim=0, algorithm='lanczos')
    w = out[0].to_dense().reshape(-1)
    A = A.to_dense()
    sA = A.shape
    A = A.reshape((sA[0]*sA[1], sA[2]*sA[3]))
    v = v.to_dense().reshape(-1)
    w_np = scipy.linalg.expm(tau * A) @ v
    w_np /= scipy.linalg.norm(w_np)
    assert np.allclose(w, w_np, rtol=1e-10, atol=1e-10)

def run_expmv_nonhermitian(A, v, tau, eigs_tol, exp_tol, kp):
    A /= A.norm()
    v /= v.norm()
    Av = lambda x: yast.tensordot(A, x, axes=((2, 3), (0, 1)))
    At = A.transpose(axes=(2, 3, 0, 1)).conj()
    Bv = lambda x: yast.tensordot(At, x, axes=((2, 3), (0, 1)))

    out = yast.expmv(Av=Av, init=[v], Bv=Bv, dt=tau, eigs_tol=eigs_tol,
                     exp_tol=exp_tol, k=kp, hermitian=False, cost_estim=0)
    w = out[0].to_dense().reshape(-1)
    A = A.to_dense()
    a1, a2, a3, a4 = A.shape
    A = A.reshape((a1 * a2, a3 * a4))
    v = v.to_dense().reshape(-1)
    w_np = scipy.linalg.expm(tau*A).dot(v)
    w_np /= scipy.linalg.norm(w_np)
    print('FULL: Average error vector entry[%]:', sum(abs((w-w_np)))/len(w_np))
    assert np.allclose(w, w_np, rtol=1e-10, atol=1e-10)

@pytest.mark.parametrize("D, k, tau, eigs_tol, exp_tol", [(6, 6, 1., 1e-14, 1e-14), (6, 6, 1j, 1e-14, 1e-14)])
def test_expmv_hermitian(D, k, tau, eigs_tol, exp_tol):
    # dense
    Ds = D
    kp = k**2
    A = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=[Ds, Ds, Ds, Ds], dtype='complex128')
    v = yast.rand(config=config_dense, s=(-1, 1), D=[Ds, Ds], dtype='complex128')
    run_expm_hermitian(A, v, tau, eigs_tol, exp_tol, kp)

    # Z2
    ts = (0, 1)
    Ds = (D//2, D//2)
    kp = (2*(k//2))**2
    A = yast.rand(config=config_Z2, s=(-1, 1, 1, -1), t=[ts, ts, ts, ts], D=[Ds, Ds, Ds, Ds], dtype='complex128')
    v = yast.rand(config=config_Z2, s=(-1, 1), t=[ts, ts], D=[Ds, Ds], dtype='complex128')
    run_expm_hermitian(A, v, tau, eigs_tol, exp_tol, kp)

    # U1
    ts = (-1, 0, 1)
    Ds = (D//4, D//2, D//4)
    kp = (2*(k//4)+(k//2))**2
    A = yast.rand(config=config_U1, s=(-1, 1, 1, -1), t=[ts, ts, ts, ts], D=[Ds, Ds, Ds, Ds], dtype='complex128')
    v = yast.rand(config=config_U1, s=(-1, 1), t=[ts, ts], D=[Ds, Ds], dtype='complex128')
    run_expm_hermitian(A, v, tau, eigs_tol, exp_tol, kp)


@pytest.mark.parametrize("D, k, tau, eigs_tol, exp_tol", [(6, 6, 1., 1e-14, 1e-14), (6, 6, 1j, 1e-14, 1e-14)])
def test_expmv_non_hermitian(D, k, tau, eigs_tol, exp_tol):
    # dense
    ts = ()
    Ds = D
    kp = k**2
    A = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=[Ds, Ds, Ds, Ds], dtype='complex128')
    v = yast.rand(config=config_dense, s=(-1, 1), D=[Ds, Ds], dtype='complex128')
    run_expmv_nonhermitian(A, v, tau, eigs_tol, exp_tol, kp)

    # Z2
    ts = (0, 1)
    Ds = (int(D/2), int(D/2),)
    kp = (2*int(k/2))**2
    A = yast.rand(config=config_Z2, s=(-1, 1, 1, -1), t=[ts, ts, ts, ts], D=[Ds, Ds, Ds, Ds], dtype='complex128')
    v = yast.rand(config=config_Z2, s=(-1, 1), t=[ts, ts], D=[Ds, Ds], dtype='complex128')
    run_expmv_nonhermitian(A, v, tau, eigs_tol, exp_tol, kp)

    # U1
    ts = (-1, 0, 1)
    Ds = (int(D/4), int(D/2), int(D/4),)
    kp = (2*int(k/4)+int(k/2))**2
    A = yast.rand(config=config_U1, s=(-1, 1, 1, -1), t=[ts, ts, ts, ts], D=[Ds, Ds, Ds, Ds], dtype='complex128')
    v = yast.rand(config=config_U1, s=(-1, 1), t=[ts, ts], D=[Ds, Ds], dtype='complex128')
    run_expmv_nonhermitian(A, v, tau, eigs_tol, exp_tol, kp)


if __name__ == '__main__':
    test_expmv_hermitian(6, 6, 1., 1e-14, 1e-14)
    # test_expmv_hermitian(6, 6, 1j, 1e-14, 1e-14)
    # test_expmv_non_hermitian(6, 6, 1., 1e-14, 1e-14)
    # test_expmv_non_hermitian(6, 6, 1j, 1e-14, 1e-14)



