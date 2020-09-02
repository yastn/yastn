import yamps.tensor as tensor
import settings_full
import settings_Z2
import settings_U1
import settings_Z2_U1
import settings_U1_U1
import pytest
import numpy as np
import scipy.linalg as LA


settings_full.dtype = 'complex128'
settings_Z2.dtype = 'complex128'
settings_U1.dtype = 'complex128'


@pytest.mark.parametrize("D, k, tau, eigs_tol, exp_tol", [(6, 6, 1., 1e-18, 1e-14), (6, 6, 1j*1., 1e-18, 1e-14)])
def test_hermitian(D, k, tau, eigs_tol, exp_tol):
    print('\nHermitian')
    hermitian = True
    cost_estim = 0
    def Av(x): return A.dot(x, axes=((2, 3, ), (0, 1, )))
    Bv = None

    # full
    ts = ()
    Ds = D
    kp = k**2
    A = tensor.rand(settings=settings_full, s=(-1, 1, 1, -1),
                    t=[ts, ts, ts, ts, ], D=[Ds, Ds, Ds, Ds, ])
    A = A.apxb(A.conj().transpose(axes=(2, 3, 0, 1)))*.5
    v = tensor.rand(settings=settings_full, s=(-1, 1),
                    t=[ts, ts, ], D=[Ds, Ds, ])
    A *= (1./A.norm())
    v *= (1./v.norm())
    out = tensor.expmw(Av=Av, init=[v], Bv=Bv, dt=tau, eigs_tol=eigs_tol, exp_tol=exp_tol, k=kp,
                       hermitian=hermitian, cost_estim=cost_estim)
    w = out[0]
    A = A.to_numpy()
    A = A.reshape((A.shape[0]*A.shape[1], A.shape[2]*A.shape[3]))
    v = v.to_numpy()
    v = v.reshape((v.shape[0]*v.shape[1]))
    w = w.to_numpy()
    w = w.reshape((w.shape[0]*w.shape[1]))/LA.norm(w)
    w_np = LA.expm(tau*A).dot(v)
    w_np *= 1./LA.norm(w_np)
    print('FULL: Average error vector entry[%]:', sum(abs((w-w_np)))/len(w_np))
    assert (abs((w-w_np)) < 1e-14).all()

    # Z2
    ts = (0, 1)
    Ds = (int(D/2), int(D/2),)
    kp = (2*int(k/2))**2
    A = tensor.rand(settings=settings_Z2, s=(-1, 1, 1, -1),
                    t=[ts, ts, ts, ts, ], D=[Ds, Ds, Ds, Ds, ])
    A = A.apxb(A.conj().transpose(axes=(2, 3, 0, 1)))*.5
    v = tensor.rand(settings=settings_Z2, s=(-1, 1),
                    t=[ts, ts, ], D=[Ds, Ds, ])
    A *= (1./A.norm())
    v *= (1./v.norm())
    out = tensor.expmw(Av=Av, init=[v], Bv=Bv, dt=tau, eigs_tol=eigs_tol, exp_tol=exp_tol, k=kp,
                       hermitian=hermitian, cost_estim=cost_estim)
    w = out[0]
    A = A.to_numpy()
    A = A.reshape((A.shape[0]*A.shape[1], A.shape[2]*A.shape[3]))
    v = v.to_numpy()
    v = v.reshape((v.shape[0]*v.shape[1]))
    w = w.to_numpy()
    w = w.reshape((w.shape[0]*w.shape[1]))/LA.norm(w)
    w_np = LA.expm(tau*A).dot(v)
    w_np *= 1./LA.norm(w_np)
    print('Z2: Average error vector entry[%]:', sum(abs((w-w_np)))/len(w_np))
    assert (abs((w-w_np)) < 1e-14).all()

    # U1
    ts = (-1, 0, 1)
    Ds = (int(D/4), int(D/2), int(D/4),)
    kp = (2*int(k/4)+int(k/2))**2
    A = tensor.rand(settings=settings_U1, s=(-1, 1, 1, -1),
                    t=[ts, ts, ts, ts, ], D=[Ds, Ds, Ds, Ds, ])
    A = A.apxb(A.conj().transpose(axes=(2, 3, 0, 1)))*.5
    v = tensor.rand(settings=settings_U1, s=(-1, 1),
                    t=[ts, ts, ], D=[Ds, Ds, ])
    A *= (1./A.norm())
    v *= (1./v.norm())
    out = tensor.expmw(Av=Av, init=[v], Bv=Bv, dt=tau, eigs_tol=eigs_tol, exp_tol=exp_tol, k=kp,
                       hermitian=hermitian, cost_estim=cost_estim)
    w = out[0]
    A = A.to_numpy()
    A = A.reshape((A.shape[0]*A.shape[1], A.shape[2]*A.shape[3]))
    v = v.to_numpy()
    v = v.reshape((v.shape[0]*v.shape[1]))
    w = w.to_numpy()
    w = w.reshape((w.shape[0]*w.shape[1]))/LA.norm(w)
    w_np = LA.expm(tau*A).dot(v)
    w_np *= 1./LA.norm(w_np)
    print('U1: Average error vector entry[%]:', sum(abs((w-w_np)))/len(w_np))
    assert (abs((w-w_np)) < 1e-14).all()


@pytest.mark.parametrize("D, k, tau, eigs_tol, exp_tol", [(6, 6, 1., 1e-18, 1e-14), (6, 6, 1j*1., 1e-18, 1e-14)])
def test_non_hermitian(D, k, tau, eigs_tol, exp_tol):
    print('\nNon-Hermitian')
    hermitian = False
    cost_estim = 0
    def Av(x): return A.dot(x, axes=((2, 3, ), (0, 1, )))
    def Bv(x): return A.conj().transpose(
        axes=(2, 3, 0, 1)).dot(x, axes=((2, 3, ), (0, 1, )))

    # full
    ts = ()
    Ds = D
    kp = k**2
    A = tensor.rand(settings=settings_full, s=(-1, 1, 1, -1),
                    t=[ts, ts, ts, ts, ], D=[Ds, Ds, Ds, Ds, ])
    v = tensor.rand(settings=settings_full, s=(-1, 1),
                    t=[ts, ts, ], D=[Ds, Ds, ])
    A *= (1./A.norm())
    v *= (1./v.norm())
    out = tensor.expmw(Av=Av, init=[v], Bv=Bv, dt=tau, eigs_tol=eigs_tol, exp_tol=exp_tol, k=kp,
                       hermitian=hermitian, cost_estim=cost_estim)
    w = out[0]
    A = A.to_numpy()
    A = A.reshape((A.shape[0]*A.shape[1], A.shape[2]*A.shape[3]))
    v = v.to_numpy()
    v = v.reshape((v.shape[0]*v.shape[1]))
    w = w.to_numpy()
    w = w.reshape((w.shape[0]*w.shape[1]))/LA.norm(w)
    w_np = LA.expm(tau*A).dot(v)
    w_np *= 1./LA.norm(w_np)
    print('FULL: Average error vector entry[%]:', sum(abs((w-w_np)))/len(w_np))
    assert (abs((w-w_np)) < 1e-14).all()

    # Z2
    ts = (0, 1)
    Ds = (int(D/2), int(D/2),)
    kp = (2*int(k/2))**2
    A = tensor.rand(settings=settings_Z2, s=(-1, 1, 1, -1),
                    t=[ts, ts, ts, ts, ], D=[Ds, Ds, Ds, Ds, ])
    v = tensor.rand(settings=settings_Z2, s=(-1, 1),
                    t=[ts, ts, ], D=[Ds, Ds, ])
    A *= (1./A.norm())**k
    v *= (1./v.norm())
    out = tensor.expmw(Av=Av, init=[v], Bv=Bv, dt=tau, eigs_tol=eigs_tol, exp_tol=exp_tol, k=kp,
                       hermitian=hermitian, cost_estim=cost_estim)
    w = out[0]
    A = A.to_numpy()
    A = A.reshape((A.shape[0]*A.shape[1], A.shape[2]*A.shape[3]))
    v = v.to_numpy()
    v = v.reshape((v.shape[0]*v.shape[1]))
    w = w.to_numpy()
    w = w.reshape((w.shape[0]*w.shape[1]))/LA.norm(w)
    w_np = LA.expm(tau*A).dot(v)
    w_np *= 1./LA.norm(w_np)
    print('Z2: Average error vector entry[%]:', sum(abs((w-w_np)))/len(w_np))
    assert (abs((w-w_np)) < 1e-14).all()

    # U1
    ts = (-1, 0, 1)
    Ds = (int(D/4), int(D/2), int(D/4),)
    kp = (2*int(k/4)+int(k/2))**2
    A = tensor.rand(settings=settings_U1, s=(-1, 1, 1, -1),
                    t=[ts, ts, ts, ts, ], D=[Ds, Ds, Ds, Ds, ])
    v = tensor.rand(settings=settings_U1, s=(-1, 1),
                    t=[ts, ts, ], D=[Ds, Ds, ])
    A *= (1./A.norm())
    v *= (1./v.norm())
    out = tensor.expmw(Av=Av, init=[v], Bv=Bv, dt=tau, eigs_tol=eigs_tol, exp_tol=exp_tol, k=kp,
                       hermitian=hermitian, cost_estim=cost_estim)
    w = out[0]
    A = A.to_numpy()
    A = A.reshape((A.shape[0]*A.shape[1], A.shape[2]*A.shape[3]))
    v = v.to_numpy()
    v = v.reshape((v.shape[0]*v.shape[1]))
    w = w.to_numpy()
    w = w.reshape((w.shape[0]*w.shape[1]))/LA.norm(w)
    w_np = LA.expm(tau*A).dot(v)
    w_np *= 1./LA.norm(w_np)
    print('U1: Average error vector entry[%]:', sum(abs((w-w_np)))/len(w_np))
    assert (abs((w-w_np)) < 1e-14).all()


def test_eigs():

    D = 6  # if k**2 in fact
    k = D
    tau = 1
    eigs_tol = 1e-18
    exp_tol = 1e-14
    print('\ntau-real:')
    test_hermitian(D=D, k=k, tau=tau, eigs_tol=eigs_tol, exp_tol=exp_tol)
    test_non_hermitian(D=D, k=k, tau=tau, eigs_tol=eigs_tol, exp_tol=exp_tol)
    print('\ntau-imag:')
    test_hermitian(D=D, k=k, tau=1j*tau, eigs_tol=eigs_tol, exp_tol=exp_tol)
    test_non_hermitian(D=D, k=k, tau=1j*tau,
                       eigs_tol=eigs_tol, exp_tol=exp_tol)


if __name__ == '__main__':
    test_eigs()
