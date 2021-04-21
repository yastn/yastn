"""
Krylov-based methods for yast tensor.
Based on:
Niesen, J., and W. Wright. "A Krylov subspace algorithm for evaluating the φ-functions in exponential integrators." arXiv preprint arXiv:0907.4631 (2009).(http://www1.maths.leeds.ac.uk/~jitse/phikrylov.pdf)
"""

import numpy as np
from scipy import linalg as LA
from ._auxliary import YastError

__all__ = ['arnoldi', 'lanczos', 'krylov']

_select_dtype = {'float64': np.float64, 'complex128': np.complex128}

# leader


def krylov(init, Av, Bv, tol, k, algorithm, hermitian, ncv=5, bi_orth=False, sigma=None, which=None, return_eigenvectors=True, tau=False):
    if algorithm == 'lanczos':
        T, Q, P, good = lanczos(init, Av, Bv, tol, ncv, hermitian, bi_orth)
    else:
        T, Q, P, good = arnoldi(init, Av, tol, ncv)
    if not tau:
        val, Y = eigs_aug(T, Q, P, k, hermitian, sigma,
                          which, return_eigenvectors)
        return val, Y, good
    else:
        err, Y = expm_aug(T, Q, tau, good[0])
        return err, Y, good


# Arnoldi method
def arnoldi(init, Av, tol, k):
    beta = None
    Q = [None] * (k + 1)
    Q[0] = init[0]
    H = np.zeros((k + 1, k + 1), dtype=Q[0].config.dtype)
    #
    for jt in range(k):
        w = Av(Q[jt])
        for it in range(jt+1):
            H[it, jt] = w.vdot(Q[it])
            w = w.apxb(Q[it], x=-H[it, jt])
        H[jt+1, jt] = w.norm()
        if H[jt+1, jt] < tol:
            beta, happy = 0, True
            break
        Q[jt+1] = w*(1./H[jt+1, jt])
    if beta == None:
        beta, happy = H[jt+1, jt], 0
    H = H[:(jt+1), :(jt+1)]
    Q = Q[:(jt+1)]
    return H, Q, None, (beta, happy, len(Q))


# Lanczos method
def lanczos(init, Av, Bv, tol, k, hermitian, bi_orth):
    # Lanczos method
    if not hermitian:
        return lanczos_her(init[0], Av, tol, k)
    else:
        return lanczos_nher(init, Av, Bv, tol, k, bi_orth)


def lanczos_her(init, Av, tol, k):
    # Lanczos algorithm for hermitian matrices
    beta = None
    Q = [None] * (k + 1)
    Q[0] = init
    a = np.zeros(k + 1, dtype=Q[0].config.dtype)
    b = np.zeros(k + 1, dtype=Q[0].config.dtype)
    #
    r = Av(Q[0])
    for it in range(k):
        a[it] = Q[it].vdot(r)
        r = r.apxb(Q[it], x=-a[it])
        b[it] = r.norm()
        if b[it] < tol:
            beta, happy = 0, 1
            break
        Q[it+1] = (1. / b[it])*r
        r = Av(Q[it+1])
        r = r.apxb(Q[it], x=-b[it])
    if beta == None:
        a[it+1] = Q[it+1].vdot(r)
        r = r.apxb(Q[it+1], x=-a[it+1])
        beta = r.norm()
        if beta < tol:
            beta, happy = 0, 1
            it += 1
        else:
            happy = 0
    a = a[:(it+1)]
    b = b[:(it)]
    Q = Q[:(it+1)]
    return make_tridiag(a=a, b=b, c=b), Q, None, (beta, happy, len(a))


def lanczos_nher(init, Av, Bv, tol, k, bi_orth):
    # Lanczos algorithm for non-hermitian matrices
    beta = None
    a = np.zeros(k + 1, dtype=init[0].config.dtype)
    b = np.zeros(k + 1, dtype=init[0].config.dtype)
    c = np.zeros(k + 1, dtype=init[0].config.dtype)
    Q = [None] * (k + 1)
    P = [None] * (k + 1)
    #
    if len(init) == 1:
        P[0], Q[0] = init[0], init[0]
    else:
        P[0], Q[0] = init[0], init[1]
    r = Av(Q[0])
    s = Bv(P[0])
    for it in range(k):
        a[it] = P[it].vdot(r)
        r = r.apxb(Q[it], x=-a[it])
        s = s.apxb(P[it], x=-a[it].conj())
        if r.norm() < tol or s.norm() < tol:
            beta, happy = 0, 1
            break
        w = r.vdot(s)
        if abs(w) < tol:
            beta, happy = 0, 1
            break
        b[it] = np.sqrt(abs(w)).real
        c[it] = w.conj() / b[it]
        Q[it+1] = (1. / b[it])*r
        P[it+1] = (1. / c[it].conj())*s
        # bi_orthogonalization
        if bi_orth:
            for io in range(it):
                c1 = P[io].vdot(Q[it])
                Q[it] = Q[it].apxb(Q[io], x=-c1)
                c2 = Q[io].vdot(P[it])
                P[it] = P[it].apxb(P[io], x=-c2)
        r = Av(Q[it+1])
        s = Bv(P[it+1])
        r = r.apxb(Q[it], x=-c[it])
        s = s.apxb(P[it], x=-b[it].conj())
    if beta == None:
        a[it+1] = P[it+1].vdot(r)
        r = r.apxb(Q[it+1], x=-a[it+1])
        s = s.apxb(P[it+1], x=-a[it+1].conj())
        if r.norm() < tol or s.norm() < tol:
            beta, happy = 0, 1
        else:
            w = r.vdot(s)
            if abs(w) < tol:
                beta, happy = 0, 1
                it += 1
            else:
                beta, happy = np.sqrt(abs(w)), 0
    a = a[:(it+1)]
    b = b[:(it)]
    c = c[:(it)]
    Q = Q[:(it+1)]
    P = P[:(it+1)]
    return make_tridiag(a=a, b=b, c=c), Q, P, (beta, happy, len(a))


# Shared functions
def make_tridiag(a, b, c):
    # build tridiagonal matrix
    out = np.diag(a, 0) + np.diag(b, -1) + np.diag(c, +1)
    return out


def enlarged_aug_mat(T, p, dtype):
    # build elarged augment matrix following Saad idea [1992, 1998]
    m = len(T)
    out2 = np.zeros((m+p, m+p), dtype=dtype)
    out2[:m, :m] = T
    out2[0, m] = 1.
    if p>1:
        out2[range(m, m+p, 1), m+1] += 1.
    return out2


def expm_aug(T, Q, tau, beta):
    p = 1  # order of the Phi-function
    dtype = Q[0].config.dtype
    tau = tau[0] * tau[1]
    phi_p = enlarged_aug_mat(T=tau*T, p=p, dtype=_select_dtype[dtype])
    expT = LA.expm(phi_p)[:len(T), 0]
    expT = expT/LA.norm(expT)
    Y = None
    for it in expT.nonzero()[0]:
        if Y:
            Y = Y.apxb(Q[it], x=expT[it])
        else:
            Y = expT[it]*Q[it]
    Y *= (1./Y.norm())
    phi_p1 = enlarged_aug_mat(T=tau*T, p=p+1, dtype=_select_dtype[dtype])
    err = LA.expm(phi_p1)
    err = abs(beta*err[len(expT)-1, len(expT)])
    return err, [Y]


def eigs_aug(T, Q, P=None, k=None, hermitian=True, sigma=None, which=None, return_eigenvectors=True):
    Y = [None] * k if k else [None] * len(Q)
    if hermitian:
        val, vr = LA.eigh(T)
    else:
        val, _, vr = LA.eig(T, left=True, right=True)

    whicher = {
        'LM':   np.argsort(abs(val))[::-1],  # ‘LM’ : largest magnitude
        'SM':   np.argsort(abs(val)),  # ‘SM’ : smallest magnitude
        'LR':   np.argsort(val.real)[::-1],  # ‘LR’ : largest real part
        'SR':   np.argsort(val.real),  # ‘SR’ : smallest real part
        'LI':   np.argsort(val.imag)[::-1],  # ‘LI’ : largest imaginary part
        'SI':   np.argsort(val.imag)        # ‘SI’ : smallest imaginary part
    }

    if sigma != None:  # target val closest to sigma on Re-Im plane
        id = np.argsort(abs(val-sigma))
    else:
        id = whicher.get(which, np.argsort(val))  # is 'SR' if not specified

    val, vr = val[id], vr[:, id]
    if return_eigenvectors:
        for it in range(len(Y)):
            sit = vr[:, it]
            for i1 in sit.nonzero()[0]:
                if Y[it]:
                    Y[it] = Y[it].apxb(Q[i1], x=sit[i1])
                else:
                    Y[it] = sit[i1]*Q[i1]
            Y[it] *= (1./Y[it].norm())
    return val[:len(Y)], Y
