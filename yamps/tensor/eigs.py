import logging
import numpy as np
from scipy import linalg as LA
import time
import tracemalloc


class FatalError(Exception):
    pass


logger = logging.getLogger('yamps.tensor.eigs')


_select_dtype = {'float64': np.float64,
                 'complex128': np.complex128}


def expmw(Av, init, Bv=None, dt=1, eigs_tol=1e-14, exp_tol=1e-14, k=5, hermitian=False, bi_orth=True, NA=None, cost_estim=0, algorithm='arnoldi'):
    return expA(Av=Av, Bv=Bv, init=init, dt=dt, eigs_tol=eigs_tol, exp_tol=exp_tol, k=k, hermitian=hermitian, bi_orth=bi_orth, NA=NA, cost_estim=cost_estim, algorithm=algorithm)


def expA(Av, init, Bv, dt, eigs_tol, exp_tol, k, hermitian, bi_orth, NA, cost_estim, algorithm):
    if not hermitian and not Bv:
        logger.exception(
            'expA: For non-hermitian case provide Av and Bv. In addition you can start with two')
        raise FatalError
    k_max = min([20, init[0].get_size()])
    if not NA:
        # n - cost of vector init
        # NA - cost of matrix Av
        # v_i - cost of exponatiation of size m(m-krylov dim)
        T = LA.expm(np.diag(np.random.rand(k_max-1), -1) + np.diag(
            np.random.rand(k_max), 0) + np.diag(np.random.rand(k_max-1), 1))
        if cost_estim == 0:  # approach 0: defaoult based on matrix size
            # in units of n
            n = init[0].get_size()
            NA = round(4.*n**(0.5), 2)
            v_i = k_max**2/n
            n = 1.
        elif cost_estim == 1:  # approach 1: based on time usage. normalized to time of saving copy of the initial vector. Don't make measures to small
            # in units of v_i
            start_time = time.time()
            init[0].copy()
            n = time.time() - start_time

            start_time = time.time()
            LA.expm(T)
            v_i = time.time() - start_time

            start_time = time.time()
            Av(init[0]).copy()
            NA = (time.time() - start_time)/v_i

            n *= (1./v_i)
            NA *= (1./v_i)
            v_i = 1.
        elif cost_estim == 2:  # approach 2: based on memory usage
            # in units of v_i
            tracemalloc.start()
            init[0].copy()
            n, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            tracemalloc.start()
            LA.expm(T)
            v_i, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            tracemalloc.start()
            Av(init[0])
            NA, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            n *= (1./v_i)
            NA *= (1./v_i)
            v_i = 1.
    # Krylov parameters
    mmax = k_max
    m = max([1, k])

    # Initialize variables
    step = 0
    ireject = 0
    reject = 0
    happy = 0
    sgn = np.sign(dt).real*1j if dt.real == 0. else np.sign(dt).real*1
    tnow = 0
    tout = abs(dt)
    j = 0
    tau = abs(dt)

    # Set safety factors
    gamma = 0.8
    delta = 1.2

    # Initial condition
    w = init
    oldm, oldtau, omega = np.nan, np.nan, np.nan
    orderold, kestold = True, True

    # Iterate until we reach the final t
    while tnow < tout:
        # Compute the exponential of the augmented matrix
        err, evec, good = expm(Av=Av, Bv=Bv, init=w, tol=eigs_tol, k=m, hermitian=hermitian,
                               bi_orth=bi_orth, tau=(sgn, tau.real), algorithm=algorithm)
        happy = good[0]
        j = good[1]

        # Error per unit step
        oldomega = omega
        omega = (tout/tau)*(err/exp_tol)

        # Estimate order
        if m == oldm and tau != oldtau and ireject > 0:
            order = max([1, np.log(omega/oldomega)/np.log(tau/oldtau)])
            orderold = False
        elif orderold or ireject == 0:
            orderold = True
            order = j*.25
        else:
            orderold = True

        # Estimate k
        if m != oldm and tau == oldtau and ireject > 0:
            kest = max([1.1, (omega/oldomega)**(1./(oldm-m))]
                       ) if omega > 0 else 1.1
            kestold = False
        elif kestold or ireject == 0:
            kestold = True
            kest = 2
        else:
            kestold = True

        # This if statement is the main difference between fixed and variable m
        oldtau, oldm = tau, m
        if happy == 1:
            # Happy breakdown; wrap up
            omega = 0
            taunew, mnew = tau, m
        elif j == mmax and omega > delta:
            # Krylov subspace to small and stepsize to large
            taunew, mnew = tau*(omega/gamma)**(-1./order), j
        else:
            # Determine optimal tau and m
            tauopt = tau*(omega/gamma)**(-1./order)
            mopt = max([1, np.ceil(j+np.log(omega/gamma)/np.log(kest))])

            # evaluate Cost functions
            Ctau = (m * NA + 3 * m * n + (m/mmax)**2*v_i * (10 + 3 * (m - 1))
                    * (m + 1) ** 3) * np.ceil((tout - tnow) / tauopt)

            Ck = (mopt * NA + 3 * mopt * n + (mopt/mmax)**2*v_i * (10 + 3 * (mopt - 1))
                  * (mopt + 1) ** 3) * np.ceil((tout - tnow) / tau)
            if Ctau < Ck:
                taunew, mnew = tauopt, m
            else:
                taunew, mnew = tau, mopt

        # Check error against target
        if omega <= delta:  # use this one
            reject += ireject
            step += 1
            w = evec
            # update time
            tnow += tau
            ireject = 0
        else:  # try again
            ireject += 1

        # Another safety factors
        tau = min([tout - tnow, max([.2 * tau, min([taunew, 2 * tau])])])
        m = int(
            max([1, min([mmax, max([np.floor(.75 * m), min([mnew, np.ceil(1.3333 * m)])])])]))

    if abs(tnow/tout) < 1.:
        logger.error('eigs/expA: Failed to approximate matrix exponent with given parameters.\nLast update of omega/delta = '+omega /
                     delta+'\nRemaining time = '+abs(1.-tnow/tout)+'\nChceck: max_iter - number of iteractions,\nk - Krylov dimension,\ndt - time step.')
        raise FatalError
    return (w[0], step, j, tnow,)


def expm(Av, init, tau, Bv=None, tol=1e-14, k=5, algorithm='arnoldi', bi_orth=False, hermitian=True):
    norm = init[0].norm()
    init = [(1. / it.norm())*it for it in init]
    if algorithm == 'arnoldi':
        T, Q, beta, good = arnoldi(Av=Av, init=init[0], k=k, tol=tol)
    else:  # Lanczos
        if hermitian:
            T, Q, beta, good = lanczos_her(Av=Av, init=init[0], k=k, tol=tol)
        else:
            T, Q, P, beta, good = lanczos_nher(
                Av=Av, Bv=Bv, init=init, k=k, tol=tol, bi_orth=bi_orth)
    val, Y = expm_aug(T=T, Q=Q, tau=tau, beta=beta)
    return val, [norm*Y[it] for it in range(len(Y))], good


def eigh(Av, init, tol=1e-14, k=5, algorithm='arnoldi'):
    norm = init[0].norm()
    init = [(1. / it.norm())*it for it in init]
    if algorithm == 'arnoldi':
        T, Q, _, good = arnoldi(Av=Av, init=init[0], k=k, tol=tol)
    else:  # Lanczos
        T, Q, _, good = lanczos_her(Av=Av, init=init[0], k=k, tol=tol)
    val, Y = eigs_aug(T=T, Q=Q, hermitian=False)
    return val, [norm*Y[it] for it in range(len(Y))], good


def eig(Av, init, Bv=None, tol=1e-14, k=5, bi_orth=True, algorithm='arnoldi'):
    norm = init[0].norm()
    init = [(1. / it.norm())*it for it in init]
    if algorithm == 'arnoldi':
        T, Q, _, good = arnoldi(Av=Av, init=init[0], k=k, tol=tol)
        P = None
    else:  # Lanczos
        T, Q, P, _, good = lanczos_nher(
            Av=Av, Bv=Bv, init=init, k=k, tol=tol, bi_orth=bi_orth)
    val, Y = eigs_aug(T=T, Q=Q, P=P, hermitian=False)
    return val, [norm*Y[it] for it in range(len(Y))], good


# Algorithms based on Krylov methods


def arnoldi(Av, init, tol=1e-14, k=5):
    # Lanczos algorithm for hermitian matrices
    beta = None
    Q = [None] * (k + 1)
    H = np.zeros((k + 1, k + 1), dtype=init.conf.dtype)
    #
    Q[0] = init
    for jt in range(k):
        w = Av(Q[jt])
        for it in range(jt+1):
            H[it, jt] = w.scalar(Q[it])
            w = w.apxb(Q[it], x=-H[it, jt])
        H[jt+1, jt] = w.norm()
        if H[jt+1, jt] < tol:
            beta, happy = 0, True
            break
        Q[jt+1] = w*(1./H[jt+1, jt])
    if beta is None:
        beta, happy = H[jt+1, jt], 0
    H = H[:(jt+1), :(jt+1)]
    Q = Q[:(jt+1)]
    return H, Q, beta, (happy, len(Q))


def lanczos_her(Av, init, tol=1e-14, k=5):
    # Lanczos algorithm for hermitian matrices
    beta = None
    Q = [None] * (k + 1)
    a = np.zeros(k + 1, dtype=init.conf.dtype)
    b = np.zeros(k + 1, dtype=init.conf.dtype)
    #
    Q[0] = init
    r = Av(Q[0])
    for it in range(k):
        a[it] = Q[it].scalar(r)
        r = r.apxb(Q[it], x=-a[it])
        b[it] = r.norm()
        if b[it] < tol:
            beta, happy = 0, 1
            break
        Q[it+1] = (1. / b[it])*r
        r = Av(Q[it+1])
        r = r.apxb(Q[it], x=-b[it])
    if beta is None:
        a[it+1] = Q[it+1].scalar(r)
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
    return make_tridiag(a=a, b=b, c=b), Q, beta, (happy, len(a))


def lanczos_nher(Av, Bv, init, tol=1e-14, k=5, bi_orth=True):
    # Lanczos algorithm for non-hermitian matrices
    beta = None
    a = np.zeros(k + 1, dtype=init[0].conf.dtype)
    b = np.zeros(k + 1, dtype=init[0].conf.dtype)
    c = np.zeros(k + 1, dtype=init[0].conf.dtype)
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
        a[it] = P[it].scalar(r)
        r = r.apxb(Q[it], x=-a[it])
        s = s.apxb(P[it], x=-a[it].conj())
        if r.norm() < tol or s.norm() < tol:
            beta, happy = 0, 1
            break
        w = r.scalar(s)
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
                c1 = P[io].scalar(Q[it])
                Q[it] = Q[it].apxb(Q[io], x=-c1)
                c2 = Q[io].scalar(P[it])
                P[it] = P[it].apxb(P[io], x=-c2)
        r = Av(Q[it+1])
        s = Bv(P[it+1])
        r = r.apxb(Q[it], x=-c[it])
        s = s.apxb(P[it], x=-b[it].conj())
    if beta is None:
        a[it+1] = P[it+1].scalar(r)
        r = r.apxb(Q[it+1], x=-a[it+1])
        s = s.apxb(P[it+1], x=-a[it+1].conj())
        if r.norm() < tol or s.norm() < tol:
            beta, happy = 0, 1
        else:
            w = r.scalar(s)
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
    return make_tridiag(a=a, b=b, c=c), Q, P, beta, (happy, len(a))


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
    for n in range(p-1):
        out2[m+n, m+1] = 1.  # ?
    return out2


def expm_aug(T, Q, tau, beta, P=None):
    p = 1  # order of the Phi-function
    dtype = Q[0].conf.dtype
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


def eigs_aug(T, Q, P=None, hermitian=True):
    dtype = Q[0].conf.dtype
    Y = [None] * len(Q)
    if hermitian:
        m, n = LA.eigh(T, right=True)
        val, vr = m.astype(dtype), n.astype(dtype)
        for it in range(len(Q)):
            sit = vr[:, it]
            for i1 in sit.nonzero()[0]:
                if Y[it]:
                    Y[it] = Y[it].apxb(Q[i1], x=sit[i1])
                else:
                    Y[it] = sit[i1]*Q[i1]
            Y[it] *= (1./Y[it].norm())
    else:
        m, n, p = LA.eig(T, left=True, right=True)
        val, vl, vr = m.astype(dtype), n.astype(dtype), p.astype(dtype)
        for it in range(len(Q)):
            sit = vr[:, it]
            for i1 in sit.nonzero()[0]:
                if Y[it]:
                    Y[it] = Y[it].apxb(Q[i1], x=sit[i1])
                else:
                    Y[it] = sit[i1]*Q[i1]
            Y[it] *= (1./Y[it].norm())
    return val, Y
