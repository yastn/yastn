import numpy as np
import scipy as sp
import time
import tracemalloc


class EigsError(Exception):
    pass


def expmw(Av, init, Bv=None, dt=1, tol=1e-14, k=5, hermitian=False, bi_orth=True,  dtype='complex128', NA=None, cost_estim=1):
    def exp_A(x): return expA(Av=Av, Bv=Bv, init=x, dt=dt, tol=tol, k=k,
                              hermitian=hermitian, bi_orth=bi_orth,  dtype=dtype, NA=NA, cost_estim=cost_estim)
    return exp_A(init)


def expA(Av, init, Bv=None, dt=1, tol=1e-14, k=5, hermitian=False, bi_orth=True,  dtype='complex128', NA=None, cost_estim=1):
    if not hermitian and not Bv:
        print('expA: For non-hermitian case provide Av and Bv. In addition you can start with two')
    if NA == None:
        if cost_estim == 1:  # approach 1: based on time usage. normalized to time of saving copy of the initial vector. Don't make measures to small
            start_time = time.time()
            init[0].copy()
            n = time.time() - start_time

            start_time = time.time()
            Av(init[0]).copy() #NA = time.time() - start_time
            NA = (time.time() - start_time)/n
            v_i = 1 #v_i = n / np.prod(init[0].get_shape())
        elif cost_estim == 2:  # approach 2: based on memory usage
            tracemalloc.start()
            init[0].copy()
            n, _ = tracemalloc.get_traced_memory()  # B

            Av(init[0]).copy()
            NA, _ = tracemalloc.get_traced_memory()  # B
            tracemalloc.stop()
            v_i = 1
    # initial values
    qt = 0  # holds current time
    sgn = np.sign(dt.real)*1. + np.sign(dt.imag) * 1j
    dt = abs(dt)
    vec = init
    tau = dt
    k = max([1, k])

    # safety factors
    k_max = 10 * k
    gamma = .8
    max_try = 10  # Upper bound on tdvp trial iterations. Can be increased if needed
    delta = 1.2
    comp_q = True
    comp_kappa = True
    k_old = k_max
    omega = 1e+10
    tau_old = tau

    it = 0
    itry = 0
    while qt < dt and itry < max_try:
        evec, err, happy = eigs(Av=Av, Bv=Bv, init=vec, tol=tol, k=k,
                                hermitian=hermitian, bi_orth=bi_orth,  dtype=dtype, tau=(tau.real, sgn))
        omega_old = omega
        omega = dt / tau * err / tol
        if (happy == 0 and err == 0) or omega_old == 0:  # < err?, omega_old==0
            itry += 1
            comp_q = True
            comp_kappa = True
            if omega_old == 0:
                tau_new = tau
                k_new = k
            else:
                tau_new = tau
                k_new = max([1, min([k_max, int(1.3333 * k) + 1])])
        else:
            if k == k_old and tau != tau_old and itry > 0:
                q = max(
                    [1, np.log(omega / omega_old) / np.log(tau / tau_old)])
                comp_q = False
            elif comp_q or itry == 0:
                comp_q = True
                q = .25 * k + 1
            else:
                comp_q = True

            if k != k_old and tau == tau_old and itry > 0:
                kappa = max([1.1, (omega / omega_old) ** (1. / (k_old - k))])
                comp_kappa = False
            elif comp_kappa or itry == 0:
                comp_kappa = True
                kappa = 2.
            else:
                comp_kappa = True

            tau_old = tau
            k_old = k
            if happy == 1:# or (dt - qt)<1e-3*dt: ERR maybe neglect very small intervals? rather not 
                omega = 0
                tau_new = dt - qt
                k_new = k  # len(eval)
            elif k == k_max and omega > delta:
                tau_new = tau * (omega / delta) ** (-1. / q)
                k_new = k
            else:
                tau_opt = tau * (omega / delta) ** (-1. / q)
                k_opt = max([1, 1 + int(k + np.log(omega / gamma) / np.log(kappa))])
                # evaluate cost functions for possible options
                Ctau = (k * NA + 3 * k * n + v_i * (10 + 3 * (k - 1))
                        * (k + 1) ** 3) * (int((dt - qt).real / tau_opt) + 1)
                Ck = (k_opt * NA + 3 * k_opt * n + v_i * (10 + 3 * (k_opt - 1))
                      * (k_opt + 1) ** 3) * (int((dt - qt).real / tau) + 1)
                if Ctau < Ck:
                    tau_new = tau_opt
                    k_new = k
                else:
                    k_new = k_opt
                    tau_new = tau
        if omega < delta:
            # save result of exp(tau*A) evolution
            it += 1
            qt += tau
            vec = [evec]
        else:
            if itry == max_try - 1:
                it += 1
                qt += dt - qt
                vec = [evec]
            itry += 1
        #update (tau, k)
        tau = min([dt - qt, max([.2 * tau, min([tau_new, 2 * tau])])]).real
        k = max(
            [1, min([k_max, max([int(.75 * k), min([k_new, int(1.3333 * k) + 1])])])]).real
    if qt < dt:
        raise EigsError(
            'eigs/expA: Failed to approximate matrix exponent with given parameters. \nChceck: max_iter - number of iteractions,\nk - Krylov dimension,\ndt - time step.')
    return (vec[0], it, k, qt,)


def eigs(Av, init, Bv=None, tau=None, tol=1e-14, k=5, hermitian=False, bi_orth=True,  dtype='complex128'):
    # solve eigenproblem using Lanczos
    init = [it.__mul__(1. / (it.norm(ord='fro'))) for it in init]
    if hermitian == True:
        out = lanczos_her(
            Av=Av, init=init[0], k=k, tol=tol, dtype=dtype, tau=tau)
    elif hermitian == False:
        out = lanczos_nher(Av=Av, Bv=Bv, init=init, k=k,
                           tol=tol, bi_orth=bi_orth, dtype=dtype, tau=tau)
    return out


def lanczos_her(Av, init, tau=None, tol=1e-14, k=5, dtype='complex128'):
    # Lanczos algorithm for hermitian matrices
    beta = False
    q = init
    a = np.zeros(k + 1, dtype=dtype)
    b = np.zeros(k + 1, dtype=dtype)
    Q = [None] * (k + 1)
    r = Av(q)
    for it in range(0, k):
        a[it] = q.scalar(r)
        Q[it] = q
        r = r.axpb(q, x=-a[it])
        b[it] = r.norm(ord='fro')
        if b[it] < tol:
            beta = 0
            happy = 1
            break
        v = q
        q = r.__mul__(1. / b[it])
        r = Av(q)
        r = r.axpb(v, x=-b[it])
    if not beta:
        tmp = q.scalar(r)
        a[it + 1] = tmp
        Q[it + 1] = q
        r = r.axpb(q, x=-a[it + 1])
        beta = r.norm(ord='fro')
        if beta < tol:
            happy = 1
        else:
            happy = 0
    a = a[range(it + 1)]
    k = len(a)
    b = b[range(k - 1)]
    Q = Q[:k]
    out = solve_tridiag(a=a, b=b, Q=Q, hermitian=True,
                        tau=tau, beta=beta, dtype=dtype)
    return out + (happy,)


def lanczos_nher(Av, Bv, init, tau=None, tol=1e-14, k=5, dtype='complex128', bi_orth=True):
    # Lanczos algorithm for non-hermitian matrices
    if len(init) == 1:
        p, q = init[0], init[0]
    else:
        p, q = init[0], init[1]
    beta = False
    a = np.zeros(k + 1, dtype=dtype)
    b = np.zeros(k + 1, dtype=dtype)
    c = np.zeros(k + 1, dtype=dtype)
    Q = [None] * (k + 1)
    P = [None] * (k + 1)
    r = Av(q)
    s = Bv(p)
    for it in range(k):
        if r.norm(ord='fro') < tol or s.norm(ord='fro') < tol:
            beta = 0
            happy = 1
            break
        tmp = p.scalar(r)
        a[it] = tmp
        Q[it] = q
        P[it] = p
        r = r.axpb(q, x=-a[it])
        s = s.axpb(p, x=-a[it].conjugate())
        w = r.scalar(s)
        if abs(w) < tol:
            beta = 0
            happy = 1
            break
        b[it] = np.sqrt(abs(w))
        c[it] = w.conjugate() / b[it]
        pp = p
        qp = q
        q = r.__mul__(1. / c[it])
        p = s.__mul__((1. / b[it]).conjugate())
        # bi_orthogonalization
        if bi_orth == True:
            for io in range(it):
                c1 = P[io].scalar(q)
                q = q.axpb(Q[io], x=-c1)
                c2 = Q[io].scalar(p)
                p = p.axpb(P[io], x=-c2)
        r = Av(q)
        s = Bv(p)
        r = r.axpb(qp, x=-b[it])
        s = s.axpb(pp, x=-c[it].conjugate())
    if not beta:
        tmp = p.scalar(r)
        a[it + 1] = tmp
        Q[it + 1] = q
        P[it + 1] = p
        r = r.axpb(q, x=-a[it + 1])
        s = s.axpb(p, x=-a[it + 1].conjugate())
        w = r.scalar(s)
        beta = np.sqrt(abs(w))
        if beta < tol:
            happy = 1
        else:
            happy = 0
    a = a[range(it + 1)]
    k = len(a)
    b = b[range(k - 1)]
    c = c[range(k - 1)]
    Q = Q[:k]
    P = P[:k]
    out = solve_tridiag(a=a, b=b, c=c, Q=Q, V=P,
                        hermitian=False, tau=tau, beta=beta, dtype=dtype)
    return out + (happy,)


def make_tridiag(a, b, c=None, hermitian=False):
    # build tridiagonal matrix
    if hermitian == True:
        out = np.diag(a, 0) + np.diag(b, +1) + np.diag(b, -1)
    else:
        out = np.diag(a, 0) + np.diag(b, +1) + np.diag(c, -1)
    return out


def solve_tridiag(a, b, Q, c=None, V=None, tau=None, beta=None, hermitian=False, dtype='complex128'):
    # find approximate eigenvalues and eigenvectors using tridiagonal matrix
    # and Krylov vectors Q and V
    if tau == None or beta == None:
        k = len(a)
        Y = [None] * k
        if hermitian == True:
            T = make_tridiag(a=a, b=b, hermitian=True)
            val, vec = np.linalg.eig(T)
            for it in range(k):  # yit =sum_j q_j*sit_j, right eigenvectors
                sit = vec[:, it]
                tmp = Q[0].__mul__(sit[0])
                for i1 in range(1, k):
                    tmp = tmp.axpb(Q[i1], x=sit[i1])
                Y[it] = tmp
        else:
            T = make_tridiag(a=a, b=b, c=c)
            val, vec = np.linalg.eig(T)
            for it in range(k):  # yit =sum_j q_j*sit_j, right eigenvectors
                sit = vec[:, it]
                tmp = Q[0].__mul__(sit[0])
                for i1 in range(1, k):
                    tmp = tmp.axpb(Q[i1], x=sit[i1])
                tmp = tmp.__mul__(1. / tmp.norm())
                Y[it] = tmp
        out = val, Y
    else:
        if hermitian == True:
            T = make_tridiag(a=a, b=b, hermitian=True)
        else:
            T = make_tridiag(a=a, b=b, c=c)

        # calculate new vector for expmv: expA*v
        tau = tau[1] * tau[0]
        expT = sp.linalg.expm(tau * T)[:, 0]
        expT = expT / np.linalg.norm(expT)

        Y = None
        for it in expT.nonzero()[0]:
            if Y != None:
                Y = Y.__add__(expT[it] * Q[it])
            else:
                Y = Q[it].__mul__(expT[it])
        Y = Y.__mul__(1. / Y.norm())

        # calculate an error for expmv
        tmp=sp.linalg.expm(tau*T)-np.identity(len(expT)) #exp(A)-1
        #phi_p+1
        #WARNING #tmp=1e-15 # (expA-1)/A at {end, 0} # not always find the inverse:  see example in 16_test*
        tmp=sum( tmp[-1,:]*np.linalg.inv(tau*T)[:,0] ) # (expA-1)/A at {end, 0} # not always find the inverse:  see example in 16_test*
        err=abs(beta*tmp) #ERR < why absolutr value
        out = Y, err
    return out