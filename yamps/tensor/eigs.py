import numpy as np

def expmw(Av, init, Bv=None, dt=1, tol=1e-14, k=5, hermitian=False, biorth=True,  dtype='complex128', NA=None):
    exp_A = lambda x: expA(Av=Av, Bv=Bv, init=x, dt=dt, tol=tol, k=k, hermitian=hermitian, biorth=biorth,  dtype=dtype, NA=NA)
    return exp_A(init)

def expA(Av, init, Bv=None, dt=1, tol=1e-14, k=5, hermitian=False, biorth=True,  dtype='complex128', NA=None):
    if hermitian == False and Bv == None:
        print('expA: For non-hermitian provide Av and Bv.')
    #set default cost of Av(init) operation
    if NA == None:
        dims = init[0].get_shape()
        n = np.prod(dims)
        NA = 4 * round(n ** (1.5))
    #initial values
    qt = 0 #qt holds current time step
    sgn = np.sign(dt)
    dt = abs(dt)
    vec = init
    tau = dt
    k = max([1,k])
    #safety factors
    k_max = 10 * k
    gamma = .8
    max_try = 1e+3
    delta = 1.2
    comp_q = True
    comp_kappa = True
    k_old = k_max
    err_old = 1e+10
    omega = 1e+10
    tau_old = tau
    it = 0
    itry = 0
    while qt < abs(dt) and itry < max_try:
        eval, evec, err, happy = eigs(Av=Av, Bv=Bv, init=vec, tol=tol, k=k, hermitian=hermitian, biorth=biorth,  dtype=dtype, tau=(tau,True))
        omega_old = omega
        omega = dt / tau * err / tol
        if happy == 0 and err == 0:
            itry+=1
            comp_q = True
            comp_kappa = True
            tau_new = tau
            k_new = max([1, min([k_max, int(1.3333 * k) + 1])])
        else:
            if k == k_old and tau != tau_old and itry > 0:
                q = max([1, np.log(omega / omega_old) / np.log(tau / tau_old)])
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
            if happy == 1:
                omega = 0
                tau_new = dt - qt
                k_new = len(eval)
            elif k == k_max and omega > delta:
                tau_new = tau * (omega / delta) ** (-1. / q)
            else:
                tau_opt = tau * (omega / delta) ** (-1. / q)
                k_opt = max([1, 1 + int(k + np.log(omega / gamma) / np.log(kappa))])
                #evaluate cost functions for possible options
                Ctau = (k * NA + 3. * k * n + 5. * k ** 3) * (int((dt - qt) / tau_opt) + 1)
                Ck = (k_opt * NA + 3. * k_opt * n + 5. * k_opt ** 3) * (int((dt - qt) / tau) + 1)
                if Ctau < Ck:
                    tau_new = tau_opt
                    k_new = k
                else:
                    k_new = k_opt
                    tau_new = tau

            if omega < delta or itry > max_try:
                # save result of exp(tau*A) evolution
                it+=1
                qt+=tau
                vec = [evec]
            else:
                itry+=1
   
            #update (tau, k)
            tau = min([dt - qt, max([.2 * tau, min([tau_new, 2 * tau])])])
            k = max([1, min([k_max, max([int(.75 * k),min([k_new, int(1.3333 * k) + 1])])])])

    return (vec[0],it, k,qt,)

def eigs(Av, init, Bv=None, tau=None, tol=1e-14, k=5, hermitian=False, biorth=True,  dtype='complex128'):
    # solve eigenproblem using Lanczos
    init = [ it.__mul__(1. / (it.norm(ord='fro'))) for it in init]
    if hermitian == True:
        out = lanczos_her(Av=Av, init=init[0], k=k, tol=tol, dtype=dtype, tau=tau)
    elif hermitian == False:
        out = lanczos_nher(Av=Av, Bv=Bv, init=init, k=k, tol=tol, biorth=biorth, dtype=dtype, tau=tau)
    return out

def lanczos_her(Av, init, tau=None, tol=1e-14, k=5, dtype='complex128'):
    # Lanczos algorithm for hermitian matrices
    beta = False
    q = init
    a = np.zeros(k + 1, dtype=dtype)
    b = np.zeros(k + 1, dtype=dtype)
    Q = [None] * (k + 1)
    r = Av(q)
    for it in range(0,k):
        tmp = q.scalar(r)
        a[it] = tmp 
        Q[it] = q
        r = r.apxb(q, x=-a[it])
        b[it] = r.norm(ord='fro')
        if b[it] < tol:
            beta = 0
            happy = 1
            break
        v = q
        q = r.__mul__(1. / b[it])
        r = Av(q)
        r = r.apxb(v, x=-b[it])
    if not beta:
        tmp = q.scalar(r)
        a[it + 1] = tmp 
        Q[it + 1] = q
        r = r.apxb(q, x=-a[it + 1])
        beta = r.norm(ord='fro')
        if beta < tol:
            happy = 1
        else:
            happy = 0
    a = a[range(it + 1)]
    k = len(a)
    b = b[range(k - 1)]
    Q = Q[:k]
    out = solve_tridiag(a=a, b=b, Q=Q, hermitian=True, tau=tau, beta=beta, dtype=dtype)
    return out + (happy,)

def lanczos_nher(Av, Bv, init, tau=None, tol=1e-14, k=5, dtype='complex128', biorth=True):
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
        r = r.apxb(q, x=-a[it])
        s = s.apxb(p, x=-a[it].conjugate())
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
        # biothogonalization
        if biorth == True:
            for io in range(it):
                c1 = P[io].scalar(q)
                q = q.apxb(Q[io], x=-c1)
                c2 = Q[io].scalar(p)
                p = p.apxb(P[io], x=-c2)
        r = Av(q)
        s = Bv(p)
        r = r.apxb(qp, x=-b[it])
        s = s.apxb(pp, x=-c[it].conjugate())
    if not beta:
        tmp = p.scalar(r)
        a[it + 1] = tmp 
        Q[it + 1] = q
        P[it + 1] = p
        r = r.apxb(q, x=-a[it + 1])
        s = s.apxb(p, x=-a[it + 1].conjugate())
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
    out = solve_tridiag(a=a, b=b, c=c, Q=Q, V=P, hermitian=False, tau=tau, beta=beta, dtype=dtype)
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
    k = len(a)
    Y = [None] * k
    if hermitian == True:
        T = make_tridiag(a=a, b=b, hermitian=True)
        val, vec = np.linalg.eig(T)
        for it in range(k): #yit =sum_j q_j*sit_j
            sit = vec[:,it]
            tmp = Q[0].__mul__(sit[0]) 
            for i1 in range(1,k):
                tmp = tmp.apxb(Q[i1], x=sit[i1]) 
            Y[it] = tmp 
    else:
        T = make_tridiag(a=a, b=b, c=c)
        val, vec = np.linalg.eig(T)
        U = vec
        UT = vec.transpose().conj()
        for it in range(k): #yit =sum_j q_j*sit_j RIGHT EIGENVECTORS
            sit = U[:,it]
            tmp = Q[0].__mul__(sit[0]) 
            for i1 in range(1, k):
                tmp = tmp.apxb(Q[i1], x=sit[i1]) 
            tmp = tmp.__mul__(1. / tmp.norm())
            Y[it] = tmp

    if tau != None and beta != None:
        # calculate error and vector for expmv
        if isinstance(tau,tuple):
            comp_exp = tau[1]
            tau = tau[0]
        else: 
            comp_exp = False
        D1 = tau * val
        D2 = np.expm1(D1) / D1
        if comp_exp:
            if hermitian == True:
                UT = U.transpose().conj()
            
            v0L = UT[:,0]
            vRm1 = U[:,-1] #ok?
            Dexp = np.exp(D1) * v0L
            Dexp = U.dot(Dexp)
            Dexp = Dexp / np.linalg.norm(Dexp)

            w = Q[0].__mul__(Dexp[0])
            for it in range(1,len(Dexp)):
                w = w.__add__(Dexp[it] * Q[it])
            w = w.__mul__(1. / w.norm())
            Y = w
        tmp = vRm1.dot(D2 * v0L)
        err = abs(beta * tau * tmp)
        out = val, Y, err
    else:
        out = val, Y
    return out