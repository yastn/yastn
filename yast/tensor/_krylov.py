"""
Krylov-based methods for yast tensor.
Based on:
Niesen, J., and W. Wright. "A Krylov subspace algorithm for evaluating the φ-functions in exponential integrators." arXiv preprint arXiv:0907.4631 (2009).(http://www1.maths.leeds.ac.uk/~jitse/phikrylov.pdf)
"""

from ._auxliary import YastError
from ._contractions import vdot

# __all__ = ['arnoldi', 'lanczos', 'krylov']
# leader
# def krylov(Av, init, tol, ncv=5, sigma=None, which=None, return_eigenvectors=True, tau=False):
#     if hermitian:
#         T, Q, P, good = lanczos_her(init, Av, tol, k)
#     else:
#         T, Q, P, good = arnoldi(init, Av, tol, ncv)
#     if not tau:
#         val, Y = eigs_aug(T, Q, P, k, hermitian, sigma,
#                           which, return_eigenvectors)
#         return val, Y, good
#     else:
#         err, Y = expm_aug(T, Q, tau, good[0])
#         return err, Y, good


# Arnoldi method
def arnoldi(init, Av, tol, k):
    beta = None
    Q = [None] * (k + 1)
    Q[0] = init[0]
    H = Q[0].config.backend.zeros((k + 1, k + 1))
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


def lanczos_her(init, Av, tol, k):
    """ Lanczos algorithm for hermitian matrices """
    beta = None
    Q, a, b = [init], [], []
    r = Av(Q[0])
    for it in range(k):
        a.append(vdot(Q[it], r))  # a[it]
        r = r.apxb(Q[it], x=-a[it])
        b.append(r.norm())  # b[it]
        if b[it] < tol:
            beta, happy = 0, True
            break
        Q.append(r / b[it])
        r = Av(Q[it+1])
        r = r.apxb(Q[it], x=-b[it])
    if beta is None:
        a.append(vdot(Q[it+1], r))
        r = r.apxb(Q[it+1], x=-a[it+1])
        beta = r.norm()
        if beta < tol:
            beta, happy = 0, True
            it += 1
        else:
            happy = False
    backend = init.config.backend
    a = backend.to_tensor(a[:it + 1])
    b = backend.to_tensor(b[:it])
    Q = Q[:it + 1]

    T = Q[0].config.backend.diag_create(a, 0) + \
        Q[0].config.backend.diag_create(b, -1) + \
        Q[0].config.backend.diag_create(b, +1)

    return T, Q, None, (beta, happy, len(a))



# Shared functions
def enlarged_aug_mat(T, p, backend):
    # build elarged augment matrix following Saad idea [1992, 1998]
    m = len(T)
    out2 = backend.zeros((m+p, m+p))
    out2[:m, :m] = T
    out2[0, m] = 1.
    if p>1:
        out2[range(m, m+p, 1), m+1] += 1.
    return out2


def expm_aug(T, Q, tau, beta):
    p = 1  # order of the Phi-function
    tau = tau[0] * tau[1]
    phi_p = enlarged_aug_mat(tau*T, p, Q[0].config.backend)
    expT = Q[0].config.backend.expm(phi_p)[:len(T), 0]
    expT = expT/sum(abs(expT)**2)**.5
    Y = None
    for it in expT.nonzero()[0]:
        if Y:
            Y = Y.apxb(Q[it], x=expT[it])
        else:
            Y = expT[it]*Q[it]
    Y *= (1./Y.norm())
    phi_p1 = enlarged_aug_mat(tau*T, p+1, Q[0].config.backend)
    err = Q[0].config.backend.expm(phi_p1)
    err = abs(beta*err[len(expT)-1, len(expT)])
    return err, [Y]


def eigs_aug(T, Q, P=None, k=None, hermitian=True, sigma=None, which=None, return_eigenvectors=True):
    Y = [None] * k if k else [None] * len(Q)
    if hermitian:
        val, vr = Q[0].config.backend.eigh(T)
    else:
        val, vr = Q[0].config.backend.eig(T)

    whicher = {
        'LM':   -(-abs(val)).argsort(),  # ‘LM’ : largest magnitude
        'SM':   abs(val).argsort(),  # ‘SM’ : smallest magnitude
        'LR':   -(-Q[0].config.backend.real(val)).argsort(),  # ‘LR’ : largest real part
        'SR':   Q[0].config.backend.real(val).argsort(),  # ‘SR’ : smallest real part
        'LI':   -(-Q[0].config.backend.imag(val)).argsort(),  # ‘LI’ : largest imaginary part
        'SI':   Q[0].config.backend.imag(val).argsort()        # ‘SI’ : smallest imaginary part
    }

    if sigma != None:  # target val closest to sigma on Re-Im plane
        id = abs(val-sigma).argsort()
    else:
        id = whicher.get(which, val.argsort())  # is 'SR' if not specified

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
