""" Building Krylov space. """
import numpy as np
from ..tensor import YastError

__all__ = ['expmv', 'eigs']


def expand_krylov_space_tensor(f, tol, ncv, hermitian, V, H=None, info=None):
    """
    Expand the Krylov base up to ncv states or until reaching tolerance tol. """
    if H is None:
        H = {}
    happy = False
    for j in range(len(V)-1, ncv):
        w = f(V[-1])
        if info is not None:
            info['krylov_steps'] += 1
        if not hermitian:  # Arnoldi
            for i in range(j + 1):
                H[(i, j)] = V[i].vdot(w)
                w = w.apxb(V[i], x=-H[(i, j)])
        else:  # Lanczos
            if j > 0:
                H[(j - 1, j)] = H[(j, j - 1)]
                w = w.apxb(V[j - 1], x=-H[(j - 1, j)])
            H[(j, j)] = V[j].vdot(w)
            w = w.apxb(V[j], x=-H[(j, j)])
        H[(j + 1, j)] = w.norm()
        if H[(j + 1, j)] < tol:
            happy = True
            H.pop((j + 1, j))
            break
        V.append(w / H[(j + 1, j)])
    return V, H, happy


def sum_tensor(V, F):
    v = F[0] * V[0]
    for it in range(1, len(F)):
        v = v.apxb(V[it], x=F[it])
    return v


# Krylov based methods, handled by anonymous function decribing action of matrix on a vector
def expmv(f, v, t=1., tol=1e-12, ncv=10, hermitian=False, normalize=False, return_info=False, generic_expand=None, generic_sum=None):
    r"""
    Calculate :math:`e^{(tF)}v`, where `v` is Tensor, and F(v) is linear operator acting on `v`.

    Employs the algorithm of: J. Niesen, W. M. Wright, ACM Trans. Math. Softw. 38, 22 (2012), 
    Algorithm 919: A Krylov subspace algorithm for evaluating the phi-functions appearing in exponential integrators.

    Parameters
    ----------
        F: Callable[[yast.Tensor],yast.Tensor]
            defines an action of a 'square matrix' on Tensor.
            F(v) should preserve the signatures of `v`.

        v: Tensor

        t: number

        tol: number
            targeted tolerance; it is used to update the time-step and size of Krylov space.
           The result should have better tolerance, as corrected result is outputed.

        ncv: int
            Initial guess for the size of the Krylov space

        hermitian: bool
            Assume that f is a hermitian operator, in which case Lanczos iterations are used.
            Otherwise Arnoldi iterations are used to span the Krylov space.

        normalize: bool
            The result is normalized to unity using 2-norm.

        return_info: bool
            if true, returns (yast.Tensor, info), where
                info.ncv : guess of the Krylov-space size,
                info.error : estimate of error (likely over-estimate)
                info.krylov_steps : number of execution of f(x),
                info.steps : number of steps to reach t,

        generic_expand, generic_sum : bool
            Allows to inject other function expanding krylov space and calculating linear combination of vectors.

    Returns
    -------
    yast.Tensor
    """
    backend = v.config.backend
    ncv, ncv_max = max(1, ncv), min([30, v.size])  # Krylov space parameters
    t_now, t_out = 0, abs(t)
    sgn = t / t_out if t_out > 0 else 0
    tau = t_out  # initial quess for a time-step
    gamma, delta = 0.8, 1.2  # Safety factors
    V, H = None, None  # reset Krylov space
    ncv_old, tau_old, omega = None, None, None
    reject, order_computed, ncv_computed = False, False, False
    info = {'ncv': ncv, 'error': 0., 'krylov_steps': 0, 'steps': 0}

    if generic_expand is None:
        generic_expand = expand_krylov_space_tensor
    if generic_sum is None:
        generic_sum = sum_tensor

    normv = v.norm()
    if normv == 0:
        if normalize:
            raise YastError('expmv got zero vector that cannot be normalized')
        t_out = 0
    else:
        v = v / normv

    while t_now < t_out:
        if V is None:
            V = [v]
        V, H, happy = generic_expand(f, tol, ncv, hermitian, V, H, info)
        if happy:
            tau = t_out - t_now
            m = len(V)
            h = 0
        else:
            m = len(V) - 1
            h = H.pop((m, m - 1))
        H[(0, m)] = backend.ones((), dtype=v.yast_dtype, device=v.device)
        T = backend.square_matrix_from_dict(H, m + 1, device=v.device)
        F = backend.expm((sgn * tau) * T)
        err = abs(h * F[m - 1, m]).item()

        # renormalized error per unit step
        omega_old, omega = omega, (t_out / tau) * (err / tol)

        # Estimate order
        if ncv == ncv_old and tau != tau_old and reject:
            order = max([1., np.log(omega / omega_old) / np.log(tau / tau_old)])
            order_computed = True
        elif reject and order_computed:
            order_computed = False
        else:
            order_computed = False
            order = 0.25 * m

        # Estimate ncv
        if ncv != ncv_old and tau == tau_old and reject:
            ncv_est = max([1.1, (omega / omega_old) ** (1. / (ncv_old - ncv))]) if omega > 0 else 1.1
            ncv_computed = True
        elif reject and ncv_computed:
            ncv_computed = False
        else:
            ncv_computed = False
            ncv_est = 2

        tau_old, ncv_old = tau, ncv
        if happy:
            omega = 0
            tau_new, ncv_new = tau, ncv
        elif m == ncv_max and omega > delta:
            tau_new, ncv_new = tau * (omega / gamma) ** (-1. / order), ncv_max
        else:
            tau_opt = tau * (omega / gamma) ** (-1. / order) if omega > 0 else t_out - t_now
            ncv_opt = int(max([1, np.ceil(m + np.log(omega / gamma) / np.log(ncv_est))])) if omega > 0 else 1
            C1 = ncv * int(np.ceil((t_out - t_now) / tau_opt))
            C2 = ncv_opt * int(np.ceil((t_out - t_now) / tau))
            tau_new, ncv_new = (tau_opt, m) if C1 < C2 else (tau, ncv_opt)

        if omega <= delta:  # Check error against target
            F[m, 0] = F[m - 1, m] * h
            F = F[:, 0]
            normF = backend.norm_matrix(F)
            normv = normv * normF
            F = F / normF
            v = generic_sum(V, F[:len(V)])
            t_now += tau
            info['steps'] += 1
            info['error'] += err
            V, H, reject = None, None, False
        else:
            reject = True
            H[(m, m - 1)] = h
            H.pop((0, m))
        tau = min(max(0.2 * tau, tau_new), t_out - t_now, 2 * tau)
        ncv = int(max(1, min(ncv_max, np.ceil(1.3333 * m), max(np.floor(0.75 * m), ncv_new))))
    info['ncv'] = ncv
    if not normalize:
        v = normv * v
    return (v, info) if return_info else v


def eigs(f, v0, k=1, which='SR', ncv=10, maxiter=None, tol=1e-13, hermitian=True, generic_expand=None, generic_sum=None):
    r"""
    Search for dominant eigenvalues of linear operator f using Arnoldi algorithm.
    Economic implementation (without restart) for internal use within :meth:`yast.tn.dmrg_`.

    Parameters
    ----------
        f: function
            define an action of a 'square matrix' on the 'vector' `v0`.
            f(v0) should preserve the signature of v0.

        v0: Tensor
            Initial guess, 'vector' to span the Krylov space.

        k: int
            Number of desired eigenvalues and eigenvectors. default is 1.

        which: str 
            One of [‘LM’, ‘SM’, ‘LR’, ‘SR’] specifying which k eigenvectors and eigenvalues to find:
            ‘LM’ : largest magnitude, ‘SM’ : smallest magnitude, ‘LR’ : largest real part, ‘SR’ : smallest real part

        ncv: int
            Dimension of the employed Krylov space. Default is 10.
            Must be greated than k.

        maxiter: int
            Maximal number of restarts; 

        tol: float
            Stopping criterion for Krylov subspace. Default is 1e-13.

        hermitian: bool
            Assume that f is a hermitian operator, in which case Lanczos iterations are used.
            Otherwise Arnoldi iterations are used to span the Krylov space.
    """
    # Maximal number of restarts - NOT IMPLEMENTED FOR NOW.
    # ONLY A SINGLE ITERATION FOR NOW
    backend = v0.config.backend
    normv = v0.norm()
    if normv == 0:
        raise YastError('Initial vector v0 of eigs should be nonzero.')

    if generic_expand is None:
        generic_expand = expand_krylov_space_tensor
    if generic_sum is None:
        generic_sum = sum_tensor

    V = [v0 / normv]
    V, H, happy = generic_expand(f, 1e-13, ncv, hermitian, V)  # tol=1e-13
    m = len(V) if happy else len(V) - 1

    T = backend.square_matrix_from_dict(H, m, device=v0.device)
    val, vr = backend.eigh(T) if hermitian else backend.eig(T)
    ind = backend.eigs_which(val, which)

    val, vr = val[ind], vr[:, ind]
    Y = []
    for it in range(k):
        sit = vr[:, it]
        Y.append(generic_sum(V[:len(sit)], sit))
    return val[:len(Y)], Y
