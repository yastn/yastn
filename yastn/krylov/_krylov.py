# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Building Krylov space. """
from __future__ import annotations
import numpy as np
import scipy.sparse.linalg as spla
from ..tensor import YastnError, Leg, einsum
from .. import zeros, decompress_from_1d, Tensor

__all__ = ['expmv', 'eigs', 'svds']


# Krylov based methods, handled by anonymous function decribing action of matrix on a vector
def expmv(f, v, t=1., tol=1e-12, ncv=10, hermitian=False, normalize=False, return_info=False, **kwargs) -> vector:
    r"""
    Calculate :math:`e^{(tF)}v`, where :math:`v` is a vector, and :math:`F(v)` is linear operator acting on :math:`v`.

    The algorithm of: J. Niesen, W. M. Wright, ACM Trans. Math. Softw. 38, 22 (2012),
    Algorithm 919: A Krylov subspace algorithm for evaluating the phi-functions appearing in exponential integrators.

    Parameters
    ----------
        f: Callable[[vector], vector]
            defines an action of a "square matrix" on vector.

        v: vector
            input vector to apply exponential map onto.

        t: number
            exponent amplitude.

        tol: number
           targeted tolerance; it is used to update the time-step and size of Krylov space.
           The returned result should have better tolerance, as correction is included.

        ncv: int
            Initial guess for the size of the Krylov space.

        hermitian: bool
            Assume that ``f`` is a hermitian operator, in which case Lanczos iterations are used.
            Otherwise Arnoldi iterations are used to span the Krylov space.

        normalize: bool
            Whether to normalize the result to unity using 2-norm.

        return_info: bool
            if ``True``, returns ``(vector, info)``, where

            * ``info.ncv`` : guess of the Krylov-space size,
            * ``info.error`` : estimate of error (likely over-estimate)
            * ``info.krylov_steps`` : number of execution of ``f(x)``,
            * ``info.steps`` : number of steps to reach ``t``,

        kwargs: any
            Further parameters that are passed to :func:`expand_krylov_space` and :func:`linear_combination`.
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

    normv = v.norm()
    if normv == 0:
        if normalize:
            raise YastnError('expmv() got zero vector that cannot be normalized')
        t_out = 0
    else:
        v = v / normv

    while t_now < t_out:
        if V is None:
            V = [v]
        lenV = len(V)
        V, H, happy = v.expand_krylov_space(f, tol, ncv, hermitian, V, H, **kwargs)
        info['krylov_steps'] += len(V) - lenV + happy

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
            if happy:
                F = F[:-1]
            normF = backend.norm_matrix(F)
            normv = normv * normF
            F = F / normF
            v = V[0].linear_combination(*V[1:], amplitudes=F, **kwargs)
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


def eigs(f, v0, k=1, which='SR', ncv=10, maxiter=None, tol=1e-13, hermitian=False, **kwargs) -> tuple[array, Sequence[vectors]]:
    r"""
    Search for dominant eigenvalues of linear operator ``f`` using Arnoldi algorithm.
    Economic implementation (without restart) for internal use within :meth:`yastn.tn.mps.dmrg_`.

    Parameters
    ----------
        f: function
            define an action of a 'square matrix' on the 'vector' ``v0``.
            ``f(v0)`` should preserve the signature of ``v0``.

        v0: Tensor
            Initial guess, 'vector' to span the Krylov space.

        k: int
            Number of desired eigenvalues and eigenvectors. The default is 1.

        which: str
            One of [``‘LM’``, ``‘LR’``, ``‘SR’``] specifying which `k` eigenvectors and eigenvalues to find:
            ``‘LM’`` : largest magnitude,
            ``‘SM’`` : smallest magnitude,
            ``‘LR’`` : largest real part,
            ``‘SR’`` : smallest real part.

        ncv: int
            Dimension of the employed Krylov space. The default is 10.
            Must be greated than `k`.

        maxiter: int
            Maximal number of restarts; (not implemented for now)

        tol: float
            Stopping criterion for an expansion of the Krylov subspace.
            The default is ``1e-13``. (not implemented for now)

        hermitian: bool
            Assume that ``f`` is a hermitian operator, in which case Lanczos iterations are used.
            Otherwise Arnoldi iterations are used to span the Krylov space.
    """
    # Maximal number of restarts - NOT IMPLEMENTED FOR NOW.
    # ONLY A SINGLE ITERATION FOR NOW
    backend = v0.config.backend
    normv = v0.norm()
    if normv == 0:
        raise YastnError('Initial vector v0 of eigs should be nonzero.')


    V = [v0 / normv]
    V, H, happy = v0.expand_krylov_space(f, 1e-13, ncv, hermitian, V, **kwargs)  # tol=1e-13
    m = len(V) if happy else len(V) - 1
    V = V[:m]

    T = backend.square_matrix_from_dict(H, m, device=v0.device)
    val, vr = backend.eigh(T) if hermitian else backend.eig(T)
    ind = backend.eigs_which(val, which)

    val, vr = val[ind], vr[:, ind]
    Y = []
    for it in range(k):
        sit = vr[:, it]
        Y.append(V[0].linear_combination(*V[1:], amplitudes=sit, **kwargs))
    return val[:k], Y


def svds(A, k=1, sU=1, nU=True, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True, \
         solver='arpack', rng=None, options=None) -> tuple[Sequence[vectors], array, Sequence[vectors]]:
    r"""
    Search for dominant singular values of linear operator ``f`` using Arnoldi algorithm.
    Economic implementation (without restart) for internal use within :meth:`yastn.tn.mps.dmrg_`.

    Parameters
    ----------
        A: function
            define an action of a 'square matrix' on the 'vector' ``v0``.
            ``f(v0)`` should preserve the signature of ``v0``.

        sU: int
            Signature of the new leg in `U`; equal to 1 or -1. The default is 1.
            `V` is going to have the opposite signature on the connecting leg.

        nU: bool
            Whether or not to attach the charge of ``a`` to `U`.
            If ``False``, it is attached to `V`. The default is ``True``.

        v0: Tensor
            Initial guess, 'vector' to span the Krylov space.

        k: int
            Number of desired singular values and singular vectors. The default is 1.

        which: str
            One of [``‘LM’``, ``‘LR’``, ``‘SR’``] specifying which `k` singular vectors and singular values to find:
            ``‘LM’`` : largest magnitude,
            ``‘SM’`` : smallest magnitude,
            ``‘LR’`` : largest real part,
            ``‘SR’`` : smallest real part.
    """
    assert len(A.get_legs()) == 2, "A has to be of rank 2"
    rows, cols= A.get_legs()

    # 
    v0_row= zeros(config=A.config, legs=(rows.conj(), Leg(sym=rows.sym, s= rows.s, t=rows.t, D= (1,)*len(rows.t))))
    _, row_meta= v0_row.compress_to_1d(meta=None)
    v0_col= zeros(config=A.config, legs=(cols.conj(), Leg(sym=cols.sym, s= cols.s, t=cols.t, D= (1,)*len(cols.t))))
    _, col_meta= v0_col.compress_to_1d(meta=None)

    def mv(v):
        col= decompress_from_1d(v, col_meta)
        Mcol= einsum('ij,jx->ix',A,col)
        row, Mcol_meta= Mcol.compress_to_1d(meta=None)
        return row
    
    def vm(v):
        row= decompress_from_1d(v, row_meta)
        rowM= einsum('ix,ij->jx',row,A)
        col, rowM_meta= rowM.compress_to_1d(meta=None)
        return col

    # step 2: invoke dense svds
    lop_A= spla.LinearOperator((v0_row.size, v0_col.size), matvec=mv, rmatvec=vm)
    U, S, Vh= spla.svds(lop_A, k=k, ncv=ncv, tol=tol, which=which, v0=None, maxiter=maxiter, \
                                    return_singular_vectors=return_singular_vectors, solver=solver, options=options,) #rng=None)
    
    # Individual singular vectors are ordered by magnitude in ascending manner [scipy], across all charge sectors. 
    # Instead, we want to have them ordered by charge sectors, and then by magnitude within each sector.
    # Locate the charge sectors of the singular vector by position of the largest element in the vector.
    #
    # A = U S Vh , while vA = v0_row A with vA having a structure of rows a
    #              and A v0_col = Av with Av having s structure of cols
    #
    rowA= v0_row.conj()
    colA= v0_col.conj()
    U_sorted= {}

    # ISSUE: in case of degeneracy, the singular vectors can be mixed-up across sectors
    #        null-space is completely degenerate
    # TODO: check if degeneracy, else post-process to separate sectors
    index_to_charge= []
    for c_block,slc in zip(rowA.get_blocks_charge(), rowA.slices):
        c_sector= (c_block[:rows.sym.NSYM],)
        index_to_charge += [c_sector]*slc.Dp
        U_sorted[c_sector]= []

    for i in range(len(S)):
        pos = np.argmax(np.abs(U[:, i]))
        c= index_to_charge[pos]
        U_sorted[c].append(i)

    # Step X: construct internal leg
    t_row, D_i= zip(*((c, len(U_sorted[c])) for c in U_sorted if len(U_sorted[c]) > 0))
    n_i= tuple( (A.n,)*len(t_row) if nU else (rows.sym.zero(),)*len(t_row) )
    t_i_nU= rows.sym.fuse(np.concatenate((
            np.array(t_row, dtype=np.int64).reshape((len(t_row), 1, rows.sym.NSYM)),
            np.array(n_i, dtype=np.int64).reshape((len(t_row), 1, rows.sym.NSYM))), axis=1), 
            (rows.s, -1), -sU)
    t_i_nU= tuple(map(tuple, t_i_nU.tolist()))
    
    leg_internal= Leg(sym=rows.sym, s= sU, t=t_i_nU, D= D_i)

    symU= zeros(config=A.config, legs=(rows, leg_internal), n=(A.n if nU else None))
    symS= zeros(config=A.config, legs=(leg_internal.conj(), leg_internal), isdiag=True)
    symVh= zeros(config=A.config, legs=(leg_internal.conj(), cols), n=(A.n if not nU else None))
    
    # embed singular triples into blocks of symmetric tensors in descending order of magnitude
    U_sectors= dict(zip(( (c[:rows.sym.NSYM],) for c in rowA.get_blocks_charge()),rowA.slices))
    Vh_sectors= dict(zip(( (c[:rows.sym.NSYM],) for c in colA.get_blocks_charge()),colA.slices))
    row_to_col_sector= { (c[:rows.sym.NSYM],): (c[cols.sym.NSYM:],) for c in A.get_blocks_charge() }
    for c in symU.get_blocks_charge():
        row_sector, i_sector, col_sector= (c[:rows.sym.NSYM],), (c[rows.sym.NSYM:],), \
            row_to_col_sector[(c[:rows.sym.NSYM],)]
        if len(U_sorted[row_sector])<1:
            continue
        inds= U_sorted[row_sector]
        symU[c]= U[slice(*U_sectors[row_sector].slcs[0]),inds[::-1]]
        symS[(i_sector,i_sector)]= S[inds[::-1]]
        symVh[(i_sector,col_sector)]= Vh[inds[::-1],slice(*Vh_sectors[col_sector].slcs[0])]

    return symU, symS, symVh
