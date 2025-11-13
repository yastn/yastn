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
from itertools import islice
from typing import Sequence, TypeVar

import numpy as np
import scipy.sparse.linalg as spla

from ..initialize import zeros
from ..tensor import YastnError, Leg, LegMeta, einsum, truncation_mask, Tensor
from ..tensor._auxliary import _clear_axes, _unpack_axes, _flatten
from ..tensor._tests import _test_axes_all
from .._split_combine_dict import split_data_and_meta, combine_data_and_meta

__all__ = ['expmv', 'eigs', 'lin_solver', 'svds']

Vector = TypeVar('Vector')


# Krylov based methods, handled by anonymous function decribing action of matrix on a vector
def expmv(f, v, t=1., tol=1e-12, ncv=10, hermitian=False, normalize=False, return_info=False, **kwargs) -> Vector:
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
            Further parameters that are passed to :func:`expand_krylov_space` and :func:`add`.
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
        H[(0, m)] = backend.ones((), dtype=v.yastn_dtype, device=v.device)
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
            v = V[0].add(*V[1:], amplitudes=F, **kwargs)
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


def eigs(f, v0, k=1, which='SR', ncv=10, maxiter=None, tol=1e-13, hermitian=False, **kwargs) -> tuple[Sequence[float], Sequence[Vector]]:
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
        Y.append(V[0].add(*V[1:], amplitudes=sit, **kwargs))
    return val[:k], Y


def lin_solver(f, b, v0, ncv=10, tol=1e-13, pinv_tol=1e-13, hermitian=False, **kwargs) -> tuple[Sequence[float], Sequence[Vector]]:
    r"""
    Search for solution of the linear equation ``f(x) = b``, where ``x`` is estimated vector and ``f(x)`` is matrix-vector operation.
    Implementation based on pseudoinverse of Krylov expansion [1].

    Ref.[1] https://www.math.iit.edu/~fass/577_ch4_app.pdf

    Parameters
    ----------
        f: function
            define an action of a 'square matrix' on the 'vector' ``v0``.
            ``f(v0)`` should preserve the signature of ``v0``.

        b: Tensor
            free term for the linear equation.

        v0: Tensor
            Initial guess span the Krylov space.

        ncv: int
            Dimension of the employed Krylov space. The default is 10.

        tol: float
            Stopping criterion for an expansion of the Krylov subspace.
            The default is ``1e-13``. TODO Not implemented yet.

        pinv_tol: float
            Cutoff for pseudoinverve. Sets lower bound on inverted Schmidt values.
            The default is ``1e-13``.

        hermitian: bool
            Assume that ``f`` is a hermitian operator, in which case Lanczos iterations are used.
            Otherwise Arnoldi iterations are used to span the Krylov space.

    Results
    ----------
        vf: Tensor
            Approximation of ``v`` in ``f(v) = b`` problem.

        res: float
            norm of the resudual vector ``r = norm(f(vf) - b)``.
    """
    backend = v0.config.backend

    q0 = b - f(v0)
    normv = q0.norm()
    if normv == 0:
        raise YastnError('Initial vector v0 of lin_solver should be nonzero.')
    Q = [q0 / normv]
    Q, H, happy = q0.expand_krylov_space(f, tol, ncv, hermitian, Q, **kwargs)
    m = len(Q) if happy else len(Q) - 1
    H[(m,m-1)] = H[(0,0)] * 0 + tol if happy else H[(m,m-1)]
    Q = Q[:m]

    T = backend.square_matrix_from_dict(H, m+1, device = v0.device)
    T = T[:(m+1),:m]

    be1 = backend.to_tensor([normv]+[0]*(m), device = v0.device)

    Tpinv = backend.pinv(T, rcond = pinv_tol)
    y = Tpinv @ be1

    vf = v0.add(*Q, amplitudes = [1,*y], **kwargs)
    res = f(vf) - b
    return vf, res.norm()


def svds(A : Tensor, axes=(0, 1), k=1, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True, \
         solver='arpack', rng=None, options=None, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Search for dominant singular values of tensor A using Arnoldi algorithm.

    Parameters
    ----------
        axes: tuple[int, int] | tuple[Sequence[int], Sequence[int]]
            Specify two groups of legs between which to perform SVD, as well as
            their final order.

        v0: Tensor
            Initial guess, 'vector' to span the Krylov space.

        k: int = float('inf')
            Number of singular values and singular vectors to compute. Must satisfy 1 <= k <= kmax,
            where kmax=min(M, N) for solver='propack' and kmax=min(M, N) - 1 otherwise.

        tol: float, optional
            Tolerance for singular values. Zero (default) means machine precision.

        which: str
            One of [``‘LM’``, ``‘LR’``, ``‘SR’``] specifying which `D_total` singular vectors and singular values to find:
            ``‘LM’`` : largest magnitude,
            ``‘SM’`` : smallest magnitude,
            ``‘LR’`` : largest real part,
            ``‘SR’`` : smallest real part.

        Additional kwargs:

            sU: int = 1
                Signature of the new leg in `U`; equal to 1 or -1. The default is 1.
                `V` is going to have the opposite signature on the connecting leg.

            nU: bool = True
                Whether or not to attach the charge of ``a`` to `U`.
                If ``False``, it is attached to `V`. The default is ``True``.

            Uaxis, Vaxis: int, int = -1, 0
                specify which leg of `U` and `V` tensors are connecting with `S`. By default,
                it is the last leg of `U` and the first of `V`.

            compute_uv: bool = None
                alias for return_singular_vectors. When provided, it overrides the value of return_singular_vectors.
                For syntax consistency with :meth:`yastn.linalg.svd` and :meth:`yastn.linalg.svd_with_truncation`.

            fix_signs: bool = True
                Whether or not to fix phases in `U` and `V`,
                so that the largest element in each column of `U` is positive.
                Provide uniqueness of decomposition for non-degenerate cases.
                The default is ``False``.

            These parameters govern the truncation of singular triples after leading-k singular triples are found.

            reltol: float = 0
                relative tolerance of singular values below which to truncate across all blocks.

            reltol_block: float = 0
                relative tolerance of singular values below which to truncate within individual blocks.

            D_total: int = None
                alias for k. When provided, it overrides the value of k.
                For syntax consistency with :meth:`yastn.linalg.svd` and :meth:`yastn.linalg.svd_with_truncation`.

            D_block: int = float('inf')
                largest number of singular values to keep in a single block. Default is to keep all.

            truncate_multiplets: bool = False
                If ``True``, enlarge the truncation range specified by other arguments by shifting
                the cut to the largest gap between to-be-truncated singular values across all blocks.
                It provides a heuristic mechanism to avoid truncating part of a multiplet.
                The default is ``False``.

            mask_f: function[yastn.Tensor] -> yastn.Tensor
                custom truncation-mask function.
                If provided, it overrides all other truncation-related arguments.
    """
    k= kwargs.get('D_total', k)
    return_singular_vectors= kwargs.get('compute_uv', return_singular_vectors)
    sU= kwargs.get('sU', 1)
    nU= kwargs.get('nU', True)
    Uaxis= kwargs.get('Uaxis', -1)
    Vaxis= kwargs.get('Vaxis', 0)
    fix_signs= kwargs.get('fix_signs', True)
    eps_multiplet= kwargs.get('eps_multiplet', 1e-13)

    _test_axes_all(A, axes)
    lout_l, lout_r = _clear_axes(*axes)
    axes = _unpack_axes(A.mfs, lout_l, lout_r)

    A_mat= A.fuse_legs(axes=axes, mode='hard')
    rows, cols= A_mat.get_legs()

    def make_dummy_leg(l):
        from dataclasses import replace
        if type(l) is Leg:
            return Leg(sym=l.sym, s= l.s, t=l.t, D= (1,)*len(l.t))
        if type(l) is LegMeta:
            return replace(l, D= (1,)*len(l.t), legs=(make_dummy_leg(il) for il in l.legs))
        raise YastnError('Leg type not recognized')

    v0_row = zeros(config=A.config, legs=(rows.conj(), make_dummy_leg(rows)) )
    _, row_meta = split_data_and_meta(v0_row.to_dict(level=0))
    v0_col = zeros(config=A.config, legs=(cols.conj(), make_dummy_leg(cols)) )
    _, col_meta = split_data_and_meta(v0_col.to_dict(level=0))

    # take care of negative strides
    to_tensor = lambda x: A_mat.config.backend.to_tensor(x if np.sum(np.array(x.strides)<0)==0 else x.copy(), dtype=A_mat.yastn_dtype, device=A_mat.device)
    to_numpy = lambda x: A_mat.config.backend.to_numpy(x)

    def mv(v): # Av
        col = Tensor.from_dict(combine_data_and_meta(to_tensor(v), col_meta))
        res = einsum('ij,jx->ix',A_mat,col)
        row, res_meta = split_data_and_meta(res.to_dict(level=0))
        return to_numpy(row)

    def vm(v): # A^\dag v  vs  (v* A)^\dag = A^\dag v
        row = Tensor.from_dict(combine_data_and_meta(to_tensor(v).conj(), row_meta))
        res = einsum('ix,ij->jx', row, A_mat)
        col, res_meta= split_data_and_meta(res.to_dict(level=0))
        return to_numpy(col.conj())

    # step 2: invoke dense svds
    lop_A = spla.LinearOperator((v0_row.size, v0_col.size), matvec=mv, rmatvec=vm)
    U, S, Vh = spla.svds(lop_A, k=k, ncv=ncv, tol=tol, which=which, v0=None, maxiter=maxiter, \
                                    return_singular_vectors=return_singular_vectors, solver=solver, options=options,) #rng=None)

    # Individual singular vectors are ordered by magnitude in ascending manner [scipy], across all charge sectors.
    # Instead, we want to have them ordered by charge sectors, and then by magnitude within each sector.
    # Locate the charge sectors of the singular vector by position of the largest element in the vector.
    #
    # A = U S Vh , while vA = v0_row A with vA having a structure of rows a
    #              and A v0_col = Av with Av having s structure of cols
    #
    rowA = v0_row.conj()
    colA = v0_col.conj()
    U_sorted= {}

    # ISSUE: in case of degeneracy, the singular vectors can be mixed-up across sectors
    #        null-space is completely degenerate
    # TODO: check if degeneracy, else post-process to separate sectors
    # gaps= _find_gaps(S, tol=kwargs.get('reltol',0), eps_multiplet=eps_multiplet, which=which)
    # if sum(gaps<eps_multiplet)>0:
    #     raise NotImplemented('Resolving degeneracies in svds not implemented yet')

    index_to_charge= []
    for c_block,slc in zip(rowA.get_blocks_charge(), rowA.slices):
        c_sector= (c_block[:rows.sym.NSYM],)
        index_to_charge += [c_sector]*slc.Dp
        U_sorted[c_sector]= []

    # charge density per sector for multiplet of dim d located at i-d+1:i+1
    n_occ= lambda i,d : np.asarray([[np.linalg.norm(U[slice(*slc.slcs[0]),i-m])**2 \
                 for c,slc in zip(rowA.get_blocks_charge(), rowA.slices)] for m in range(d)])

    # overlap matrix for single non-zero charge sector, iterate over subspace
    # with ascending index (compatible with sliciing of U below)
    def overlaps_per_sector(c_sec,d):
        ic_slc= dict(zip(rowA.get_blocks_charge(), rowA.slices))[c_sec]
        overlaps= np.asarray([[ U[slice(*ic_slc.slcs[0]),i+1-d+m_row].conj() @ U[slice(*ic_slc.slcs[0]),i+1-d+m_col] \
                               for m_col in range(d)] for m_row in range(d)])
        return overlaps

    def get_sharp_sectors(overlap_diag):
                assert np.all(np.isclose(overlap_diag, 0, atol=1e-12) | np.isclose(overlap_diag, 1, atol=1e-12)), \
                    "The degeneracy within the sector is not resolved"
                return np.rint(overlap_diag)

    # Relate dense singular vectors to charge sectors. Traverse singular vectors in descending order of magnitude.
    isvals= iter(range(len(S)-1,-1,-1))
    for i in isvals:
        if S[i]<kwargs.get('reltol',0)*S[-1]:
            break # For finite reltol, this eliminates the kernel of A and its degeneracy
        # look-ahead at the dimension d of the degenerate subspace
        d= sum(S[i]-S[:i+1] < eps_multiplet)
        # print(f"{i} {S[i]} {d} {S[i-d:i+1]}")
        if d>1: # Treat degenerate subspace

            # 1) find components of degen. subspace and compute charge density per charge sector C_secs
            C= n_occ(i,d)
            C_secs= np.asarray([[c for c,slc in zip(rowA.get_blocks_charge(), rowA.slices)] for m in range(d)])
            assert np.allclose(np.sum(C, axis=1), 1, rtol=1.0e-14,  atol=1.0e-14), 'Degenerate subspace not properly separated'

            # 2) identify charge sectors that contain the degenerate subspace
            mask= C > 1.0e-14
            C0= C[np.ix_(np.any(mask, axis=1), np.any(mask, axis=0))]

            # non-zero C_sectors. In each row, there are nzC_secs.shape[1] sectors, with charges in last dimension of nzC_secs
            nzC_secs= C_secs[np.ix_(np.any(mask, axis=1), np.any(mask, axis=0))]
            if nzC_secs.shape[0]<nzC_secs.shape[1]: # more non-empty charge sectors than degenerate singular triples
                raise YastnError('Singular triples i-d+1:i+1 are a part of incomplete multiplet. Charge cannot be well-defined.')

            # 3) find basis in which singular triples are charge density operator eigenstates
            nv= nzC_secs.shape[0]
            full_overlap= np.zeros((nv*nzC_secs.shape[1],)*2, dtype=U.dtype)
            for nz_sec in range(nzC_secs.shape[1]):
                full_overlap[nz_sec*nv:(nz_sec+1)*nv,nz_sec*nv:(nz_sec+1)*nv]= overlaps_per_sector(tuple(nzC_secs[0,nz_sec,:]),d)

            D_0, B_0 = np.linalg.eigh(full_overlap) # o1 = B @ np.diag(D) @ B.
            sec_mask= get_sharp_sectors(D_0)
            assert sum(sec_mask)==nv, f"We should resolve multiplet of size {nv}, instead {sum(sec_mask)}"

            # build unitary from non-zero sections of B_0 corresponding to sharp sectors (D_0=1)
            UB= np.asarray([ B_0[:,-u][abs(B_0[:,-u])>1.0e-14] for u in range(nv,0,-1) ]).T
            U[:,i-d+1:i+1]= U[:,i-d+1:i+1] @ UB
            Vh[i-d+1:i+1,:]= UB.conj().T @ Vh[i-d+1:i+1,:]

            for x in range(d):
                U_sorted[ index_to_charge[ np.argmax(np.abs(U[:, i-d+1+x])) ] ].append(i-d+1+x)

            next(islice(isvals, d-1, d-1), None) # skip ahead
            continue
        if i==0:
            # special case of last singular triple being a part of a multiplet
            # 1) find components of degen. subspace and compute charge density per charge sector C_secs
            C= n_occ(i,1)
            assert np.allclose(np.sum(C, axis=1), 1, rtol=1.0e-14,  atol=1.0e-14), 'Degenerate subspace not properly separated'

            # 2) identify charge sectors that contain the degenerate subspace
            mask= C > 1.0e-14
            if np.sum(mask)>1: # its a part of a multiplet
                if kwargs.get('truncate_multiplets',False): continue
                YastnError('Last singular triple is part of a multiplet without well-defined charge')
        U_sorted[ index_to_charge[ np.argmax(np.abs(U[:, i])) ] ].append(i)

    # Step X: construct internal leg
    t_row, D_i= zip(*((c, len(U_sorted[c])) for c in U_sorted if len(U_sorted[c]) > 0))
    n_i= tuple( (A_mat.n,)*len(t_row) if nU else (rows.sym.zero(),)*len(t_row) )
    t_i_nU= rows.sym.fuse(np.concatenate((
            np.array(t_row, dtype=np.int64).reshape((len(t_row), 1, rows.sym.NSYM)),
            np.array(n_i, dtype=np.int64).reshape((len(t_row), 1, rows.sym.NSYM))), axis=1),
            (rows.s, -1), -sU)
    t_i_nU= tuple(map(tuple, t_i_nU.tolist()))

    leg_internal= Leg(sym=rows.sym, s= sU, t=t_i_nU, D= D_i)

    U, S, Vh= to_tensor(U), to_tensor(S), to_tensor(Vh)
    symU= zeros(config=A.config, legs=(rows, leg_internal), n=(A_mat.n if nU else None), dtype=A_mat.yastn_dtype)
    symS= zeros(config=A.config, legs=(leg_internal.conj(), leg_internal), isdiag=True,)
    symVh= zeros(config=A.config, legs=(leg_internal.conj(), cols), n=(A_mat.n if not nU else None), dtype=A_mat.yastn_dtype)

    # embed singular triples into blocks of symmetric tensors in descending order of magnitude
    U_sectors= dict(zip(( (c[:rows.sym.NSYM],) for c in rowA.get_blocks_charge()),rowA.slices))
    Vh_sectors= dict(zip(( (c[:rows.sym.NSYM],) for c in colA.get_blocks_charge()),colA.slices))
    row_to_col_sector= { (c[:rows.sym.NSYM],): (c[cols.sym.NSYM:],) for c in A_mat.get_blocks_charge() }
    for c in symU.get_blocks_charge():
        row_sector, i_sector, col_sector= (c[:rows.sym.NSYM],), (c[rows.sym.NSYM:],), \
            row_to_col_sector[(c[:rows.sym.NSYM],)]
        if len(U_sorted[row_sector])<1:
            continue
        inds= U_sorted[row_sector]
        symU[c]= U[slice(*U_sectors[row_sector].slcs[0]),inds]
        symS[(i_sector,i_sector)]= S[inds].real
        symVh[(i_sector,col_sector)]= Vh[inds,slice(*Vh_sectors[col_sector].slcs[0])]

    # fix relative phases of singular vectors
    if fix_signs:
        # associate left and right singular vectors (slices) of the same charge sector
        get_c_of_Vh = lambda c_of_U: tuple( _flatten(c_of_U[rows.sym.NSYM:]+row_to_col_sector[(c_of_U[:rows.sym.NSYM],)]) )
        mU= dict(zip(symU.struct.t,symU.slices))
        mVh= dict(zip(symVh.struct.t,symVh.slices))
        iterlist= ((mU[c].slcs[0], mU[c].D, mVh[get_c_of_Vh(c)].slcs[0],mVh[get_c_of_Vh(c)].D) for c in symU.get_blocks_charge())

        symU._data, symVh._data= A.config.backend.fix_svd_signs(symU._data, symVh._data, \
            ((None,None,slU,DU,None,slVh,DVh) for slU,DU,slVh,DVh in iterlist) )

    symU= symU.unfuse_legs(axes=0)
    symVh= symVh.unfuse_legs(axes=1)

    # Additional truncation
    Smask = truncation_mask(symS, tol=kwargs.get('reltol',0), tol_block=kwargs.get('reltol_block',0),
                            D_block=kwargs.get('D_block',float('inf')), D_total=k,
                            truncate_multiplets=kwargs.get('truncate_multiplets',False),
                            mask_f=kwargs.get('mask_f',None))
    symU, symS, symVh = Smask.apply_mask(symU, symS, symVh, axes=(-1, 0, 0))

    symU = symU.moveaxis(source=-1, destination=Uaxis)
    symVh = symVh.moveaxis(source=0, destination=Vaxis)

    return symU, symS, symVh
