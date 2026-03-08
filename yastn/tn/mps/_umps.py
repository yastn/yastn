# Copyright 2026 The YASTN Authors. All Rights Reserved.
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

from typing import Sequence, Union
from ...tensor import Leg, ncon, Tensor, block, tensordot
from ...initialize import rand, eye
from ..._from_dict import from_dict, combine_data_and_meta, split_data_and_meta
from ...mps import mps

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs


def apply_transfer_matrix(
        umps: Union[mps.Mps,Sequence["Tensor"]],
        v: "Tensor",
        umps_conj: Union[mps.Mps,Sequence["Tensor"]] = None
    ) -> "yastn.Tensor":
    """
    Applies the transfer matrix of a uniform MPS (uMPS) to a tensor `v`.

    This contracts the sequence of uMPS tensors (and optionally different conjugate tensors)
    with `v`, effectively computing the action of the transfer matrix on `v`.
    NOTE / TODO Intial guess, v, must be in a zero sector.

    ::
    
               -- A  -- 2  2 --A --- 5 5 -- A --- 8 8 -- A --- 11 ...
      \rho  = |                3            6            9
               -- A+ -- 1  1 --A+ -- 4 4 -- A+ -- 7 7 -- A+ -- 10 ...

    Parameters
    ----------
    umps :
        List of MPS tensors (rank-3, typically with axes (left, phys, right)).
    v : yastn.Tensor
        The tensor to which the transfer matrix is applied. Should be rank-2 or rank-3.
        In case of rank-3, the first leg is treated as a batch dimension.
    umps_conj :
        List of different conjugate MPS tensors. If None, uses the conjugate of `umps`.

    Returns
    -------
    yastn.Tensor
        The result of applying the transfer matrix to `v`.
    """
    # solve for leading eigenvector of transfer matrix via arnoldi method
    # assume v is rank-2 or rank-3 in which case first leg is batch 
    #
    
    #     
    if umps_conj is None:
        umps_conj= [umps[n] for n in range(len(umps))]
        conjs= [0,]+sum([[1,0] for n in range(len(umps))], [])
    else:
        conjs= None
    ts= [v,]+sum([[umps_conj[n],umps[n]] for n in range(len(umps))], [])
    idxs= ([[1,2],] if v.ndim == 2 else [[-1,1,2],] ) + \
        sum([ [[1+3*n, 3*(n+1) ,1+3*(n+1)],[2+3*n,3*(n+1),2+3*(n+1)] ] for n in range(len(umps))], [])
    idxs[-2][-1]= -1 - ( v.ndim == 3 )
    idxs[-1][-1]= -2 - ( v.ndim == 3 )

    vNAA= ncon(ts,idxs,conjs=conjs)
    return vNAA


def eigs_implicit(umps:Union[mps.Mps,Sequence["Tensor"]], 
                  umps_conj:Union[mps.Mps,Sequence["Tensor"]]=None, 
                  k=1, eigenvectors=False, V0:"Tensor"=None):
    """
    Compute the k leading eigenvalues (and optionally eigenvectors) of the
    transfer matrix of a uMPS using an implicit Arnoldi method (scipy ``eigs``).

    The transfer matrix is defined by :func:`apply_transfer_matrix`. It is
    never formed explicitly; matrix-vector products are evaluated on-the-fly
    by packing/unpacking the YASTN tensor structure around the flat array
    expected by ``scipy.sparse.linalg.eigs``.

    Parameters
    ----------
    umps :
        ket layer of the transfer matrix.
    umps_conj :
        bra layer. Pass a different sequence to obtain a biorthogonal/mixed transfer matrix.
        If none, the conjugate of `umps` is used.
    k : int, optional
        Number of leading eigenvalues to compute. Default is 1.
    eigenvectors : bool, optional
        If ``True``, also return the eigenvectors as a batched YASTN tensor.
        Default is ``False``.
    V0 : yastn.Tensor, optional
        Initial guess for the dominant eigenvector (rank-2).
        When ``None`` a random tensor with the correct leg structure is created.

    Returns
    -------
    TODO return eigenvalues as yastn.Tensor
    vals : ndarray, shape (k,)
        Leading eigenvalues sorted by decreasing magnitude.
    Vecs : yastn.Tensor or None
        eigenvectors when ``eigenvectors=True``,
        ``None`` otherwise. The batch axis (axis 0) indexes the k eigenvectors.
    """
    cfg= umps[0].config

    if V0 is None:
        # initial guess
        V0= rand(umps[0].config, legs=[umps[0].get_legs(axes=0), 
            umps[0].get_legs(axes=0).conj() if umps_conj is None else umps_conj[0].get_legs(axes=0)])
    V0_data, V0_meta= split_data_and_meta(V0.to_dict())

    def _mv(v):
        V= from_dict(combine_data_and_meta(cfg.backend.to_tensor(v,
            dtype=cfg.default_dtype,
            device=cfg.default_device),
            meta=V0_meta))
        MV= apply_transfer_matrix(umps,V,umps_conj=umps_conj)
        MV_data, MV_meta= split_data_and_meta(MV.to_dict(level=2))
        return MV_data[0]

    T= LinearOperator((V0.size,V0.size), matvec=_mv, dtype=cfg.default_dtype)
    if not eigenvectors:
        vals= eigs(T, k=k, v0=None, return_eigenvectors=False)
        return vals, None
    
    vals, vecs= eigs(T, k=k, v0=None, return_eigenvectors=True)
    Vecs= block( {col: from_dict(combine_data_and_meta(cfg.backend.to_tensor(vecs[:,col],
        dtype=cfg.default_dtype,
        device=cfg.default_device),
        meta=V0_meta)).add_leg(axis=0) for col in range(vecs.shape[1])}, common_legs=tuple(i+1 for i in range(len(V0.get_legs()))) ) 
    
    return vals, Vecs


def eigs_implicit_v2(mv:callable, config=None, legs:Sequence[Leg]=None,
                    k=1, eigenvectors=False, V0:"Tensor"=None):
    """
    Compute the k leading eigenvalues (and optionally eigenvectors) of matrix-vector operator 
    using an implicit Arnoldi method (scipy ``eigs``).

    Matrix-vector products are evaluated on-the-fly
    by packing/unpacking the YASTN tensor structure around the flat array
    expected by ``scipy.sparse.linalg.eigs``.

    Parameters
    ----------
    mv : callable(ndarray)->ndarray
        A function that takes a 1D numpy.ndarray `v` and returns the result of applying some, 
        here YASTN-based linear operator to `v`, returning 1D numpy.ndarray.
    k : int, optional
        Number of leading eigenvalues to compute. Default is 1.
    eigenvectors : bool, optional
        If ``True``, also return the eigenvectors as a batched YASTN tensor.
        Default is ``False``.
    legs : Sequence[Leg], optional
        Leg structure of the eigenvectors. Required if `V0` is None.
    V0 : yastn.Tensor, optional
        Initial guess for the dominant eigenvector (rank-2).
        When ``None`` a random tensor with the correct leg structure is created.

    Returns
    -------
    TODO return eigenvalues as yastn.Tensor
    vals : ndarray, shape (k,)
        Leading eigenvalues sorted by decreasing magnitude.
    Vecs : yastn.Tensor or None
        eigenvectors when ``eigenvectors=True``,
        ``None`` otherwise. The batch axis (axis 0) indexes the k eigenvectors.
    """
    assert V0 or (config and legs), "Either V0 or (config and legs) must be provided to determine the shape of the eigenvectors."
    if V0 is None:
        # initial guess
        V0= rand(config=config, legs=legs)
    config= V0.config
    V0_data, V0_meta= split_data_and_meta(V0.to_dict())

    def _mv(v):
        V= from_dict(combine_data_and_meta(config.backend.to_tensor(v,
            dtype=config.default_dtype,
            device=config.default_device),
            meta=V0_meta))
        MV= mv(V)
        MV_data, MV_meta= split_data_and_meta(MV.to_dict(level=2))
        return MV_data[0]

    T= LinearOperator((V0.size,V0.size), matvec=_mv, dtype=config.default_dtype)
    if not eigenvectors:
        vals= eigs(T, k=k, v0=None, return_eigenvectors=False)
        return vals, None
    
    vals, vecs= eigs(T, k=k, v0=None, return_eigenvectors=True)
    Vecs= block( {col: from_dict(combine_data_and_meta(config.backend.to_tensor(vecs[:,col],
        dtype=config.default_dtype,
        device=config.default_device),
        meta=V0_meta)).add_leg(axis=0) for col in range(vecs.shape[1])}, common_legs=tuple(i+1 for i in range(len(V0.get_legs()))) ) 
    
    return vals, Vecs


def qr_sweep_all_cyclic(umps, C_init=None):
    """
    Compute all N cyclic variants in two O(N) sweeps.

    Variant k uses C_k = R_{k-1} (the R entering site k in sweep 1).
    Variant k produces:
        qs  = [Q[k], ..., Q[N-1], Q'[0], ..., Q'[k-1]]
        R   = R'[k]

    Returns
    -------
    qs1, qs2 : lists of Q tensors from sweep 1 and sweep 2
    Rs1, Rs2 : lists of R matrices (Rs1[k] = R entering site k in sweep 1)
    """
    N = len(umps)

    def _sweep(umps, C):
        R = C
        qs, Rs = [], [C]
        for i in range(len(umps)):
            RA = tensordot(R, umps[i], axes=((1,), (0,)))
            Q, R = RA.qr(axes=((0, 1), 2), sQ=1, Qaxis=2)
            qs.append(Q)
            Rs.append(R)
            print(f"{i} {umps[i].get_legs(0)} {R}")
        return qs, Rs  # Rs[k] = R entering site k; Rs[N] = final R

    if C_init is None:
        legs = [umps[0].get_legs(axes=0), umps[0].get_legs(axes=0).conj()]
        C_init = eye(umps[0].config, legs=legs)
    else:
        _, C_init= C_init.qr(axes=(0, 1))  # ensure C_init is isometric

    # (C_init<=>R_0) A_0 -> Q_0 R_1
    #                           R_1 A_1 -> Q_1 R_2
    #                                      ... R_{N-1} A_{N-1} -> Q_{N-1} R_N 
    # R_N A_0 -> Q_0 R_1
    #                R_1 A_1 -> Q_1 R_2
    #                           ... R_{N-1} A_{N-1} -> Q_{N-1} R_N
    #
    qs1, Rs1 = _sweep(umps, C_init)     # sweep 1 with C = I
    qs2, Rs2 = _sweep(umps, Rs1[N])     # sweep 2 with C = R_N

    results = []
    for k in range(N):
        results.append({
            'shift': k,
            'qs': qs1[k:] + qs2[:k],   # [Q[k..N-1], Q'[0..k-1]] 
            'R':  Rs2[k],               # R'[k]
        })
        # k = 0 : [Q[0..N-1], Q'[0..-1]] = [Q[0..N-1], ]  -> R'[0] = R_N
        # k = 1 : [Q[1..N-1], Q'[0..0]] = [Q[1..N-1], Q'[0]] -> R'[1]
        # k = 2 : [Q[2..N-1], Q'[0..1]] = [Q[2..N-1], Q'[0], Q'[1]] -> R'[2]
    return results


def isogauge_left(umps, C_init=None, eps=1e-12):
    # generate first set of candidates
    results= qr_sweep_all_cyclic(umps, C_init=C_init)
    R= [ res['R'] for res in results ]
    lambdas= [ r.norm() for r in R ]
    R_old= [ r * (1/l) for l,r in zip(lambdas,R) ]
    R= [ r for r in R ]
    Delta= [ float('inf') ] * len(R_old)
    n_iter= 0
    while max(Delta) > eps and n_iter < 20:
        
        for i, res in enumerate(results):
            vals, vecs = eigs_implicit([ umps[(i+j) % len(umps)] for j in range(len(umps)) ], 
                                         umps_conj= [q.conj() for q in results[i]['qs']], 
                                         k=1, eigenvectors=True, V0=R[i])
            print(vals, vecs.get_legs())
            _,r= vecs.remove_leg(0).qr(axes=(0, 1))
            R_old[i]= r * (1/r.norm())

        results= qr_sweep_all_cyclic(umps, C_init=R_old[0])
        R= [ res['R'] for res in results ]
        lambdas= [ r.norm() for r in R ]
        R= [ r * (1/l) for l,r in zip(lambdas,R) ]
        Delta= [ (r-r_old).norm() for r,r_old in zip(R,R_old) ]
        

        print(f"Iteration {n_iter}: max Delta = {max(Delta):.2e}, lambdas = {[f'{l:.4f}' for l in lambdas]} {[ (r-r_old).norm() for r,r_old in zip(R,R_old) ]}")
        for r in R:
            print(r.trace().to_number())
        n_iter += 1

        for i, res in enumerate(results):
            res['R']= R[i]
            res['lambda']= lambdas[i]
    return results


def qr_sweep(umps:Union[mps.Mps,Sequence["Tensor"]], C_init:"Tensor"=None):
    """
    Execute series of QR decompositions:  
    
    R[i-1] @ umps[i] = Q[i] R[i],  with R[-1] = C.
                            R[i] @ umps[i+1] = Q[i+1] R[i+1],  with R[N-1] @ umps[0] = Q[0] R[0] 
                                                      ...
    Parameters
    ----------
    umps : Mps
    C    : initial guess for R[-1] as a rank-2 tensor. If None, uses identity.

    Returns
    -------
    qs : list of Q tensors, each with axes (left, phys, right)
    R  : list of R tensors
    """
    if C_init is None:
        legs = [umps[0].get_legs(axes=0), umps[0].get_legs(axes=0).conj()]
        C_init = eye(umps[0].config, legs=legs)
    else:
        _, C_init= C_init.qr(axes=(0, 1))  # ensure C_init is isometric

    Qs, Rs, R = [], [], C_init
    for i in range(len(umps)):
        A= umps[i]
        # R: (D, D_left),  A: (D_left, d, D_right)  ->  RA: (D, d, D_right)
        RA = tensordot(R, A, axes=((1,), (0,)))
        # QR: axes (0,1) -> Q with new bond at Qaxis=2,  axis 2 -> R with bond at Raxis=0
        Q, R = RA.qr(axes=((0, 1), 2), sQ=1, Qaxis=2)
        Qs.append(Q)
        Rs.append(R)
    return Qs, Rs


def isogauge_left_v2(umps, C_init=None, eps=1e-12):
    # generate first set of candidates
    #
    #  -- Rs[0] -- [A0 -- A1 -- ... -- AN-1] -- = lambda -- [Q0 ... QN-1] -- Rs[0] --
    #  -- Rs[1] -- [A1 -- ... -- AN-1 -- A0] -- = lambda -- [Q1 -- Q2 -- ... QN-1 -- Q0] -- Rs[1] -- 
    #  ...
    #  -- Rs[N-1] -- [AN-1 -- A0 -- A1 -- ... -- AN-2] -- = lambda -- [QN-1 -- Q0 -- ... QN-2] -- Rs[N-1] --
    #
    Qs, Rs= qr_sweep(umps, C_init=C_init)
    for i in range(len(umps)):
        print(f"R[{i}] shape: {Rs[i]}")
    l = Rs[-1].norm()
    R= Rs[-1] * (1/l)
    Delta= float('inf')
    n_iter= 0
    while Delta > eps and n_iter < 1000:
        vals, vecs = eigs_implicit( umps, umps_conj= [q.conj() for q in Qs], 
                                    k=1, eigenvectors=True, V0=R)
        print(vals, vecs.get_legs())
        _,R= vecs.remove_leg(0).qr(axes=(0, 1))
        R_old= R * (1/R.norm())

        Qs, Rs= qr_sweep(umps, C_init=R_old)
        l= Rs[-1].norm()
        R= Rs[-1] * (1/l)
        Delta= (R-R_old).norm()
        
        print(f"Iteration {n_iter}: Delta = {Delta}, lambda = {l}")
        n_iter += 1

    Rs= [R,] + Rs[:-1]
    return Qs, Rs


def biorthogonalize_left(umps_top:Union[mps.Mps,Sequence["Tensor"]], 
                         umps_bottom:Union[mps.Mps,Sequence["Tensor"]], 
                         C_init="Tensor", pinv_cutoff=1e-12, eps=1e-12):
    """
    Find the left biorthogonal gauge for a pair of uMPS (ket/bra) layers.

    ::

        -- C_LU  -- A_t --     -- P_L -- C_LU --
                    |       =     |
        -- C_DL  -- A_b --     -- Pbar_L -- C_DL -- 

    Iteratively computes gauge matrices ``C_LU`` and ``C_DL`` such that the
    single-site tensors ``P_L`` (ket) and ``Pbar_L`` (bra) satisfy the
    biorthogonality condition::

        sum_{s} P_L[s] · Pbar_L[s]^T  =  λ · I

    The algorithm follows the left biorthogonalization procedure:

    1. Find the dominant left eigenvector ``X_L`` of the mixed transfer matrix
       built from ``umps_top`` (ket) and ``umps_bottom`` (bra).
    2. SVD-factorize ``X_L = U_L Σ_L V_L†`` to initialize::

           C_LU  = √Σ_L V_L†        (upper gauge factor)
           C_DL  = U_L √Σ_L         (lower gauge factor)

    3. Absorb the gauges into single-site tensors::

           P_L    = C_LU · A_top[0] · C_LU⁺
           Pbar_L = C_DL^T · A_bot[0] · C_DL⁻ᵀ

    4. Iterate: find the dominant eigenvector ``Y_L`` of the single-site mixed
       transfer matrix ``{P_L, Pbar_L}``, SVD-factorize, and update all gauge
       matrices and single-site tensors until ``‖I_approx − I‖ < eps``.

    Parameters
    ----------
    umps_top :
        Sequence of rank-3 ket tensors (axes: left, phys, right).
    umps_bottom :
        Sequence of rank-3 bra tensors with the same structure.
    C_init :
        Unused placeholder for a future warm-start initial gauge. Default ``None``.
    pinv_cutoff : float, optional
        Singular-value cutoff for the pseudo-inverses of gauge matrices.
        Default ``1e-12``.
    eps : float, optional
        Convergence threshold on ``‖I_approx − I‖``. Default ``1e-12``.

    Returns
    -------
    P_L : yastn.Tensor
        Biorthogonalized single-site ket tensor (rank-3).
    Pbar_L : yastn.Tensor
        Biorthogonalized single-site bra tensor (rank-3).
    C_LU : yastn.Tensor
        Upper gauge matrix (rank-2) transforming the ket bond index.
    C_DL : yastn.Tensor
        Lower gauge matrix (rank-2) transforming the bra bond index.
    """
    eval, evecs= eigs_implicit(umps_top, umps_conj=umps_bottom, k=1, eigenvectors=True, V0=C_init)

    U_L, S_L, V_Ldag= evecs.remove_leg(0).svd(axes=(0, 1), sU=1, nU=True, compute_uv=True,
        Uaxis=-1, Vaxis=0, policy='fullrank', fix_signs=False)

    #CLU ≡ ΣLV †  L and  CDL ≡ ULΣL
    C_LU= S_L.sqrt() @ V_Ldag
    C_DL= U_L @ S_L.sqrt()
    C_LU_pinv= (V_Ldag.H).consume_transpose() @ S_L.rsqrt(cutoff=pinv_cutoff) # TODO relative cutoff
    C_DL_pinv= S_L.rsqrt(cutoff=pinv_cutoff) @ (U_L.H).consume_transpose()

    P_L = (C_LU @ umps_top[0]) @ C_LU_pinv
    Pbar_L = (C_DL.transpose() @ umps_bottom[0]) @ C_DL_pinv.transpose()
    # Pbar_L = (C_DL_pinv @ umps_bottom[0]) @ C_DL

    Delta, n_iter= float('inf'), 0
    while Delta>eps and n_iter < 10:
        eval, evecs= eigs_implicit([P_L], umps_conj=[Pbar_L], k=1, eigenvectors=True)
        
        U_L, S_L, V_Ldag= evecs.remove_leg(0).svd(axes=(0, 1), sU=1, nU=True, compute_uv=True,
                                                Uaxis=-1, Vaxis=0, policy='fullrank', fix_signs=False) # Y_L
        
        Y_LU = S_L.sqrt() @ V_Ldag
        Y_DL = U_L @ S_L.sqrt()
        Y_LU_pinv= (V_Ldag.H).consume_transpose() @ S_L.rsqrt(cutoff=pinv_cutoff) # TODO relative cutoff
        Y_DL_pinv= S_L.rsqrt(cutoff=pinv_cutoff) @ (U_L.H).consume_transpose()

        # Ps_L → Y_LU Ps_L Y+_LU, 
        # [P−L]s → Y+_DL [P−L]s Y_DL, 
        # C_LU → Y_LU C_LU, and  C_DL → Y_DL C_DL
        P_L= (Y_LU @ P_L) @ Y_LU_pinv
        Pbar_L= (Y_DL.transpose() @ Pbar_L) @ Y_DL_pinv.transpose()
        C_LU= Y_LU @ C_LU
        C_DL= Y_DL @ C_DL

        # check biorthogonality of P_L and Pbar_L
        I_approx= P_L.tensordot(Pbar_L, axes=((0, 1), (0, 1)))*(1/eval)
        Delta= (I_approx - eye(I_approx.config, legs=I_approx.get_legs(), isdiag=False)).norm()
        print(f"Iteration {n_iter}: eval = {eval}, ||I_approx - I|| = {Delta}")
        n_iter += 1

    return P_L, Pbar_L, C_LU, C_DL