# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
""" Routines for time evolution with nn gates on a 2D lattice. """

from ... import tensordot, vdot, svd_with_truncation, YastnError, Tensor
from ._peps import Peps2Layers
from ._gates_auxiliary import Gate, gate_from_mpo
from ..mps import MpsMpoOBC
from typing import NamedTuple
from itertools import pairwise


class BondMetric(NamedTuple):
    g: Tensor = None


class BipartiteBondMetric(NamedTuple):
    gL: Tensor = None
    gR: Tensor = None


class Evolution_out(NamedTuple):
    """ All errors and eigenvalues are relative. """
    bond: tuple = None
    truncation_error: float = 0
    best_method: str = ''
    nonhermitian_part: float = 0
    min_eigenvalue: float = None
    wrong_eigenvalues: float = None
    eat_metric_error: float = None
    truncation_errors: dict[str, float] = ()
    iterations: dict[str, int] = ()
    pinv_cutoffs: dict[str, float] = ()


def evolution_step_(env, gates, opts_svd, symmetrize=True, method='mpo',
                    fix_metric=0,
                    pinv_cutoffs=(1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
                    max_iter=100, tol_iter=1e-13, initialization="EAT_SVD",
                    ):
    r"""
    Perform a single step of PEPS evolution by applying a list of gates.
    Truncate bond dimension after each application of a two-site gate.

    Parameters
    ----------
    env: EnvNTU | EnvCTM | EnvApproximate
        Environment class containing PEPS state (updated in place),
        and a method to calculate bond metric tensors employed during truncation.
    gates: yastn.tn.fpeps.Gates
        The gates to be applied to PEPS.
    opts_svd: dict | Sequence[dict]
        Options passed to :meth:`yastn.linalg.svd_with_truncation` which are used
        to initially truncate the bond dimension before it is further optimized iteratively.
        In particular, it fixes bond dimensions (potentially, sectorial).
        It is possible to provide a list of dicts (with decreasing bond dimensions),
        in which case the truncation is done gradually in a few steps.
    symmetrize: bool
        Whether to iterate through provided gates forward and then backward, resulting in a 2nd order method.
        In that case, each gate should correspond to half of the desired timestep. The default is ``True``.
    method: str
        For ``'NN'``, split multi-site gates into a series of nearest-neighbor gates.
        Otherwise, apply mpo-gate first, and then sequentially truncate enlarged bonds (the default).
    fix_metric: int | None
        Error measure of the metric tensor is a sum of: the norm of its non-hermitian part
        and absolute value of the most negative eigenvalue (if present).
        If ``fix_metric`` is a number, replace eigenvalues smaller than the error_measure with fix_metric * error_measure.
        Sensible values of ``fix_metric`` are :math:`0` and :math:`1`. The default is :math:`0`.
        If ``None``, do not perform eigh to test for negative eigenvalues and do not fix metric.
    pinv_cutoffs: Sequence[float] | float
        List of pseudo-inverse cutoffs.
        The one that gives the smallest truncation error is used during iterative optimizations and EAT initialization.
    max_iter: int
        The maximal number of iterative steps for each truncation optimization.
    tol_iter: int
        Tolerance of truncation_error to stop iterative optimization.
    initialization: str
        Tested initializations of iterative optimization. The one resulting in the smallest error is selected.
        Possible options are 'SVD' (svd initialization only), 'EAT' (EAT optimization only), 'SVD_EAT' (tries both).

    Returns
    -------
    list[Evolution_out(NamedTuple)]
        List of Namedtuple containing fields:
            * ``bond`` bond where the gate is applied.
            * ``truncation_error`` relative norm of the difference between untruncated and truncated bonds, calculated in metric specified by env.
            * ``best_method`` initialization/optimization method giving the best truncation_error. Possible values are 'eat', 'eat_opt', 'svd', 'svd_opt'.
            * ``nonhermitian_part`` norm of the non-hermitian part of the bond metric, normalized by the bond metric norm. Estimator of metric error.
            * ``min_eigenvalue`` the smallest bond metric eigenvalue, normalized by the bond metric norm. Can be negative which gives an estimate of metric error.
            * ``wrong_eigenvalues`` a fraction of bond metrics eigenvalues that were below the error threshold; and were modified according to ``fix_metric`` argument.
            * ``eat_metric_error`` error of approximating metric tensor by a product via SVD-1 in EAT initialization.
            * ``truncation_errors`` dict with truncation_error-s for all tested initializations/optimizations.
            * ``iterations`` dict with number of iterations to converge iterative optimization.
            * ``pinv_cutoffs`` dict with optimal pinv_cutoffs for methods where pinv appears.
    """
    psi = env.psi
    if isinstance(psi, Peps2Layers):
        psi = psi.ket  # to make it work with CtmEnv

    infos = []
    if symmetrize:
        gates = gates + gates[::-1]

    if 'nn' in method.lower():
        gates = [Gate(gate_from_mpo(gate.G), gate.sites) if isinstance(gate.G, MpsMpoOBC) else gate  for gate in gates]
        gates = [ng for og in gates for ng in split_gate_2site(og)]

    for gate in gates:
        psi.apply_gate_(gate)

        for s0, s1 in pairwise(gate.sites[-1::-1]):
            env.pre_truncation_((s0, s1))
        if len(gate.sites) > 2:
            for s0, s1 in pairwise(gate.sites):
                env.pre_truncation_((s0, s1))

        for s0, s1 in pairwise(gate.sites):
            info = truncate_(env, opts_svd, (s0, s1), fix_metric, pinv_cutoffs, max_iter, tol_iter, initialization)
            infos.append(info)

    return infos


def split_gate_2site(gate):
    """
    Split gate-mpo into a series of 2-site gates using SVD.
    """
    if len(gate.G) < 3:
        return [gate]
    new_gates = []
    g0, s0 = gate.G[0], gate.sites[0]
    for g1, s1 in zip(gate.G[1:-1], gate.sites[1:-1]):
        U, S, V = g1.svd_with_truncation(axes=((1, 2), (0, 3)), Uaxis=0, Vaxis=1, tol=1e-14)
        S = S.sqrt()
        U, V = S.broadcast(U, V, axes=(0, 1))
        new_gates.append(Gate(G=(g0, U), sites=(s0, s1)))
        g0, s0 = V, s1
    new_gates.append(Gate(G=(g0, gate.G[-1]), sites=(s0, gate.sites[-1])))
    return new_gates


def truncate_(env, opts_svd, bond=None,
              fix_metric=0,
              pinv_cutoffs=(1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
              max_iter=100, tol_iter=1e-13, initialization="EAT_SVD"):
    r"""
    Truncate virtual bond dimensions of PEPS.

    Parameters
    ----------
    env: EnvNTU | EnvCTM | EnvApproximate
        Environment class containing PEPS state (updated in place),
        and a method to calculate bond metric tensors employed during truncation.
    opts_svd: dict | Sequence[dict]
        Options passed to :meth:`yastn.linalg.svd_with_truncation` which are used
        to initially truncate the bond dimension before it is further optimized iteratively.
        In particular, it fixes bond dimensions (potentially, sectorial).
        It is possible to provide a list of dicts (with decreasing bond dimensions),
        in which case the truncation is done gradually in a few steps.
    bond: tuple[Site, Site] | None
        Specify single bond to be truncated. If None, truncate all bonds. The default is None.
    fix_metric: int | None
        Error measure of the metric tensor is a sum of: the norm of its non-hermitian part
        and absolute value of the most negative eigenvalue (if present).
        If ``fix_metric`` is a number, replace eigenvalues smaller than the error_measure with fix_metric * error_measure.
        Sensible values of ``fix_metric`` are :math:`0` and :math:`1`. The default is :math:`0`.
        If ``None``, do not perform eigh to test for negative eigenvalues and do not fix metric.
    pinv_cutoffs: Sequence[float] | float
        List of pseudo-inverse cutoffs.
        The one that gives the smallest truncation error is used during iterative optimizations and EAT initialization.
    max_iter: int
        The maximal number of iterative steps for each truncation optimization.
    tol_iter: int
        Tolerance of truncation_error to stop iterative optimization.
    initialization: str
        Tested initializations of iterative optimization. The one resulting in the smallest error is selected.
        Possible options are 'SVD' (svd initialization only), 'EAT' (EAT optimization only), 'SVD_EAT' (tries both).

    Returns
    -------
    Evolution_out(NamedTuple)
        Namedtuple containing fields:
            * ``bond`` bond where the gate is applied.
            * ``truncation_error`` relative norm of the difference between untruncated and truncated bonds, calculated in metric specified by env.
            * ``best_method`` initialization/optimization method giving the best truncation_error. Possible values are 'eat', 'eat_opt', 'svd', 'svd_opt'.
            * ``nonhermitian_part`` norm of the non-hermitian part of the bond metric, normalized by the bond metric norm. Estimator of metric error.
            * ``min_eigenvalue`` the smallest bond metric eigenvalue, normalized by the bond metric norm. Can be negative which gives an estimate of metric error.
            * ``wrong_eigenvalues`` a fraction of bond metrics eigenvalues that were below the error threshold; and were modified according to ``fix_metric`` argument.
            * ``eat_metric_error`` error of approximating metric tensor by a product via SVD-1 in EAT initialization.
            * ``truncation_errors`` dict with truncation_error-s for all tested initializations/optimizations.
            * ``iterations`` dict with number of iterations to converge iterative optimization.
            * ``pinv_cutoffs`` dict with optimal pinv_cutoffs for methods where pinv appears.
    """
    psi = env.psi
    if isinstance(psi, Peps2Layers):
        psi = psi.ket  # to make it work with CtmEnv

    if bond is None:
        bonds = psi.bonds()
    else:
        bonds = [bond]

    infos = []
    for bond in bonds:
        dirn2, l_ordered = psi.nn_bond_type(bond)
        dirn = psi.nn_bond_dirn(*bond)
        if dirn in ('rl', 'bt'):
            bond, dirn = bond[::-1], dirn[::-1]  # 'lr' or 'tb'

        s0, s1 = bond
        info = {'bond': bond}

        if dirn == 'lr':  # Horizontal gate, "lr" ordered
            Q0, R0 = psi[s0].qr(axes=((0, 1, 2, 4), 3), sQ=-1, Qaxis=3)  # t l b rr sa @ rr r
            Q1, R1 = psi[s1].qr(axes=((0, 2, 3, 4), 1), sQ=1, Qaxis=1, Raxis=-1)  # t ll b r sa @ l ll
        else:  # dirn == 'tb':  Vertical gate, "tb" ordered
            Q0, R0 = psi[s0].qr(axes=((0, 1, 3, 4), 2), sQ=1, Qaxis=2)  # t l bb r sa @ bb b
            Q1, R1 = psi[s1].qr(axes=((1, 2, 3, 4), 0), sQ=-1, Qaxis=0, Raxis=-1)  # tt l b r sa @ t tt

        fgf = env.bond_metric(Q0, Q1, s0, s1, dirn2)

        if isinstance(fgf, BipartiteBondMetric):  # bipartite bond metric
            M0, M1, info = truncate_bipartite_(fgf, R0, R1, opts_svd, pinv_cutoffs, info)
        elif isinstance(fgf, BondMetric):  # bipartite bond metric
            M0, M1, info = truncate_optimize_(fgf, R0, R1, opts_svd, fix_metric, pinv_cutoffs, max_iter, tol_iter, initialization, info)

        if dirn == 'lr':
            psi[s0] = tensordot(Q0, M0, axes=(3, 0)).transpose(axes=(0, 1, 2, 4, 3))  # t l b r sa
            psi[s1] = tensordot(M1, Q1, axes=(1, 1)).transpose(axes=(1, 0, 2, 3, 4))  # t l b r sa
        else:  # dirn == 'tb':
            psi[s0] = tensordot(Q0, M0, axes=(2, 0)).transpose(axes=(0, 1, 4, 2, 3))  # t l b r sa
            psi[s1] = tensordot(M1, Q1, axes=(1, 0)) # t l b r sa

        env.post_truncation_(bond)
        infos.append(Evolution_out(**info))
    return infos[0] if len(bonds) == 1 else infos


def truncate_optimize_(g, R0, R1, opts_svd, fix_metric, pinv_cutoffs, max_iter, tol_iter, initialization, info):
    # enforce hermiticity
    fgf = g.g
    fgfH = fgf.H
    nonhermitian = (fgf - fgfH).norm() / 2
    fgf = (fgf + fgfH) / 2
    #
    fgf_norm = fgf.norm()
    info['nonhermitian_part'] = (nonhermitian / fgf_norm).item()
    #
    # check metric tensor eigenvalues
    if fix_metric is not None:
        S, U = fgf.eigh(axes=(0, 1))
        smin = min(S._data)
        info['min_eigenvalue'] = (smin / fgf_norm).item()
        g_error = max(-smin, 0) + nonhermitian
        info['wrong_eigenvalues'] = sum(S._data < g_error).item() / len(S._data)
        S._data[S._data < g_error] = g_error * fix_metric
        fgf = U @ S @ U.H
    #
    fRR = (R0 @ R1).fuse_legs(axes=[(0, 1)])
    fgRR = fgf @ fRR
    RRgRR = abs(vdot(fRR, fgRR).item())
    #
    pinv_cutoffs = sorted(pinv_cutoffs)
    M0, M1 = R0, R1
    for opts in [opts_svd] if isinstance(opts_svd, dict) else opts_svd:

        Ms, error2s, pinvs, iters = {}, {}, {}, {}

        if 'EAT' in initialization:
            key = 'eat'
            Ms[key], error2s[key], pinvs[key], info["eat_metric_error"] = initial_truncation_EAT(M0, M1, fgf, fRR, RRgRR, opts, pinv_cutoffs)
            key = 'eat_opt'
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['eat'], error2s['eat'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
        if 'SVD' in initialization:
            key = 'svd'
            Ms[key] = symmetrized_svd(M0, M1, opts, normalize=False)
            error2s[key] = calculate_truncation_error2(Ms[key][0] @ Ms[key][1], fgf, fRR, RRgRR)
            key = 'svd_opt'
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['svd'], error2s['svd'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
        if len(Ms) == 0:
            raise YastnError(f"{initialization=} not recognized. Should contain 'SVD' or 'EAT'.")

        error2s = {k: v ** 0.5 for k, v in error2s.items()}
        key = min(error2s, key=error2s.get)
        info['best_method'] = key
        info['truncation_errors'] = error2s
        info['truncation_error'] = error2s[key]
        info['pinv_cutoffs'] = pinvs
        info['iterations'] = iters
        M0, M1 = Ms[key]

    M0, M1 = symmetrized_svd(M0, M1, opts, normalize=True)
    return M0, M1, info


def truncate_bipartite_(g, R0, R1, opts_svd, pinv_cutoffs, info):
    #
    E0, E1 = g.gL, g.gR
    E0 = (E0 + E0.H) / 2
    E1 = (E1 + E1.H) / 2
    #
    RR = (R0 @ R1)
    gRR = E0 @ RR @ E1
    RRgRR = abs(vdot(RR, gRR).item())
    #
    F0 = R0.H @ E0 @ R0
    F1 = R1 @ E1 @ R1.H
    #
    S0, U0 = F0.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs))
    S1, U1 = F1.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs))
    #
    W0, W1 = symmetrized_svd(S0.sqrt() @ U0.H, U1 @ S1.sqrt(), opts_svd, normalize=False)
    p0, p1 = R0 @ U0, U1.H @ R1
    #
    error2, s0max, s1max = 100, max(S0._data), max(S1._data)
    vs0o, vs1o = len(S0._data) + 1, len(S1._data) + 1
    for c_off in sorted(pinv_cutoffs):
        vs0 = sum(S0._data > c_off * s0max).item()
        vs1 = sum(S1._data > c_off * s1max).item()
        if vs0 < vs0o or vs1 < vs1o:
            vs0o, vs1o = vs0, vs1
            M0_tmp = p0 @ S0.reciprocal(cutoff=c_off * s0max).sqrt() @ W0
            M1_tmp = W1 @ S1.reciprocal(cutoff=c_off * s1max).sqrt() @ p1
            delta = RR - M0_tmp @ M1_tmp
            error2_tmp = abs(vdot(delta, E0 @ delta @ E1).item()) / RRgRR
            if error2_tmp < error2:
                M0, M1, error2, pinv_c = M0_tmp, M1_tmp, error2_tmp, c_off

    info['best_method'] = 'bipartite'
    info['truncation_errors'] = {'bipartite': error2 ** 0.5}
    info['truncation_error'] = error2 ** 0.5
    info['pinv_cutoffs'] = pinv_c
    return M0, M1, info


def accumulated_truncation_error(infoss, statistics='mean'):
    r"""
    Return accumulated truncation error :math:`\Delta` calcuated from evolution output statistics.

    :math:`\Delta = \sum_{steps} statistics_{bond} [\sum_{gate \in bond} truncation\_error(gate, step)]`,
    where statistics is mean or max.

    Gives an estimate of errors accumulated during time evolution.

    Parameters
    ----------
    infoss: Sequence[Sequence[Evolution_out]]
        list of outputs of :meth:`evolution_step_`.
    statistics: str
        'max' or 'mean', whether to take the maximal value or a mean over the bonds in the lattice.

    Example
    -------

    ::

        infoss = []
        for step in range(number_of_steps):
            infos = fpeps.evolution_step_(env, gates, opt_svd)
            infoss.append(infos)

        # Accumulated runcation error
        Delta = fpeps.accumulated_truncation_error(infoss)
    """
    if statistics == 'max':
        stat = lambda x: max(x)
    elif statistics == 'mean':
        stat = lambda x: sum(x) / len(x)
    else:
        raise YastnError(f"{statistics=} in accumulated_truncation_error not recognized; Should be 'max' or 'mean'.")

    Delta = 0
    for infos in infoss:
        accum = {}
        for info in infos:
            accum[info.bond] = accum.get(info.bond, 0) + info.truncation_error
        Delta += stat(accum.values())
    return Delta


def symmetrized_svd(R0, R1, opts_svd, normalize=False):
    """ SVD truncation of central tensor; divide singular values symmetrically between tensors. """
    Q0, R0 = R0.qr(axes=(0, 1))
    Q1, R1 = R1.qr(axes=(1, 0), Qaxis=0, Raxis=1)
    U, S, V = svd_with_truncation(R0 @ R1, sU=R0.s[1], **opts_svd)
    if normalize:
        S = S / S.norm(p='inf')
    S = S.sqrt()
    M0, M1 = S.broadcast(U, V, axes=(1, 0))
    return Q0 @ M0, M1 @ Q1


def calculate_truncation_error2(fMM, fgf, fRR, RRgRR):
    """ Calculate squared truncation error. """
    if fMM.ndim == 2:  # if legs of MM not fused into vector
        fMM = fMM.fuse_legs(axes=[(0, 1)])
    delta = fRR - fMM
    return abs(vdot(delta, fgf @ delta).item()) / RRgRR


def initial_truncation_EAT(R0, R1, fgf, fRR, RRgRR, opts_svd, pinv_cutoffs):
    """
    Truncate R0 @ R1 to bond dimension specified in opts_svd
    including information from a product approximation of bond metric.
    """
    G = fgf.unfuse_legs(axes=(0, 1))
    #
    # rank-1 approximation
    Gremove = G.remove_zero_blocks()
    G0, S, G1 = svd_with_truncation(Gremove, axes=((0, 2), (3, 1)), policy='lowrank', D_block=1, D_total=1)
    fid = (S.norm() / G.norm()).item()
    eat_metric_error = (max(0., 1 - fid ** 2)) ** 0.5
    #
    G0 = G0.remove_leg(axis=2)
    G1 = G1.remove_leg(axis=0)
    #
    # make sure it is hermitian
    G0 = G0 / G0.trace().to_number()
    G1 = G1 / G1.trace().to_number()
    G0 = (G0 + G0.H) / 2
    G1 = (G1 + G1.H) / 2
    #
    F0 = R0.H @ G0 @ R0
    F1 = R1 @ G1 @ R1.H
    #
    S0, U0 = F0.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs))
    S1, U1 = F1.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs))
    #
    W0, W1 = symmetrized_svd(S0.sqrt() @ U0.H, U1 @ S1.sqrt(), opts_svd, normalize=False)
    p0, p1 = R0 @ U0, U1.H @ R1
    #
    error2, s0max, s1max = 100, max(S0._data), max(S1._data)
    vs0o, vs1o = len(S0._data) + 1, len(S1._data) + 1
    for c_off in pinv_cutoffs:
        vs0 = sum(S0._data > c_off * s0max).item()
        vs1 = sum(S1._data > c_off * s1max).item()
        if vs0 < vs0o or vs1 < vs1o:
            vs0o, vs1o = vs0, vs1
            M0_tmp = p0 @ S0.reciprocal(cutoff=c_off * s0max).sqrt() @ W0
            M1_tmp = W1 @ S1.reciprocal(cutoff=c_off * s1max).sqrt() @ p1
            error2_tmp = calculate_truncation_error2(M0_tmp @ M1_tmp, fgf, fRR, RRgRR)
            if error2_tmp < error2:
                M0, M1, error2, pinv_c = M0_tmp, M1_tmp, error2_tmp, c_off
    return (M0, M1), error2, pinv_c, eat_metric_error


def optimize_truncation(M0, M1, error2_old, fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter):
    r"""
    Optimizes the matrices M0 and M1 by minimizing
    truncation error using least square optimization.
    """
    gf = fgf.unfuse_legs(axes=0)
    gRR =  fgRR.unfuse_legs(axes=0)

    for iter in range(1, max_iter+1):
        # fix M1 and optimize M0
        j0 = (gRR @ M1.H).fuse_legs(axes=[(0, 1)])
        g0 = tensordot(M1.conj(), gf, axes=(1, 1)).fuse_legs(axes=((1, 0), 2))
        g0 = g0.unfuse_legs(axes=1)
        g0 = tensordot(g0, M1, axes=(2, 1)).fuse_legs(axes=(0, (1, 2)))
        error_fun = lambda x: calculate_truncation_error2(x @ M1, fgf, fRR, RRgRR)
        M0, error2, pinv_c = optimal_pinv(g0, j0, pinv_cutoffs, error_fun)

        # fix M0 and optimize M1
        j1 = (M0.H @ gRR).fuse_legs(axes=[(0, 1)])
        g1 = tensordot(M0.conj(), gf, axes=(0, 0)).fuse_legs(axes=((0, 1), 2))
        g1 = g1.unfuse_legs(axes=1)
        g1 = tensordot(g1, M0, axes=(1, 0)).fuse_legs(axes=(0, (2, 1)))
        error_fun = lambda x: calculate_truncation_error2(M0 @ x, fgf, fRR, RRgRR)
        M1, error2, pinv_c = optimal_pinv(g1, j1, pinv_cutoffs, error_fun)

        # convergence condition
        if abs(error2 - error2_old) < tol_iter:
            break
        error2_old = error2

    return (M0, M1), error2, pinv_c, iter


def optimal_pinv(g, j, pinv_cutoffs, error_fun):
    r"""
    M = pinv(g) * j, where pinv cutoff is optimized based on error_fun.
    """
    S, U = g.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs))
    smax = max(S._data)
    UHJ = U.H @ j
    error2, values_old = 100, len(S._data) + 1
    for cutoff in pinv_cutoffs:
        values = sum(S._data > cutoff * smax).item()
        if values < values_old:
            values_old = values
            Sr = S.reciprocal(cutoff=cutoff * smax)
            SrUHJ = Sr.broadcast(UHJ, axes=0)
            M_tmp = (U @ SrUHJ).unfuse_legs(axes=0)
            error2_tmp = error_fun(M_tmp)
            if error2_tmp < error2:
                M_new, error2, pinv_c = M_tmp, error2_tmp, cutoff
    return M_new, error2, pinv_c
