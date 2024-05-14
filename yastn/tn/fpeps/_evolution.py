""" Routines for time evolution with nn gates on a 2D lattice. """

from ... import tensordot, vdot, svd_with_truncation, YastnError
from ._peps import Peps2Layers
from ._gates_auxiliary import apply_gate_onsite, apply_gate_nn, gate_fix_order, apply_bond_tensors
from typing import NamedTuple


class Evolution_out(NamedTuple):
    """ All errors and eigenvalues are relative. """
    bond: tuple = None
    truncation_error: float = 0
    truncation_error_init: float = 0
    truncation_error_iter: float = 0
    nonhermitian_part: float = 0
    min_eigenvalue: float = 0
    second_eigenvalue: float = 0
    wrong_eigenvalues: float = 0
    iterations: int = 0
    pinv_cutoff: float = 0
    exit_code: int = 0


def evolution_step_(env, gates, opts_svd, symmetrize=True,
                    initialization="EAT", fix_metric=0,
                    pinv_cutoffs=(1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
                    max_iter=100, tol_iter=1e-14):
    r"""
    Perform a single step of PEPS evolution by applying a list of gates.
    Truncate bond dimension after each application of a two-site gate.

    Parameters
    ----------
    env: EnvNTU | ...
        Environment class containing PEPS state (updated in place),
        and a method to calculate bond metric tensors employed during truncation.
    gates: Gates
        The gates to be applied to PEPS.
    opts_svd: dict | Sequence[dict]
        Options passed to :meth:`yastn.linalg.svd_with_truncation` which are used
        to initially truncate the bond dimension before it is further optimized iteratively.
        In particular, it fixes bond dimensions (potentially, sectorial).
        It is possible to provide a list of dicts (with decreasing bond dimensions),
        in which case the truncation is done gradually in a few steps.
    symmetrize: bool
        Whether to iterate through provided gates forward and then backward, resulting in a 2nd order method.
        In that case, each gate should correspond to half of the desired timestep.
    initialization: str
        "SVD" or "EAT". Type of procedure initializing the optimization of truncated PEPS tensors
        after application of a two-site gate. Employ plain SVD, or EAT, which is an SVD truncation
        including a product approximation of bond metric tensor. By default uses "EAT".
    fix_metric: int | None:
        Error_measure of the metric tensor is a sum of: the norm of its non-hermitian part,
        and absolute value of the most negative eigenvalue (if present).
        If fix_metric is not None, replace eigenvalues smaller than the error_measure
        by fix_metric * error_measure. Sensible values of fix_metric are 0 and 1. Default is 0.
    pinv_cutoffs : Sequence[float] | float
        List of pseudo-inverse cutoffs.
        The one that gives the smallest truncation error is used during iterative optimization and EAT initialization.
    max_iter : int
        The maximal number of iterative steps for each truncation optimization.
    max_tol : int
        Tolerance of truncation_error to stop iterative optimization.

    Returns
    -------
    Evolution_out(NamedTuple)
        Namedtuple containing fields:
            * :code:`bond` bond where the gate is applied
            * :code:`truncation_error` relative norm of the difference between untruncated and truncated bonds, calculated in metric specified by env. It is the smallest number of the two below.
            * :code:`truncation_error_init` error after initialization.
            * :code:`truncation_error_iter` error after iterative optimization.
            * :code:`nonhermitian_part` norm of the non-hermitian part of the bond metric, normalized by its largest eigenvalue. Estimator of a metric error.
            * :code:`min_eigenvalue` a ratio of the smallest and the largest bond metric eigenvalue. Can be negative which gives an estimate of a metric error.
            * :code:`second_eigenvalue` a ratio of the second largest and the largest bond metric eigenvalue.
            * :code:`wrong_eigenvalues` a fraction of bond metrics eigenvalues that were below the error threshold; and were modified according to :code:`fix_metric` argument.
            * :code:`iterations` a number of iterations to iterative optimization convergence
            * :code:`pinv_cutoff` the optimal cutoff used in the last iteration.
            * :code:`exit_code`. Adds 1 if max_iter reached;  Adds 2 if initial truncation gives better error than subsequent iterative procedure.
    """
    infos = []

    psi = env.psi
    if isinstance(psi, Peps2Layers):
        psi = psi.ket  # to make it work with CtmEnv

    for gate in gates.local:
        psi[gate.site] = apply_gate_onsite(psi[gate.site], gate.G)
    for gate in gates.nn:
        info = apply_nn_truncate_optimize_(env, psi, gate, opts_svd, initialization, fix_metric, pinv_cutoffs, max_iter, tol_iter)
        infos.append(info)
    if symmetrize:
        for gate in gates.nn[::-1]:
            info = apply_nn_truncate_optimize_(env, psi, gate, opts_svd, initialization, fix_metric, pinv_cutoffs, max_iter, tol_iter)
            infos.append(info)
        for gate in gates.local[::-1]:
            psi[gate.site] = apply_gate_onsite(psi[gate.site], gate.G)
    return infos


def apply_nn_truncate_optimize_(env, psi, gate, opts_svd,
                    initialization="EAT", fix_metric=0,
                    pinv_cutoffs=(1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
                    max_iter=100, tol_iter=1e-14):
    r"""
    Applies a nearest-neighbor gate to a PEPS tensor, truncate, and
    optimize the resulting tensors using alternate least squares.
    """
    info = {'bond': gate.bond}

    dirn, l_ordered = psi.nn_bond_type(gate.bond)
    f_ordered = psi.f_ordered(gate.bond)
    s0, s1 = gate.bond if l_ordered else gate.bond[::-1]

    G0, G1 = gate_fix_order(gate.G0, gate.G1, l_ordered, f_ordered)
    Q0, Q1, R0, R1, Q0f, Q1f = apply_gate_nn(psi[s0], psi[s1], G0, G1, dirn)

    fgf = env.bond_metric(Q0, Q1, s0, s1, dirn)

    # enforce hermiticity
    fgfH = fgf.H
    nonhermitian = (fgf - fgfH).norm() / 2
    fgf = (fgf + fgfH) / 2

    # check metric tensor eigenvalues
    S, U = fgf.eigh(axes=(0, 1))
    smin, smax = min(S._data), max(S._data)

    info['nonhermitian_part'] = nonhermitian / smax
    info['min_eigenvalue'] = smin / smax
    info['second_eigenvalue'] = max(*(x for x in S._data if x < smax), smin) / smax

    g_error = max(-smin, 0) + nonhermitian
    info['wrong_eigenvalues'] = sum(S._data < g_error).item() / len(S._data)
    if fix_metric is not None:
        S._data[S._data < g_error] = g_error * fix_metric

    fgf = U @ S @ U.H

    fRR = (R0 @ R1).fuse_legs(axes=[(0, 1)])
    fgRR = fgf @ fRR
    RRgRR = vdot(fRR, fgRR)

    pinv_cutoffs = sorted(pinv_cutoffs)
    M0, M1 = R0, R1
    for opts in [opts_svd] if isinstance(opts_svd, dict) else opts_svd:
        info['exit_code'] = 0
        M0i, M1i, error2_init = initial_truncation(M0, M1, fgf, fRR, RRgRR, initialization, opts, pinv_cutoffs)
        M0, M1, error2_iter = optimize_truncation(M0i, M1i, fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter, info)
        info['truncation_error_init'] = error2_init ** 0.5
        info['truncation_error_iter'] = error2_iter ** 0.5
        if error2_init * 1.0001 < error2_iter:
            M0, M1 = M0i, M1i
            info['exit_code'] += 2

    M0, M1 = symmetrized_svd(M0, M1, opts, normalize=True)
    psi[s0], psi[s1] = apply_bond_tensors(Q0f, Q1f, M0, M1, dirn)

    info['truncation_error'] = min(error2_init, error2_iter) ** 0.5
    return Evolution_out(**info)


def accumulated_truncation_error(infoss, mode='bonds', statistics='max', symmetrized=True):
    r"""
    Return accumulated truncation error calcuated from evolution output statistics.

    Gives some estimate of errors accumulated during time evolution.

    Parameters
    ----------
    infoss: Sequence[Sequence[Evolution_out]]
        list of outputs of :meth:`evolution_step_`.
    mode: str
        'gates' or 'bonds', whether to accumulate truncation error for each gate, or for each lattice bond.
    statistics: str
        'max' or 'mean', whether to take the maximal value of a mean over the lattice.
    symmetrized: bool
        Whether application of the gates has been symmetrized in :meth:`evolution_step_`.
    """
    tmpss = [[info.truncation_error for info in infos] for infos in infoss]
    Ng = len(tmpss[0])
    if symmetrized:
        tmpss = [[x + y for x, y in zip(tmps[:Ng//2], tmps[-1: -Ng//2 - 1: -1])] for tmps in tmpss]
    accum = [sum(col) for col in zip(*tmpss)]
    if mode=='bonds':
        bonds = [info.bond for info in infoss[0]]
        baccum = {}
        for b, v in zip(bonds, accum):
            baccum[b] = baccum.get(b, 0) + v
        accum = baccum.values()
    if statistics == 'max':
        return max(accum)
    elif statistics == 'mean':
        return sum(accum) / len(accum)
    raise YastnError(f"{statistics=} in accumulated_truncation_error not recognized; Should be 'max' or 'mean'.")


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
    return vdot(delta, fgf @ delta) / RRgRR


def initial_truncation(R0, R1, fgf, fRR, RRgRR, initialization, opts_svd, pinv_cutoffs):
    """
    Truncate R0 @ R1 to bond dimension specified in opts_svd,
    Uses plain SVD for initialization == 'SVD', and includes information from
    product approximation of bond metric when initialization == 'EAT'.
    """
    if initialization == 'EAT':
        G = fgf.unfuse_legs(axes=1)
        G = tensordot(G, R0, axes=(1, 0))
        G = tensordot(G, R1, axes=(1, 1))
        G = G.fuse_legs(axes=((0, (1, 2))))
        G = G.unfuse_legs(axes=0)
        G = tensordot(R1.conj(), G, axes=(1, 1))
        G = tensordot(R0.conj(), G, axes=(0, 1))
        G = G.unfuse_legs(axes=2)
        #
        # rank-1 approximation
        G0, _, G1 = svd_with_truncation(G, axes=((0, 2), (3, 1)), D_total=1)
        G0 = G0.remove_leg(axis=2)
        G1 = G1.remove_leg(axis=0)
        #
        # make sure it is hermitian
        G0 = G0 / G0.trace().to_number()
        G1 = G1 / G1.trace().to_number()
        G0 = (G0 + G0.H) / 2
        G1 = (G1 + G1.H) / 2
        #
        S0, U0 = G0.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs))
        S1, U1 = G1.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs))
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
                    M0, M1, error2 = M0_tmp, M1_tmp, error2_tmp
    elif initialization == 'SVD':
        M0, M1 = symmetrized_svd(R0, R1, opts_svd)
        error2 = calculate_truncation_error2(M0 @ M1, fgf, fRR, RRgRR)
    return M0, M1, error2


def optimize_truncation(M0, M1, fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter, info):
    r"""
    Optimizes the matrices M0 and M1 by minimizing
    truncation error using least square optimization.
    """
    gf = fgf.unfuse_legs(axes=0)
    gRR =  fgRR.unfuse_legs(axes=0)

    error2_old = 0
    for iter in range(1, max_iter+1):
        # fix M1 and optimize M0
        j0 = (gRR @ M1.H).fuse_legs(axes=[(0, 1)])
        g0 = tensordot(M1.conj(), gf, axes=(1, 1)).fuse_legs(axes=((1, 0), 2))
        g0 = g0.unfuse_legs(axes=1)
        g0 = tensordot(g0, M1, axes=(2, 1)).fuse_legs(axes=(0, (1, 2)))
        error_fun = lambda x: calculate_truncation_error2(x @ M1, fgf, fRR, RRgRR)
        M0, error2, pinv_cutoff = optimal_pinv(g0, j0, pinv_cutoffs, error_fun)

        # fix M0 and optimize M1
        j1 = (M0.H @ gRR).fuse_legs(axes=[(0, 1)])
        g1 = tensordot(M0.conj(), gf, axes=(0, 0)).fuse_legs(axes=((0, 1), 2))
        g1 = g1.unfuse_legs(axes=1)
        g1 = tensordot(g1, M0, axes=(1, 0)).fuse_legs(axes=(0, (2, 1)))
        error_fun = lambda x: calculate_truncation_error2(M0 @ x, fgf, fRR, RRgRR)
        M1, error2, pinv_cutoff = optimal_pinv(g1, j1, pinv_cutoffs, error_fun)

        # convergence condition
        if abs(error2 - error2_old) < tol_iter:
            break
        error2_old = error2

    info['pinv_cutoff'] = pinv_cutoff
    info['iterations'] = iter
    info['exit_code'] += (iter == max_iter)
    return M0, M1, error2


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
