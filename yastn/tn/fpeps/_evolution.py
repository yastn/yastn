""" Routines for time evolution with nn gates on a 2D lattice. """

from ... import tensordot, vdot, svd_with_truncation, truncation_mask, YastnError
from ._peps import Peps2Layers
from ._gates_auxiliary import apply_gate_onsite, apply_gate_nn, gate_fix_order, apply_bond_tensors
from typing import NamedTuple


class Evolution_out(NamedTuple):
    """ All errors and eigenvalues are relative. """
    bond: tuple = None
    truncation_error: float = 0
    truncation_errors: tuple[float] = ()
    iterations: tuple[int] = ()
    nonhermitian_part: float = 0
    min_eigenvalue: float = 0
    wrong_eigenvalues: float = 0
    EAT_error: float = 0
    pinv_cutoffs: tuple[float] = ()

    def __str__(self):
        txt = f"({self.bond}, error={self.truncation_error:0.2e}"
        txt += ", errors=(" + ", ".join(format(f, '.2e') for f in self.truncation_errors) + ")"
        txt += f", EAT_err={self.EAT_error:0.1e}"
        txt += f", nonh_err={self.nonhermitian_part:0.1e}, min_eig={self.min_eigenvalue:0.1e}, fixed={self.wrong_eigenvalues:0.1e}"
        txt += ", pinv_c=(" + ", ".join(format(f, '.0e') for f in self.pinv_cutoffs) + ")"
        txt += f", iters={self.iterations})"
        return txt


def evolution_step_(env, gates, opts_svd, symmetrize=True,
                    fix_metric=0,
                    pinv_cutoffs=(1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
                    max_iter=100, tol_iter=1e-15, safe_mode=False):
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
    fix_metric: int | None:
        Error_measure of the metric tensor is a sum of: the norm of its non-hermitian part,
        and absolute value of the most negative eigenvalue (if present).
        If fix_metric is not None, replace eigenvalues smaller than the error_measure
        by fix_metric * error_measure. Sensible values of fix_metric are 0 and 1. Default is 0.
    pinv_cutoffs : Sequence[float] | float
        List of pseudo-inverse cutoffs.
        The one that gives the smallest truncation error is used during iterative optimizations and EAT initialization.
    max_iter : int
        The maximal number of iterative steps for each truncation optimization.
    max_tol : int
        Tolerance of truncation_error to stop iterative optimization.
    safe_mode: bool
        True: Do both SVD initialization and EAT initialization and pick up the one with the smallest error; False: Do EAT initialization only.

    Returns
    -------
    Evolution_out(NamedTuple)
        Namedtuple containing fields:
            * :code:`bond` bond where the gate is applied
            * :code:`truncation_error` relative norm of the difference between untruncated and truncated bonds, calculated in metric specified by env. The best value.
            * :code:`truncation_errors` (error2_eat, error2_eat_iter, error2_svd, error2_svd_iter)
            * :code:`iterations` a number of iterations to iterative optimization convergence (eat_iterations, svd_iterations)
            * :code:`nonhermitian_part` norm of the non-hermitian part of the bond metric, normalized by its largest eigenvalue. Estimator of a metric error.
            * :code:`min_eigenvalue` a ratio of the smallest and the largest bond metric eigenvalue. Can be negative which gives an estimate of a metric error.
            * :code:`wrong_eigenvalues` a fraction of bond metrics eigenvalues that were below the error threshold; and were modified according to :code:`fix_metric` argument.
            * :code:`EAT_error` a ratio of the second largest and the largest bond metric eigenvalue.
            * :code:`pinv_cutoffs` Optimal pinv_cutoff in (eat, eat_iter, svd_iter).
    """
    infos = []

    psi = env.psi
    if isinstance(psi, Peps2Layers):
        psi = psi.ket  # to make it work with CtmEnv

    for gate in gates.local:
        psi[gate.site] = apply_gate_onsite(psi[gate.site], gate.G)
    for gate in gates.nn:
        info = apply_nn_truncate_optimize_(env, psi, gate, opts_svd, fix_metric, pinv_cutoffs, max_iter, tol_iter, safe_mode=safe_mode)
        infos.append(info)
    if symmetrize:
        for gate in gates.nn[::-1]:
            info = apply_nn_truncate_optimize_(env, psi, gate, opts_svd, fix_metric, pinv_cutoffs, max_iter, tol_iter, safe_mode=safe_mode)
            infos.append(info)
        for gate in gates.local[::-1]:
            psi[gate.site] = apply_gate_onsite(psi[gate.site], gate.G)
    return infos


def apply_nn_truncate_optimize_(env, psi, gate, opts_svd,
                    fix_metric=0,
                    pinv_cutoffs=(1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
                    max_iter=100, tol_iter=1e-15, safe_mode=False):
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

    g_error = max(-smin, 0) + nonhermitian
    info['wrong_eigenvalues'] = sum(S._data < g_error).item() / len(S._data)
    if fix_metric is not None:
        S._data[S._data < g_error] = g_error * fix_metric

    fgf = U @ S @ U.H

    fRR = (R0 @ R1).fuse_legs(axes=[(0, 1)])
    fgRR = fgf @ fRR
    RRgRR = abs(vdot(fRR, fgRR).item())

    pinv_cutoffs = sorted(pinv_cutoffs)
    M0, M1 = R0, R1
    for opts in [opts_svd] if isinstance(opts_svd, dict) else opts_svd:
        info['pinv_cutoffs'] = []
        info['iterations'] = []
        if safe_mode:
            M0_svd, M1_svd = symmetrized_svd(M0, M1, opts, normalize=False)
            error2_svd = calculate_truncation_error2(M0_svd @ M1_svd, fgf, fRR, RRgRR)
        M0_eat, M1_eat, error2_eat = initial_truncation_EAT(M0, M1, fgf, fRR, RRgRR, opts, pinv_cutoffs, info)
        M0_ite, M1_ite, error2_ite = optimize_truncation(M0_eat, M1_eat, error2_eat, fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter, info)
        if safe_mode:
            M0_its, M1_its, error2_its = optimize_truncation(M0_svd, M1_svd, error2_svd, fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter, info)
        else:
            M0_its = M0_ite
            M1_its = M1_ite
            M0_svd = M0_eat
            M1_svd = M1_eat
            error2_its = error2_ite
            error2_svd = error2_eat

        Ms = [(M0_eat, M1_eat), (M0_ite, M1_ite), (M0_svd, M1_svd), (M0_its, M1_its)]
        error2s = [(error2_eat, 0), (error2_ite, 1), (error2_svd, 2), (error2_its, 3)]
        error2, ind = min(error2s)

        M0, M1 = Ms[ind]

        info['truncation_errors'] = tuple(x ** 0.5 for x, _ in error2s)
        info['truncation_error'] = error2 ** 0.5
        info['pinv_cutoffs'] = tuple(info['pinv_cutoffs'])
        info['iterations'] = tuple(info['iterations'])

    M0, M1 = symmetrized_svd(M0, M1, opts, normalize=True)
    psi[s0], psi[s1] = apply_bond_tensors(Q0f, Q1f, M0, M1, dirn)

    return Evolution_out(**info)


def accumulated_truncation_error(infoss, statistics='mean'):
    r"""
    Return accumulated truncation error Delta calcuated from evolution output statistics.

    Delta = sum_steps{statistics(sum_bond truncation_error(bond, step))}
    Gives an estimate of errors accumulated during time evolution.

    Parameters
    ----------
    infoss: Sequence[Sequence[Evolution_out]]
        list of outputs of :meth:`evolution_step_`.
    statistics: str
        'max' or 'mean', whether to take the maximal value of a mean over the lattice.
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


def initial_truncation_EAT(R0, R1, fgf, fRR, RRgRR, opts_svd, pinv_cutoffs, info):
    """
    Truncate R0 @ R1 to bond dimension specified in opts_svd,
    includes information from a product approximation of bond metric.
    """
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
    G0, S, G1 = G.svd(axes=((0, 2), (3, 1)))

    maskS = truncation_mask(S, D_total=1)
    G0, S1, G1 = maskS.apply_mask(G0, S, G1, axes=(-1, 0, 0))

    nS = S.norm()
    info['EAT_error'] = (nS**2 - S1.norm()**2)**0.5 / nS

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
                M0, M1, error2, pinv_c = M0_tmp, M1_tmp, error2_tmp, c_off
    info['pinv_cutoffs'].append(pinv_c)
    return M0, M1, error2


def optimize_truncation(M0, M1, error2_old, fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter, info):
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

    info['pinv_cutoffs'].append(pinv_c)
    info['iterations'].append(iter)
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
