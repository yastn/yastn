"""
Routines for tensor update (tu) timestep on a 2D lattice.
tu supports fermions though application of swap-gates.
PEPS tensors have 5 legs: (top, left, bottom, right, system)
In case of purification, system leg is a fusion of (ancilla, system)
"""
from ... import tensordot, vdot, svd_with_truncation, svd, ncon
from ._peps import Peps2Layers
from ._gates_auxiliary import apply_gate_onsite, apply_gate_nn, gate_fix_order, apply_bond_tensors
from typing import NamedTuple


class Evolution_out(NamedTuple):
    """ All errors and eigenvalues are relative. """
    bond: tuple = None
    truncation_error: float = 0
    nonhermitian_part: float = 0
    min_eigenvalue: float = 0
    second_eigenvalue: float = 0
    fixed_eigenvalues: float = 0
    iterations: int = 0
    pinv_cutoff: float = 0
    exit_code: int = 0


def evolution_step_(env, gates, opts_svd, symmetrize=True,
                    initialization="EAT", fix=0,
                    pinv_cutoffs=(1e-12, 1e-10, 1e-8, 1e-6),
                    max_iter=100, tol_iter=1e-13):
    r"""
    Perform a single step of PEPS evolution by applying a list of gates,
    truncate bond-dimension after each application of a two-site gate.

    Parameters
    ----------
    env: EnvNTU | ...
        Environment class containing PEPS state (updated in place),
        and a method to calculate bond metric tensors employed during truncation.
    gates: Gates
        The gates to be applied to PEPS.
    opts_svd: dict | Sequence[dict]
        Options passed to :meth:`yastn.linalg.svd_with_truncation` which are used
        to initially truncate the bond dimension, before it is further optimized iteratively.
        In particular, it fixes bond dimensions (potentially, sectorial).
        It is possible to provide a list of dicts (with decreasing bond dimensions),
        in which case the truncation is done gradually in a few steps.
    symmetrize: bool
        Whether to iterate through provided gates forward and then backward, resulting in a 2nd order method.
        In that case, each gate should correspond to half of desired timestep.
    initialization: str
        "SVD" or "EAT". Type of procedure initializing the optimization of truncated PEPS tensors
        after application of two-site gate. Employ plain SVD, or SVD that includes
        a product approximation of bond metric tensor. By default uses "EAT".
    fix: int | None:
        Error_measure of the metric tensor is a sum of the norm of its non-hermitian part
        and absolute value of the most negative eigenvalue (if present).
        If fix is not None, replace eigenvalues smaller than error_measure by fix * error_measure.
        The sensible values of fix are 0 and 1.
    pinv_cutoffs : Sequence[float] | float
        List of pseudo-inverse cutoffs. During iterative optimization, the one that gives the smallest error is used.
    max_iter : int
        A maximal number of iterative steps for each truncation optimization.
    max_tol : int
        Tolerance of truncation_error to stop iterative optimization.

    Returns
    -------
    Evolution_out(NamedTuple)
        Namedtuple containing fields:
            * :code:`bond`
            * :code:`truncation_error` relative norm of the difference between untruncated and truncated bond, calculated according to metric specified by env.
            * :code:`nonhermitian_part`
            * :code:`min_eigenvalue`
            * :code:`second_eigenvalue`
            * :code:`fixed_eigenvalues`
            * :code:`iterations`
            * :code:`pinv_cutoff`
            * :code:`exit_code`
    """
    infos = []

    psi = env.psi
    if isinstance(psi, Peps2Layers):
        psi = psi.ket  # to make it work with CtmEnv

    for gate in gates.local:
        psi[gate.site] = apply_gate_onsite(psi[gate.site], gate.G)
    for gate in gates.nn:
        info = apply_nn_truncate_optimize_(env, psi, gate, opts_svd, initialization, fix, pinv_cutoffs, max_iter, tol_iter)
        infos.append(info)
    if symmetrize:
        for gate in gates.nn[::-1]:
            info = apply_nn_truncate_optimize_(env, psi, gate, opts_svd, initialization, fix, pinv_cutoffs, max_iter, tol_iter)
            infos.append(info)
        for gate in gates.local[::-1]:
            psi[gate.site] = apply_gate_onsite(psi[gate.site], gate.G)
    return Evolution_out(*zip(*infos))


def apply_nn_truncate_optimize_(env, psi, gate, opts_svd,
                    initialization="EAT", fix=0,
                    pinv_cutoffs=(1e-12, 1e-10, 1e-8, 1e-6),
                    max_iter=1000, tol_iter=1e-13):
    r"""
    Applies a nearest-neighbor gate to a PEPS tensor, truncate, and
    optimize the resulting tensors using alternate least squares.
    """
    pinv_cutoffs = sorted(pinv_cutoffs)
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

    g_error = max(-smin, 0 * smin) + nonhermitian

    info['fixed_eigenvalues'] = 0.
    if fix is not None:
        info['fixed_eigenvalues'] = sum(S._data < g_error).item() / len(S._data)
        S._data[S._data < g_error] = g_error * fix

    fgf = U @ S @ U.H

    fRR = (R0 @ R1).fuse_legs(axes=[(0, 1)])
    fgRR = fgf @ fRR
    RRgRR = vdot(fRR, fgRR)

    M0, M1 = R0, R1
    for opts in [opts_svd] if isinstance(opts_svd, dict) else opts_svd:
        M0i, M1i, i_error2 = initial_truncation(M0, M1, fgf, fRR, RRgRR, initialization, opts, pinv_cutoffs)
        M0, M1, error2, pinv_cutoff, iters = optimize_truncation(M0i, M1i, fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
        if i_error2 < error2:
            M0, M1 = M0i, M1i
            error2 = i_error2

    M0, M1 = symmetrize_truncation(M0, M1, opts, normalize=True)
    psi[s0], psi[s1] = apply_bond_tensors(Q0f, Q1f, M0, M1, dirn)

    info['truncation_error'] = error2 ** 0.5
    info['pinv_cutoff'] = pinv_cutoff
    info['iterations'] = iters
    return Evolution_out(**info)


def symmetrize_truncation(RA, RB, opts_svd, normalize=False):
    """ svd truncation of central tensor; divide singular values symmetrically between tensors. """
    U, S, V = svd_with_truncation(RA @ RB, sU=RA.s[1], **opts_svd)
    if normalize:
        S = S / S.norm(p='inf')
    S = S.sqrt()
    MA, MB = S.broadcast(U, V, axes=(1, 0))
    return MA, MB


def calculate_truncation_error2(fMM, fgf, fRR, RRgRR):
    """ Calculate squared truncation error. """
    if fMM.ndim == 2:  # if legs of MM not fused into vector
        fMM = fMM.fuse_legs(axes=[(0, 1)])
    delta = fRR - fMM
    return vdot(delta, fgf @ delta) / RRgRR


def initial_truncation(R0, R1, fgf, fRR, RRgRR, initialization, opts_svd, pinv_cutoffs):
    """
    Truncate R0 @ R1 to bond dimension specified in opts_svd, wither using SVD,
    or including information from a product approximation the bond metric
    """

    if initialization == 'EAT':
        g = fgf.unfuse_legs(axes=(0, 1))
        G = ncon((g, R0, R1, R0.conj(), R1.conj()), ((1, 2, 3, 4), (3, -0), (-2, 4), (1, -1), (-3, 2)))
        [ul, _, vr] = svd_with_truncation(G, axes=((0, 1), (2, 3)), D_total=1)
        ul = ul.remove_leg(axis=2)
        vr = vr.remove_leg(axis=0)
        GL, GR = ul.transpose(axes=(1, 0)), vr
        _, SL, UL = svd(GL)
        UR, SR, _ = svd(GR)
        XL, XR = SL.sqrt() @ UL, UR @ SR.sqrt()
        XRRX = XL @ XR
        U, L, V = svd_with_truncation(XRRX, sU=R0.get_signature()[1], **opts_svd)
        mA, mB = U @ L.sqrt(), L.sqrt() @ V

        results = []
        for c_off in pinv_cutoffs:
            XL_inv = tensordot(UL.conj(), SL.sqrt().reciprocal(cutoff=c_off), axes=(0, 0))
            XR_inv = tensordot(SR.sqrt().reciprocal(cutoff=c_off), UR.conj(), axes=(1, 1))
            pA, pB = XL_inv @ mA, mB @ XR_inv
            M0, M1 = R0 @ pA, pB @ R1
            error2 = calculate_truncation_error2(M0 @ M1, fgf, fRR, RRgRR)
            results.append((error2, c_off, M0, M1))
        error2, c_off, M0, M1 = min(results, key=lambda x: x[0])

    elif initialization == 'SVD':
        M0, M1 = symmetrize_truncation(R0, R1, opts_svd)
        error2 = calculate_truncation_error2(M0 @ M1, fgf, fRR, RRgRR)

    return M0, M1, error2


def optimize_truncation(M0, M1, fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter):
    """
    Optimizes the matrices M0 and M1 by minimizing the truncation error using least square optimization.
    """
    gf = fgf.unfuse_legs(axes=0)
    gRR =  fgRR.unfuse_legs(axes=0)

    error2_old = 0
    for iter in range(max_iter):
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

    return M0, M1, error2, pinv_cutoff, iter + 1


def optimal_pinv(g, j, pinv_cutoffs, error_fun):
    """
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
