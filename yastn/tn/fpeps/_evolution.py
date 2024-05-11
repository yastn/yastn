"""
Routines for tensor update (tu) timestep on a 2D lattice.
tu supports fermions though application of swap-gates.
PEPS tensors have 5 legs: (top, left, bottom, right, system)
In case of purification, system leg is a fusion of (ancilla, system)
"""
from ... import tensordot, vdot, svd_with_truncation, svd, ncon, eigh_with_truncation
from ._peps import Peps2Layers
from ._gates_auxiliary import apply_gate_onsite, apply_gate_nn, gate_fix_order, apply_bond_tensors
from typing import NamedTuple


class Evolution_out(NamedTuple):
    """ All errors and eigenvalues are relative. """
    bond: tuple = None
    truncation_error: float = 0
    nonhermitian_error: float = 0
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
    performing bond-dimension truncation after each application of a two-site gate.

    Parameters
    ----------
    env: EnvNTU | ...
        Environment class containing PEPS state, which is updated in place,
        and a method to calculate bond metric tensors employed during truncation.
    gates: Gates
        The gates that will be applied to PEPS.
    symmetrize : bool
        Whether to iterate through provided gates forward and then backward, resulting in a 2nd order method.
        In that case, each gate should correspond to half of desired timestep.
    opts_svd : dict | Sequence[dict]
        Options passed to :meth:`yastn.linalg.svd_with_truncation` which are used
        to initially truncate the bond dimension, before it is further optimized iteratively.
        In particular, it fixes bond dimensions (potentially, sectorial).
        It is possible to prvide a list of dicts (with decreasing bond dimensions),
        in which case the truncation is done step by step.
    initialization : str
        "SVD" or "EAT". Type of procedure initializing the optimization of truncated PEPS tensors
        after application of two-site gate. Employ plain SVD, or SVD that includes
        a product approximation of bond metric tensor. The default is "EAT".
    pinv_cutoffs : Sequence[float] | float
        List of pseudo-inverse cutoffs. During iterative optimization, use the one that gives the smallest error.
    max_iter : int
        A maximal number of iterative steps for each truncation optimization.

    Returns
    -------
    Evolution_out(NamedTuple)
        Namedtuple containing fields:
            * :code:`truncation_error` for all applied gates, calculated according to metric specified by env.
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
    info = {'bond': gate.bond}
    dirn, l_ordered = psi.nn_bond_type(gate.bond)
    f_ordered = psi.f_ordered(gate.bond)
    s0, s1 = gate.bond if l_ordered else gate.bond[::-1]

    G0, G1 = gate_fix_order(gate.G0, gate.G1, l_ordered, f_ordered)
    Q0, Q1, R0, R1, Q0f, Q1f = apply_gate_nn(psi[s0], psi[s1], G0, G1, dirn)

    fgf = env.bond_metric(Q0, Q1, s0, s1, dirn)

    # enforce hermiticity
    nonh = (fgf - fgf.H).norm()
    fgf = (fgf + fgf.H) / 2

    # check metric tensor eigenvalues
    S, U = fgf.eigh(axes=(0, 1))
    smin, smax = min(S._data), max(S._data)

    info['nonhermitian_error'] = nonh / smax
    info['min_eigenvalue'] = smin / smax
    info['second_eigenvalue'] = max(*(x for x in S._data if x < smax), smin) / smax

    g_error = max(-smin / smax, 0 * smax) + nonh / smax

    if fix is not None:
        S._data[S._data < g_error] = g_error * fix
        info['fixed_eigenvalues'] = sum(S._data < g_error).item() / len(S._data)
    else:
        info['fixed_eigenvalues'] = 0.

    fgf = U @ S @ U.H

    R01 = R0 @ R1
    R01 = R01.fuse_legs(axes=[(0, 1)])
    gf = fgf.unfuse_legs(axes=0)
    fgR01 = fgf @ R01
    gR01 =  gf @ R01
    gRR = vdot(R01, fgR01).item()

    M0, M1 = R0, R1
    for opts in [opts_svd] if isinstance(opts_svd, dict) else opts_svd:
        M0, M1, svd_error2 = initial_truncation(gRR, fgf, fgR01, M0, M1, initialization, opts, pinv_cutoffs)
        M0, M1, truncation_error2, pinv_cutoff, iters = optimize_truncation(M0, M1, gR01, gf, gRR, svd_error2, pinv_cutoffs, max_iter, tol_iter)
        M0, M1 = symmetrize_truncation(R0, R1, opts, normalize=True)

    psi[s0], psi[s1] = apply_bond_tensors(Q0f, Q1f, M0, M1, dirn)

    info['truncation_error'] = truncation_error2 ** 0.5
    info['pinv_cutoff'] = pinv_cutoff
    info['iterations'] = iters

    return Evolution_out(**info)


def symmetrize_truncation(RA, RB, opts_svd, normalize=False):
    """ svd truncation of central tensor """

    U, S, V = svd_with_truncation(RA @ RB, sU=RA.s[1], **opts_svd)
    if normalize:
        S = S / S.norm(p='inf')
    S = S.sqrt()
    MA, MB = S.broadcast(U, V, axes=(1, 0))
    return MA, MB


def initial_truncation(gRR, fgf, fgRAB, RA, RB, initialization, opts_svd, pinv_cutoffs):
    """
    truncation_mode = 'optimal' is for implementing EAT algorithm only applicable for symmetric
    tensors and offers no advantage for dense tensors.

    Returns
    -------
        MA, MB: truncated pair of tensors before alternate least square optimization
        svd_error2: here just implies the error2 incurred for the initial truncation
               before the optimization
    """

    if initialization == 'EAT':
        g = fgf.unfuse_legs(axes=(0, 1))
        G = ncon((g, RA, RB, RA, RB), ([1, 2, 3, 4], [3, -1], [-3, 4], [1, -2], [-4, 2]), conjs=(0, 0, 0, 1, 1))
        [ul, _, vr] = svd_with_truncation(G, axes=((0, 1), (2, 3)), D_total=1)
        ul = ul.remove_leg(axis=2)
        vr = vr.remove_leg(axis=0)
        GL, GR = ul.transpose(axes=(1, 0)), vr
        _, SL, UL = svd(GL)
        UR, SR, _ = svd(GR)
        XL, XR = SL.sqrt() @ UL, UR @ SR.sqrt()
        XRRX = XL @ XR
        U, L, V = svd_with_truncation(XRRX, sU=RA.get_signature()[1], **opts_svd)
        mA, mB = U @ L.sqrt(), L.sqrt() @ V
        MA, MB, svd_error2, _ = optimal_initial_pinv(mA, mB, RA, RB, gRR, SL, UL, SR, UR, fgf, fgRAB, pinv_cutoffs)
        return MA, MB, svd_error2
    elif initialization == 'SVD':
        MA, MB = symmetrize_truncation(RA, RB, opts_svd)
        MAB = MA @ MB
        MAB = MAB.fuse_legs(axes=[(0, 1)])
        gMM = vdot(MAB, fgf @ MAB).item()
        gMR = vdot(MAB, fgRAB).item()
        svd_error2 = abs((gMM + gRR - gMR - gMR.conjugate()) / gRR)

    return MA, MB, svd_error2


def optimize_truncation(MA, MB, gRAB, gf, gRR, svd_error2, pinv_cutoffs, max_iter, tol_iter):

    """ Optimizes the matrices MA and MB by minimizing the truncation error using least square optimization """

    MA_guess, MB_guess = MA, MB
    truncation_error2_old_A, truncation_error2_old_B = 0, 0
    epsilon1, epsilon2 = 0, 0

    for iter in range(max_iter):
        # fix MB and optimize MA
        j_A = tensordot(gRAB, MB, axes=(1, 1), conj=(0, 1)).fuse_legs(axes=[(0, 1)])
        g_A = tensordot(MB, gf, axes=(1, 1), conj=(1, 0)).fuse_legs(axes=((1, 0), 2))
        g_A = g_A.unfuse_legs(axes=1)
        g_A = tensordot(g_A, MB, axes=(2, 1)).fuse_legs(axes=(0, (1, 2)))
        truncation_error2, optimal_pinv_cutoff, MA = optimal_pinv(g_A, j_A, gRR, pinv_cutoffs)

        epsilon1 = abs(truncation_error2_old_A - truncation_error2)
        truncation_error2_old_A = truncation_error2
        count1, count2 = 0, 0
        if truncation_error2 >= svd_error2:
            MA = MA_guess
            epsilon1, truncation_error2_old_A = 0, 0
            count1 += 1

         # fix MA and optimize MB
        j_B = tensordot(MA, gRAB, axes=(0, 0), conj=(1, 0)).fuse_legs(axes=[(0, 1)])
        g_B = tensordot(MA, gf, axes=(0, 0), conj=(1, 0)).fuse_legs(axes=((0, 1), 2))
        g_B = g_B.unfuse_legs(axes=1)
        g_B = tensordot(g_B, MA, axes=(1, 0)).fuse_legs(axes=(0, (2, 1)))
        truncation_error2, optimal_pinv_cutoff, MB = optimal_pinv(g_B, j_B, gRR, pinv_cutoffs)

        epsilon2 = abs(truncation_error2_old_B - truncation_error2)
        truncation_error2_old_B = truncation_error2

        if truncation_error2 >= svd_error2:
            MB = MB_guess
            epsilon2, truncation_error2_old_B = 0, 0
            count2 += 1

        count = count1 + count2
        if count > 0:
            break

        epsilon = max(epsilon1, epsilon2)
        if epsilon < tol_iter:  # convergence condition
            break

    return MA, MB, truncation_error2, optimal_pinv_cutoff, iter + 1


def optimal_initial_pinv(mA, mB, RA, RB, gRR, SL, UL, SR, UR, fgf, fgRAB, pinv_cutoffs):

    """ function for choosing the optimal initial cutoff for the inverse which gives the least svd_error2 """

    results = []
    for c_off in pinv_cutoffs:
        XL_inv, XR_inv = tensordot(UL.conj(), SL.sqrt().reciprocal(cutoff=c_off), axes=(0, 0)), tensordot(SR.sqrt().reciprocal(cutoff=c_off), UR.conj(), axes=(1, 1))
        pA, pB = XL_inv @ mA, mB @ XR_inv
        MA, MB = RA @ pA, pB @ RB
        MAB = MA @ MB
        MAB = MAB.fuse_legs(axes=[(0, 1)])
        gMM = vdot(MAB, fgf @ MAB).item()
        gMR = vdot(MAB, fgRAB).item()
        svd_error2 = abs((gMM + gRR - gMR - gMR.conjugate()) / gRR)
        results.append((svd_error2, c_off, MA, MB))

    svd_error2, c_off, MA, MB = min(results, key=lambda x: x[0])

    return MA, MB, svd_error2, c_off


def optimal_pinv(gg, J, gRR, pinv_cutoffs):
    """ solve pinv(gg) * J, optimizing pseudoinverse cutoff for tu error2. """

    assert (gg - gg.conj().transpose(axes=(1, 0))).norm() < 1e-12 * gg.norm()
    S, U = eigh_with_truncation(gg, axes=(0, 1), tol=1e-14)
    UdJ = tensordot(J, U, axes=(0, 0), conj=(0, 1))

    results = []
    for c_off in pinv_cutoffs:
        Sd = S.reciprocal(cutoff=c_off)
        SdUdJ = Sd.broadcast(UdJ, axes=0)
        # Mnew = tensordot(SdUdJ, V, axes=(0, 0), conj=(0, 1))
        Mnew = U @ SdUdJ

        # calculation of errors with respect to metric
        met_newA = vdot(Mnew, gg @ Mnew).item()
        met_mixedA = vdot(Mnew, J).item()
        error2 = abs((met_newA + gRR - met_mixedA - met_mixedA.conjugate()) / gRR)
        results.append((error2, c_off, Mnew))

    truncation_error2, optimal_pinv_cutoff, Mnew = min(results, key=lambda x: x[0])

    Mnew = Mnew.unfuse_legs(axes=0)
    return truncation_error2, optimal_pinv_cutoff, Mnew
