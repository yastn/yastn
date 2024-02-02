"""
Routines for tensor update (tu) timestep on a 2D lattice.
tu supports fermions though application of swap-gates.
PEPS tensors have 5 legs: (top, left, bottom, right, system)
In case of purification, system leg is a fusion of (ancilla, system)
"""
from yastn.tn.fpeps.gates._gates import match_ancilla_1s, match_ancilla_2s
from yastn import tensordot, vdot, svd_with_truncation, svd, qr, ncon, eigh_with_truncation
from typing import NamedTuple


class Gate_nn(NamedTuple):
    """ A should be before B in the fermionic order. """
    A : tuple = None
    B : tuple = None
    bond : tuple = None


class Gate_local(NamedTuple):
    A : tuple = None
    site : tuple = None


class Gates(NamedTuple):
    local : list = None   # list of Gate_local
    nn : list = None   # list of Gate_nn


class Evolution_out(NamedTuple):
    truncation_errors : list = None
    optimal_cutoffs : list = None


def evolution_step_(env, gates, symmetrize=True,
                    opts_svd=None, initialization="EAT",
                    pinv_cutoffs=(1e-12, 1e-10, 1e-8, 1e-6, 1e-4), max_iter=1000):
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
    opts_svd : dict
        Options passed to :meth:`yastn.linalg.svd_with_truncation` which are used
        to initially truncate the bond dimension, before it is further optimized iteratively.
        In particular, it fixes bond dimensions (potentially, sectorial).
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
    if opts_svd is None:
        opts_svd = {'D_total': 4, 'tol_block': 1e-15}
    infos = []

    for gate in gates.local:
        apply_local_gate_(env, gate)
    for gate in gates.nn:
        infos.append(apply_nn_gate_and_truncate_(env, gate, opts_svd, initialization, pinv_cutoffs, max_iter))
    if symmetrize:
        for gate in gates.nn[::-1]:
            infos.append(apply_nn_gate_and_truncate_(env, gate, opts_svd, initialization, pinv_cutoffs, max_iter))
        for gate in gates.local[::-1]:
            apply_local_gate_(env, gate)

    trunction_errors = [record['truncation'] for record in infos]
    optimal_cutoffs = [record['optimal_cutoff'] for record in infos]

    return Evolution_out(trunction_errors, optimal_cutoffs)



def apply_nn_gate_and_truncate_(env, gate, opts_svd=None, initialization="EAT",
                   pinv_cutoffs=(1e-12, 1e-10, 1e-8, 1e-6, 1e-4), max_iter=1000):
    r"""
    Applies a nearest-neighbor gate to a PEPS tensor and optimizes the resulting tensor using alternate
    least squares.

    Parameters
    ----------
    env             : class ClusterEnv
    gate            : A Gate object representing the nearest-neighbor gate to apply.
    truncation_mode : str
                    The mode to use for truncation of the environment tensors. Can be
                    'normal' or 'optimal'.
    step            : str
                    The optimization step to perform. Can be 'svd-update', 'one-step', or 'two-step'.
    env_type        : str
                    The type of environment to use for optimization. Can be 'NTU' (neighborhood tensor update),
                        'FU'(full update - to be added).
    opts_svd        : dict, optional
                    A dictionary with options for the SVD truncation. Default is None.

    Returns
    -------
        peps : The optimized PEPS tensor (yastn.fpeps.Lattice).
        info : dict
             A dictionary with information about the optimization. Contains the following keys:
             - 'svd_error': The SVD truncation error.
             - 'tu_error': The truncation error after optimization.
             - 'optimal_cutoff': The optimal cutoff value used for inverse.

    """

    if opts_svd is None:
        opts_svd = {'D_total':4, 'tol_block':1e-15}

    QA, QB, RA, RB, QAf, QBf = apply_nn_gate(env.psi, gate)

    info = {}
    # if env.which != 'NN': # svd-upddate
    #     MA, MB = truncation_step(RA, RB, opts_svd=opts_svd, normalize=True)
    #     env.psi[gate.bond.site0], env.psi[gate.bond.site1] = form_new_peps_tensors(QAf, QBf, MA, MB, gate.bond)
    #     info['truncation'] = 0
    #     return info

    fgf = env.bond_metric(gate.bond, QA, QB)
    MA, MB, opt_error, optim, svd_error = truncate_and_optimize(fgf, RA, RB, initialization, opts_svd, pinv_cutoffs, max_iter)
    env.psi[gate.bond.site0], env.psi[gate.bond.site1] = form_new_peps_tensors(QAf, QBf, MA, MB, gate.bond)
    info.update({'truncation': opt_error, 'optimal_cutoff': optim})
    return info


def truncate_and_optimize(fgf, RA, RB, initialization, opts_svd, pinv_cutoffs, max_iter):
    """
    First we truncate RA and RB tensors based on the input truncation_mode with
    function environment_aided_truncation_step. Then the truncated
    MA and MB tensors are subjected to least square optimization to minimize
     the truncation error with the function tu_single_optimization.
    """
    RAB = RA @ RB
    RAB = RAB.fuse_legs(axes=[(0, 1)])
    gf = fgf.unfuse_legs(axes=0)
    fgRAB = fgf @ RAB
    gRAB =  gf @ RAB
    gRR = vdot(RAB, fgRAB).item()

    MA, MB, svd_error = environment_aided_truncation_step(gRR, fgf, fgRAB, RA, RB, initialization, opts_svd, pinv_cutoffs)
    MA, MB, tu_errorB, optimal_cf  = tu_single_optimization(MA, MB, gRAB, gf, gRR, svd_error, pinv_cutoffs, max_iter)
    MA, MB = truncation_step(MA, MB, opts_svd, normalize=True)
    return MA, MB, tu_errorB, optimal_cf, svd_error


def truncation_step(RA, RB, opts_svd, normalize=False):
    """ svd truncation of central tensor """

    U, S, V = svd_with_truncation(RA @ RB, sU=RA.s[1], **opts_svd)
    if normalize:
        S = S / S.norm(p='inf')
    S = S.sqrt()
    MA, MB = S.broadcast(U, V, axes=(1, 0))
    return MA, MB


def environment_aided_truncation_step(gRR, fgf, fgRAB, RA, RB, initialization, opts_svd, pinv_cutoffs):

    """
    truncation_mode = 'optimal' is for implementing EAT algorithm only applicable for symmetric
    tensors and offers no advantage for dense tensors.

    Returns
    -------
        MA, MB: truncated pair of tensors before alternate least square optimization
        svd_error: here just implies the error incurred for the initial truncation
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
        MA, MB, svd_error, _ = optimal_initial_pinv(mA, mB, RA, RB, gRR, SL, UL, SR, UR, fgf, fgRAB, pinv_cutoffs)
        return MA, MB, svd_error
    elif initialization == 'SVD':
        MA, MB = truncation_step(RA, RB, opts_svd)
        MAB = MA @ MB
        MAB = MAB.fuse_legs(axes=[(0, 1)])
        gMM = vdot(MAB, fgf @ MAB).item()
        gMR = vdot(MAB, fgRAB).item()
        svd_error = abs((gMM + gRR - gMR - gMR.conjugate()) / gRR)

    return MA, MB, svd_error


def tu_single_optimization(MA, MB, gRAB, gf, gRR, svd_error, pinv_cutoffs, max_iter):

    """ Optimizes the matrices MA and MB by minimizing the truncation error using least square optimization """

    MA_guess, MB_guess = MA, MB
    tu_errorA_old, tu_errorB_old = 0, 0
    epsilon1, epsilon2 = 0, 0

    for _ in range(max_iter):
        # fix MB and optimize MA
        j_A = tensordot(gRAB, MB, axes=(1, 1), conj=(0, 1)).fuse_legs(axes=[(0, 1)])
        g_A = tensordot(MB, gf, axes=(1, 1), conj=(1, 0)).fuse_legs(axes=((1, 0), 2))
        g_A = g_A.unfuse_legs(axes=1)
        g_A = tensordot(g_A, MB, axes=(2, 1)).fuse_legs(axes=(0, (1, 2)))
        tu_errorA, optimal_cf, MA = optimal_pinv(g_A, j_A, gRR, pinv_cutoffs)

        epsilon1 = abs(tu_errorA_old - tu_errorA)
        tu_errorA_old = tu_errorA
        count1, count2 = 0, 0
        if abs(tu_errorA) >= abs(svd_error):
            MA = MA_guess
            epsilon1, tu_errorA_old = 0, 0
            count1 += 1

         # fix MA and optimize MB
        j_B = tensordot(MA, gRAB, axes=(0, 0), conj=(1, 0)).fuse_legs(axes=[(0, 1)])
        g_B = tensordot(MA, gf, axes=(0, 0), conj=(1, 0)).fuse_legs(axes=((0, 1), 2))
        g_B = g_B.unfuse_legs(axes=1)
        g_B = tensordot(g_B, MA, axes=(1, 0)).fuse_legs(axes=(0, (2, 1)))
        tu_errorB, optimal_cf, MB = optimal_pinv(g_B, j_B, gRR, pinv_cutoffs)

        epsilon2 = abs(tu_errorB_old - tu_errorB)
        tu_errorB_old = tu_errorB

        if abs(tu_errorB) >= abs(svd_error):
            MB = MB_guess
            epsilon2, tu_errorB_old = 0, 0
            count2 += 1

        count = count1 + count2
        if count > 0:
            break

        epsilon = max(epsilon1, epsilon2)
        if epsilon < 1e-14: ### convergence condition
            break

    return MA, MB, tu_errorB, optimal_cf


def optimal_initial_pinv(mA, mB, RA, RB, gRR, SL, UL, SR, UR, fgf, fgRAB, pinv_cutoffs):

    """ function for choosing the optimal initial cutoff for the inverse which gives the least svd_error """

    results = []
    for c_off in pinv_cutoffs:
        XL_inv, XR_inv = tensordot(UL.conj(), SL.sqrt().reciprocal(cutoff=c_off), axes=(0, 0)), tensordot(SR.sqrt().reciprocal(cutoff=c_off), UR.conj(), axes=(1, 1))
        pA, pB = XL_inv @ mA, mB @ XR_inv
        MA, MB = RA @ pA, pB @ RB
        MAB = MA @ MB
        MAB = MAB.fuse_legs(axes=[(0, 1)])
        gMM = vdot(MAB, fgf @ MAB).item()
        gMR = vdot(MAB, fgRAB).item()
        svd_error = abs((gMM + gRR - gMR - gMR.conjugate()) / gRR)
        results.append((svd_error, c_off, MA, MB))
    svd_error, c_off, MA, MB = min(results, key=lambda x: x[0])

    return MA, MB, svd_error, c_off


def optimal_pinv(gg, J, gRR, pinv_cutoffs):
    """ solve pinv(gg) * J, optimizing pseudoinverse cutoff for tu error. """

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
        tu_error = abs((met_newA + gRR - met_mixedA - met_mixedA.conjugate()) / gRR)
        results.append((tu_error, c_off, Mnew))

    tu_error, c_off, Mnew = min(results, key=lambda x: x[0])

    Mnew = Mnew.unfuse_legs(axes=0)
    return tu_error, c_off, Mnew


def apply_local_gate_(env, gate):
    """ apply local gates on PEPS tensors """
    A = match_ancilla_1s(gate.A, env.psi[gate.site])
    env.psi[gate.site] = tensordot(env.psi[gate.site], A, axes=(2, 1)) # [t l] [b r] [s a]


def apply_nn_gate(psi, gate):
    """ Apply nearest neighbor gate to PEPS tensors. """
    bd = gate.bond
    dirn = bd.dirn
    A = psi[bd.site0]  # [t l] [b r] sa
    B = psi[bd.site1]  # [t l] [b r] sa

    if dirn == "h":  # Horizontal gate
        GA_an = match_ancilla_2s(gate.A, A, dir='l')
        int_A = tensordot(A, GA_an, axes=(2, 1)) # [t l] [b r] sa c
        int_A = int_A.fuse_legs(axes=((0, 2), 1, 3))  # [[t l] sa] [b r] c
        int_A = int_A.unfuse_legs(axes=1)  # [[t l] sa] b r c
        int_A = int_A.swap_gate(axes=(1, 3))  # b X c
        int_A = int_A.fuse_legs(axes=((0, 1), (2, 3)))  # [[[t l] sa] b] [r c]
        QAf, RA = qr(int_A, axes=(0, 1), sQ=-1)  # [[[t l] sa] b] rr @ rr [r c]
        QA = QAf.unfuse_legs(axes=0)  # [[t l] sa] b rr
        QA = QA.fuse_legs(axes=(0, (1, 2)))  # [[t l] sa] [b rr]
        QA = QA.unfuse_legs(axes=0)  # [t l] sa [b rr]
        QA = QA.transpose(axes=(0, 2, 1))  # [t l] [b rr] sa

        GB_an = match_ancilla_2s(gate.B, B, dir='r')
        int_B = tensordot(B, GB_an, axes=(2, 1)) # [t l] [b r] sa c
        int_B = int_B.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] sa] c
        int_B = int_B.unfuse_legs(axes=0)  # t l [[b r] sa] c
        int_B = int_B.fuse_legs(axes=((0, 2), (1, 3)))  # [t [[b r] sa]] [l c]
        QBf, RB = qr(int_B, axes=(0, 1), sQ=1, Qaxis=0, Raxis=-1)  # ll [t [[b r] sa]]  @  [l c] ll
        QB = QBf.unfuse_legs(axes=1)  # ll t [[b r] sa]
        QB = QB.fuse_legs(axes=((1, 0), 2))  # [t ll] [[b r] sa]
        QB = QB.unfuse_legs(axes=1)  # [t ll] [b r] sa

    elif dirn == "v":  # Vertical gate
        GA_an = match_ancilla_2s(gate.A, A, dir='l')
        int_A = tensordot(A, GA_an, axes=(2, 1)) # [t l] [b r] sa c
        int_A = int_A.fuse_legs(axes=((0, 2), 1, 3))  # [[t l] sa] [b r] c
        int_A = int_A.unfuse_legs(axes=1)  # [[t l] sa] b r c
        int_A = int_A.fuse_legs(axes=((0, 2), (1, 3)))  # [[[t l] sa] r] [b c]
        QAf, RA = qr(int_A, axes=(0, 1), sQ=1)  # [[[t l] sa] r] bb  @  bb [b c]
        QA = QAf.unfuse_legs(axes=0)  # [[t l] sa] r bb
        QA = QA.fuse_legs(axes=(0, (2, 1)))  # [[t l] sa] [bb r]
        QA = QA.unfuse_legs(axes=0)  # [t l] sa [bb r]
        QA = QA.transpose(axes=(0, 2, 1))  # [t l] [bb r] sa

        GB_an = match_ancilla_2s(gate.B, B, dir='r')
        int_B = tensordot(B, GB_an, axes=(2, 1)) # [t l] [b r] sa c
        int_B = int_B.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] sa] c
        int_B = int_B.unfuse_legs(axes=0)  # t l [[b r] sa] c
        int_B = int_B.swap_gate(axes=(1, 3))  # l X c
        int_B = int_B.fuse_legs(axes=((1, 2), (0, 3)))  # [l [[b r] sa]] [t c]
        QBf, RB = qr(int_B, axes=(0, 1), sQ=-1, Qaxis=0, Raxis=-1)  # tt [l [[b r] sa]]  @  [t c] tt
        QB = QBf.unfuse_legs(axes=1)  # t l [[b r] sa]
        QB = QB.fuse_legs(axes=((0, 1), 2))  # [t l] [[b r] sa]
        QB = QB.unfuse_legs(axes=1)  # [t l] [b r] sa

    return QA, QB, RA, RB, QAf, QBf


def form_new_peps_tensors(QAf, QBf, MA, MB, bond):
    """ combine unitaries in QA with optimized MA to form new peps tensors. """
    if bond.dirn == "h":
        A = QAf @ MA  # [[[[t l] sa] b] r
        A = A.unfuse_legs(axes=0)  # [[t l] sa] b r
        A = A.fuse_legs(axes=(0, (1, 2)))  # [[t l] sa] [b r]
        A = A.unfuse_legs(axes=0)  # [t l] sa [b r]
        A = A.transpose(axes=(0, 2, 1))  # [t l] [b r] sa

        B = MB @ QBf  # l [t [[b r] s]]
        B = B.unfuse_legs(axes=1)  # l t [[b r] sa]
        B = B.fuse_legs(axes=((1, 0), 2))  # [t l] [[b r] sa]
        B = B.unfuse_legs(axes=1)  # [t l] [b r] sa
    elif bond.dirn == "v":
        A = QAf @ MA  # [[[t l] sa] r] b
        A = A.unfuse_legs(axes=0)  # [[t l] sa] r b
        A = A.fuse_legs(axes=(0, (2, 1)))  # [[t l] sa] [b r]
        A = A.unfuse_legs(axes=0)  # [t l] sa [b r]
        A = A.transpose(axes=(0, 2, 1))  # [t l] [b r] sa

        B = MB @ QBf  # t [l [[b r] sa]]
        B = B.unfuse_legs(axes=1)  # t l [[b r] sa]
        B = B.fuse_legs(axes=((0, 1), 2))  # [t l] [[b r] sa]
        B = B.unfuse_legs(axes=1)  # [t l] [b r] sa
    return A, B


def gates_homogeneous(peps, nn, local):
    """
    Generate a list of gates that is homogeneous over the lattice.

    Parameters
    ----------
    peps      : class Lattice
    nn : list
              A list of two-tuples, each containing the tensors that form a two-site
              nearest-neighbor gate.
    local : A two-tuple containing the tensors that form the single-site gate.

    Returns
    -------
    Gates: The generated gates. The NamedTuple 'Gates` named tuple contains a list of
      local and nn gates along with info where they should be applied.
    """
    gates_nn = []   # nn = [(GA, GB), (GA, GB)]   [(GA, GB, GA, GB)]
    for bd in peps.bonds():
        for i in range(len(nn)):
            gates_nn.append(Gate_nn(A=nn[i][0], B=nn[i][1], bond=bd))
    gates_loc = []
    for site in peps.sites():
        gates_loc.append(Gate_local(A=local, site=site))
    return Gates(local=gates_loc, nn=gates_nn)
