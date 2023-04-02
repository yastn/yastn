"""
Routines for tensor update (tu) timestep on a 2D lattice. 
tu supports fermions though application of swap-gates.
PEPS tensors have 5 legs: (top, left, bottom, right, system)
In case of purification, system leg is a fusion of (ancilla, system)
"""
import logging
import yastn
from yastn.tn.fpeps.operators.gates import trivial_tensor, match_ancilla_1s, match_ancilla_2s
from yastn import tensordot, vdot, svd_with_truncation, svd, qr, ncon, eigh_with_truncation
from ._ntu import env_NTU

def evol_machine(peps, gate, truncation_mode, step, env_type, opts_svd=None):

    r"""
    Applies a nearest-neighbor gate to a PEPS tensor and optimizes the resulting tensor using alternate
    least squares. 

    Parameters
    ----------
        peps            : class Peps

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
        peps : The optimized PEPS tensor.
        info : dict
             A dictionary with information about the optimization. Contains the following keys:
             - 'svd_error': The SVD truncation error.
             - 'tu_error': The truncation error after optimization.
             - 'optimal_cutoff': The optimal cutoff value used for inverse.
            
    """

    if opts_svd is None:
        opts_svd = {'D_total':10, 'tol_block':1e-15}  # D_total = 10 chosen arbitrarily

    QA, QB, RA, RB = apply_nn_gate(peps, gate)

    if step == "svd-update":
        MA, MB = truncation_step(RA, RB, opts_svd=opts_svd, normalize=True)
        peps[gate.bond.site_0], peps[gate.bond.site_1] = form_new_peps_tensors(QA, QB, MA, MB, gate.bond)
        info = {}
        return peps, info
    else:
        if env_type=='NTU':
            g = env_NTU(peps, gate.bond, QA, QB, dirn=gate.bond.dirn)
        info={}
        MA, MB, opt_error, optim, svd_error = truncate_and_optimize(g, RA, RB, truncation_mode, opts_svd=opts_svd)
        if step == 'two-step':  # else 'one-step'
            opts_svd_2 = {'D_total':int(opts_svd['D_total']*2), 'tol_block':opts_svd['tol_block']}
            MA_int, MB_int, _, _, _ = truncate_and_optimize(g, RA, RB, truncation_mode, opts_svd=opts_svd_2)
            MA_2, MB_2, opt_error_2, optim_2, svd_error_2 = truncate_and_optimize(g, MA_int, MB_int, truncation_mode, opts_svd=opts_svd)
            if opt_error < opt_error_2:
                logging.info("1-step update; truncation errors 1-and 2-step %0.5e,  %0.5e; svd error %0.5e,  %0.5e" % (opt_error, opt_error_2, svd_error, svd_error_2))
            else:
                MA, MB = MA_2, MB_2
                logging.info("2-step update; truncation errors 1-and 2-step %0.5e,  %0.5e; svd error %0.5e,  %0.5e " % (opt_error, opt_error_2, svd_error, svd_error_2))
                opt_error, optim, svd_error = opt_error_2, optim_2, svd_error_2
        peps[gate.bond.site_0], peps[gate.bond.site_1] = form_new_peps_tensors(QA, QB, MA, MB, gate.bond)
        if env_type == 'NTU':
            info.update({'ntu_error': opt_error, 'optimal_cutoff': optim, 'svd_error': svd_error})

        return peps, info

############################
##### gate application  ####
############################

def apply_local_gate_(peps, gate):
    """ apply local gates on PEPS tensors """
    A = match_ancilla_1s(gate.A, peps[gate.site]) 
    peps[gate.site] = tensordot(peps[gate.site], A, axes=(2, 1)) # [t l] [b r] [s a]
    return peps


def apply_nn_gate(peps, gate):

    """ apply nn gates on PEPS tensors. """
    A, B = peps[gate.bond.site_0], peps[gate.bond.site_1]  # A = [t l] [b r] s

    dirn = gate.bond.dirn
    if dirn == "h":  # Horizontal gate
        GA_an = match_ancilla_2s(gate.A, A, dir='l') 
        int_A = tensordot(A, GA_an, axes=(2, 1)) # [t l] [b r] s c
        int_A = int_A.fuse_legs(axes=((0, 2), 1, 3))  # [[t l] s] [b r] c
        int_A = int_A.unfuse_legs(axes=1)  # [[t l] s] b r c
        int_A = int_A.swap_gate(axes=(1, 3))  # b X c
        int_A = int_A.fuse_legs(axes=((0, 1), (2, 3)))  # [[[t l] s] b] [r c]
        QA, RA = qr(int_A, axes=(0, 1), sQ=-1)  # [[[t l] s] b] rr @ rr [r c]

        GB_an = match_ancilla_2s(gate.B, B, dir='r')
        int_B = tensordot(B, GB_an, axes=(2, 1)) # [t l] [b r] s c
        int_B = int_B.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] s] c
        int_B = int_B.unfuse_legs(axes=0)  # t l [[b r] s] c
        int_B = int_B.fuse_legs(axes=((0, 2), (1, 3)))  # [t [[b r] s]] [l c]
        QB, RB = qr(int_B, axes=(0, 1), sQ=1, Qaxis=0, Raxis=-1)  # ll [t [[b r] s]]  @  [l c] ll       

    elif dirn == "v":  # Vertical gate
            
        GA_an = match_ancilla_2s(gate.A, A, dir='l') 
        int_A = tensordot(A, GA_an, axes=(2, 1)) # [t l] [b r] s c
        int_A = int_A.fuse_legs(axes=((0, 2), 1, 3))  # [[t l] s] [b r] c
        int_A = int_A.unfuse_legs(axes=1)  # [[t l] s] b r c
        int_A = int_A.fuse_legs(axes=((0, 2), (1, 3)))  # [[[t l] s] r] [b c]
        QA, RA = qr(int_A, axes=(0, 1), sQ=1)  # [[[t l] s] r] bb  @  bb [b c]

        GB_an = match_ancilla_2s(gate.B, B, dir='r')
        int_B = tensordot(B, GB_an, axes=(2, 1)) # [t l] [b r] s c
        int_B = int_B.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b r] s] c
        int_B = int_B.unfuse_legs(axes=0)  # t l [[b r] s] c
        int_B = int_B.swap_gate(axes=(1, 3))  # l X c
        int_B = int_B.fuse_legs(axes=((1, 2), (0, 3)))  # [l [[b r] s]] [t c]
        QB, RB = qr(int_B, axes=(0, 1), sQ=-1, Qaxis=0, Raxis=-1)  # tt [l [[b r] s]]  @  [t c] tt

    return QA, QB, RA, RB


def truncation_step(RA, RB, opts_svd, normalize=False):
    """ svd truncation of central tensor """

    theta = RA @ RB
    if isinstance(opts_svd['D_total'], dict):
        opts_svd['D_total'] = sum(opts_svd['D_total'].values())
    UA, S, UB = svd_with_truncation(theta, sU=RA.get_signature()[1], **opts_svd)
    if normalize:
        S = S / S.norm(p='inf')
    sS = S.sqrt()
    MA = sS.broadcast(UA, axes=1)
    MB = sS.broadcast(UB, axes=0)
    return MA, MB


def form_new_peps_tensors(QA, QB, MA, MB, bond):
    """ combine unitaries in QA with optimized MA to form new peps tensors. """
    if bond.dirn == "h":
        A = QA @ MA  # [[[[t l] s] b] r
        A = A.unfuse_legs(axes=0)  # [[t l] s] b r
        A = A.fuse_legs(axes=(0, (1, 2)))  # [[t l] s] [b r]
        A = A.unfuse_legs(axes=0)  # [t l] s [b r]
        A = A.transpose(axes=(0, 2, 1))  # [t l] [b r] s

        B = MB @ QB  # l [t [[b r] s]]
        B = B.unfuse_legs(axes=1)  # l t [[b r] s]
        B = B.fuse_legs(axes=((1, 0), 2))  # [t l] [[b r] s]
        B = B.unfuse_legs(axes=1)  # [t l] [b r] s
    elif bond.dirn == "v":
        A = QA @ MA  # [[[t l] s] r] b
        A = A.unfuse_legs(axes=0)  # [[t l] s] r b
        A = A.fuse_legs(axes=(0, (2, 1)))  # [[t l] s] [b r]
        A = A.unfuse_legs(axes=0)  # [t l] s [b r]
        A = A.transpose(axes=(0, 2, 1))  # [t l] [b r] s

        B = MB @ QB  # t [l [[b r] s]]
        B = B.unfuse_legs(axes=1)  # t l [[b r] s]
        B = B.fuse_legs(axes=((0, 1), 2))  # [t l] [[b r] s]
        B = B.unfuse_legs(axes=1)  # [t l] [b r] s
    return A, B


def environment_aided_truncation_step(g, gRR, fgf, fgRAB, RA, RB, truncation_mode, opts_svd):

    """ 
    truncation_mode = 'optimal' is for implementing EAT algorithm only applicable for symmetric
    tensors and offers no advantage for dense tensors.

    Returns
    -------
    MA, MB: truncated pair of tensors before alternate least square optimization
    svd_error: here just implies the error incurred for the initial truncation 
               before the optimization
    """
    
    if truncation_mode == 'optimal':
        G = ncon((g, RA, RB, RA, RB), ([1, 2, 3, 4], [1, -1], [-3, 3], [2, -2], [-4, 4]), conjs=(0, 0, 0, 1, 1))
        [ul, _, vr] = svd_with_truncation(G, axes=((0, 1), (2, 3)), tol_block=1e-15, D_total=1)
        ul = ul.remove_leg(axis=2)
        vr = vr.remove_leg(axis=0)
        GL, GR = ul.transpose(axes=(1, 0)), vr
        _, SL, UL = svd(GL)
        UR, SR, _ = svd(GR)
        XL, XR = SL.sqrt() @ UL, UR @ SR.sqrt()
        XRRX = XL @ XR
        U, L, V = svd_with_truncation(XRRX, sU=RA.get_signature()[1], **opts_svd)
        mA, mB = U @ L.sqrt(), L.sqrt() @ V
        MA, MB, svd_error, _ = optimal_initial_pinv(mA, mB, RA, RB, gRR, SL, UL, SR, UR, fgf, fgRAB)
        return MA, MB, svd_error

    elif truncation_mode == 'normal':

        MA, MB = truncation_step(RA, RB, opts_svd)
        MAB = MA @ MB
        MAB = MAB.fuse_legs(axes=[(0, 1)])
        gMM = vdot(MAB, fgf @ MAB).item()
        gMR = vdot(MAB, fgRAB).item()
        svd_error = abs((gMM + gRR - gMR - gMR.conjugate()) / gRR)
         
    return MA, MB, svd_error


def optimal_initial_pinv(mA, mB, RA, RB, gRR, SL, UL, SR, UR, fgf, fgRAB):

    """ function for choosing the optimal initial cutoff for the inverse which gives the least svd_error """

    cutoff_list = [10**n for n in range(-14, -5)]
    results = []
    for c_off in cutoff_list:
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


def tu_single_optimization(MA, MB, gRAB, gf, gRR, svd_error, max_iter):

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
        tu_errorA, optimal_cf, MA = optimal_pinv(g_A, j_A, gRR)

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
        tu_errorB, optimal_cf, MB = optimal_pinv(g_B, j_B, gRR)

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


def truncate_and_optimize(g, RA, RB, truncation_mode, opts_svd):

    """ 
    First we truncate RA and RB tensors based on the input truncation_mode with 
    function environment_aided_truncation_step. Then the truncated
    MA and MB tensors are subjected to least square optimization to minimize
     the truncation error with the function tu_single_optimization.
    """
    max_iter = 1000 # max no of tu optimization loops
    assert (g.fuse_legs(axes=((0, 2), (1, 3))) - g.fuse_legs(axes=((0, 2), (1, 3))).conj().transpose(axes=(1, 0))).norm() < 1e-14 * g.fuse_legs(axes=((0, 2), (1, 3))).norm()
    
    RAB = RA @ RB
    RAB = RAB.fuse_legs(axes=[(0, 1)])
    gf = g.fuse_legs(axes=(1, 3, (0, 2)))
    fgf = gf.fuse_legs(axes=((0, 1), 2))
    fgRAB = fgf @ RAB
    gRAB =  gf @ RAB
    gRR = vdot(RAB, fgRAB).item()
    
    MA, MB, svd_error = environment_aided_truncation_step(g, gRR, fgf, fgRAB, RA, RB, truncation_mode, opts_svd)
    MA, MB, tu_errorB, optimal_cf  = tu_single_optimization(MA, MB, gRAB, gf, gRR, svd_error, max_iter)
    MA, MB = truncation_step(MA, MB, opts_svd, normalize=True)
    return MA, MB, tu_errorB, optimal_cf, svd_error

def optimal_pinv(gg, J, gRR):
    """ solve pinv(gg) * J, optimizing pseudoinverse cutoff for tu error. """
    
    assert (gg - gg.conj().transpose(axes=(1, 0))).norm() < 1e-12 * gg.norm()
    S, U = eigh_with_truncation(gg, axes=(0, 1), tol=1e-14)
    UdJ = tensordot(J, U, axes=(0, 0), conj=(0, 1))
    cutoff_list = [10**n for n in range(-14, -5)]
    results = []
    for c_off in cutoff_list:
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
