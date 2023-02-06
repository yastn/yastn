"""
Routines for NTU timestep on a checkerboard lattice.
NTU supports fermions though application of swap-gates.
PEPS tensors have 5 legs: (top, left, bottom, right, system)
In case of purification, system leg is a fusion of (ancila, system)
"""
import logging
import yast
from yast.tn.peps.operators.gates import trivial_tensor, match_ancilla_1s, match_ancilla_2s
from yast import tensordot, vdot, svd_with_truncation, svd, qr, swap_gate, fuse_legs, ncon, eigh_with_truncation, eye
from ._ntu import env_NTU

def ntu_machine(gamma, gate, Ds, truncation_mode, step, env_type):
    # step can be svd-step, one-step or two-step
    # application of nearest neighbor gate and subsequent optimization of peps tensor using NTU
    QA, QB, RA, RB = apply_nn_gate(gamma, gate)

    if step == "svd-update":
        MA, MB = truncation_step(RA, RB, Ds, normalize=True)
        gamma[gate.bond.site_0], gamma[gate.bond.site_1] = form_new_peps_tensors(QA, QB, MA, MB, gate.bond)
        info = {}
        return gamma, info
    else:
        if env_type=='NTU':
            g = env_NTU(gamma, gate.bond, QA, QB, dirn=gate.bond.dirn)
        info={}
        MA, MB, opt_error, optim, svd_error = truncate_and_optimize(g, RA, RB, Ds, truncation_mode)
        if step == 'two-step':  # else 'one-step'
            MA_int, MB_int, _, _, _ = truncate_and_optimize(g, RA, RB, int(2*Ds), truncation_mode)
            MA_2, MB_2, opt_error_2, optim_2, svd_error_2 = truncate_and_optimize(g, MA_int, MB_int, Ds, truncation_mode)
            if opt_error < opt_error_2:
                logging.info("1-step update; truncation errors 1-and 2-step %0.5e,  %0.5e; svd error %0.5e,  %0.5e" % (opt_error, opt_error_2, svd_error, svd_error_2))
            else:
                MA, MB = MA_2, MB_2
                logging.info("2-step update; truncation errors 1-and 2-step %0.5e,  %0.5e; svd error %0.5e,  %0.5e " % (opt_error, opt_error_2, svd_error, svd_error_2))
                opt_error, optim, svd_error = opt_error_2, optim_2, svd_error_2
        gamma[gate.bond.site_0], gamma[gate.bond.site_1] = form_new_peps_tensors(QA, QB, MA, MB, gate.bond)
        info.update({'ntu_error': opt_error, 'optimal_cutoff': optim, 'svd_error': svd_error})

        return gamma, info

############################
##### gate application  ####
############################

def apply_local_gate_(gamma, gate):
    """ apply local gates on PEPS tensors """
    A = match_ancilla_1s(gate.A, gamma[gate.site])
    gamma[gate.site] = tensordot(gamma[gate.site], A, axes=(2, 1)) # [t l] [b r] [s a]
    return gamma



def apply_nn_gate(gamma, gate):

    """ apply nn gates on PEPS tensors. """
    A, B = gamma[gate.bond.site_0], gamma[gate.bond.site_1]  # A = [t l] [b r] s

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


def truncation_step(RA, RB, Ds, normalize=False):
    """ svd truncation of central tensor """
    theta = RA @ RB
    if isinstance(Ds, dict):
        Ds = sum(Ds.values())
    UA, S, UB = svd_with_truncation(theta, sU=RA.get_signature()[1], tol_block=1e-15, D_total=Ds)
    if normalize:
        S = S / S.norm(p='inf')
    sS = S.sqrt()
    MA = sS.broadcast(UA, axis=1)
    MB = sS.broadcast(UB, axis=0)
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

def forced_sectorial_truncation(U, L, V, Ds):
    # truncates bond dimensions of different symmetry sectors according to agiven distribution Ds
    F = eye(config=U.config, legs=U.get_legs(1).conj())
    discard_block_weight= {}
    for k in L.get_leg_structure(axis=0).keys():
        if k in Ds.keys():
            v = Ds.get(k)
            discard_block_weight[k] = sum(L.A[k+k][v:]**2)
        elif k not in Ds.keys():
            discard_block_weight[k] = sum(L.A[k+k]**2)
    for k, v in Ds.items():
        if k in F.get_leg_structure(axis=0).keys():
            F.A[k+k][v:] = 0
    for k, v in F.get_leg_structure(axis=0).items():
        if k not in Ds.keys(): 
            F.A[k+k][0:v] = 0
    U = U.mask(F, axis=1)
    new_L = L.mask(F, axis=0)
    V = V.mask(F, axis=0)
    return U, new_L, V, discard_block_weight


def environment_aided_truncation_step(g, gRR, fgf, fgRAB, RA, RB, Ds, truncation_mode):

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
        
        if isinstance(Ds, dict):
            Dn = sum(Ds.values())
        else:
            Dn = Ds
        U, L, V = svd_with_truncation(XRRX, sU=RA.get_signature()[1], D_total=Dn, tol_block=1e-15)
        mA, mB = U @ L.sqrt(), L.sqrt() @ V
        MA, MB, svd_error, _ = optimal_initial_pinv(mA, mB, RA, RB, gRR, SL, UL, SR, UR, fgf, fgRAB)
        return MA, MB, svd_error

    elif truncation_mode == 'normal':

        MA, MB = truncation_step(RA, RB, Ds)
        MAB = MA @ MB
        MAB = MAB.fuse_legs(axes=[(0, 1)])
        gMM = vdot(MAB, fgf @ MAB).item()
        gMR = vdot(MAB, fgRAB).item()
        svd_error = abs((gMM + gRR - gMR - gMR.conjugate()) / gRR)
         
    return MA, MB, svd_error


def optimal_initial_pinv(mA, mB, RA, RB, gRR, SL, UL, SR, UR, fgf, fgRAB):

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


def ntu_single_optimization(MA, MB, gRAB, gf, gRR, svd_error, max_iter):
    MA_guess, MB_guess = MA, MB
    ntu_errorA_old, ntu_errorB_old = 0, 0
    epsilon1, epsilon2 = 0, 0

    for _ in range(max_iter):
        # fix MB and optimize MA
        j_A = tensordot(gRAB, MB, axes=(1, 1), conj=(0, 1)).fuse_legs(axes=[(0, 1)])
        g_A = tensordot(MB, gf, axes=(1, 1), conj=(1, 0)).fuse_legs(axes=((1, 0), 2))
        g_A = g_A.unfuse_legs(axes=1)
        g_A = tensordot(g_A, MB, axes=(2, 1)).fuse_legs(axes=(0, (1, 2)))
        ntu_errorA, optimal_cf, MA = optimal_pinv(g_A, j_A, gRR)

        epsilon1 = abs(ntu_errorA_old - ntu_errorA)
        ntu_errorA_old = ntu_errorA
        count1, count2 = 0, 0
        if abs(ntu_errorA) >= abs(svd_error):
            MA = MA_guess
            epsilon1, ntu_errorA_old = 0, 0
            count1 += 1

         # fix MA and optimize MB
        j_B = tensordot(MA, gRAB, axes=(0, 0), conj=(1, 0)).fuse_legs(axes=[(0, 1)])
        g_B = tensordot(MA, gf, axes=(0, 0), conj=(1, 0)).fuse_legs(axes=((0, 1), 2))
        g_B = g_B.unfuse_legs(axes=1)
        g_B = tensordot(g_B, MA, axes=(1, 0)).fuse_legs(axes=(0, (2, 1)))
        ntu_errorB, optimal_cf, MB = optimal_pinv(g_B, j_B, gRR)

        epsilon2 = abs(ntu_errorB_old - ntu_errorB)
        ntu_errorB_old = ntu_errorB

        if abs(ntu_errorB) >= abs(svd_error):
            MB = MB_guess
            epsilon2, ntu_errorB_old = 0, 0
            count2 += 1

        count = count1 + count2
        if count > 0:
            break

        epsilon = max(epsilon1, epsilon2)
        if epsilon < 1e-14: ### convergence condition
            break

    return MA, MB, ntu_errorB, optimal_cf

###############################################################################
########### given the environment and the seeds, optimization #################
########################## of MA and MB #######################################
###############################################################################

def truncate_and_optimize(g, RA, RB, Ds, truncation_mode):
    """ optimize truncated MA and MB tensors, using NTU metric. """
    max_iter = 1000 # max no of NTU optimization loops
    assert (g.fuse_legs(axes=((0, 2), (1, 3))) - g.fuse_legs(axes=((0, 2), (1, 3))).conj().transpose(axes=(1, 0))).norm() < 1e-14 * g.fuse_legs(axes=((0, 2), (1, 3))).norm()
    
    RAB = RA @ RB
    RAB = RAB.fuse_legs(axes=[(0, 1)])
    gf = g.fuse_legs(axes=(1, 3, (0, 2)))
    fgf = gf.fuse_legs(axes=((0, 1), 2))
    fgRAB = fgf @ RAB
    gRAB =  gf @ RAB
    gRR = vdot(RAB, fgRAB).item()
    
    MA, MB, svd_error = environment_aided_truncation_step(g, gRR, fgf, fgRAB, RA, RB, Ds, truncation_mode)
    MA, MB, ntu_errorB, optimal_cf  = ntu_single_optimization(MA, MB, gRAB, gf, gRR, svd_error, max_iter)
    MA, MB = truncation_step(MA, MB, Ds, normalize=True)
    return MA, MB, ntu_errorB, optimal_cf, svd_error

def optimal_pinv(gg, J, gRR):
    """ solve pinv(gg) * J, optimizing pseudoinverse cutoff for NTU error. """
    
    assert (gg - gg.conj().transpose(axes=(1, 0))).norm() < 1e-12 * gg.norm()
    S, U = eigh_with_truncation(gg, axes=(0, 1), tol=1e-14)
    UdJ = tensordot(J, U, axes=(0, 0), conj=(0, 1))
    cutoff_list = [10**n for n in range(-14, -5)]
    results = []
    for c_off in cutoff_list:
        Sd = S.reciprocal(cutoff=c_off)
        SdUdJ = Sd.broadcast(UdJ, axis=0)
        # Mnew = tensordot(SdUdJ, V, axes=(0, 0), conj=(0, 1))
        Mnew = U @ SdUdJ

        # calculation of errors with respect to metric
        met_newA = vdot(Mnew, gg @ Mnew).item()
        met_mixedA = vdot(Mnew, J).item()
        ntu_error = abs((met_newA + gRR - met_mixedA - met_mixedA.conjugate()) / gRR)
        results.append((ntu_error, c_off, Mnew))

    ntu_error, c_off, Mnew = min(results, key=lambda x: x[0])

    Mnew = Mnew.unfuse_legs(axes=0)
    return ntu_error, c_off, Mnew
