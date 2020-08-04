import numpy as np
import yamps.mps as mps
import yamps.ops.settings_Z2 as settings_Z2
import yamps.ops.settings_U1 as settings_U1
import yamps.tensor as tensor
from yamps.tensor.ncon import ncon
import transport_vectorization_general as general


# STATE
def thermal_state(LSR, io, ww, temp, basis, dtype='float64'):
    r"""
    Generate vectorization of a thermal state according Fermi-Dirac distribution.
    Output is of MPS form with nr_aux=0 and nr_phys=4.

    Parameters
    ----------
    NL: int
        Number of states in a lead with occupancies acc. to Fermi-Dirac distribution.
    io: list of size NS
        List of occupancies in the impurity.
    temp: list
        Temperature on sites, zero for the impurity
    ww: list
        List of energies in a chain. Impurity energies are ignored - io used instead.
    """
    N = len(LSR)
    im = 0
    psi = mps.Mps(N, nr_phys=1)
    psi.normalize = False
    for n in range(N):  # empty tensors
        if abs(LSR[n]) == 1:
            wk = ww[n]
            tp = temp[n]
            occ = 1./(np.exp(wk/tp)+1.) if tp > 1e-6 else (1.-np.sign(wk))*.5
        else:
            occ = io[im]
            im += 1

        if basis == 0:  # choose I, Z, X, Y basis
            psi.A[n] = tensor.Tensor(
                settings=settings_Z2, s=(1, 1, -1), dtype=dtype, n=0)
            psi.A[n].set_block(ts=(0, 0, 0), val=np.array(
                [1., 1.-2.*occ]), Ds=(1, 2, 1))
        elif basis == 1:  # choose cp c, c cp, c, cp
            psi.A[n] = tensor.Tensor(
                settings=settings_U1, s=(1, 1, -1), dtype=dtype, n=0)
            psi.A[n].set_block(ts=(0, 0, 0), val=np.array(
                [occ, 1.-occ]), Ds=(1, 2, 1))
    return psi


# Operator
def Lindbladian_1AIM_mixed(NL, LSR, wk, temp, vk, dV, gamma, basis, AdagA=False, dtype='float64'):

    oo, ii, q_z, z_q, _, _, _, _, c_q_cp, cp_q_c, z_q_z, c_q, cp_q, q_c, q_cp, ccp_q__p__q_ccp, n_q__p__q_n, m1j_n_q__m__q_n = general.generate_operator_basis(
        basis)

    N = 2 * NL + 1  # total number of sites
    n1 = np.argwhere(LSR == 2)[0, 0]  # impurity position

    if basis == 0:
        II = tensor.Tensor(settings=settings_Z2, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        II.set_block(ts=(0, 0, 0, 0), val=ii[:2, :2], Ds=(1, 2, 2, 1))
        II.set_block(ts=(0, 1, 1, 0), val=ii[2:, 2:], Ds=(1, 2, 2, 1))

        OO = tensor.Tensor(settings=settings_Z2, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        OO.set_block(ts=(0, 0, 0, 0), val=oo[:2, :2], Ds=(1, 2, 2, 1))
        OO.set_block(ts=(0, 1, 1, 0), val=oo[2:, 2:], Ds=(1, 2, 2, 1))

        N_Q__P__Q_N = tensor.Tensor(
            settings=settings_Z2, s=(1, 1, -1, -1), dtype=dtype, n=0)
        N_Q__P__Q_N.set_block(
            ts=(0, 0, 0, 0), val=n_q__p__q_n[:2, :2], Ds=(1, 2, 2, 1))
        N_Q__P__Q_N.set_block(
            ts=(0, 1, 1, 0), val=n_q__p__q_n[2:, 2:], Ds=(1, 2, 2, 1))

        CCP_Q__P__Q_CCP = tensor.Tensor(
            settings=settings_Z2, s=(1, 1, -1, -1), dtype=dtype, n=0)
        CCP_Q__P__Q_CCP.set_block(
            ts=(0, 0, 0, 0), val=ccp_q__p__q_ccp[:2, :2], Ds=(1, 2, 2, 1))
        CCP_Q__P__Q_CCP.set_block(
            ts=(0, 1, 1, 0), val=ccp_q__p__q_ccp[2:, 2:], Ds=(1, 2, 2, 1))

        m1j_N_Q__M__Q_N = tensor.Tensor(
            settings=settings_Z2, s=(1, 1, -1, -1), dtype=dtype, n=0)
        m1j_N_Q__M__Q_N.set_block(
            ts=(0, 0, 0, 0), val=m1j_n_q__m__q_n[:2, :2], Ds=(1, 2, 2, 1))
        m1j_N_Q__M__Q_N.set_block(
            ts=(0, 1, 1, 0), val=m1j_n_q__m__q_n[2:, 2:], Ds=(1, 2, 2, 1))

        Z_Q_Z = tensor.Tensor(settings=settings_Z2, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        Z_Q_Z.set_block(ts=(0, 0, 0, 0), val=z_q_z[:2, :2], Ds=(1, 2, 2, 1))
        Z_Q_Z.set_block(ts=(0, 1, 1, 0), val=z_q_z[2:, 2:], Ds=(1, 2, 2, 1))

        C_Q_CP = tensor.Tensor(settings=settings_Z2, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        C_Q_CP.set_block(ts=(0, 0, 0, 0), val=c_q_cp[:2, :2], Ds=(1, 2, 2, 1))

        CP_Q_C = tensor.Tensor(settings=settings_Z2, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        CP_Q_C.set_block(ts=(0, 0, 0, 0), val=c_q_cp[:2, :2], Ds=(1, 2, 2, 1))
        #
        Q_Z = tensor.Tensor(settings=settings_Z2, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        Q_Z.set_block(ts=(1, 0, 0, 1), val=q_z[:2, :2], Ds=(1, 2, 2, 1))
        Q_Z.set_block(ts=(1, 1, 1, 1), val=q_z[2:, 2:], Ds=(1, 2, 2, 1))

        Z_Q = tensor.Tensor(settings=settings_Z2, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        Z_Q.set_block(ts=(1, 0, 0, 1), val=z_q[:2, :2], Ds=(1, 2, 2, 1))
        Z_Q.set_block(ts=(1, 1, 1, 1), val=z_q[2:, 2:], Ds=(1, 2, 2, 1))
        #
        Q_CP_right = tensor.Tensor(
            settings=settings_Z2, s=(1, 1, -1, -1), dtype=dtype, n=0)
        Q_CP_right.set_block(
            ts=(0, 0, 1, 1), val=q_cp[:2, 2:], Ds=(1, 2, 2, 1))
        Q_CP_right.set_block(
            ts=(0, 1, 0, 1), val=q_cp[2:, :2], Ds=(1, 2, 2, 1))

        CP_Q_right = tensor.Tensor(
            settings=settings_Z2, s=(1, 1, -1, -1), dtype=dtype, n=0)
        CP_Q_right.set_block(
            ts=(0, 0, 1, 1), val=cp_q[:2, 2:], Ds=(1, 2, 2, 1))
        CP_Q_right.set_block(
            ts=(0, 1, 0, 1), val=cp_q[2:, :2], Ds=(1, 2, 2, 1))

        Q_C_right = tensor.Tensor(
            settings=settings_Z2, s=(1, 1, -1, -1), dtype=dtype, n=0)
        Q_C_right.set_block(ts=(0, 0, 1, 1), val=q_c[:2, 2:], Ds=(1, 2, 2, 1))
        Q_C_right.set_block(ts=(0, 1, 0, 1), val=q_c[2:, :2], Ds=(1, 2, 2, 1))

        C_Q_right = tensor.Tensor(
            settings=settings_Z2, s=(1, 1, -1, -1), dtype=dtype, n=0)
        C_Q_right.set_block(ts=(0, 0, 1, 1), val=c_q[:2, 2:], Ds=(1, 2, 2, 1))
        C_Q_right.set_block(ts=(0, 1, 0, 1), val=c_q[2:, :2], Ds=(1, 2, 2, 1))
        #
        Q_CP_left = tensor.Tensor(
            settings=settings_Z2, s=(1, 1, -1, -1), dtype=dtype, n=0)
        Q_CP_left.set_block(ts=(1, 0, 1, 0), val=q_cp[:2, 2:], Ds=(1, 2, 2, 1))
        Q_CP_left.set_block(ts=(1, 1, 0, 0), val=q_cp[2:, :2], Ds=(1, 2, 2, 1))

        CP_Q_left = tensor.Tensor(
            settings=settings_Z2, s=(1, 1, -1, -1), dtype=dtype, n=0)
        CP_Q_left.set_block(ts=(1, 0, 1, 0), val=cp_q[:2, 2:], Ds=(1, 2, 2, 1))
        CP_Q_left.set_block(ts=(1, 1, 0, 0), val=cp_q[2:, :2], Ds=(1, 2, 2, 1))

        Q_C_left = tensor.Tensor(settings=settings_Z2,
                                 s=(1, 1, -1, -1), dtype=dtype, n=0)
        Q_C_left.set_block(ts=(1, 0, 1, 0), val=q_c[:2, 2:], Ds=(1, 2, 2, 1))
        Q_C_left.set_block(ts=(1, 1, 0, 0), val=q_c[2:, :2], Ds=(1, 2, 2, 1))

        C_Q_left = tensor.Tensor(settings=settings_Z2,
                                 s=(1, 1, -1, -1), dtype=dtype, n=0)
        C_Q_left.set_block(ts=(1, 0, 1, 0), val=c_q[:2, 2:], Ds=(1, 2, 2, 1))
        C_Q_left.set_block(ts=(1, 1, 0, 0), val=c_q[2:, :2], Ds=(1, 2, 2, 1))

    elif basis == 1:
        # set 0,1,2 for charges -1,0,1.
        ch = (range(2, 3), range(0, 2), range(3, 4))
        Ds = (1, 2, 1)
        ts = (-1, 0, 1)
        #
        virt_ch = (0, 0)  # virtual charges left/right

        op = ii         # operator in full version
        II = tensor.Tensor(settings=settings_U1, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (-1, -1)]  # physical charges ket, bra
        II.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                     val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (0, 0)]  # physical charges ket, bra
        II.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                     val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 1)]  # physical charges ket, bra
        II.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                     val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        op = n_q__p__q_n         # operator in full version
        N_Q__P__Q_N = tensor.Tensor(
            settings=settings_U1, s=(1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (-1, -1)]  # physical charges ket, bra
        N_Q__P__Q_N.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                              val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (0, 0)]  # physical charges ket, bra
        N_Q__P__Q_N.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                              val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 1)]  # physical charges ket, bra
        N_Q__P__Q_N.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                              val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        op = m1j_n_q__m__q_n         # operator in full version
        m1j_N_Q__M__Q_N = tensor.Tensor(
            settings=settings_U1, s=(1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (-1, -1)]  # physical charges ket, bra
        m1j_N_Q__M__Q_N.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                                  val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 1)]  # physical charges ket, bra
        m1j_N_Q__M__Q_N.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                                  val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        op = ccp_q__p__q_ccp         # operator in full version
        CCP_Q__P__Q_CCP = tensor.Tensor(
            settings=settings_U1, s=(1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (-1, -1)]  # physical charges ket, bra
        CCP_Q__P__Q_CCP.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                                  val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (0, 0)]  # physical charges ket, bra
        m1j_N_Q__M__Q_N.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                                  val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 1)]  # physical charges ket, bra
        m1j_N_Q__M__Q_N.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                                  val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        # ? check ones again
        virt_ch = (0, 0)  # virtual charges left/right
        op = z_q_z         # operator in full version
        Z_Q_Z = tensor.Tensor(settings=settings_U1, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (-1, -1)]  # physical charges ket, bra
        Z_Q_Z.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                        val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (0, 0)]  # physical charges ket, bra
        m1j_N_Q__M__Q_N.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                                  val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 1)]  # physical charges ket, bra
        m1j_N_Q__M__Q_N.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                                  val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        virt_ch = (0, 0)  # virtual charges left/right
        op = c_q_cp         # operator in full version
        C_Q_CP = tensor.Tensor(settings=settings_U1, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (0, 0)]  # physical charges ket, bra
        C_Q_CP.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                         val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        virt_ch = (0, 0)  # virtual charges left/right
        op = cp_q_c         # operator in full version
        CP_Q_C = tensor.Tensor(settings=settings_U1, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (0, 0)]  # physical charges ket, bra
        CP_Q_C.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                         val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))
        #
        virt_ch = (1, 1)  # virtual charges left/right

        op = z_q         # operator in full version
        Z_Q = tensor.Tensor(settings=settings_U1, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (-1, -1)]  # physical charges ket, bra
        Z_Q.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                      val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (0, 0)]  # physical charges ket, bra
        Z_Q.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                      val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 1)]  # physical charges ket, bra
        Z_Q.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                      val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        virt_ch = (-1, -1)
        phys_ch = [it+1 for it in (-1, -1)]  # physical charges ket, bra
        Z_Q.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                      val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (0, 0)]  # physical charges ket, bra
        Z_Q.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                      val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 1)]  # physical charges ket, bra
        Z_Q.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                      val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        op = q_z         # operator in full version
        Q_Z = tensor.Tensor(settings=settings_U1, s=(
            1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (-1, -1)]  # physical charges ket, bra
        Q_Z.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                      val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (0, 0)]  # physical charges ket, bra
        Q_Z.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                      val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 1)]  # physical charges ket, bra
        Q_Z.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                      val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        virt_ch = (-1, -1)
        phys_ch = [it+1 for it in (-1, -1)]  # physical charges ket, bra
        Q_Z.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                      val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (0, 0)]  # physical charges ket, bra
        Q_Z.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                      val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 1)]  # physical charges ket, bra
        Q_Z.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                      val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        #

        op = q_cp         # operator in full version
        virt_ch = (0, 1)
        Q_CP_right = tensor.Tensor(
            settings=settings_U1, s=(1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (0, -1)]  # physical charges ket, bra
        Q_CP_right.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                             val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 0)]  # physical charges ket, bra
        Q_CP_right.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                             val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        op = cp_q         # operator in full version
        virt_ch = (0, 1)
        CP_Q_right = tensor.Tensor(
            settings=settings_U1, s=(1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (0, -1)]  # physical charges ket, bra
        CP_Q_right.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                             val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 0)]  # physical charges ket, bra
        CP_Q_right.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                             val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        op = q_c         # operator in full version
        virt_ch = (0, -1)
        Q_C_right = tensor.Tensor(
            settings=settings_U1, s=(1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (-1, 0)]  # physical charges ket, bra
        Q_C_right.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                            val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (0, 1)]  # physical charges ket, bra
        Q_C_right.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                            val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        op = c_q         # operator in full version
        virt_ch = (0, -1)
        C_Q_right = tensor.Tensor(
            settings=settings_U1, s=(1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (-1, 0)]  # physical charges ket, bra
        C_Q_right.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                            val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (0, 1)]  # physical charges ket, bra
        C_Q_right.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                            val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))
        #
        op = q_cp         # operator in full version
        virt_ch = (-1, 0)
        Q_CP_left = tensor.Tensor(
            settings=settings_U1, s=(1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (0, -1)]  # physical charges ket, bra
        Q_CP_left.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                            val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 0)]  # physical charges ket, bra
        Q_CP_left.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                            val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        op = cp_q         # operator in full version
        virt_ch = (-1, 0)
        CP_Q_left = tensor.Tensor(
            settings=settings_U1, s=(1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (0, -1)]  # physical charges ket, bra
        CP_Q_left.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                            val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (1, 0)]  # physical charges ket, bra
        CP_Q_left.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                            val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        op = q_c         # operator in full version
        virt_ch = (1, 0)
        Q_C_left = tensor.Tensor(settings=settings_U1,
                                 s=(1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (-1, 0)]  # physical charges ket, bra
        Q_C_left.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                           val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (0, 1)]  # physical charges ket, bra
        Q_C_left.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                           val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        op = c_q         # operator in full version
        virt_ch = (1, 0)
        C_Q_left = tensor.Tensor(settings=settings_U1,
                                 s=(1, 1, -1, -1), dtype=dtype, n=0)
        phys_ch = [it+1 for it in (-1, 0)]  # physical charges ket, bra
        C_Q_left.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                           val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))

        phys_ch = [it+1 for it in (0, 1)]  # physical charges ket, bra
        C_Q_left.set_block(ts=(virt_ch[0], ts[phys_ch[0]], ts[phys_ch[1]], virt_ch[1]),
                           val=op[ch[phys_ch[0]], :][:, ch[phys_ch[1]]], Ds=(1, Ds[phys_ch[0]], Ds[phys_ch[1]], 1))
        #

    H = mps.Mps(N, nr_phys=2)
    for n in range(N):
        wn = wk[n]
        if LSR[n] == -1:  # L
            vL, vR = (-1j)*vk[n], 0
        elif LSR[n] == +1:  # R
            vL, vR = 0, (-1j)*vk[n]

        # local operator - including dissipation
        if abs(LSR[n]) == 1:
            en = wk[n] + dV[n]
            p = 1. / \
                (1. + np.exp(en / temp[n])
                 ) if temp[n] > 1e-6 else (1. - np.sign(en))*.5
            gp = gamma[n]*p
            gm = gamma[n]*(1. - p)
            #
            On_Site = (wn * m1j_N_Q__M__Q_N).apxb(N_Q__P__Q_N,
                                                  x=(- gm*.5)).apxb(CCP_Q__P__Q_CCP, x=(- gp*.5))
            diss_off = (gp*CP_Q_C).apxb(C_Q_CP, x=gm)
        else:  # impurity
            On_Site = wn * m1j_N_Q__M__Q_N

        if abs(LSR[n]) == 1 and n < n1:
            if n == 0:
                H.A[n] = tensor.block({(0, 0): On_Site+diss_off,
                                       (0, 1): +vL*C_Q_right,  (0, 2): -vL*Q_C_right,
                                       (0, 3): +vL*CP_Q_right,  (0, 4): -vL*Q_CP_right,
                                       (0, 5): +vR*C_Q_right,  (0, 6): -vR*Q_C_right,
                                       (0, 7): +vR*CP_Q_right,  (0, 8): -vR*Q_CP_right,
                                       (0, 9): Z_Q_Z, (0, 10): II}, common_legs=(1, 2))
            else:
                H.A[n] = tensor.block({(0, 0): On_Site, (-1, 0): diss_off,
                                       (0, 1): +vL*C_Q_right,  (0, 2): -vL*Q_C_right,
                                       (0, 3): +vL*CP_Q_right,  (0, 4): -vL*Q_CP_right,
                                       (0, 5): +vR*C_Q_right,  (0, 6): -vR*Q_C_right,
                                       (0, 7): +vR*CP_Q_right,  (0, 8): -vR*Q_CP_right,
                                       (-10, 0): II, (-9, 1): Z_Q, (-7, 3): Z_Q, (-5, 5): Z_Q, (-3, 7): Z_Q, (-1, 9): Z_Q_Z,
                                       (-8, 2): Q_Z, (-6, 4): Q_Z, (-4, 6): Q_Z, (-2, 8): Q_Z, (0, 10): II
                                       }, common_legs=(1, 2))
        elif abs(LSR[n]) == 1 and n > n1:
            if n == N - 1:
                H.A[n] = tensor.block({(-10, 0): II,
                                       (-9, 0): +vL * CP_Q_left,
                                       (-8, 0): -vL * Q_CP_left,
                                       (-7, 0): +vL * C_Q_left,
                                       (-6, 0): -vL * Q_C_left,
                                       (-5, 0): +vR * CP_Q_left,
                                       (-4, 0): -vR * Q_CP_left,
                                       (-3, 0): +vR * C_Q_left,
                                       (-2, 0): -vR * Q_C_left,
                                       (-1, 0): diss_off,
                                       (0, 0): On_Site
                                       }, common_legs=(1, 2))
            else:
                H.A[n] = tensor.block({(-9, 0): +vL * CP_Q_left,
                                       (-8, 0): -vL * Q_CP_left,
                                       (-7, 0): +vL * C_Q_left,
                                       (-6, 0): -vL * Q_C_left,
                                       (-5, 0): +vR * CP_Q_left,
                                       (-4, 0): -vR * Q_CP_left,
                                       (-3, 0): +vR * C_Q_left,
                                       (-2, 0): -vR * Q_C_left,
                                       (-1, 0): diss_off,
                                       (0, 0): On_Site,
                                       (-10, 0): II, (-9, 1): Z_Q, (-7, 3): Z_Q, (-5, 5): Z_Q, (-3, 7): Z_Q, (-1, 9): Z_Q_Z,
                                       (-8, 2): Q_Z, (-6, 4): Q_Z, (-4, 6): Q_Z, (-2, 8): Q_Z, (0, 10): II
                                       }, common_legs=(1, 2))
        elif n == n1:  # site 1 of S in LSR
            H.A[n] = tensor.block({(-10, 0): II,
                                   (-9, 0): CP_Q_left,
                                   (-8, 0): Q_CP_left,
                                   (-7, 0): C_Q_left,
                                   (-6, 0): Q_C_left,
                                   (-5, 0): CP_Q_left,
                                   (-4, 0): Q_CP_left,
                                   (-3, 0): C_Q_left,
                                   (-2, 0): Q_C_left,
                                   (0, 0): On_Site,
                                   (0, 1): C_Q_right,  (0, 2): Q_C_right,
                                   (0, 3): CP_Q_right,  (0, 4): Q_CP_right,
                                   (0, 5): C_Q_right,  (0, 6): Q_C_right,
                                   (0, 7): CP_Q_right,  (0, 8): Q_CP_right,
                                   (-1, 9): Z_Q_Z, (0, 10): II}, common_legs=(1, 2))

    HdagH = general.stack_MPOs(H, H) if AdagA else None
    return H, HdagH


# MEASURE
def current(LSR, vk, cut, basis):
    N = len(LSR)  # total number of sites
    n1 = np.argwhere(LSR == 2)[0, 0]  # impurity position

    vII, _, vc, vcp, vz = general.generate_vectorized_basis(basis)

    if basis == 0:  # choose I, Z, X, Y basis
        dtype = 'complex128'
        II = tensor.Tensor(settings=settings_Z2,
                           s=(1, 1, -1), dtype=dtype, n=0)
        II.set_block(ts=(0, 0, 0), val=vII[:2], Ds=(1, 2, 1))

        z = tensor.Tensor(settings=settings_Z2, s=(1, 1, -1), dtype=dtype, n=0)
        z.set_block(ts=(1, 0, 1), val=vz[:2], Ds=(1, 2, 1))

        c_right = tensor.Tensor(settings=settings_Z2,
                                s=(1, 1, -1), dtype=dtype, n=0)
        c_right.set_block(ts=(0, 1, 1), val=vc[2:], Ds=(1, 2, 1))

        cp_right = tensor.Tensor(settings=settings_Z2,
                                 s=(1, 1, -1), dtype=dtype, n=0)
        cp_right.set_block(ts=(0, 1, 1), val=vcp[2:], Ds=(1, 2, 1))

        c_left = tensor.Tensor(settings=settings_Z2,
                               s=(1, 1, -1), dtype=dtype, n=0)
        c_left.set_block(ts=(1, 1, 0), val=vc[2:], Ds=(1, 2, 1))

        cp_left = tensor.Tensor(settings=settings_Z2,
                                s=(1, 1, -1), dtype=dtype, n=0)
        cp_left.set_block(ts=(1, 1, 0), val=vcp[2:], Ds=(1, 2, 1))
    elif basis == 1:
        II = tensor.Tensor(settings=settings_U1, s=(1, 1, -1), n=0)
        II.set_block(ts=(0, 0, 0), val=vII[:2], Ds=(1, 2, 1))

        z = tensor.Tensor(settings=settings_U1, s=(1, 1, -1), n=0)
        z.set_block(ts=(1, 0, 1), val=vz[:2], Ds=(1, 2, 1))
        z.set_block(ts=(-1, 0, -1), val=vz[:2], Ds=(1, 2, 1))

        c_right = tensor.Tensor(settings=settings_U1, s=(1, 1, -1), n=0)
        c_right.set_block(ts=(0, -1, -1), val=np.array([vc[2]]), Ds=(1, 1, 1))

        cp_right = tensor.Tensor(settings=settings_U1, s=(1, 1, -1), n=0)
        cp_right.set_block(ts=(0, 1, 1), val=np.array([vcp[3]]), Ds=(1, 1, 1))

        c_left = tensor.Tensor(settings=settings_U1, s=(1, 1, -1), n=0)
        c_left.set_block(ts=(1, -1, 0), val=np.array([vc[2]]), Ds=(1, 1, 1))

        cp_left = tensor.Tensor(settings=settings_U1, s=(1, 1, -1), n=0)
        cp_left.set_block(ts=(-1, 1, 0), val=np.array([vcp[3]]), Ds=(1, 1, 1))

    if cut == 'LS':
        ck_right, ck_left = c_right, c_left
        cs_right, cs_left = cp_right, cp_left
    elif cut == 'SR':
        cs_right, cs_left = c_right, c_left
        ck_right, ck_left = cp_right, cp_left

    H = mps.Mps(N, nr_phys=1)
    for n in range(N):
        if cut == 'LS':
            v = vk[n] if LSR[n] == -1 else 0
        if cut == 'SR':
            v = vk[n] if LSR[n] == +1 else 0

        if n == 0:
            H.A[n] = tensor.block({(0, 0): II*0, (0, 2): II, (0, 1): v*ck_right}, common_legs=(1))
        if n == N - 1:
            H.A[n] = tensor.block({(0, 0): II*0, (-2, 0): II, (-1, 0): v*ck_left}, common_legs=(1))
        else:
            if n < n1:
                H.A[n] = tensor.block({(0, 0): II*0, (0, 2): II, (0, 1): v*ck_right, (-1, 1): z}, common_legs=(1))
            if n == n1:
                H.A[n] = tensor.block({(0, 0): II*0, (0, 2): II, (0, 1): cs_right, (-2, 0): II, (-1, 0): cs_left, (-1, 1): z}, common_legs=(1))
            else:
                H.A[n] = tensor.block({(0, 0): II*0,(-2, 0): II, (-1, 0): v*ck_left, (-1, 1): z}, common_legs=(1))
    # TO PUSH:  cant blck different blocks on the same position. Overwrites.
    return H


def measure_Op(N, id, Op, basis):
    vII, vnn, _, _, _ = general.generate_vectorized_basis(basis)
    if basis == 0:  # choose I, Z, X, Y basis
        II = tensor.Tensor(settings=settings_Z2, s=(1, 1, -1), n=0)
        nn = tensor.Tensor(settings=settings_Z2, s=(1, 1, -1), n=0)
    elif basis == 1:
        II = tensor.Tensor(settings=settings_U1, s=(1, 1, -1), n=0)
        nn = tensor.Tensor(settings=settings_U1, s=(1, 1, -1), n=0)
    II.set_block(ts=(0, 0, 0), val=vII[:2], Ds=(1, 2, 1))
    if Op == 'nn':
        nn.set_block(ts=(0, 0, 0), val=vnn[:2], Ds=(1, 2, 1))
        Op = nn
    H = mps.Mps(N, nr_phys=1)
    for n in range(N):
        H.A[n] = Op if n == id else II
    return H


def measure_sumOp(choice, LSR, basis, Op):
    # sum of particles in all elements choice==LSR
    N = len(LSR)
    vII, vnn, _, _, _ = general.generate_vectorized_basis(basis)
    if basis == 0:  # choose I, Z, X, Y basis
        II = tensor.Tensor(settings=settings_Z2, s=(1, 1, -1), n=0)
        nn = tensor.Tensor(settings=settings_Z2, s=(1, 1, -1), n=0)
    elif basis == 1:
        II = tensor.Tensor(settings=settings_U1, s=(1, 1, -1), n=0)
        nn = tensor.Tensor(settings=settings_U1, s=(1, 1, -1), n=0)
    II.set_block(ts=(0, 0, 0), val=vII[:2], Ds=(1, 2, 1))
    if Op == 'nn':
        nn.set_block(ts=(0, 0, 0), val=vnn[:2], Ds=(1, 2, 1))
        Op = nn
    H = mps.Mps(N, nr_phys=1)
    for n in range(N):
        taken = Op if LSR[n] == choice else II*0
        if n == 0:
            H.A[n] = tensor.block({(0, 0): taken, (0, 1): II}, common_legs=(1))
        elif n == N-1:
            H.A[n] = tensor.block(
                {(0, 0): taken, (-1, 0): II}, common_legs=(1))
        else:
            H.A[n] = tensor.block(
                {(0, 0): taken, (0, 1): II, (-1, 0): II}, common_legs=(1))
    return H


def identity(N, basis):
    vII, _, _, _, _ = general.generate_vectorized_basis(basis)
    if basis == 0:  # choose I, Z, X, Y basis
        II = tensor.Tensor(settings=settings_Z2, s=(1, 1, -1), n=0)
    elif basis == 1:
        II = tensor.Tensor(settings=settings_U1, s=(1, 1, -1), n=0)
    II.set_block(ts=(0, 0, 0), val=vII[:2], Ds=(1, 2, 1))
    H = mps.Mps(N, nr_phys=1)
    for n in range(N):
        H.A[n] = II
    return H
