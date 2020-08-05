import numpy as np
import yamps.mps as mps
import yamps.tensor as tensor
import yamps.ops.settings_full as settings
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
    for n in range(N):  # empty tensors
        psi.A[n] = tensor.Tensor(settings=settings, s=(1, 1, -1), dtype=dtype)
        if abs(LSR[n]) == 1:
            wk = ww[n]
            tp = temp[n]
            occ = 1./(np.exp(wk/tp)+1.) if tp > 1e-6 else (1.-np.sign(wk))*.5
        else:
            occ = io[im]
            im += 1
        #
        if basis == 0:  # choose I, Z, X, Y basis
            val = np.array([1., 1.-2.*occ, 0, 0])
        elif basis == 1:  # choose cp c, c cp, c, cp
            val = np.array([occ, 1.-occ, 0, 0])
        #
        psi.A[n].set_block(val=val, Ds=(1, 4, 1))
    return psi


# HAMILTONIAN
def Lindbladian_1AIM_mixed(NL, LSR, wk, temp, vk, dV, gamma, basis, AdagA=False, dtype='float64'):
    OO, II, q_z, z_q, _, _, _, _, c_q_cp, cp_q_c, z_q_z, c_q, cp_q, q_c, q_cp, ccp_q__p__q_ccp, n_q__p__q_n, m1j_n_q__m__q_n = general.generate_operator_basis(
        basis)

    N = 2 * NL + 1  # total number of sites
    n1 = np.argwhere(LSR == 2)[0, 0]  # impurity position

    H = mps.Mps(N, nr_phys=2)

    for n in range(N):  # empty tensors
        H.A[n] = tensor.Tensor(
            settings=settings, s=(1, 1, -1, -1), dtype=dtype)

        wn = wk[n]
        v = (-1j)*vk[n]
        # local operator - including dissipation
        if abs(LSR[n]) == 1:
            en = wk[n] + dV[n]
            p = 1. / \
                (1. + np.exp(en / temp[n])
                 ) if temp[n] > 1e-6 else (1. - np.sign(en))*.5
            gp = gamma[n]*p
            gm = gamma[n]*(1. - p)
            #
            On_Site = wn * m1j_n_q__m__q_n - gp*.5 * \
                (ccp_q__p__q_ccp) - gm*.5*(n_q__p__q_n)
            diss_off = gp*cp_q_c + gm*c_q_cp
        else:
            On_Site = wn * m1j_n_q__m__q_n
        #
        if n == 0:
            tmp = np.block([On_Site+diss_off, +v * c_q, -v * q_c, +v * cp_q, -v * q_cp,
                            z_q_z, II])
            tmp = tmp.reshape((1, 4, 7, 4))
        elif n != 0 and n < n1:
            tmp = np.block([[II,  OO, OO, OO, OO, OO, OO],
                            [OO, z_q, OO, OO, OO, OO, OO],
                            [OO, OO, q_z, OO, OO, OO, OO],
                            [OO, OO, OO, z_q, OO, OO, OO],
                            [OO, OO, OO, OO, q_z, OO, OO],
                            [diss_off, OO, OO, OO, OO, z_q_z, OO],
                            [On_Site, +v * c_q, -v * q_c, +v * cp_q, -v * q_cp, OO, II]])
            tmp = tmp.reshape((7, 4, 7, 4))
        elif n == n1:
            tmp = np.block([[II,   OO, OO, OO, OO, OO, OO],
                            [cp_q, OO, OO, OO, OO, OO, OO],
                            [q_cp, OO, OO, OO, OO, OO, OO],
                            [c_q,  OO, OO, OO, OO, OO, OO],
                            [q_c,  OO, OO, OO, OO, OO, OO],
                            [OO,   OO, OO, OO, OO, z_q_z, OO],
                            [On_Site, c_q, q_c, cp_q, q_cp, OO, II]])
            tmp = tmp.reshape((7, 4, 7, 4))
        elif n > n1 and n != N-1:
            tmp = np.block([[II,   OO, OO, OO, OO, OO, OO],
                            [+v*cp_q, z_q, OO, OO, OO, OO, OO],
                            [-v*q_cp, OO, q_z, OO, OO, OO, OO],
                            [+v*c_q,  OO, OO, z_q, OO, OO, OO],
                            [-v*q_c,  OO, OO, OO, q_z, OO, OO],
                            [diss_off,   OO, OO, OO, OO, z_q_z, OO],
                            [On_Site, OO, OO, OO, OO, OO, II]])
            tmp = tmp.reshape((7, 4, 7, 4))
        elif n == N - 1:
            tmp = np.block([[II],
                            [+v*cp_q],
                            [-v*q_cp],
                            [+v*c_q],
                            [-v*q_c],
                            [diss_off],
                            [On_Site]])
            tmp = tmp.reshape((7, 4, 1, 4))
        #  #  #
        tmp = tmp.transpose((0, 1, 3, 2))
        H.A[n].set_block(val=tmp)
    HdagH = general.stack_MPOs(H, H) if AdagA else None
    return H, HdagH


# MEASURE
def current(LSR, vk, cut, basis):
    N = len(LSR)  # total number of sites
    n1 = np.argwhere(LSR == 2)[0, 0]  # impurity position
    #
    II, _, vc, vcp, z = general.generate_vectorized_basis(basis)
    OO = II*0
    #
    if cut == 'LS':
        ck = vc
        cs = vcp
    elif cut == 'SR':
        cs = vc
        ck = vcp
    #
    H = mps.Mps(N, nr_phys=1)
    for n in range(N):
        H.A[n] = tensor.Tensor(settings=settings, s=(1, 1, -1))
        #
        if cut == 'LS':
            v = 1j*vk[n] if LSR[n] == -1 else 0
        elif cut == 'SR':
            v = 1j*vk[n] if LSR[n] == +1 else 0
        #
        if n == 0:
            tmp = np.block([OO,  v * ck,  -v * cs, II])
            tmp = tmp.reshape((1, 4, 4))
        elif n < n1 and n != 0:
            tmp = np.block([[II,     OO, OO, OO],
                            [OO,      z, OO, OO],
                            [OO,     OO,  z, OO],
                            [OO, v * ck, -v * cs, II]])
            tmp = tmp.reshape((4, 4, 4))
        elif n == n1:
            tmp = np.block([[II, OO, OO, OO],
                            [cs, OO, OO, OO],
                            [ck, OO, OO, OO],
                            [OO, cs, ck, II]])
            tmp = tmp.reshape((4, 4, 4))
        elif n > n1 and n != N - 1:
            tmp = np.block([[II,     OO, OO, OO],
                            [v * ck,  z, OO, OO],
                            [-v * cs,  OO, z, OO],
                            [OO,     OO, OO, II]])
            tmp = tmp.reshape((4, 4, 4))
        elif n == N - 1:
            tmp = np.block([[II],
                            [v * ck],
                            [-v * cs],
                            [OO]])
            tmp = tmp.reshape((4, 1, 4))
        tmp = tmp.transpose((0, 2, 1))
        H.A[n].set_block(val=tmp)
    return H


def occupancy(N, id, basis):
    II, nn, _, _, _ = general.generate_vectorized_basis(basis)

    H = mps.Mps(N, nr_phys=1)
    for n in range(N):  # empty tensors
        H.A[n] = tensor.Tensor(settings=settings, s=(1, 1, -1))
        tmp = nn if n == id else II
        tmp = tmp.reshape((1, 4, 1))
        H.A[n].set_block(val=tmp)
    return H


def identity(N, basis):
    II, _, _, _, _ = general.generate_vectorized_basis(basis)
    H = mps.Mps(N, nr_phys=1)
    for n in range(N):  # empty tensors
        H.A[n] = tensor.Tensor(settings=settings, s=(1, 1, -1))
        tmp = II
        tmp = tmp.reshape((1, 4, 1))
        H.A[n].set_block(val=tmp)
    return H


def measure_Op(N, id, Op, basis):
    vII, vnn, _, _, _ = general.generate_vectorized_basis(basis)
    if Op == 'nn':
        Op = vnn
    H = mps.Mps(N, nr_phys=1)
    for n in range(N):
        H.A[n] = tensor.Tensor(settings=settings, s=(1, 1, -1))
        val = Op if n == id else vII
        H.A[n].set_block(val=val, Ds=(1, 4, 1))
    return H


def measure_sumOp(choice, LSR, basis, Op):
    # sum of particles in all elements choice==LSR
    N = len(LSR)
    II, vnn, _, _, _ = general.generate_vectorized_basis(basis)
    OO = II*0
    H = mps.Mps(N, nr_phys=1)
    for n in range(N):
        H.A[n] = tensor.Tensor(settings=settings, s=(1, 1, -1))
        taken = vnn if LSR[n] == choice else vnn*0
        if n == 0:
            tmp = np.block([taken, II])
            tmp = tmp.reshape((1*1, 2, 4))
        elif n == N-1:
            tmp = np.block([[II],
                            [taken]])
            tmp = tmp.reshape((1*2, 1, 4))
        else:
            tmp = np.block([[II,    OO],
                            [taken, II]])
            tmp = tmp.reshape((2, 2, 4))
        tmp = tmp.transpose((0, 2, 1))
        H.A[n].set_block(val=tmp)
    return H
