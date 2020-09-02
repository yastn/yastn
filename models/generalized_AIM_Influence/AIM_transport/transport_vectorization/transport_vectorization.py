import numpy as np
import yamps.mps as mps
import yamps.tensor as tensor
from yamps.tensor.ncon import ncon
from AIM_transport.transport_vectorization import settings_full
from AIM_transport.transport_vectorization import settings_Z2
from AIM_transport.transport_vectorization import settings_U1
from AIM_transport.transport_vectorization import transport_vectorization_general as general


# STATE


def thermal_state(tensor_type, basis, LSR, io, ww, temp):
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
    vii, vnn, _, _, vz = general.generate_vectorized_basis(basis)
    ii = vector_into_Tensor(tensor_type, vii, 0)
    if basis == 'Majorana':
        z = vector_into_Tensor(tensor_type, vz, 0)
    elif basis == 'Dirac':
        nn = vector_into_Tensor(tensor_type, vnn, 0)
    #
    N = len(LSR)
    im = 0
    psi = mps.Mps(N, nr_phys=1)
    psi.normalize = False
    for n in range(N):
        if abs(LSR[n]) == 1:
            wk, tp = ww[n], temp[n]
            occ = 1./(np.exp(wk/tp)+1.) if tp > 1e-6 else (1.-np.sign(wk))*.5
        else:
            occ = io[im]
            im += 1
        if basis == 'Majorana':
            psi.A[n] = ii+(1.-2.*occ)*z
        elif basis == 'Dirac':
            psi.A[n] = occ*nn+(1.-occ)*(ii-nn)
    return psi


# OPERATOR

def vectorized_Lindbladian_general(tensor_type, LSR, temp, dV, gamma, corr, basis):
    # make operator for evolution with dissipation
    _, ii, _, _, _, _, q_z, z_q, _, _, _, _, c_q_cp, cp_q_c, z_q_z, c_q, cp_q, q_c, q_cp, ccp_q__p__q_ccp, n_q__p__q_n, m1j_n_q__m__q_n = general.generate_operator_basis(
        basis)
    II = operator_into_Tensor(tensor_type, ii, 0)
    N_Q__P__Q_N = operator_into_Tensor(tensor_type, n_q__p__q_n, 0)
    CCP_Q__P__Q_CCP = operator_into_Tensor(tensor_type, ccp_q__p__q_ccp, 0)
    m1j_N_Q__M__Q_N = operator_into_Tensor(tensor_type, m1j_n_q__m__q_n, 0)
    Z_Q_Z = operator_into_Tensor(tensor_type, z_q_z, 0)
    C_Q_CP = operator_into_Tensor(tensor_type, c_q_cp, 0)
    CP_Q_C = operator_into_Tensor(tensor_type, cp_q_c, 0)
    Q_Z_for_ccp = operator_into_Tensor(tensor_type, q_z, -1)
    Z_Q_for_ccp = operator_into_Tensor(tensor_type, z_q, -1)
    Q_Z_for_cpc = operator_into_Tensor(tensor_type, q_z, 1)
    Z_Q_for_cpc = operator_into_Tensor(tensor_type, z_q, 1)
    Q_CP_right = operator_into_Tensor(tensor_type, q_cp, 0)
    Q_C_left = operator_into_Tensor(tensor_type, q_c, 1)
    Q_C_right = operator_into_Tensor(tensor_type, q_c, 0)
    Q_CP_left = operator_into_Tensor(tensor_type, q_cp, -1)
    CP_Q_right = operator_into_Tensor(tensor_type, cp_q, 0)
    C_Q_left = operator_into_Tensor(tensor_type, c_q, 1)
    C_Q_right = operator_into_Tensor(tensor_type, c_q, 0)
    CP_Q_left = operator_into_Tensor(tensor_type, cp_q, -1)
    #
    N = len(LSR)
    H = mps.Mps(N, nr_phys=2)
    # Set 1 on site energies and dissipation
    for n in range(N):
        wn = corr[n, n]
        # local operator - including dissipation
        if abs(LSR[n]) == 1:
            en = wn + dV[n]
            p = 1. / \
                (1. + np.exp(en / temp[n])
                 ) if temp[n] > 1e-6 else (1. - np.sign(en))*.5
            gp = gamma[n]*p
            gm = gamma[n]*(1. - p)
            #
            On_Site = wn * m1j_N_Q__M__Q_N - gp*.5 * \
                (CCP_Q__P__Q_CCP) - gm*.5*(N_Q__P__Q_N)
            diss_off = gp*CP_Q_C + gm*C_Q_CP
        else:
            On_Site = wn * m1j_N_Q__M__Q_N
            diss_off = 0.*II
        if n == 0:
            H.A[n] = tensor.block(
                {(0, 0): On_Site + diss_off, (0, 1): Z_Q_Z, (0, 2): II}, common_legs=(1, 2))
        elif n == N - 1:
            H.A[n] = tensor.block({(-2, 0): II,
                                   (-1, 0): diss_off,
                                   (0, 0): On_Site}, common_legs=(1, 2))
        else:
            H.A[n] = tensor.block({(-2, 0): II,
                                   (-1, 0): diss_off, (-1, 1): Z_Q_Z,
                                   (0, 0): On_Site, (0, 2): II}, common_legs=(1, 2))
    # Set 2 reservoir-impurity couplings
    for mS in np.nonzero(LSR == 2)[0]:
        for n in range(N):
            v = (-1j)*corr[mS, n]
            if n == 0:
                tmp = tensor.block({(0, 0): II*0., (0, 1): +v*C_Q_right,  (0, 2): -v*Q_C_right,
                                    (0, 3): +v*CP_Q_right, (0, 4): -v*Q_CP_right, (0, 5): II}, common_legs=(1, 2))
            elif n != 0 and n < mS:
                tmp = tensor.block({(-5, 0): II,
                                    (-4, 1): Z_Q_for_ccp,
                                    (-3, 2): Q_Z_for_ccp,
                                    (-2, 3): Z_Q_for_cpc,
                                    (-1, 4): Q_Z_for_cpc,
                                    (0, 0): II*0, (0, 1): +v*C_Q_right,  (0, 2): -v*Q_C_right, (0, 3): +v*CP_Q_right, (0, 4): -v*Q_CP_right, (0, 5): II}, common_legs=(1, 2))
            elif n == mS:
                tmp = tensor.block({(-5, 0): II,
                                    (-4, 0): CP_Q_left,
                                    (-3, 0): Q_CP_left,
                                    (-2, 0): C_Q_left,
                                    (-1, 0): Q_C_left,
                                    (0, 0): II*0., (0, 1): C_Q_right, (0, 2): Q_C_right, (0, 3): CP_Q_right, (0, 4): Q_CP_right, (0, 5): II}, common_legs=(1, 2))
            elif n > mS and n != N-1:
                tmp = tensor.block({(-5, 0): II,
                                    (-4, 0): +v*CP_Q_left, (-5, 1): Z_Q_for_ccp,
                                    (-3, 0): -v*Q_CP_left, (-4, 2): Q_Z_for_ccp,
                                    (-2, 0): +v*C_Q_left, (-3, 3): Z_Q_for_cpc,
                                    (-1, 0): -v*Q_C_left, (-2, 4): Q_Z_for_cpc,
                                    (0, 0): II*0., (0, 5): II}, common_legs=(1, 2))
            elif n == N - 1:
                tmp = tensor.block({(-5, 0): II,
                                    (-4, 0): +v*CP_Q_left,
                                    (-3, 0): -v*Q_CP_left,
                                    (-2, 0): +v*C_Q_left,
                                    (-1, 0): -v*Q_C_left,
                                    (0, 0): II*0.}, common_legs=(1, 2))
            H.A[n] = general.add_MPOs(H.A[n], tmp, common_legs=(1, 2))
    return H


def vectorized_Lindbladian_real(tensor_type, LSR, temp, dV, gamma, corr, basis):
    # make operator for evolution with dissipation
    _, ii, _, x_q, _, y_q, _, z_q, _, _, _, _, c_q_cp, cp_q_c, z_q_z, _, _, _, _, ccp_q__p__q_ccp, n_q__p__q_n, m1j_n_q__m__q_n = general.generate_operator_basis(
        basis)
    #
    II = operator_into_Tensor(tensor_type, ii, 0)
    N_Q__P__Q_N = operator_into_Tensor(tensor_type, n_q__p__q_n, 0)
    CCP_Q__P__Q_CCP = operator_into_Tensor(tensor_type, ccp_q__p__q_ccp, 0)
    m1j_N_Q__M__Q_N = operator_into_Tensor(tensor_type, m1j_n_q__m__q_n, 0)
    Z_Q_Z = operator_into_Tensor(tensor_type, z_q_z, 0)
    C_Q_CP = operator_into_Tensor(tensor_type, c_q_cp, 0)
    CP_Q_C = operator_into_Tensor(tensor_type, cp_q_c, 0)
    #
    ZR = operator_into_Tensor(tensor_type, z_q.real, 1)
    ZI = operator_into_Tensor(tensor_type, z_q.imag, 1)
    #
    XR_right = operator_into_Tensor(tensor_type, x_q.real, 0)
    XI_right = operator_into_Tensor(tensor_type, x_q.imag, 0)
    YR_right = operator_into_Tensor(tensor_type, y_q.real, 0)
    YI_right = operator_into_Tensor(tensor_type, y_q.imag, 0)
    #
    XR_left = operator_into_Tensor(tensor_type, x_q.real, 1)
    XI_left = operator_into_Tensor(tensor_type, x_q.imag, 1)
    YR_left = operator_into_Tensor(tensor_type, y_q.real, 1)
    YI_left = operator_into_Tensor(tensor_type, y_q.imag, 1)
    #
    N = len(LSR)
    H = mps.Mps(N, nr_phys=2)
    # Set 1 on site energies and dissipation
    for n in range(N):
        wn = corr[n, n]
        # local operator - including dissipation
        if abs(LSR[n]) == 1:
            en = wn + dV[n]
            p = 1. / \
                (1. + np.exp(en / temp[n])
                 ) if temp[n] > 1e-6 else (1. - np.sign(en))*.5
            gp = gamma[n]*p
            gm = gamma[n]*(1. - p)
            #
            On_Site = wn * m1j_N_Q__M__Q_N - gp*.5 * \
                (CCP_Q__P__Q_CCP) - gm*.5*(N_Q__P__Q_N)
            diss_off = gp*CP_Q_C + gm*C_Q_CP
        else:
            On_Site = wn * m1j_N_Q__M__Q_N
            diss_off = 0.*II
        if n == 0:
            H.A[n] = tensor.block(
                {(0, 0): On_Site + diss_off, (0, 1): Z_Q_Z, (0, 2): II}, common_legs=(1, 2))
        elif n == N - 1:
            H.A[n] = tensor.block({(-2, 0): II,
                                   (-1, 0): diss_off,
                                   (0, 0): On_Site}, common_legs=(1, 2))
        else:
            H.A[n] = tensor.block({(-2, 0): II,
                                   (-1, 0): diss_off, (-1, 1): Z_Q_Z,
                                   (0, 0): On_Site, (0, 2): II}, common_legs=(1, 2))
    # Set 2 reservoir-impurity couplings
    for mS in np.nonzero(LSR == 2)[0]:
        for n in range(N):
            v = corr[mS, n]
            if n == 0:
                tmp = tensor.block({(0, 0): II*0., (0, 1): v*XR_right, (0, 2): v*XI_right,
                                    (0, 3): v*YR_right,  (0, 4): v*YI_right, (0, 5): II}, common_legs=(1, 2))
            elif n != 0 and n < mS:
                tmp = tensor.block({(-5, 0): II,
                                    (-4, 1): ZR, (-4, 2): ZI,
                                    (-3, 1): (-1)*ZI, (-3, 2): ZR,
                                    (-2, 3): ZR, (-2, 4): ZI,
                                    (-1, 3): (-1)*ZI, (-1, 4): ZR,
                                    (0, 0): II*0, (0, 1): v*XR_right, (0, 2): v*XI_right, (0, 3): v*YR_right,  (0, 4): v*YI_right, (0, 5): II}, common_legs=(1, 2))
            elif n == mS:
                tmp = tensor.block({(-5, 0): II,
                                    (-4, 0): XI_left,
                                    (-3, 0): XR_left,
                                    (-2, 0): YI_left,
                                    (-1, 0): YR_left,
                                    (0, 0): II*0, (0, 1): XR_right, (0, 2): XI_right, (0, 3): YR_right,  (0, 4): YI_right, (0, 5): II}, common_legs=(1, 2))
            elif n > mS and n != N-1:
                tmp = tensor.block({(-5, 0): II,
                                    (-4, 0): v*XI_left,
                                    (-3, 0): v*XR_left,
                                    (-2, 0): v*YI_left,
                                    (-1, 0): v*YR_left,
                                    (-4, 1): ZR, (-4, 2): ZI,
                                    (-3, 1): (-1)*ZI, (-3, 2): ZR,
                                    (-2, 3): ZR, (-2, 4): ZI,
                                    (-1, 3): (-1)*ZI, (-1, 4): ZR,
                                    (0, 0): II*0, (0, 5): II}, common_legs=(1, 2))
            elif n == N - 1:
                tmp = tensor.block({(-5, 0): II,
                                    (-4, 0): v*XI_left,
                                    (-3, 0): v*XR_left,
                                    (-2, 0): v*YI_left,
                                    (-1, 0): v*YR_left,
                                    (0, 0): II*0}, common_legs=(1, 2))
            H.A[n] = general.add_MPOs(H.A[n], tmp, common_legs=(1, 2))
    return H


def Liouville_AIM_Coulomb_general(tensor_type, LSR, temp, dV, gamma, corr, basis):
    # make operator for evolution with dissipation
    _, ii, _, _, _, _, _, _, q_n, n_q, _, _, _, _, _, _, _, _, _, _, _, _ = general.generate_operator_basis(
        basis)
    II = operator_into_Tensor(tensor_type, ii, 0)
    N_Q = operator_into_Tensor(tensor_type, n_q, 0)
    Q_N = operator_into_Tensor(tensor_type, q_n, 0)
    #
    N = len(LSR)
    H = mps.Mps(N, nr_phys=2)
    # Set 1 Coulomb interaction inside impurity
    NS_sites = np.nonzero(LSR == 2)[0]
    for mS in NS_sites:
        for n in range(N):
            U = (-1j)*corr[mS, n]
            if n == min(NS_sites):
                tmp = tensor.block(
                    {(0, 0): II*0., (0, 1): U*N_Q,  (0, 2): -U*Q_N, (0, 3): II}, common_legs=(1, 2))
            elif n > min(NS_sites) and n < mS:
                tmp = tensor.block({(-3, 0): II,
                                    (-2, 1): II,
                                    (-1, 2): II,
                                    (0, 0): II*0., (0, 1): U*N_Q,  (0, 2): -U*Q_N, (0, 3): II}, common_legs=(1, 2))
            elif n == mS:
                if n == max(NS_sites):
                    tmp = tensor.block({(-3, 0): II,
                                        (-2, 0): N_Q,
                                        (-1, 0): Q_N,
                                        (0, 0): II*0.}, common_legs=(1, 2))
                else:
                    tmp = tensor.block({(-3, 0): II,
                                        (-2, 0): N_Q,
                                        (-1, 0): Q_N,
                                        (0, 0): II*0., (0, 1): N_Q,  (0, 2): Q_N, (0, 3): II}, common_legs=(1, 2))

            elif n > mS and n < max(NS_sites):
                tmp = tensor.block({(-3, 0): II,
                                    (-2, 0): U*N_Q, (-2, 1): II,
                                    (-1, 0): -U*Q_N, (-1, 2): II,
                                    (0, 0): II*0., (0, 3): II}, common_legs=(1, 2))
            elif n == max(NS_sites):
                tmp = tensor.block({(-3, 0): II,
                                    (-2, 0): U*N_Q,
                                    (-1, 0): -U*Q_N,
                                    (0, 0): II*0.}, common_legs=(1, 2))
            else:
                tmp = II
            H.A[n] = general.add_MPOs(H.A[n], tmp, common_legs=(
                1, 2)) if mS != NS_sites[0] else tmp
    return H


def Liouville_AIM_Coulomb_real(tensor_type, LSR, temp, dV, gamma, corr, basis):
    # make operator for evolution with dissipation
    _, ii, _, _, _, _, _, _, q_n, n_q, _, _, _, _, _, _, _, _, _, _, _, _ = general.generate_operator_basis(
        basis)
    II = operator_into_Tensor(tensor_type, ii, 0)
    R = operator_into_Tensor(tensor_type, n_q.real, 0)
    I = operator_into_Tensor(tensor_type, q_n.imag, 0)
    #
    N = len(LSR)
    H = mps.Mps(N, nr_phys=2)
    # Set 1 Coulomb interaction inside impurity
    NS_sites = np.nonzero(LSR == 2)[0]
    for mS in NS_sites:
        for n in range(N):
            U = 2.*corr[mS, n]
            if n == min(NS_sites):
                tmp = tensor.block(
                    {(0, 0): II*0., (0, 1): U*R,  (0, 2): U*I, (0, 3): II}, common_legs=(1, 2))
            elif n > min(NS_sites) and n < mS:
                tmp = tensor.block({(-3, 0): II,
                                    (-2, 1): II,
                                    (-1, 2): II,
                                    (0, 0): II*0., (0, 1): U*R,  (0, 2): U*I, (0, 3): II}, common_legs=(1, 2))
            elif n == mS:
                if n == max(NS_sites):
                    tmp = tensor.block({(-3, 0): II,
                                        (-2, 0): I,
                                        (-1, 0): R,
                                        (0, 0): II*0.}, common_legs=(1, 2))
                else:
                    tmp = tensor.block({(-3, 0): II,
                                        (-2, 0): I,
                                        (-1, 0): R,
                                        (0, 0): II*0., (0, 1): R,  (0, 2): I, (0, 3): II}, common_legs=(1, 2))

            elif n > mS and n < max(NS_sites):
                tmp = tensor.block({(-3, 0): II,
                                    (-2, 0): U*I, (-2, 1): II,
                                    (-1, 0): U*R, (-1, 2): II,
                                    (0, 0): II*0., (0, 3): II}, common_legs=(1, 2))
            elif n == max(NS_sites):
                tmp = tensor.block({(-3, 0): II,
                                    (-2, 0): U*I,
                                    (-1, 0): U*R,
                                    (0, 0): II*0.}, common_legs=(1, 2))
            else:
                tmp = II
            H.A[n] = general.add_MPOs(H.A[n], tmp, common_legs=(
                1, 2)) if mS != NS_sites[0] else tmp
    return H


def stack_MPOs(A, B, transpose=[0, 0], conj=[0, 0]):
    if A and B:
        C = A.copy()
        for n in range(C.N):
            C.A[n] = general.stack_MPOs(A.A[n], B.A[n], transpose, conj)
        return C
    else:
        return A if A else B


def add_MPOs(A, B, common_legs=(1, 2)):
    if A and B:
        C = A.copy()
        for n in range(C.N):
            C.A[n] = general.add_MPOs(A.A[n], B.A[n], common_legs=common_legs)
        return C
    else:
        return A if A else B


# MEASURE


def identity(tensor_type, N, basis):
    vii, _, _, _, _ = general.generate_vectorized_basis(basis)
    ii = vector_into_Tensor(tensor_type, vii, 0)
    H = mps.Mps(N, nr_phys=1)
    for n in range(N):
        H.A[n] = ii
    return H


def measure_Op(tensor_type, N, id, Op, basis):
    vii, vnn, _, _, _ = general.generate_vectorized_basis(basis)
    ii = vector_into_Tensor(tensor_type, vii, 0)
    nn = vector_into_Tensor(tensor_type, vnn, 0)
    if Op == 'nn':
        Op = nn
    H = mps.Mps(N, nr_phys=1)
    for n in range(N):
        H.A[n] = Op if n == id else ii
    return H


def measure_sumOp(tensor_type, choice, LSR, basis, Op):
    # sum of particles in all elements choice==LSR
    N = len(LSR)
    vii, vnn, _, _, _ = general.generate_vectorized_basis(basis)
    ii = vector_into_Tensor(tensor_type, vii, 0)
    nn = vector_into_Tensor(tensor_type, vnn, 0)
    if Op == 'nn':
        Op = nn
    H = mps.Mps(N, nr_phys=1)
    for n in range(N):
        taken = Op if LSR[n] == choice else 0*ii
        if n == 0:
            H.A[n] = tensor.block({(0, 0): taken, (0, 1): ii}, common_legs=(1))
        elif n == N-1:
            H.A[n] = tensor.block(
                {(0, 0): taken, (-1, 0): ii}, common_legs=(1))
        else:
            H.A[n] = tensor.block(
                {(0, 0): taken, (0, 1): ii, (-1, 0): ii}, common_legs=(1))
    return H


def current_ccp(tensor_type, sign_from, id_to, sign_list, vk, direction, basis):
    vii, _, vc, vcp, vz = general.generate_vectorized_basis(basis)
    ii = vector_into_Tensor(tensor_type, vii, 0)
    z_for_ccp = vector_into_Tensor(tensor_type, vz, -1)
    z_for_cpc = vector_into_Tensor(tensor_type, vz, 1)
    c_right = vector_into_Tensor(tensor_type, vc, 0)
    cp_right = vector_into_Tensor(tensor_type, vcp, 0)
    c_left = vector_into_Tensor(tensor_type, vc, 1)
    cp_left = vector_into_Tensor(tensor_type, vcp, -1)
    oo = ii*0
    #
    N = len(sign_list)  # total number of sites
    if direction == +1:  # from sites of sign_from to id_to
        c1_right, c1_left = c_right, c_left
        c2_right, c2_left = cp_right, cp_left
        z1, z2 = z_for_ccp, z_for_cpc
    elif direction == -1:  # from id_to to sites of sign_from
        c2_right, c2_left = c_right, c_left
        c1_right, c1_left = cp_right, cp_left
        z2, z1 = z_for_ccp, z_for_cpc
    #
    H = mps.Mps(N, nr_phys=1)
    for n in range(N):
        v = (.5*1j)*vk[n] if sign_list[n] == sign_from else 0
        if n == 0:
            H.A[n] = tensor.block(
                {(0, 0): oo, (0, 1): v*c1_right, (0, 2): -v*c2_right, (0, 3): ii}, common_legs=(1))
        elif n < id_to and n != 0:
            H.A[n] = tensor.block({(-3, 0): ii,
                                   (-2, 1): z1,
                                   (-1, 2): z2,
                                   (0, 1): v*c1_right, (0, 2): -v*c2_right, (0, 3): ii}, common_legs=(1))
        elif n == id_to:
            H.A[n] = tensor.block({(-3, 0): ii,
                                   (-2, 0): c2_left,
                                   (-1, 0): c1_left,
                                   (0, 1): c1_right, (0, 2): c2_right, (0, 3): ii}, common_legs=(1))
        elif n > id_to and n != N - 1:
            H.A[n] = tensor.block({(-3, 0): ii,
                                   (-2, 0): -v*c2_left, (-2, 1): z1,
                                   (-1, 0): v*c1_left, (-1, 2): z2,
                                   (0, 3): ii}, common_legs=(1))
        elif n == N - 1:
            H.A[n] = tensor.block({(-3, 0): ii,
                                   (-2, 0): -v*c2_left,
                                   (-1, 0): v*c1_left,
                                   (0, 0): oo}, common_legs=(1))
    return H


def current_XY(tensor_type, sign_from, id_to, sign_list, vk, direction, basis):
    vii, _, vc, vcp, vz = general.generate_vectorized_basis(basis)
    ii = vector_into_Tensor(tensor_type, vii.real, 0)
    x_right = vector_into_Tensor(tensor_type, (vcp+vc).real, 0)
    y_right = vector_into_Tensor(tensor_type, (1j*(vcp-vc)).real, 0)
    x_left = vector_into_Tensor(tensor_type, (vcp+vc).real, 1)
    y_left = vector_into_Tensor(tensor_type, (1j*(vcp-vc)).real, 1)
    z = vector_into_Tensor(tensor_type, vz.real, 1)
    oo = ii*0
    #
    N = len(sign_list)  # total number of sites
    if direction == +1:  # from sites of sign_from to id_to
        c1_right, c1_left = x_right, x_left
        c2_right, c2_left = y_right, y_left
    elif direction == -1:  # from id_to to sites of sign_from
        c2_right, c2_left = x_right, x_left
        c1_right, c1_left = y_right, y_left
    #
    H = mps.Mps(N, nr_phys=1)
    for n in range(N):
        v = .25*vk[n] if sign_list[n] == sign_from else 0
        if n == 0:
            H.A[n] = tensor.block(
                {(0, 0): oo, (0, 1): v*c1_right, (0, 2): -v*c2_right, (0, 3): ii}, common_legs=(1))
        elif n < id_to and n != 0:
            H.A[n] = tensor.block({(-3, 0): ii,
                                   (-2, 1): z,
                                   (-1, 2): z,
                                   (0, 1): v*c1_right, (0, 2): -v*c2_right, (0, 3): ii}, common_legs=(1))
        elif n == id_to:
            H.A[n] = tensor.block({(-3, 0): ii,
                                   (-2, 0): c2_left,
                                   (-1, 0): c1_left,
                                   (0, 1): c1_right, (0, 2): c2_right, (0, 3): ii}, common_legs=(1))
        elif n > id_to and n != N - 1:
            H.A[n] = tensor.block({(-3, 0): ii,
                                   (-2, 0): -v*c2_left, (-2, 1): z,
                                   (-1, 0): v*c1_left, (-1, 2): z,
                                   (0, 3): ii}, common_legs=(1))
        elif n == N - 1:
            H.A[n] = tensor.block({(-3, 0): ii,
                                   (-2, 0): -v*c2_left,
                                   (-1, 0): v*c1_left,
                                   (0, 0): oo}, common_legs=(1))
    return H


# WORKING ON TENSORS


def vector_into_Tensor(tensor_type, np_matrix, left_virtual):
    if tensor_type[0] == 'full':
        return cast_into_Tensor(settings=settings_full, s=(1, 1, -1), dims_chrgs=[(range(4), 0)], np_matrix=np_matrix, dtype=tensor_type[1])
    elif tensor_type[0] == 'Z2':
        return cast_into_Tensor(settings=settings_Z2, s=(1, 1, -1), dims_chrgs=[(range(2), 0), (range(2, 4), 1)], np_matrix=np_matrix, left_virtual=abs(left_virtual), cycle=2, n=0, dtype=tensor_type[1])
    elif tensor_type[0] == 'U1':
        return cast_into_Tensor(settings=settings_U1, s=(1, 1, -1), dims_chrgs=[(range(2), 0), ([2], -1), ([3], 1)], np_matrix=np_matrix, left_virtual=left_virtual, n=0, dtype=tensor_type[1])


def operator_into_Tensor(tensor_type, np_matrix, left_virtual):
    if tensor_type[0] == 'full':
        return cast_into_Tensor(settings=settings_full, s=(1, 1, -1, -1), dims_chrgs=[(range(4), 0)], np_matrix=np_matrix, dtype=tensor_type[1])
    elif tensor_type[0] == 'Z2':
        return cast_into_Tensor(settings=settings_Z2, s=(1, 1, -1, -1), dims_chrgs=[(range(2), 0), (range(2, 4), 1)], np_matrix=np_matrix, left_virtual=abs(left_virtual), cycle=2, n=0, dtype=tensor_type[1])
    elif tensor_type[0] == 'U1':
        return cast_into_Tensor(settings=settings_U1, s=(1, 1, -1, -1), dims_chrgs=[(range(2), 0), ([2], -1), ([3], 1)], np_matrix=np_matrix, left_virtual=left_virtual, n=0, dtype=tensor_type[1])


def cast_into_Tensor(settings, dtype, s, dims_chrgs, np_matrix, left_virtual=0, cycle=None, n=None):
    r"""
    Build 3 or 4 legged Tensor out of numpy array. Analogous to MPS or MPO tensors.

    Parameters
    ----------
    settings: module
        Module for tensor settings.
    s: tuple
        Signs of leggs.
    n: int
        Tensor charge.
        default = 0
    dims_chrgs: list of 2-element tuple
        List of 2-elements tuples with range for section and charge of a virtual legs.
    cycle: int
        adjust right virtual charge modulo cycle.
    np_matrix: numpy array
        Full version of an operator. Here is 4x4 matrix.
    left_virtual:  int
        Charge on left virtual leg.
    """
    settings.dtype = dtype
    Op = tensor.Tensor(settings=settings, s=s, n=n)
    if len(s) == 3:
        for iL, chL in dims_chrgs:
            mat = np_matrix[iL]
            if not np. all((mat == 0)):
                if n == None:
                    ts = ()
                else:
                    ts = (left_virtual, chL, int(-n+left_virtual+chL) %
                          cycle) if cycle else (left_virtual, chL, int(-n+left_virtual+chL))
                Op.set_block(ts=ts, val=mat, Ds=(1, len(iL), 1))
                del mat
    elif len(s) == 4:
        for iL, chL in dims_chrgs:
            for iR, chR in dims_chrgs:
                mat = np_matrix[:, iR][iL, :]
                if not np. all((mat == 0)):
                    if n == None:
                        ts = ()
                    else:
                        ts = (left_virtual, chL, chR, int(-n+left_virtual+chL-chR) %
                              cycle) if cycle else (left_virtual, chL, chR, int(-n+left_virtual+chL-chR))
                    Op.set_block(ts=ts, val=mat, Ds=(1, len(iL), len(iR), 1))
                    del mat
    return Op


def save_psi_to_h5py(big_file, psi):
    direction = 'state/'
    for n in range(psi.N):
        direction_g = direction + str(n) + '/'
        to_dict = psi.A[n].to_dict()
        try:
            big_file.create_group(direction_g)
        except:
            pass
        for inm, ival in to_dict.items():
            if type(ival) is dict:
                direction_sb = direction_g + inm + '/'
                it2 = 0
                for inm2, ival2 in ival.items():
                    try:
                        big_file.create_group(direction_sb+str(it2)+'/')
                    except:
                        pass
                    try:
                        big_file.create_dataset(
                            direction_sb+str(it2)+'/'+'block', data=inm2)
                        big_file.create_dataset(
                            direction_sb+str(it2)+'/'+'mat', data=ival2)
                    except:
                        del big_file[direction_sb+str(it2)+'/'+'block']
                        del big_file[direction_sb+str(it2)+'/'+'mat']
                        big_file.create_dataset(
                            direction_sb+str(it2)+'/'+'block', data=inm2)
                        big_file.create_dataset(
                            direction_sb+str(it2)+'/'+'mat', data=ival2)
                    it2 += 1
            else:
                try:
                    big_file.create_dataset(direction_g+str(inm), data=[ival])
                except:
                    del big_file[direction_g+str(inm)]
                    big_file.create_dataset(direction_g+str(inm), data=[ival])


def import_psi_from_h5py(big_file, tensor_type):
    if tensor_type[0] == 'full':
        settings = settings_full
    elif tensor_type[0] == 'Z2':
        settings = settings_Z2
    elif tensor_type[0] == 'U1':
        settings = settings_U1
    settings_full.dtype = tensor_type[1]
    direction = 'state/'
    g_mps = big_file.get(direction)
    N = len(g_mps.items())
    psi = mps.Mps(N, nr_phys=1)
    it = 0
    for inm, _ in g_mps.items():
        d = {}
        g_A = big_file.get(direction+inm)
        # GET A
        d_A = {}
        g_mat = big_file.get(direction+inm+'/A/')
        for _, ival3 in g_mat.items():
            block = tuple(ival3.get('block')[:])
            mat = ival3.get('mat')[:]
            d_A[block] = mat
        d['A'] = d_A
        # GET OTHERS
        d['s'] = g_A.get('s')[:][0]
        d['n'] = g_A.get('n')[:][0]
        d['isdiag'] = g_A.get('isdiag')[:][0]
        #
        psi.A[it] = tensor.from_dict(settings=settings, d=d)
        it += 1
    return psi
