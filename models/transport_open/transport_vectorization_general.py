import numpy as np
from yamps.tensor.ncon import ncon
import yamps.mps.measure as measure

# Majorana basis: I, Z, X, Y
# Dirac basis: cp_c, c_cp, c, cp

def generate_discretization(NL, w0, wS, mu, v, dV, tempL, tempR, method, ordered, gamma):
    if method == 0:  # spatial 1d to energy modes via sine-transformation.
        kk = np.arange(1, NL + 1, 1)
        ww = 2. * w0 * np.cos(kk * np.pi / (NL + 1))
        vk = np.sqrt(2. / (NL + 1.)) * v * np.sin(kk * np.pi / (NL + 1.))

        LSR = np.concatenate(
            (-1 + np.zeros(NL), np.array([2]), +1 + np.zeros(NL)))  # -1,0,1 = L,S,R
        wk = np.concatenate((ww - mu*.5, np.array([0]), ww + mu*.5 + 1e-14))
        vk = np.concatenate((vk, np.array([0]), vk))
        dV = np.concatenate((-dV * .5 + np.zeros(NL), np.array([0]), +dV * .5 + np.zeros(NL)))
        temp = np.concatenate(
            (tempL + np.zeros(NL), np.array([0]), tempR + np.zeros(NL)))
        gamma = np.zeros(NL*2+1)+gamma
        #
        wk_tmp = np.concatenate((ww - 1e-14, np.array([0]), ww + 1e-14))
    elif method == 1:  # Minimal model with uniform spacing and constant coupling to S
        W = 4*w0
        dw = W/NL
        ww = -W*.5+dw*.5+np.arange(NL)*dw
        vk = np.zeros(NL)+v

        LSR = np.concatenate(
            (-1 + np.zeros(NL), np.array([2]), +1 + np.zeros(NL)))  # -1,0,1 = L,S,R
        wk = np.concatenate((ww - mu*.5, np.array([0]), ww + mu*.5 + 1e-14))
        vk = np.concatenate((vk, np.array([0]), vk))
        dV = np.concatenate((-dV * .5 + np.zeros(NL), np.array([0]), +dV * .5 + np.zeros(NL)))
        temp = np.concatenate(
            (tempL + np.zeros(NL), np.array([0]), tempR + np.zeros(NL)))
        gamma = np.zeros(NL*2+1)+gamma
        #
        wk_tmp = np.concatenate((ww - 1e-14, np.array([0]), ww + 1e-14))

    if ordered:  # sort by energy before applying mu
        id = np.argsort(wk_tmp)
        LSR = LSR[id]
        wk = wk[id]
        vk = vk[id]
        dV = dV[id]
        temp = temp[id]
        gamma = gamma[id]

    wk[list(LSR).index(2)] = wS+1e-15

    return LSR, wk, temp, vk, dV, gamma


def generate_operator_basis(basis):
    if basis == 'Majorana':
        OO = np.zeros((4, 4), dtype=np.complex128)

        II = np.identity(4)

        q_z = np.block([[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1j],
                        [0, 0, -1j, 0]])

        z_q = np.block([[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, -1j],
                        [0, 0, 1j, 0]])

        q_x = np.block([[0, 0, 1, 0],
                        [0, 0, 0, -1j],
                        [1, 0, 0, 0],
                        [0, 1j, 0, 0]])

        x_q = np.block([[0, 0, 1, 0],
                        [0, 0, 0, 1j],
                        [1, 0, 0, 0],
                        [0, -1j, 0, 0]])

        q_y = np.block([[0, 0, 0, 1],
                        [0, 0, 1j, 0],
                        [0, -1j, 0, 0],
                        [1, 0, 0, 0]])

        y_q = np.block([[0, 0, 0, 1],
                        [0, 0, -1j, 0],
                        [0, 1j, 0, 0],
                        [1, 0, 0, 0]])

        n_q__p__q_n = np.block([[1, -1, 0, 0],
                                [-1, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        ccp_q__p__q_ccp = np.block([[1, 1, 0, 0],
                                    [1, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        m1j_n_q__m__q_n = np.block([[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, -1, 0]])

        z_q_z = np.block([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, -1]])

        c_q = .5*np.block([[0, 0, 1, +1j],
                          [0, 0, +1, 1j],
                          [1, -1, 0, 0],
                          [+1j, -1j, 0, 0]])

        cp_q = .5*np.block([[0, 0, 1, -1j],
                           [0, 0, -1, 1j],
                           [1, +1, 0, 0],
                           [-1j, -1j, 0, 0]])

        q_c = .5*np.block([[0, 0, 1, +1j],
                          [0, 0, -1, -1j],
                          [1, +1, 0, 0],
                          [+1j, 1j, 0, 0]])

        q_cp = .5*np.block([[0, 0, 1, -1j],
                           [0, 0, +1, -1j],
                           [1, -1, 0, 0],
                           [-1j, 1j, 0, 0]])

        c_q_cp = .5*np.block([[+1, -1, 0, 0],
                             [+1, -1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]])

        cp_q_c = .5*np.block([[+1, +1, 0, 0],
                             [-1, -1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]])

        n_q = .5*np.block([[1, -1, 0, 0],
                          [-1, 1, 0, 0],
                          [0, 0, 1, 1j],
                          [0, 0, -1j, 1]])

        q_n = .5*np.block([[1, -1, 0, 0],
                          [-1, 1, 0, 0],
                          [0, 0, 1, -1j],
                          [0, 0, 1j, 1]])

        q_ccp = .5*np.block([[1, 1, 0, 0],
                            [1, 1, 0, 0],
                            [0, 0, 1, 1j],
                            [0, 0, 1j, 1]])

        ccp_q = .5*np.block([[1, 1, 0, 0],
                            [1, 1, 0, 0],
                            [0, 0, 1, -1j],
                            [0, 0, -1j, 1]])
    elif basis == 'Dirac':
        OO = np.zeros((4, 4), dtype=np.complex128)

        II = np.identity(4, dtype=np.complex128)

        q_z = np.block([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
                        
        q_z = np.block([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])

        z_q = np.block([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, -1]])

        n_q__p__q_n = np.zeros((4, 4), dtype=np.complex128)
        n_q__p__q_n[0, 0] = 2
        n_q__p__q_n[2, 2] = 1
        n_q__p__q_n[-1, -1] = 1

        m1j_n_q__m__q_n = np.zeros((4, 4), dtype=np.complex128)
        m1j_n_q__m__q_n[2, 2] = -1
        m1j_n_q__m__q_n[-1, -1] = 1
        m1j_n_q__m__q_n *= (-1j)

        ccp_q__p__q_ccp = np.zeros((4, 4), dtype=np.complex128)
        ccp_q__p__q_ccp[1, 1] = 2
        ccp_q__p__q_ccp[2, 2] = 1
        ccp_q__p__q_ccp[-1, -1] = 1

        z_q_z = np.block([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, -1]])

        c_q = np.zeros((4, 4), dtype=np.complex128)
        c_q[2, 0] = 1
        c_q[1, -1] = 1

        cp_q = np.zeros((4, 4), dtype=np.complex128)
        cp_q[0, 2] = 1
        cp_q[-1, 1] = 1

        q_c = np.zeros((4, 4), dtype=np.complex128)
        q_c[2, 1] = 1
        q_c[0, -1] = 1

        q_cp = np.zeros((4, 4), dtype=np.complex128)
        q_cp[-1, 0] = 1
        q_cp[1, 2] = 1

        c_q_cp = np.zeros((4, 4), dtype=np.complex128)
        c_q_cp[1, 0] = 1

        cp_q_c = np.zeros((4, 4), dtype=np.complex128)
        cp_q_c[0, 1] = 1

        q_n = np.zeros((4, 4), dtype=np.complex128)
        q_n[0, 0] = 1
        q_n[2, 2] = 1

        n_q = np.zeros((4, 4), dtype=np.complex128)
        n_q[0, 0] = 1
        n_q[-1, -1] = 1

        q_ccp = np.zeros((4, 4), dtype=np.complex128)
        q_ccp[1, 1] = 1
        q_ccp[2, 2] = 1

        ccp_q = np.zeros((4, 4), dtype=np.complex128)
        ccp_q[1, 1] = 1
        ccp_q[-1, -1] = 1

        q_x = q_cp + q_c
        x_q = cp_q + c_q

        q_y = 1j*(q_cp - q_c)
        y_q = 1j*(cp_q - c_q)

    return OO, II, q_x, x_q, q_y, y_q, q_z, z_q, q_n, n_q, q_ccp, ccp_q, c_q_cp, cp_q_c, z_q_z, c_q, cp_q, q_c, q_cp, ccp_q__p__q_ccp, n_q__p__q_n, m1j_n_q__m__q_n


def generate_vectorized_basis(basis):
    if basis == 'Majorana':
        vII = np.array([1, 0, 0, 0])
        vnn = .5*np.array([1, -1, 0, 0])
        vc = .5*np.array([0, 0, 1, 1j])
        vcp = .5*np.array([0, 0, 1, -1j])
        vz = np.array([0, 1, 0, 0])
    elif basis == 'Dirac':
        vII = np.array([1, 1, 0, 0])
        vnn = np.array([1, 0, 0, 0])
        vc = np.array([0, 0, 1, 0])
        vcp = np.array([0, 0, 0, 1])
        vz = np.array([-1, +1, 0, 0])
    return vII, vnn, vc, vcp, vz


def stack_MPOs(UP, DOWN):
    new = UP.copy()
    for it in range(UP.N):
        tmp = ncon([DOWN.A[it], UP.A[it]], [[-1, 1, -3, -5], [-2, 1, -4, -6]], [1, 0])
        tmp, _ = tmp.group_legs(axes=(4, 5), new_s=-1)
        new.A[it], _ = tmp.group_legs(axes=(0, 1), new_s=1)
    return new


def stack_mpo_mps(mpo, mps):
    for it in range(mps.N):
        tmp = ncon([mps.A[it], mpo.A[it]], [[-2, 1, -5], [-1, -3, 1, -4]], [0, 0])
        tmp, _ = tmp.group_legs(axes=(3, 4), new_s=-1)
        mps.A[it], _ = tmp.group_legs(axes=(0, 1), new_s=1)
    return mps


# SAVE
def save_to_file(names, vals, file_name):
    data = {}
    for it in range(len(names)):
        data.update({names[it]: vals[it]})
    np.save(file_name, data, 'a')


def measure_overlaps(psi, list_of_ops, norm=None):
    if norm:
        norm = measure.measure_overlap(psi, norm)
    else:
        norm = 1.
    out = np.zeros(len(list_of_ops))
    for n in range(len(out)):
        out[n] = measure.measure_overlap(bra=psi, ket=list_of_ops[n])
    return out/norm, norm


def measure_MPOs(psi, list_of_ops):
    norm = measure.measure_overlap(bra=psi, ket=psi)
    out = np.zeros(len(list_of_ops))
    for n in range(len(out)):
        out[n] = measure.measure_mpo(bra=psi, op=list_of_ops[n], ket=psi)
    return out/norm, norm
