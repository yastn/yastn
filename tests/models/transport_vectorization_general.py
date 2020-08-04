import numpy as np
from yamps.tensor.ncon import ncon


def generate_discretization(NL, w0, wS, mu, v, dV, tempL, tempR, method, ordered, gamma):
    muL = +0.5 * mu
    muR = -0.5 * mu

    if method == 0:  # spatial 1d to energy modes via sine-transformation.
        kk = np.arange(1, NL + 1, 1)
        ww = 2. * w0 * np.cos(kk * np.pi / (NL + 1))
        vk = np.sqrt(2. / (NL + 1.)) * v * np.sin(kk * np.pi / (NL + 1.))

        LSR = np.concatenate(
            (-1 + np.zeros(NL), np.array([2]), +1 + np.zeros(NL)))  # -1,0,1 = L,S,R
        wk = np.concatenate((ww - muL, np.array([0]), ww - muR + 1e-14))
        vk = np.concatenate((vk, np.array([0]), vk))
        dV = np.concatenate(
            (-dV * .5 + np.zeros(NL), np.array([0]), +dV * .5 + np.zeros(NL)))
        temp = np.concatenate(
            (tempL + np.zeros(NL), np.array([0]), tempR + np.zeros(NL)))
        gamma = np.zeros(NL*2+1)+gamma
        #
        wk_tmp = np.concatenate((ww, np.array([0]), ww + 1e-14))
    elif method == 1:  # Minimal model with uniform spacing and constant coupling to S
        W = 4*w0
        dw = W/NL
        ww = -W*.5+dw*.5+np.arange(NL)*dw
        vk = np.zeros(NL)+v

        LSR = np.concatenate(
            (-1 + np.zeros(NL), np.array([2]), +1 + np.zeros(NL)))  # -1,0,1 = L,S,R
        wk = np.concatenate((ww - muL, np.array([0]), ww - muR + 1e-14))
        vk = np.concatenate((vk, np.array([0]), vk))
        dV = np.concatenate(
            (-dV * .5 + np.zeros(NL), np.array([0]), +dV * .5 + np.zeros(NL)))
        temp = np.concatenate(
            (tempL + np.zeros(NL), np.array([0]), tempR + np.zeros(NL)))
        gamma = np.zeros(NL*2+1)+gamma
        #
        wk_tmp = np.concatenate((ww, np.array([0]), ww + 1e-14))

    if ordered:  # sort by energy before applying mu
        id = np.argsort(wk_tmp)
        LSR = LSR[id]
        wk = wk[id]
        vk = vk[id]
        dV = dV[id]
        temp = temp[id]
        gamma = gamma[id]

    wk[list(LSR).index(2)] = wS

    return LSR, wk, temp, vk, dV, gamma


def generate_operator_basis(basis):
    if basis == 0:  # choose I, Z, X, Y basis
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
    elif basis == 1:  # choose cp c, c cp, c, cp
        OO = np.zeros((4, 4), dtype=np.complex128)

        II = np.identity(4)

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
        q_c[1, -1] = 1

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

    return OO, II, q_z, z_q, q_n, n_q, q_ccp, ccp_q, c_q_cp, cp_q_c, z_q_z, c_q, cp_q, q_c, q_cp, ccp_q__p__q_ccp, n_q__p__q_n, m1j_n_q__m__q_n


def generate_vectorized_basis(basis):
    if basis == 0:  # choose I, Z, X, Y basis
        vII = np.array([1, 0, 0, 0])
        vnn = np.array([.5, -.5, 0, 0])
        vc = np.array([0, 0, 1, 1j])
        vcp = np.array([0, 0, 1, -1j])
        vz = np.array([0, 1, 0, 0])
    elif basis == 1:  # choose cp c, c cp, c, cp
        vII = np.array([1, 1, 0, 0])
        vnn = np.array([1, 0, 0, 0])
        vc = np.array([0, 0, 1, 0])
        vcp = np.array([0, 0, 0, 1])
        vz = np.array([-1, +1, 0, 0])
    return vII, vnn, vc, vcp, vz


def stack_MPOs(UP, DOWN):
    for it in range(UP.N):
        tmp = ncon([DOWN.A[it], UP.A[it]], [[-1, 1, -3, -5], [-2, 1, -4, -6]], [1, 0])
        tmp, _ = tmp.group_legs(axes=(4, 5), new_s=-1)
        UP.A[it], _ = tmp.group_legs(axes=(0, 1), new_s=1)
    return UP


# SAVE TO FILE
def save_to_file(names, vals, file_name):
    data = {}
    for it in range(len(names)):
        data.update({names[it]: vals[it]})
    np.save(file_name, data, 'a')
