import numpy as np
from yamps.tensor.ncon import ncon
import yamps.mps.measure as measure
from AIM_transport import basis as basis

# Majorana basis: I, Z, X, Y
# Dirac basis: cp_c, c_cp, c, cp


def generate_discretization(Nj, wj, HS, muj, dVj, tempj, vj, method, ordered, gamma, Fcoup=1, Frel=1):
    """
    Nj: list of ints
        Nj = [NL, NS, NR]
        Number of sites in the L/S/R regions for spatial_1D
        .. or ...
        Nj = [TscaleL, NS, TscaleR] for
    wj: list of floats
        wj = [wL, wR]
        Hopping amplitude in non-interacting L/R lead
    HS: NSxNS numpy array
        Interation within impurity. written in (c_dag_S1, c_dag_S2, ...) (.) (c_S1, c_S2, ...)^T
    muj: list
        muj = [muL, muR]
        Chemical potentials in L/R leads.
    dVj: list
        dVj = [dVL, dVR]
        Bias voltage in L/R leads.
    tempj: list
        tempj = [tempL, tempR]
        Temperature in L/R leads.
    vj: list
        vj = [v_1, v_2, ..]
        Coupling of each impurity site to the L.
    lambdaj: list
        lambdaj = [lambda_1, lambda_2, ..]
        Rescaling factor for coupling of each impurity site to the R.
    """
    wL, wR = wj[0], wj[-1]
    muL, muR = muj[0], muj[-1]
    dVL, dVR = dVj[0], dVj[-1]
    tempL, tempR = tempj[0], tempj[-1]
    if method == 'influence_LinLin':  # linear-linear influence
        TscaleL, NS, TscaleR = Nj[0], Nj[1], Nj[-1]

        def Influence(
            w, W, u): return basis.influence_functions.influence_LinLin(w, W, u)
        NL, scaleL, wkL, dkL, vkL, gkL = basis.wdk_AS(
            Influence=Influence, Tscale=TscaleL, W=4.*wL, u=abs(muR-muL), Fcoup=Fcoup, Frel=Frel)
        NR, scaleR, wkR, dkR, vkR, gkR = basis.wdk_AS(
            Influence=Influence, Tscale=TscaleR, W=4.*wR, u=abs(muR-muL), Fcoup=Fcoup, Frel=Frel)
    elif method == 'influence_LinInv':  # linear-inv influence
        TscaleL, NS, TscaleR = Nj[0], Nj[1], Nj[-1]

        def Influence(
            w, W, u): return basis.influence_functions.influence_LinInv(w, W, u)
        NL, scaleL, wkL, dkL, vkL, gkL = basis.wdk_AS(
            Influence=Influence, Tscale=TscaleL, W=4.*wL, u=abs(muR-muL), Fcoup=Fcoup, Frel=Frel)
        NR, scaleR, wkR, dkR, vkR, gkR = basis.wdk_AS(
            Influence=Influence, Tscale=TscaleR, W=4.*wR, u=abs(muR-muL), Fcoup=Fcoup, Frel=Frel)
    elif method == 'influence_LinInv1D':  # linear-inv influence 1D
        TscaleL, NS, TscaleR = Nj[0], Nj[1], Nj[-1]

        def Influence(
            w, W, u): return basis.influence_functions.influence_LinInv1D(w, W, u)
        NL, scaleL, wkL, dkL, vkL, gkL = basis.wdk_AS(
            Influence=Influence, Tscale=TscaleL, W=4.*wL, u=abs(muR-muL), Fcoup=Fcoup, Frel=Frel)
        NR, scaleR, wkR, dkR, vkR, gkR = basis.wdk_AS(
            Influence=Influence, Tscale=TscaleR, W=4.*wR, u=abs(muR-muL), Fcoup=Fcoup, Frel=Frel)
    elif method == 'influence_LinLog':  # linear-inv influence
        TscaleL, NS, TscaleR = Nj[0], Nj[1], Nj[-1]

        def Influence(
            w, W, u): return basis.influence_functions.influence_LinLog(w, W, u)
        NL, scaleL, wkL, dkL, vkL, gkL = basis.wdk_AS(
            Influence=Influence, Tscale=TscaleL, W=4.*wL, u=abs(muR-muL), Fcoup=Fcoup, Frel=Frel)
        NR, scaleR, wkR, dkR, vkR, gkR = basis.wdk_AS(
            Influence=Influence, Tscale=TscaleR, W=4.*wR, u=abs(muR-muL), Fcoup=Fcoup, Frel=Frel)
    elif method == 'influence_LinLog1D':  # linear-inv influence 1D
        TscaleL, NS, TscaleR = Nj[0], Nj[1], Nj[-1]
        def Influence(
            w, W, u): return basis.influence_functions.influence_LinLog1D(w, W, u)
        NL, scaleL, wkL, dkL, vkL, gkL = basis.wdk_AS(
            Influence=Influence, Tscale=TscaleL, W=4.*wL, u=abs(muR-muL), Fcoup=Fcoup, Frel=Frel)
        NR, scaleR, wkR, dkR, vkR, gkR = basis.wdk_AS(
            Influence=Influence, Tscale=TscaleR, W=4.*wR, u=abs(muR-muL), Fcoup=Fcoup, Frel=Frel)
    else:  # spatial 1d to energy modes via sine-transformation.
        NL, NS, NR = Nj[0], Nj[1], Nj[-1]
        kL = np.arange(1, NL + 1, 1)
        kR = np.arange(1, NR + 1, 1)
        wkL = 2. * wL * np.cos(kL * np.pi / (NL + 1))
        wkR = 2. * wR * np.cos(kR * np.pi / (NR + 1))
        vkL = np.sqrt(2. / (NL + 1.)) * np.sin(kL * np.pi / (NL + 1.))
        vkR = np.sqrt(2. / (NR + 1.)) * np.sin(kR * np.pi / (NR + 1.))
        gkL = np.zeros(NL)+gamma
        gkR = np.zeros(NR)+gamma

    N = NL+NS+NR
    LSR = np.zeros(N)
    LSR[:NL] += -1
    LSR[NL:NL+NS] += 2
    LSR[NL+NS:] += 1

    gamma = np.zeros(N)
    gamma[:NL] = gkL
    gamma[NL+NS:] = gkR

    temp = np.zeros(N)
    temp[:NL] += tempL
    temp[NL+NS:] += tempR

    dV = np.zeros(N)
    dV[:NL] += dVL
    dV[NL+NS:] += dVR

    corr = np.diag(np.concatenate((wkL+muL, np.zeros(NS), wkR+muR)))
    corr[NL:NL+NS, NL:NL+NS] = np.tril(HS)
    for n in range(NS):
        corr[NL+n, :NL] = vkL * vj[n][0]
        corr[NL+n, NL+NS:] = vkR * vj[n][1]
        corr[:NL, NL+n] = vkL * vj[n][0]
        corr[NL+NS:, NL+n] = vkR * vj[n][1]

    if ordered:  # sort by energy, leave impurity ordering intact
        id = np.argsort(np.concatenate(
            (wkL + muL - 1e-14, np.arange(NS)*1e-16, wkR + muR+1e-14)))
        LSR = LSR[id]
        dV = dV[id]
        temp = temp[id]
        gamma = gamma[id]
        corr = corr[id, :][:, id]

    return LSR, temp, dV, gamma, corr


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
        tmp = ncon([DOWN.A[it], UP.A[it]], [
                   [-1, 1, -3, -5], [-2, 1, -4, -6]], [1, 0])
        tmp, _ = tmp.group_legs(axes=(4, 5), new_s=-1)
        new.A[it], _ = tmp.group_legs(axes=(0, 1), new_s=1)
    return new


def stack_mpo_mps(mpo, mps):
    for it in range(mps.N):
        tmp = ncon([mps.A[it], mpo.A[it]], [
                   [-2, 1, -5], [-1, -3, 1, -4]], [0, 0])
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
        norm = measure.measure_overlap(norm, psi)
    else:
        norm = 1.
    out = np.zeros(len(list_of_ops))
    for n in range(len(out)):
        out[n] = measure.measure_overlap(bra=list_of_ops[n], ket=psi)
    return out/norm, norm


def measure_MPOs(psi, list_of_ops):
    norm = measure.measure_overlap(bra=psi, ket=psi)
    out = np.zeros(len(list_of_ops))
    for n in range(len(out)):
        out[n] = measure.measure_mpo(bra=psi, op=list_of_ops[n], ket=psi)
    return out/norm, norm
