import logging
import yastn
from yastn.tn.fpeps.operators.gates import trivial_tensor
from yastn import tensordot, swap_gate, fuse_legs, ncon

###################################
##### creating ntu environment ####
###################################

def env_NTU(peps, bd, QA, QB, dirn):
    r"""
    Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.

    Parameters
    ----------
    peps : class PEPS

    bd  : class Bond
        bond around which the tensor environment is to be calculated

    QA : yastn.Tensor
        Fixed isometry to be attached to the metric positioned on the left/top side of the bond.
        Input obtained from QR decomposition of the PEPS to the left/top of the bond being optimized

    QB : yastn.Tensor
        Fixed isometry to be attached to the metric positioned on the right/bottom side of the bond.
        Input obtained from QR decomposition of the PEPS to the right/bottom of the bond being optimized

    dirn : str
        The direction of the bond. Can be "h" (horizontal) or "v" (vertical).

    Returns
    -------
    Tensor
        The environment tensor g .
    """

    env = peps.tensors_NtuEnv(bd)
    G={}
    for ms in env.keys():
        if env[ms] is None:
            leg = peps[(0, 0)].get_legs(axes=-1)
            leg, _ = yastn.leg_undo_product(leg) # last leg of A should be fused
            fid = yastn.eye(config=peps[(0,0)].config, legs=[leg, leg.conj()]).diag()
            G[ms] = trivial_tensor(fid)
        else:
            G[ms] = peps[env[ms]]

    if dirn == "h":
        m_tl, m_l, m_bl = con_tl(G['tl']), con_l(G['l']), con_bl(G['bl'])
        m_tr, m_r, m_br = con_tr(G['tr']), con_r(G['r']), con_br(G['br'])
        env_l = con_Q_l(QA, m_l)  # [t t'] [b b'] [rr rr']
        env_r = con_Q_r(QB, m_r)  # [ll ll'] [t t'] [b b']
        env_t = m_tl @ m_tr  # [tl tl'] [tr tr']
        env_b = m_br @ m_bl  # [br br'] [bl bl']
        g = ncon((env_t, env_l, env_r, env_b), ((1, 4), (1, 3, -1), (-2, 4, 2), (2, 3)))  # [ll ll'] [rr rr']
    elif dirn == "v":
        m_tl, m_t, m_tr = con_tl(G['tl']), con_t(G['t']), con_tr(G['tr'])
        m_bl, m_b, m_br = con_bl(G['bl']), con_b(G['b']), con_br(G['br'])
        env_t = con_Q_t(QA, m_t)  # [l l'] [r r'] [bb bb']
        env_b = con_Q_b(QB, m_b)  # [tt tt'] [l l'] [r r']
        env_l = m_bl @ m_tl  # [bl bl'] [tl tl']
        env_r = m_tr @ m_br  # [tr tr'] [br br']
        g = ncon((env_l, env_t, env_b, env_r), ((4, 1), (1, 3, -1), (-2, 4, 2), (3, 2)))  # [tt tt'] [bb bb']
    return g.unfuse_legs(axes=(0, 1))


def con_tl(A):  # A -> [t l] [b r] s
    """ top-left env tensor """
    A = fuse_legs(A, axes=((0, 2), 1))  # [[t l] s] [b r]
    m_tl = tensordot(A, A, axes=(0, 0), conj=(0, 1))  # [b r] [b' r']
    m_tl = m_tl.unfuse_legs(axes=(0, 1))  # b r b' r'
    m_tl = m_tl.swap_gate(axes=((0, 2), 3))  # b b' X r'
    m_tl = m_tl.fuse_legs(axes=((0, 2), (1, 3)))  # [b b'] [r r']
    return m_tl


def con_tr(A):  # A = [t l] [b r] s
    """ top-right env tensor """
    A = A.unfuse_legs(axes=(0, 1))
    A = swap_gate(A, axes=(0, 1, 2, 3))  # t X l, b X r
    A = fuse_legs(A, axes=((1, 2), (0, 3, 4)))  # [l b] [t r s]
    m_tr = tensordot(A, A, axes=(1, 1), conj=(0, 1))  # [l b] [l' b']
    m_tr = m_tr.unfuse_legs(axes=(0, 1))  # l b l' b'
    m_tr = m_tr.fuse_legs(axes=((0, 2), (1, 3)))  # [l l'] [b b']
    return m_tr


def con_br(A):  # A = [t l] [b r] s
    """ bottom-right env tensor """
    A = fuse_legs(A, axes=(0, (1, 2)))  # [t l] [[b r] s]
    m_br = tensordot(A, A, axes=(1, 1), conj=(0, 1))  # [t l] [t' l']
    m_br = m_br.unfuse_legs(axes=(0, 1))  # t l t' l'
    m_br = m_br.swap_gate(axes=((1, 3), 2))  # l l' X t'
    m_br = m_br.fuse_legs(axes=((0, 2), (1, 3)))  # [t t'] [l l']
    return m_br


def con_bl(A):  # A = [t l] [b r] s
    """ bottom-left env tensor """
    A = A.unfuse_legs(axes=(0, 1))  # t l b r s
    A = fuse_legs(A, axes=((0, 3), (1, 2, 4)))  # [t r] [b l s]
    m_bl = tensordot(A, A, axes=(1, 1), conj=(0, 1))  # [t r] [t' r']
    m_bl = m_bl.unfuse_legs(axes=(0, 1))  # t r t' r'
    m_bl = m_bl.fuse_legs(axes=((1, 3), (0, 2)))  # [r r'] [t t']
    return m_bl


def con_l(A):  # A = [t l] [b r] s
    """ left env tensor """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    A = A.fuse_legs(axes=((0, 1, 3), 2))  # [[t l] b s] r
    m_l = tensordot(A, A, axes=(0, 0), conj=(1, 0))  # r' r
    return m_l


def con_r(A):  # A = [t l] [b r] s
    """ right env tensor """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    A = A.fuse_legs(axes=(1, (0, 2, 3)))  # l [t [b r] s]
    m_r = tensordot(A, A, axes=(1, 1), conj=(1, 0))  # [l' l]
    return m_r


def con_b(A):  # A = [t l] [b r] s
    """ bottom env tensor """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    A = A.fuse_legs(axes=(0, (1, 2, 3)))  # t [l [b r] s]
    m_b = tensordot(A, A, axes=(1, 1), conj=(1, 0))  # [t' t]
    return m_b


def con_t(A):
    """ top env tensor """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    A = A.fuse_legs(axes=((0, 2, 3), 1))  # [[t l] r s] b
    m_t = tensordot(A, A, axes=(0, 0), conj=(1, 0))  # b' b
    return m_t



def con_Q_l(QA, m_l):  # QA -> [[[t l] s] b] rr
    """ env_l where left legs of double-layer Q tensors are contracted with m_l """
    QA = QA.unfuse_legs(axes=0)  # [[t l] s] b rr
    QA = QA.swap_gate(axes=(1, 2))  # b X rr
    QA = QA.fuse_legs(axes=(0, (1, 2)))   # [[t l] s] [b rr]
    QA = QA.unfuse_legs(axes=0)  # [t l] s [b rr]
    QA = QA.unfuse_legs(axes=0)  # t l s [b rr]
    QA = QA.fuse_legs(axes=(1, 2, (0, 3)))  # l s [t [b rr]]
    mlQA = m_l @ QA  # l' s [t [b rr]]
    env_l = tensordot(mlQA, QA, axes=((0, 1), (0, 1)), conj=(0, 1))  # [t [b rr]] [t' [b' rr']]
    env_l = env_l.unfuse_legs(axes=(0, 1))  # t [b rr] t' [b' rr']
    env_l = env_l.fuse_legs(axes=((0, 2), 1, 3))  # [t t'] [b rr] [b' rr']
    env_l = env_l.unfuse_legs(axes=(1, 2))  # [t t'] b rr b' rr'
    env_l = env_l.swap_gate(axes=(1, (2, 4)))  # b x rr rr'
    env_l = env_l.fuse_legs(axes=(0, (1, 3), (2, 4)))  # [t t'] [b b'] [rr rr']
    return env_l


def con_Q_r(QB, m_r):  # QB -> ll [t [[b r] s]]
    """ env_r where right legs of double-layer Q tensors are contracted with m_r """
    QB = QB.unfuse_legs(axes=1)  # ll t [[b r] s]
    QB = QB.fuse_legs(axes=((0, 1), 2))  # [ll t] [[b r] s]
    QB = QB.unfuse_legs(axes=1)  # [ll t] [b r] s
    QB = QB.unfuse_legs(axes=1)  # [ll t] b r s
    QB = QB.swap_gate(axes=(1, 2))  # b X r
    QB = QB.fuse_legs(axes=(2, 3, (0, 1)))  # r s [[ll t] b]
    mrsQB = m_r @ QB  # r' s [ll [t b]]
    env_r = tensordot(mrsQB, QB, axes=((0, 1), (0, 1)), conj=(0, 1))  # [[ll t] b] [[ll' t'] b']
    env_r = env_r.unfuse_legs(axes=(0, 1))  # [ll t] b [ll' t'] b'
    env_r = env_r.fuse_legs(axes=(0, 2, (1, 3)))  # [ll t] [ll' t'] [b b']
    env_r = env_r.unfuse_legs(axes=(0, 1))  # ll t ll' t' [b b']
    env_r = env_r.swap_gate(axes=((0, 2), 3))  # ll ll' X t'
    env_r = env_r.fuse_legs(axes=((0, 2), (1, 3), 4))  # [ll ll'] [t t'] [b b']
    return env_r


def con_Q_t(QA, m_t):  # QA -> [[[t l] s] r] bb
    """ env_t where top legs of double-layer Q tensors are contracted with m_t """
    QA = QA.unfuse_legs(axes=0)  # [[t l] s] r bb
    QA = QA.fuse_legs(axes=(0, (1, 2)))  # [[t l] s] [r bb]
    QA = QA.unfuse_legs(axes=0)  # [t l] s [r bb]
    QA = QA.unfuse_legs(axes=0)  # t l s [r bb]
    QA = QA.swap_gate(axes=(0, 1))  # t X l
    QA = QA.fuse_legs(axes=(0, 2, (1, 3)))  # t s [l [r bb]]
    mtsQA = m_t @ QA  # t' s [l [r bb]]
    env_t = tensordot(mtsQA, QA, axes=((0, 1), (0, 1)), conj=(0, 1))  # [l [r bb]] [l' [r' bb']]
    env_t = env_t.unfuse_legs(axes=(0, 1))  # l [r bb] l' [r' bb']
    env_t = env_t.fuse_legs(axes=((0, 2), 1, 3))  # [l l'] [r bb] [r' bb']
    env_t = env_t.unfuse_legs(axes=(1, 2))  # [l l'] r bb r' bb'
    env_t = env_t.swap_gate(axes=(3, (2, 4)))  # r' X bb bb'
    env_t = env_t.fuse_legs(axes=(0, (1, 3), (2, 4))) # [l l'] [r r'] [bb bb']
    return env_t


def con_Q_b(QB, m_b):  # QB -> t [l [[b r] s]]
    """ env_t where top legs of double-layer Q tensors are contracted with m_t """
    QB = QB.unfuse_legs(axes=1)  # tt l [[b r] s]
    QB = QB.swap_gate(axes=(0, 1))  # tt X l
    QB = QB.fuse_legs(axes=((0, 1), 2))  # [tt l] [[b r] s]
    QB = QB.unfuse_legs(axes=1)  # [tt l] [b r] s
    QB = QB.unfuse_legs(axes=1)  # [tt l] b r s
    QB = QB.fuse_legs(axes=(1, 3, (0, 2)))  # b s [[tt l] r]
    mbQB = m_b @ QB  # b' s [[tt l] r]
    env_b = tensordot(mbQB, QB, axes=((0, 1), (0, 1)), conj=(0, 1))  # [[tt l] r] [[tt' l'] r']
    env_b = env_b.unfuse_legs(axes=(0, 1))  # [tt l] r [tt' l'] r'
    env_b = env_b.fuse_legs(axes=(0, 2, (1, 3)))  # [tt l] [tt' l'] [r r']
    env_b = env_b.unfuse_legs(axes=(0, 1))  # tt l tt' l' [r r']
    env_b = env_b.swap_gate(axes=((0, 2), 1)) # tt tt' X l
    env_b = env_b.fuse_legs(axes=((0, 2), (1, 3), 4))  # [tt tt'] [l l'] [r r']
    return env_b


