from yastn import fuse_legs, tensordot, swap_gate

__all__ = ['con_tl', 'con_tr', 'con_br', 'con_bl',
           'con_l', 'con_r', 'con_b', 'con_t',
           'con_Al', 'con_Ar', 'con_At', 'con_Ab']

def con_tl(A):  # A -> [t l] [b r] s
    """ top-left env tensor """
    A = fuse_legs(A, axes=((0, 2), 1))  # [[t l] s] [b r]
    mtl = tensordot(A, A.conj(), axes=(0, 0))  # [b r] [b' r']
    mtl = mtl.unfuse_legs(axes=(0, 1))  # b r b' r'
    mtl = mtl.swap_gate(axes=((0, 2), 3))  # b b' X r'
    mtl = mtl.fuse_legs(axes=((0, 2), (1, 3)))  # [b b'] [r r']
    return mtl


def con_tr(A):  # A = [t l] [b r] s
    """ top-right env tensor """
    A = A.unfuse_legs(axes=(0, 1))  # t l b r s
    A = swap_gate(A, axes=(0, 1, 2, 3))  # t X l, b X r
    A = fuse_legs(A, axes=((1, 2), (0, 3, 4)))  # [l b] [t r s]
    mtr = tensordot(A, A.conj(), axes=(1, 1))  # [l b] [l' b']
    mtr = mtr.unfuse_legs(axes=(0, 1))  # l b l' b'
    mtr = mtr.fuse_legs(axes=((0, 2), (1, 3)))  # [l l'] [b b']
    return mtr


def con_br(A):  # A = [t l] [b r] s
    """ bottom-right env tensor """
    A = fuse_legs(A, axes=(0, (1, 2)))  # [t l] [[b r] s]
    mbr = tensordot(A, A.conj(), axes=(1, 1))  # [t l] [t' l']
    mbr = mbr.unfuse_legs(axes=(0, 1))  # t l t' l'
    mbr = mbr.swap_gate(axes=((1, 3), 2))  # l l' X t'
    mbr = mbr.fuse_legs(axes=((0, 2), (1, 3)))  # [t t'] [l l']
    return mbr


def con_bl(A):  # A = [t l] [b r] s
    """ bottom-left env tensor """
    A = A.unfuse_legs(axes=(0, 1))  # t l b r s
    A = fuse_legs(A, axes=((0, 3), (1, 2, 4)))  # [t r] [b l s]
    mbl = tensordot(A, A.conj(), axes=(1, 1))  # [t r] [t' r']
    mbl = mbl.unfuse_legs(axes=(0, 1))  # t r t' r'
    mbl = mbl.fuse_legs(axes=((1, 3), (0, 2)))  # [r r'] [t t']
    return mbl


def con_l(A):  # A = [t l] [b r] s
    """ left env tensor """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    A = A.fuse_legs(axes=((0, 1, 3), 2))  # [[t l] b s] r
    ml = tensordot(A.conj(), A, axes=(0, 0))  # r r'
    return ml


def con_r(A):  # A = [t l] [b r] s
    """ right env tensor """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    A = A.fuse_legs(axes=(1, (0, 2, 3)))  # l [t [b r] s]
    mr = tensordot(A.conj(), A, axes=(1, 1))  # l l'
    return mr


def con_b(A):  # A = [t l] [b r] s
    """ bottom env tensor """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    A = A.fuse_legs(axes=(0, (1, 2, 3)))  # t [l [b r] s]
    mb = tensordot(A.conj(), A, axes=(1, 1))  # t t'
    return mb


def con_t(A):
    """ top env tensor """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    A = A.fuse_legs(axes=((0, 2, 3), 1))  # [[t l] r s] b
    m_t = tensordot(A.conj(), A, axes=(0, 0))  # b b'
    return m_t


def con_Al(A, ml=None):  # A = [t l] [b r] s
    """ env_l where left legs of double-layer A tensors are contracted with ml """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    if ml is None:
        A = A.fuse_legs(axes=((1, 3), (0, 2)))  # [l s] [t [b r]]
        env_l = tensordot(A, A.conj(), axes=(0, 0))  # [t [b r]] [t' [b' r']]
    else:
        A = A.fuse_legs(axes=(1, 3, (0, 2)))  # l s [t [b r]]
        mlA = ml @ A  # l' s [t [b r]]
        env_l = tensordot(mlA, A.conj(), axes=((0, 1), (0, 1)))  # [t [b r]] [t' [b' r']]
    env_l = env_l.unfuse_legs(axes=(0, 1))  # t [b r] t' [b' r']
    env_l = env_l.fuse_legs(axes=(1, 3, (0, 2)))  # [b r] [b' r'] [t t']
    env_l = env_l.unfuse_legs(axes=(0, 1))  # b r b' r' [t t']
    env_l = env_l.swap_gate(axes=((0, 2), 3))  # b b' X r'
    env_l = env_l.fuse_legs(axes=((0, 2), (1, 3), 4))  # [b b'] [r r'] [t t']
    return env_l


def con_Ar(A, mr=None):  # A = [t l] [b r] s
    """ env_r where right legs of double-layer A tensors are contracted with m_r """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    A = A.swap_gate(axes=(1, 2))  # b X r
    if mr is None:
        A = A.fuse_legs(axes=((2, 3), (0, 1)))  # [r s] [[t l] b]
        env_r = tensordot(A, A.conj(), axes=(0, 0))  # [[t l] b] [[t' l'] b']
    else:
        A = A.fuse_legs(axes=(2, 3, (0, 1)))  # r s [[t l] b]
        mrA = mr @ A  # r' s [[t l] b]
        env_r = tensordot(mrA, A.conj(), axes=((0, 1), (0, 1)))  # [[t l] b] [[t' l'] b']
    env_r = env_r.unfuse_legs(axes=(0, 1))  # [t l] b [t' l'] b'
    env_r = env_r.fuse_legs(axes=(0, 2, (1, 3)))  # [t l] [t' l'] [b b']
    env_r = env_r.unfuse_legs(axes=(0, 1))  # t l t' l' [b b']
    env_r = env_r.swap_gate(axes=((1, 3), 2))  # l l' X t'
    env_r = env_r.fuse_legs(axes=((0, 2), (1, 3), 4))  # [t t'] [l l'] [b b']
    return env_r


def con_At(A, mt=None):  # QA = [t l] [b r] s
    """ env_t where top legs of double-layer Q tensors are contracted with m_t """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    A = A.swap_gate(axes=(0, 1))  # t X l
    if mt is None:
        A = A.fuse_legs(axes=((0, 3), (1, 2)))  # [t s] [l [b r]]
        env_t = tensordot(A, A.conj(), axes=(0, 0))  # [l [b r]] [l' [b' r']]
    else:
        A = A.fuse_legs(axes=(0, 3, (1, 2)))  # t s [l [b r]]
        mtA = mt @ A  # t' s [l [b r]]
        env_t = tensordot(mtA, A.conj(), axes=((0, 1), (0, 1)))  # [l [b r]] [l' [b' r']]
    env_t = env_t.unfuse_legs(axes=(0, 1))  # l [b r] l' [b' r']
    env_t = env_t.fuse_legs(axes=((0, 2), 1, 3))  # [l l'] [b r] [b' r']
    env_t = env_t.unfuse_legs(axes=(1, 2))  # [l l'] b r b' r'
    env_t = env_t.swap_gate(axes=((1, 3), 4))  # b b' X r'
    env_t = env_t.fuse_legs(axes=(0, (1, 3), (2, 4))) # [l l'] [b b'] [r r']
    return env_t


def con_Ab(A, mb=None):  # A = [t l] [b r] s
    """ env_t where top legs of double-layer Q tensors are contracted with m_t """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    if mb is None:
        A = A.fuse_legs(axes=((1, 3), (0, 2)))  # [b s] [[t l] r]
        env_b = tensordot(A, A.conj(), axes=(0, 0))  # [[t l] r] [[t' l'] r']
    else:
        A = A.fuse_legs(axes=(1, 3, (0, 2)))  # b s [[t l] r]
        mbA = mb @ A  # b' s [[t l] r]
        env_b = tensordot(mbA, A.conj(), axes=((0, 1), (0, 1)))  # [[t l] r] [[t' l'] r']
    env_b = env_b.unfuse_legs(axes=(0, 1))  # [t l] r [t' l'] r'
    env_b = env_b.fuse_legs(axes=((1, 3), 0, 2))  # [r r'] [t l] [t' l']
    env_b = env_b.unfuse_legs(axes=(1, 2))  # [r r'] t l t' l'
    env_b = env_b.swap_gate(axes=((2, 4), 3)) # l l' X t'
    env_b = env_b.fuse_legs(axes=(0, (1, 3), (2, 4)))  # [r r'] [t t'] [l l']
    return env_b


# def con_Q_l(QA, m_l):  # QA -> [[[t l] s] b] rr
#     """ env_l where left legs of double-layer Q tensors are contracted with m_l """
#     QA = QA.unfuse_legs(axes=0)  # [[t l] s] b rr
#     QA = QA.swap_gate(axes=(1, 2))  # b X rr
#     QA = QA.fuse_legs(axes=(0, (1, 2)))   # [[t l] s] [b rr]
#     QA = QA.unfuse_legs(axes=0)  # [t l] s [b rr]
#     QA = QA.unfuse_legs(axes=0)  # t l s [b rr]
#     QA = QA.fuse_legs(axes=(1, 2, (0, 3)))  # l s [t [b rr]]
#     mlQA = m_l @ QA  # l' s [t [b rr]]
#     env_l = tensordot(mlQA, QA, axes=((0, 1), (0, 1)), conj=(0, 1))  # [t [b rr]] [t' [b' rr']]
#     env_l = env_l.unfuse_legs(axes=(0, 1))  # t [b rr] t' [b' rr']
#     env_l = env_l.fuse_legs(axes=((0, 2), 1, 3))  # [t t'] [b rr] [b' rr']
#     env_l = env_l.unfuse_legs(axes=(1, 2))  # [t t'] b rr b' rr'
#     env_l = env_l.swap_gate(axes=(1, (2, 4)))  # b X rr rr'
#     env_l = env_l.fuse_legs(axes=(0, (1, 3), (2, 4)))  # [t t'] [b b'] [rr rr']
#     return env_l


# def con_Q_r(QB, m_r):  # QB -> ll [t [[b r] s]]
#     """ env_r where right legs of double-layer Q tensors are contracted with m_r """
#     QB = QB.unfuse_legs(axes=1)  # ll t [[b r] s]
#     QB = QB.fuse_legs(axes=((0, 1), 2))  # [ll t] [[b r] s]
#     QB = QB.unfuse_legs(axes=1)  # [ll t] [b r] s
#     QB = QB.unfuse_legs(axes=1)  # [ll t] b r s
#     QB = QB.swap_gate(axes=(1, 2))  # b X r
#     QB = QB.fuse_legs(axes=(2, 3, (0, 1)))  # r s [[ll t] b]
#     mrsQB = m_r @ QB  # r' s [ll [t b]]
#     env_r = tensordot(mrsQB, QB, axes=((0, 1), (0, 1)), conj=(0, 1))  # [[ll t] b] [[ll' t'] b']
#     env_r = env_r.unfuse_legs(axes=(0, 1))  # [ll t] b [ll' t'] b'
#     env_r = env_r.fuse_legs(axes=(0, 2, (1, 3)))  # [ll t] [ll' t'] [b b']
#     env_r = env_r.unfuse_legs(axes=(0, 1))  # ll t ll' t' [b b']
#     env_r = env_r.swap_gate(axes=((0, 2), 3))  # ll ll' X t'
#     env_r = env_r.fuse_legs(axes=((0, 2), (1, 3), 4))  # [ll ll'] [t t'] [b b']
#     return env_r


# def con_Q_t(QA, m_t):  # QA -> [[[t l] s] r] bb
#     """ env_t where top legs of double-layer Q tensors are contracted with m_t """
#     QA = QA.unfuse_legs(axes=0)  # [[t l] s] r bb
#     QA = QA.fuse_legs(axes=(0, (1, 2)))  # [[t l] s] [r bb]
#     QA = QA.unfuse_legs(axes=0)  # [t l] s [r bb]
#     QA = QA.unfuse_legs(axes=0)  # t l s [r bb]
#     QA = QA.swap_gate(axes=(0, 1))  # t X l
#     QA = QA.fuse_legs(axes=(0, 2, (1, 3)))  # t s [l [r bb]]
#     mtsQA = m_t @ QA  # t' s [l [r bb]]
#     env_t = tensordot(mtsQA, QA, axes=((0, 1), (0, 1)), conj=(0, 1))  # [l [r bb]] [l' [r' bb']]
#     env_t = env_t.unfuse_legs(axes=(0, 1))  # l [r bb] l' [r' bb']
#     env_t = env_t.fuse_legs(axes=((0, 2), 1, 3))  # [l l'] [r bb] [r' bb']
#     env_t = env_t.unfuse_legs(axes=(1, 2))  # [l l'] r bb r' bb'
#     env_t = env_t.swap_gate(axes=(3, (2, 4)))  # r' X bb bb'
#     env_t = env_t.fuse_legs(axes=(0, (1, 3), (2, 4))) # [l l'] [r r'] [bb bb']
#     return env_t


# def con_Q_b(QB, m_b):  # QB -> t [l [[b r] s]]
#     """ env_t where top legs of double-layer Q tensors are contracted with m_t """
#     QB = QB.unfuse_legs(axes=1)  # tt l [[b r] s]
#     QB = QB.swap_gate(axes=(0, 1))  # tt X l
#     QB = QB.fuse_legs(axes=((0, 1), 2))  # [tt l] [[b r] s]
#     QB = QB.unfuse_legs(axes=1)  # [tt l] [b r] s
#     QB = QB.unfuse_legs(axes=1)  # [tt l] b r s
#     QB = QB.fuse_legs(axes=(1, 3, (0, 2)))  # b s [[tt l] r]
#     mbQB = m_b @ QB  # b' s [[tt l] r]
#     env_b = tensordot(mbQB, QB, axes=((0, 1), (0, 1)), conj=(0, 1))  # [[tt l] r] [[tt' l'] r']
#     env_b = env_b.unfuse_legs(axes=(0, 1))  # [tt l] r [tt' l'] r'
#     env_b = env_b.fuse_legs(axes=(0, 2, (1, 3)))  # [tt l] [tt' l'] [r r']
#     env_b = env_b.unfuse_legs(axes=(0, 1))  # tt l tt' l' [r r']
#     env_b = env_b.swap_gate(axes=((0, 2), 1)) # tt tt' X l
#     env_b = env_b.fuse_legs(axes=((0, 2), (1, 3), 4))  # [tt tt'] [l l'] [r r']
#     return env_b
