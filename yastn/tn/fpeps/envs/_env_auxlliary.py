from yastn import fuse_legs, tensordot, swap_gate

__all__ = ['leaf_t', 'leaf_l', 'leaf_b', 'leaf_r',
           'cor_tl', 'cor_bl', 'cor_br', 'cor_tr',
           'edge_t', 'edge_l', 'edge_b', 'edge_r',
           'append_vec_tl', 'append_vec_br']


def leaf_t(A):
    """ top leaf tensor """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    A = A.fuse_legs(axes=((0, 2, 3), 1))  # [[t l] r s] b
    lft = tensordot(A.conj(), A, axes=(0, 0))  # b' b
    return lft  # b' b


def leaf_l(A):  # A = [t l] [b r] s
    """ left leaf tensor """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    A = A.fuse_legs(axes=((0, 1, 3), 2))  # [[t l] b s] r
    lfl = tensordot(A.conj(), A, axes=(0, 0))  # r' r
    return lfl  # r' r


def leaf_b(A):  # A = [t l] [b r] s
    """ bottom leaf tensor """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    A = A.fuse_legs(axes=(0, (1, 2, 3)))  # t [l [b r] s]
    lfb = tensordot(A.conj(), A, axes=(1, 1))  # t' t
    return lfb  # t' t


def leaf_r(A):  # A = [t l] [b r] s
    """ right leaf tensor """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    A = A.fuse_legs(axes=(1, (0, 2, 3)))  # l [t [b r] s]
    lfr = tensordot(A.conj(), A, axes=(1, 1))  # l' l
    return lfr  # l' l


def cor_tl(A):  # A -> [t l] [b r] s
    """ top-left corner tensor """
    A = fuse_legs(A, axes=((0, 2), 1))  # [[t l] s] [b r]
    ctl = tensordot(A, A.conj(), axes=(0, 0))  # [b r] [b' r']
    ctl = ctl.unfuse_legs(axes=(0, 1))  # b r b' r'
    ctl = ctl.swap_gate(axes=((0, 2), 3))  # b b' X r'
    ctl = ctl.fuse_legs(axes=((0, 2), (1, 3)))  # [b b'] [r r']
    return ctl  # [b b'] [r r']


def cor_bl(A):  # A = [t l] [b r] s
    """ bottom-left corner tensor """
    A = A.unfuse_legs(axes=(0, 1))  # t l b r s
    A = fuse_legs(A, axes=((0, 3), (1, 2, 4)))  # [t r] [b l s]
    cbl = tensordot(A, A.conj(), axes=(1, 1))  # [t r] [t' r']
    cbl = cbl.unfuse_legs(axes=(0, 1))  # t r t' r'
    cbl = cbl.fuse_legs(axes=((1, 3), (0, 2)))  # [r r'] [t t']
    return cbl  # [r r'] [t t']


def cor_br(A):  # A = [t l] [b r] s
    """ bottom-right corner tensor """
    A = fuse_legs(A, axes=(0, (1, 2)))  # [t l] [[b r] s]
    cbr = tensordot(A, A.conj(), axes=(1, 1))  # [t l] [t' l']
    cbr = cbr.unfuse_legs(axes=(0, 1))  # t l t' l'
    cbr = cbr.swap_gate(axes=((1, 3), 2))  # l l' X t'
    cbr = cbr.fuse_legs(axes=((0, 2), (1, 3)))  # [t t'] [l l']
    return cbr  # [t t'] [l l']


def cor_tr(A):  # A = [t l] [b r] s
    """ top-right corner tensor """
    A = A.unfuse_legs(axes=(0, 1))  # t l b r s
    A = swap_gate(A, axes=(0, 1, 2, 3))  # t X l, b X r
    A = fuse_legs(A, axes=((1, 2), (0, 3, 4)))  # [l b] [t r s]
    ctr = tensordot(A, A.conj(), axes=(1, 1))  # [l b] [l' b']
    ctr = ctr.unfuse_legs(axes=(0, 1))  # l b l' b'
    ctr = ctr.fuse_legs(axes=((0, 2), (1, 3)))  # [l l'] [b b']
    return ctr  # [l l'] [b b']


def edge_l(A, lfl=None):  # A = [t l] [b r] s
    """ left edge tensor where left legs of double-layer A tensors can be contracted with lfl """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    if lfl is None:
        A = A.fuse_legs(axes=((1, 3), (0, 2)))  # [l s] [t [b r]]
        egl = tensordot(A, A.conj(), axes=(0, 0))  # [t [b r]] [t' [b' r']]
    else:
        A = A.fuse_legs(axes=(1, 3, (0, 2)))  # l s [t [b r]]
        lflA = lfl @ A  # l' s [t [b r]]
        egl = tensordot(lflA, A.conj(), axes=((0, 1), (0, 1)))  # [t [b r]] [t' [b' r']]
    egl = egl.unfuse_legs(axes=(0, 1))  # t [b r] t' [b' r']
    egl = egl.fuse_legs(axes=(1, 3, (0, 2)))  # [b r] [b' r'] [t t']
    egl = egl.unfuse_legs(axes=(0, 1))  # b r b' r' [t t']
    egl = egl.swap_gate(axes=((0, 2), 3))  # b b' X r'
    egl = egl.fuse_legs(axes=((0, 2), (1, 3), 4))  # [b b'] [r r'] [t t']
    return egl  # [b b'] [r r'] [t t']


def edge_t(A, lft=None):  # QA = [t l] [b r] s
    """ top edge tensor where top legs of double-layer A tensors can be contracted with lft """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    A = A.swap_gate(axes=(0, 1))  # t X l
    if lft is None:
        A = A.fuse_legs(axes=((0, 3), (1, 2)))  # [t s] [l [b r]]
        egt = tensordot(A, A.conj(), axes=(0, 0))  # [l [b r]] [l' [b' r']]
    else:
        A = A.fuse_legs(axes=(0, 3, (1, 2)))  # t s [l [b r]]
        lftA = lft @ A  # t' s [l [b r]]
        egt = tensordot(lftA, A.conj(), axes=((0, 1), (0, 1)))  # [l [b r]] [l' [b' r']]
    egt = egt.unfuse_legs(axes=(0, 1))  # l [b r] l' [b' r']
    egt = egt.fuse_legs(axes=((0, 2), 1, 3))  # [l l'] [b r] [b' r']
    egt = egt.unfuse_legs(axes=(1, 2))  # [l l'] b r b' r'
    egt = egt.swap_gate(axes=((1, 3), 4))  # b b' X r'
    egt = egt.fuse_legs(axes=(0, (1, 3), (2, 4))) # [l l'] [b b'] [r r']
    return egt  # [l l'] [b b'] [r r']


def edge_r(A, lfr=None):  # A = [t l] [b r] s
    """ right edge tensor where right legs of double-layer A tensors can be contracted with lfr """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    A = A.swap_gate(axes=(1, 2))  # b X r
    if lfr is None:
        A = A.fuse_legs(axes=((2, 3), (0, 1)))  # [r s] [[t l] b]
        egr = tensordot(A, A.conj(), axes=(0, 0))  # [[t l] b] [[t' l'] b']
    else:
        A = A.fuse_legs(axes=(2, 3, (0, 1)))  # r s [[t l] b]
        lfrA = lfr @ A  # r' s [[t l] b]
        egr = tensordot(lfrA, A.conj(), axes=((0, 1), (0, 1)))  # [[t l] b] [[t' l'] b']
    egr = egr.unfuse_legs(axes=(0, 1))  # [t l] b [t' l'] b'
    egr = egr.fuse_legs(axes=(0, 2, (1, 3)))  # [t l] [t' l'] [b b']
    egr = egr.unfuse_legs(axes=(0, 1))  # t l t' l' [b b']
    egr = egr.swap_gate(axes=((1, 3), 2))  # l l' X t'
    egr = egr.fuse_legs(axes=((0, 2), (1, 3), 4))  # [t t'] [l l'] [b b']
    return egr  # [t t'] [l l'] [b b']


def edge_b(A, lfb=None):  # A = [t l] [b r] s;  lfb = b' b
    """ bottom edge tensor where bottom legs of double-layer A tensors can be contracted with lfb """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    if lfb is None:
        A = A.fuse_legs(axes=((1, 3), (0, 2)))  # [b s] [[t l] r]
        egb = tensordot(A, A.conj(), axes=(0, 0))  # [[t l] r] [[t' l'] r']
    else:
        A = A.fuse_legs(axes=(1, 3, (0, 2)))  # b s [[t l] r]
        lfbA = lfb @ A  # b' s [[t l] r]
        egb = tensordot(lfbA, A.conj(), axes=((0, 1), (0, 1)))  # [[t l] r] [[t' l'] r']
    egb = egb.unfuse_legs(axes=(0, 1))  # [t l] r [t' l'] r'
    egb = egb.fuse_legs(axes=((1, 3), 0, 2))  # [r r'] [t l] [t' l']
    egb = egb.unfuse_legs(axes=(1, 2))  # [r r'] t l t' l'
    egb = egb.swap_gate(axes=((2, 4), 3)) # l l' X t'
    egb = egb.fuse_legs(axes=(0, (1, 3), (2, 4)))  # [r r'] [t t'] [l l']
    return egb  # [r r'] [t t'] [l l']


def append_vec_tl(A, Ac, vectl):  # A = [t l] [b r] s;  Ac = [t' l'] [b' r'] s;  vectl = x [l l'] [t t'] y
    """ Append the A and Ac tensors to the top-left vector """
    vectl = vectl.fuse_legs(axes=((0, 3), 1, 2))  # [x y] [l l'] [t t']
    vectl = vectl.unfuse_legs(axes=(1, 2))  # [x y] l l' t t'
    vectl = vectl.swap_gate(axes=((1, 2), 4))  # l l' X t'
    # vectl = vectl.fuse_legs(axes=(0, (3, 1), (4, 2)))  # [x y] [t l] [t' l']
    # vectl = vectl.tensordot(Ac.conj(), axes=(2, 0))  # [x y] [t l] [b' r'] s
    # vectl = vectl.tensordot(A, axes=((1, 3), (0, 2)))  # [x y] [b' r'] [b r]
    vectl = vectl.fuse_legs(axes=(0, (4, 2), (3, 1)))  # [x y] [t' l'] [t l]
    vectl = vectl.tensordot(A, axes=(2, 0))  # [x y] [t' l'] [b r] s
    vectl = vectl.tensordot(Ac.conj(), axes=((1, 3), (0, 2)))  # [x y] [b r] [b' r']
    #
    vectl = vectl.unfuse_legs(axes=(1, 2))  # [x y] b r b' r'
    vectl = vectl.swap_gate(axes=((1, 3), 4))  # b b' X r'
    vectl = vectl.fuse_legs(axes=(0, (1, 3), (2, 4)))  # [x y] [b b'] [r r']
    vectl = vectl.unfuse_legs(axes=0)  # x y [b b'] [r r']
    vectl = vectl.transpose(axes=(0, 2, 1, 3))  # x [b b'] y [r r']
    return vectl


def append_vec_br(A, Ac, vecbr):  # A = [t l] [b r] s;  Ac = [t' l'] [b' r'] s;  vecbr = x [r r'] [b b'] y
    """ Append the A and Ac tensors to the bottom-right vector. """
    vecbr = vecbr.fuse_legs(axes=((0, 3), 1, 2))  # [x y] [r r'] [b b']
    vecbr = vecbr.unfuse_legs(axes=(1, 2))  # [x y] r r' b b'
    vecbr = vecbr.swap_gate(axes=(2, (3, 4)))  # r' X b b'
    vecbr = vecbr.fuse_legs(axes=(0, (4, 2), (3, 1)))  # [x y] [b' r'] [b r]
    vecbr = vecbr.tensordot(A, axes=(2, 1))  # [x y] [b' r'] [t l] s
    vecbr = vecbr.tensordot(Ac.conj(), axes=((1, 3), (1, 2)))  # [x y] [t l] [t' l']
    vecbr = vecbr.unfuse_legs(axes=(1, 2))  # [x y] t l t' l'
    vecbr = vecbr.swap_gate(axes=((2, 4), 3))  # l l' X t'
    vecbr = vecbr.fuse_legs(axes=(0, (1, 3), (2, 4)))  # [x y] [t t'] [l l']
    vecbr = vecbr.unfuse_legs(axes=0)  # x y [t t'] [l l']
    vecbr = vecbr.transpose(axes=(0, 2, 1, 3))  # x [t t'] y [l l']
    return vecbr
