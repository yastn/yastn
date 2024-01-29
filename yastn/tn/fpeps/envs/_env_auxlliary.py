from yastn import fuse_legs, tensordot, swap_gate

__all__ = ['hair_t', 'hair_l', 'hair_b', 'hair_r',
           'cor_tl', 'cor_bl', 'cor_br', 'cor_tr',
           'edge_t', 'edge_l', 'edge_b', 'edge_r',
           'append_vec_tl', 'append_vec_br']


def hair_t(A, ht=None, hl=None, hr=None):
    """ top hair tensor """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    if ht is None and hl is None and hr is None:
        A = A.fuse_legs(axes=((0, 2, 3), 1))  # [[t l] r s] b
        return tensordot(A.conj(), A, axes=(0, 0))  # b' b
    Af = A.transpose(axes=(2, 0, 1, 3)) if hr is None else tensordot(hr, A, axes=(1, 2))  # r' [t l] b s
    Af = Af.unfuse_legs(axes=1)  # r' t l b s
    Af = Af.transpose(axes=(2, 0, 1, 3, 4)) if hl is None else tensordot(hl, Af, axes=(1, 2))  # l' r' t  b s
    Af = Af.transpose(axes=(2, 0, 1, 3, 4)) if ht is None else tensordot(ht, Af, axes=(1, 2))  # t' l' r' b s
    Af = Af.fuse_legs(axes=((0, 1), 2, 4, 3))  # [t' l'] r' s b
    return tensordot(A.conj(), Af, axes=((0, 2, 3), (0, 1, 2)))  # b' b


def hair_l(A, ht=None, hl=None, hb=None):  # A = [t l] [b r] s
    """ left hair tensor """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    if ht is None and hl is None and hb is None:
        A = A.fuse_legs(axes=((0, 1, 3), 2))  # [[t l] b s] r
        return tensordot(A.conj(), A, axes=(0, 0))  # r' r
    Af = A.transpose(axes=(1, 0, 2, 3)) if hb is None else tensordot(hb, A, axes=(1, 1))  # b' [t l] r s
    Af = Af.unfuse_legs(axes=1)  # b' t l r s
    Af = Af.transpose(axes=(2, 0, 1, 3, 4)) if hl is None else tensordot(hl, Af, axes=(1, 2))  # l' b' t  r s
    Af = Af.transpose(axes=(2, 0, 1, 3, 4)) if ht is None else tensordot(ht, Af, axes=(1, 2))  # t' l' b' r s
    Af = Af.fuse_legs(axes=((0, 1), 2, 4, 3))  # [t' l'] b' s r
    return tensordot(A.conj(), Af, axes=((0, 1, 3), (0, 1, 2)))  # r' r


def hair_b(A, hl=None, hb=None, hr=None):  # A = [t l] [b r] s
    """ bottom hair tensor """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    if hl is None and hb is None and hr is None:
        A = A.fuse_legs(axes=(0, (1, 2, 3)))  # t [l [b r] s]
        return tensordot(A.conj(), A, axes=(1, 1))  # t' t
    Af = A.transpose(axes=(0, 2, 3, 1)) if hl is None else tensordot(A, hl, axes=(1, 1))  # t [b r] s l'
    Af = Af.unfuse_legs(axes=1)  # t b r s l'
    Af = Af.transpose(axes=(0, 2, 3, 4, 1)) if hb is None else tensordot(Af, hb, axes=(1, 1))  # t r s l' b'
    Af = Af.transpose(axes=(0, 2, 3, 4, 1)) if hr is None else tensordot(Af, hr, axes=(1, 1))  # t s l' b' r'
    Af = Af.fuse_legs(axes=(2, (3, 4), 1, 0))  # l' [b' r'] s t
    return tensordot(A.conj(), Af, axes=((1, 2, 3), (0, 1, 2)))  # t' t


def hair_r(A, ht=None, hb=None, hr=None):  # A = [t l] [b r] s
    """ right hair tensor """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    if ht is None and hb is None and hr is None:
        A = A.fuse_legs(axes=(1, (0, 2, 3)))  # l [t [b r] s]
        return tensordot(A.conj(), A, axes=(1, 1))  # l' l
    Af = A if ht is None else tensordot(ht, A, axes=(1, 0))  # t' l [b r] s
    Af = Af.unfuse_legs(axes=2)  # t' l b r s
    Af = Af.transpose(axes=(3, 0, 1, 2, 4)) if hr is None else tensordot(hr, Af, axes=(1, 3))  # r' t' l b s
    Af = Af.transpose(axes=(3, 0, 1, 2, 4)) if hb is None else tensordot(hb, Af, axes=(1, 3))  # b' r' t' l s
    Af = Af.fuse_legs(axes=((0, 1), 2, 4, 3))  # [b' r'] t' s t
    return tensordot(A.conj(), Af, axes=((2, 0, 3), (0, 1, 2)))  # l' l


def cor_tl(A, ht=None, hl=None):  # A -> [t l] [b r] s
    """ top-left corner tensor """
    if ht is None and hl is None:
        A = fuse_legs(A, axes=((0, 2), 1))  # [[t l] s] [b r]
        ctl = tensordot(A, A.conj(), axes=(0, 0))  # [b r] [b' r']
    else:
        A = A.unfuse_legs(axes=0).transpose(axes=(0, 2, 3, 1))  # t [b r] s l
        Af = A if ht is None else ht @ A
        if hl is not None:
            Af = Af @ hl.T
        ctl = tensordot(Af, A.conj(), axes=((0, 2, 3), (0, 2, 3)))  # [b r] [b' r']
    ctl = ctl.unfuse_legs(axes=(0, 1))  # b r b' r'
    ctl = ctl.swap_gate(axes=((0, 2), 3))  # b b' X r'
    ctl = ctl.fuse_legs(axes=((0, 2), (1, 3)))  # [b b'] [r r']
    return ctl  # [b b'] [r r']


def cor_bl(A, hb=None, hl=None):  # A = [t l] [b r] s
    """ bottom-left corner tensor """
    A = A.unfuse_legs(axes=(0, 1))  # t l b r s
    if hb is None and hl is None:
        A = fuse_legs(A, axes=((0, 3), (1, 2, 4)))  # [t r] [b l s]
        cbl = tensordot(A, A.conj(), axes=(1, 1))  # [t r] [t' r']
    else:
        A = fuse_legs(A, axes=(1, (0, 3), 4, 2))  # l [t r] s b
        Af = A if hl is None else hl @ A
        if hb is not None:
            Af = Af @ hb.T
        cbl = tensordot(Af, A.conj(), axes=((0, 2, 3), (0, 2, 3)))  # [t r] [t' r']
    cbl = cbl.unfuse_legs(axes=(0, 1))  # t r t' r'
    cbl = cbl.fuse_legs(axes=((1, 3), (0, 2)))  # [r r'] [t t']
    return cbl  # [r r'] [t t']


def cor_br(A, hb=None, hr=None):  # A = [t l] [b r] s
    """ bottom-right corner tensor """
    if hb is None and hr is None:
        A = fuse_legs(A, axes=(0, (1, 2)))  # [t l] [[b r] s]
        cbr = tensordot(A, A.conj(), axes=(1, 1))  # [t l] [t' l']
    else:
        A = A.unfuse_legs(axes=1).transpose(axes=(1, 0, 3, 2))  # b [t l] s r
        Af = A if hb is None else hb @ A
        if hr is not None:
            Af = Af @ hr.T
        cbr = tensordot(Af, A.conj(), axes=((0, 2, 3), (0, 2, 3)))  # [t l] [t' l']
    cbr = cbr.unfuse_legs(axes=(0, 1))  # t l t' l'
    cbr = cbr.swap_gate(axes=((1, 3), 2))  # l l' X t'
    cbr = cbr.fuse_legs(axes=((0, 2), (1, 3)))  # [t t'] [l l']
    return cbr  # [t t'] [l l']


def cor_tr(A, ht=None, hr=None):  # A = [t l] [b r] s
    """ top-right corner tensor """
    A = A.unfuse_legs(axes=(0, 1))  # t l b r s
    A = swap_gate(A, axes=(0, 1, 2, 3))  # t X l, b X r
    if ht is None and hr is None:
        A = fuse_legs(A, axes=((1, 2), (0, 3, 4)))  # [l b] [t r s]
        ctr = tensordot(A, A.conj(), axes=(1, 1))  # [l b] [l' b']
    else:
        A = fuse_legs(A, axes=(0, (1, 2), 4, 3))  # t [l b] s r
        Af = A if ht is None else ht @ A
        if hr is not None:
            Af = Af @ hr.T
        ctr = tensordot(Af, A.conj(), axes=((0, 2, 3), (0, 2, 3)))  # [l b] [l' b']
    ctr = ctr.unfuse_legs(axes=(0, 1))  # l b l' b'
    ctr = ctr.fuse_legs(axes=((0, 2), (1, 3)))  # [l l'] [b b']
    return ctr  # [l l'] [b b']


def edge_l(A, hl=None):  # A = [t l] [b r] s
    """ left edge tensor where left legs of double-layer A tensors can be contracted with hl """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    if hl is None:
        A = A.fuse_legs(axes=((1, 3), (0, 2)))  # [l s] [t [b r]]
        egl = tensordot(A, A.conj(), axes=(0, 0))  # [t [b r]] [t' [b' r']]
    else:
        A = A.fuse_legs(axes=(1, 3, (0, 2)))  # l s [t [b r]]
        hlA = hl @ A  # l' s [t [b r]]
        egl = tensordot(hlA, A.conj(), axes=((0, 1), (0, 1)))  # [t [b r]] [t' [b' r']]
    egl = egl.unfuse_legs(axes=(0, 1))  # t [b r] t' [b' r']
    egl = egl.fuse_legs(axes=(1, 3, (0, 2)))  # [b r] [b' r'] [t t']
    egl = egl.unfuse_legs(axes=(0, 1))  # b r b' r' [t t']
    egl = egl.swap_gate(axes=((0, 2), 3))  # b b' X r'
    egl = egl.fuse_legs(axes=((0, 2), (1, 3), 4))  # [b b'] [r r'] [t t']
    return egl  # [b b'] [r r'] [t t']


def edge_t(A, ht=None):  # QA = [t l] [b r] s
    """ top edge tensor where top legs of double-layer A tensors can be contracted with ht """
    A = A.unfuse_legs(axes=0)  # t l [b r] s
    A = A.swap_gate(axes=(0, 1))  # t X l
    if ht is None:
        A = A.fuse_legs(axes=((0, 3), (1, 2)))  # [t s] [l [b r]]
        egt = tensordot(A, A.conj(), axes=(0, 0))  # [l [b r]] [l' [b' r']]
    else:
        A = A.fuse_legs(axes=(0, 3, (1, 2)))  # t s [l [b r]]
        htA = ht @ A  # t' s [l [b r]]
        egt = tensordot(htA, A.conj(), axes=((0, 1), (0, 1)))  # [l [b r]] [l' [b' r']]
    egt = egt.unfuse_legs(axes=(0, 1))  # l [b r] l' [b' r']
    egt = egt.fuse_legs(axes=((0, 2), 1, 3))  # [l l'] [b r] [b' r']
    egt = egt.unfuse_legs(axes=(1, 2))  # [l l'] b r b' r'
    egt = egt.swap_gate(axes=((1, 3), 4))  # b b' X r'
    egt = egt.fuse_legs(axes=(0, (1, 3), (2, 4))) # [l l'] [b b'] [r r']
    return egt  # [l l'] [b b'] [r r']


def edge_r(A, hr=None):  # A = [t l] [b r] s
    """ right edge tensor where right legs of double-layer A tensors can be contracted with hr """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    A = A.swap_gate(axes=(1, 2))  # b X r
    if hr is None:
        A = A.fuse_legs(axes=((2, 3), (0, 1)))  # [r s] [[t l] b]
        egr = tensordot(A, A.conj(), axes=(0, 0))  # [[t l] b] [[t' l'] b']
    else:
        A = A.fuse_legs(axes=(2, 3, (0, 1)))  # r s [[t l] b]
        hrA = hr @ A  # r' s [[t l] b]
        egr = tensordot(hrA, A.conj(), axes=((0, 1), (0, 1)))  # [[t l] b] [[t' l'] b']
    egr = egr.unfuse_legs(axes=(0, 1))  # [t l] b [t' l'] b'
    egr = egr.fuse_legs(axes=(0, 2, (1, 3)))  # [t l] [t' l'] [b b']
    egr = egr.unfuse_legs(axes=(0, 1))  # t l t' l' [b b']
    egr = egr.swap_gate(axes=((1, 3), 2))  # l l' X t'
    egr = egr.fuse_legs(axes=((0, 2), (1, 3), 4))  # [t t'] [l l'] [b b']
    return egr  # [t t'] [l l'] [b b']


def edge_b(A, hb=None):  # A = [t l] [b r] s;  hb = b' b
    """ bottom edge tensor where bottom legs of double-layer A tensors can be contracted with hb """
    A = A.unfuse_legs(axes=1)  # [t l] b r s
    if hb is None:
        A = A.fuse_legs(axes=((1, 3), (0, 2)))  # [b s] [[t l] r]
        egb = tensordot(A, A.conj(), axes=(0, 0))  # [[t l] r] [[t' l'] r']
    else:
        A = A.fuse_legs(axes=(1, 3, (0, 2)))  # b s [[t l] r]
        hbA = hb @ A  # b' s [[t l] r]
        egb = tensordot(hbA, A.conj(), axes=((0, 1), (0, 1)))  # [[t l] r] [[t' l'] r']
    egb = egb.unfuse_legs(axes=(0, 1))  # [t l] r [t' l'] r'
    egb = egb.fuse_legs(axes=((1, 3), 0, 2))  # [r r'] [t l] [t' l']
    egb = egb.unfuse_legs(axes=(1, 2))  # [r r'] t l t' l'
    egb = egb.swap_gate(axes=((2, 4), 3)) # l l' X t'
    egb = egb.fuse_legs(axes=(0, (1, 3), (2, 4)))  # [r r'] [t t'] [l l']
    return egb  # [r r'] [t t'] [l l']


def append_vec_tl(A, Ac, vectl):  # A = [t l] [b r] s;  Ac = [t' l'] [b' r'] s;  vectl = x [l l'] [t t'] y
    """ Append the A and Ac tensors to the top-left vector """
    vectl = vectl.fuse_legs(axes=(2, (0, 3), 1))  # [t t'] [x y] [l l']
    vectl = vectl.unfuse_legs(axes=(0, 2))  # t t' [x y] l l'
    vectl = vectl.swap_gate(axes=(1, (3, 4)))  # t' X l l'
    vectl = vectl.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [t l] [x y] [t' l']
    vectl = vectl.tensordot(Ac.conj(), axes=(2, 0))  # [t l] [x y] [b' r'] s
    vectl = A.tensordot(vectl, axes=((0, 2), (0, 3)))  # [b r] [x y] [b' r']
    vectl = vectl.unfuse_legs(axes=(0, 2))  # b r [x y] b' r'
    vectl = vectl.swap_gate(axes=((0, 3), 4))  # b b' X r'
    vectl = vectl.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [b b'] [x y] [r r']
    vectl = vectl.unfuse_legs(axes=1)  # [b b'] x y [r r']
    vectl = vectl.transpose(axes=(1, 0, 2, 3))  # x [b b'] y [r r']
    return vectl


def append_vec_br(A, Ac, vecbr):  # A = [t l] [b r] s;  Ac = [t' l'] [b' r'] s;  vecbr = x [r r'] [b b'] y
    """ Append the A and Ac tensors to the bottom-right vector. """
    vecbr = vecbr.fuse_legs(axes=(2, (0, 3), 1))  # [b b'] [x y] [r r']
    vecbr = vecbr.unfuse_legs(axes=(0, 2))  # b b' [x y] r r'
    vecbr = vecbr.swap_gate(axes=((0, 1), 4))  # b b' X r'
    vecbr = vecbr.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [b r] [x y] [b' r']
    vecbr = vecbr.tensordot(Ac.conj(), axes=(2, 1))  # [b r] [x y] [t' l'] s
    vecbr = A.tensordot(vecbr, axes=((1, 2), (0, 3)))  # [t l] [x y] [t' l']
    vecbr = vecbr.unfuse_legs(axes=(0, 2))  # t l [x y] t' l'
    vecbr = vecbr.swap_gate(axes=((1, 4), 3))  # l l' X t'
    vecbr = vecbr.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [t t'] [x y] [l l']
    vecbr = vecbr.unfuse_legs(axes=1)  # [t t'] x y [l l']
    vecbr = vecbr.transpose(axes=(1, 0, 2, 3))  # x [t t'] y [l l']
    return vecbr
