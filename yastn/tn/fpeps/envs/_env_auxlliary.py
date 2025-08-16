# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from .... import fuse_legs, tensordot, swap_gate, ones, Leg, eye, Tensor, YastnError
from ... import mps

__all__ = ['hair_t', 'hair_l', 'hair_b', 'hair_r',
           'cor_tl', 'cor_bl', 'cor_br', 'cor_tr',
           'edge_t', 'edge_l', 'edge_b', 'edge_r',
           'append_vec_tl', 'append_vec_tr',
           'append_vec_bl', 'append_vec_br',
           'tensors_from_psi', 'cut_into_hairs',
           'identity_tm_boundary', 'identity_boundary',
           'trivial_peps_tensor']


def trivial_peps_tensor(config):
    triv = ones(config, legs=[Leg(config, t=(config.sym.zero(),), D=(1,))])
    for s in (-1, 1, 1, -1):
        triv = triv.add_leg(axis=0, s=s)
    return triv.fuse_legs(axes=((0, 1), (2, 3), 4))


def tensors_from_psi(d, psi):
    if any(v is None for v in d.values()):
        triv = trivial_peps_tensor(psi.config)
    for k, v in d.items():
        d[k] = triv if v is None else psi[v]


def identity_tm_boundary(tmpo):
    """
    For transfer matrix MPO build of DoublePepsTensors,
    create MPS that contracts each DoublePepsTensor from the right.
    """
    phi = mps.Mps(N=tmpo.N)
    config = tmpo.config
    for n in phi.sweep(to='last'):
        legf = tmpo[n].get_legs(axes=3).conj()
        tmp = identity_boundary(config, legf)
        phi[n] = tmp.add_leg(0, s=-1).add_leg(2, s=1)
    return phi


def identity_boundary(config, leg):
    if leg.is_fused():
        return eye(config, legs=leg.unfuse_leg(), isdiag=False).fuse_legs(axes=[(0, 1)])
    else:
        return ones(config, legs=[leg])


def cut_into_hairs(A):
    """ rank-one approximation of a tensor into hairs """
    hl, _, hr = A.svd_with_truncation(axes=(0, 1), D_total=1)
    hl = hl.remove_leg(axis=1).unfuse_legs(axes=0).transpose(axes=(1, 0))
    hr = hr.remove_leg(axis=0).unfuse_legs(axes=0).transpose(axes=(1, 0))
    return hl, hr


def hair_t(Abra, ht=None, hl=None, hr=None, Aket=None):
    """ top hair tensor """
    Abra = Abra.unfuse_legs(axes=1)#.swap_gate(axes=(1, 2))  # [t l] b r s
    Aket = Abra if Aket is None else Aket.unfuse_legs(axes=1)#.swap_gate(axes=(1, 2))  # [t l] b r s

    if ht is None and hl is None and hr is None:
        return tensordot(Abra.conj(), Aket, axes=((0, 2, 3), (0, 2, 3)))  # b' b

    Af = Aket.transpose(axes=(2, 0, 1, 3)) if hr is None else tensordot(hr, Aket, axes=(1, 2))  # r' [t l] b s
    Af = Af.unfuse_legs(axes=1)  # r' t l b s
    Af = Af.transpose(axes=(2, 0, 1, 3, 4)) if hl is None else tensordot(hl, Af, axes=(1, 2))  # l' r' t  b s
    Af = Af.transpose(axes=(2, 0, 1, 3, 4)) if ht is None else tensordot(ht, Af, axes=(1, 2))  # t' l' r' b s
    Af = Af.fuse_legs(axes=((0, 1), 2, 4, 3))  # [t' l'] r' s b
    return tensordot(Abra.conj(), Af, axes=((0, 2, 3), (0, 1, 2)))  # b' b


def hair_l(Abra, ht=None, hl=None, hb=None, Aket=None):  # A = [t l] [b r] s
    """ left hair tensor """
    Abra = Abra.unfuse_legs(axes=1)  # [t' l'] b' r' s'
    Aket = Abra if Aket is None else Aket.unfuse_legs(axes=1)  # [t l] b r s

    if ht is None and hl is None and hb is None:
        return tensordot(Abra.conj(), Aket, axes=((0, 1, 3), (0, 1, 3)))  # r' r

    Af = Aket.transpose(axes=(1, 0, 2, 3)) if hb is None else tensordot(hb, Aket, axes=(1, 1))  # b' [t l] r s
    Af = Af.unfuse_legs(axes=1)  # b' t l r s
    Af = Af.transpose(axes=(2, 0, 1, 3, 4)) if hl is None else tensordot(hl, Af, axes=(1, 2))  # l' b' t  r s
    Af = Af.transpose(axes=(2, 0, 1, 3, 4)) if ht is None else tensordot(ht, Af, axes=(1, 2))  # t' l' b' r s
    Af = Af.fuse_legs(axes=((0, 1), 2, 4, 3))  # [t' l'] b' s r
    return tensordot(Abra.conj(), Af, axes=((0, 1, 3), (0, 1, 2)))  # r' r


def hair_b(Abra, hl=None, hb=None, hr=None, Aket=None):  # A = [t l] [b r] s
    """ bottom hair tensor """
    Abra = Abra.unfuse_legs(axes=0)  # t l [b r] s
    Aket = Abra if Aket is None else Aket.unfuse_legs(axes=0)  # t l [b r] s

    if hl is None and hb is None and hr is None:
        return tensordot(Abra.conj(), Aket, axes=((1, 2, 3), (1, 2, 3)))  # t' t

    Af = Aket.transpose(axes=(0, 2, 3, 1)) if hl is None else tensordot(Aket, hl, axes=(1, 1))  # t [b r] s l'
    Af = Af.unfuse_legs(axes=1)  # t b r s l'
    Af = Af.transpose(axes=(0, 2, 3, 4, 1)) if hb is None else tensordot(Af, hb, axes=(1, 1))  # t r s l' b'
    Af = Af.transpose(axes=(0, 2, 3, 4, 1)) if hr is None else tensordot(Af, hr, axes=(1, 1))  # t s l' b' r'
    Af = Af.fuse_legs(axes=(2, (3, 4), 1, 0))  # l' [b' r'] s t
    return tensordot(Abra.conj(), Af, axes=((1, 2, 3), (0, 1, 2)))  # t' t


def hair_r(Abra, ht=None, hb=None, hr=None, Aket=None):  # A = [t l] [b r] s
    """ right hair tensor """
    Abra = Abra.unfuse_legs(axes=0)#.swap_gate(axes=(0, 1))  # t l [b r] s
    Aket = Abra if Aket is None else Aket.unfuse_legs(axes=0)#.swap_gate(axes=(0, 1))  # t l [b r] s

    if ht is None and hb is None and hr is None:
        return tensordot(Abra.conj(), Aket, axes=((0, 2, 3), (0, 2, 3)))  # l' l

    Af = Aket if ht is None else tensordot(ht, Aket, axes=(1, 0))  # t' l [b r] s
    Af = Af.unfuse_legs(axes=2)  # t' l b r s
    Af = Af.transpose(axes=(3, 0, 1, 2, 4)) if hr is None else tensordot(hr, Af, axes=(1, 3))  # r' t' l b s
    Af = Af.transpose(axes=(3, 0, 1, 2, 4)) if hb is None else tensordot(hb, Af, axes=(1, 3))  # b' r' t' l s
    Af = Af.fuse_legs(axes=((0, 1), 2, 4, 3))  # [b' r'] t' s t
    return tensordot(Abra.conj(), Af, axes=((2, 0, 3), (0, 1, 2)))  # l' l


def cor_tl(A_bra, ht=None, hl=None, A_ket=None):  # A -> [t l] [b r] s
    """ top-left corner tensor """
    if ht is None and hl is None:
        A = fuse_legs(A_bra, axes=((0, 2), 1))  # [[t l] s] [b r]
        A_ket= A if A_ket is None else fuse_legs(A_ket, axes=((0, 2), 1))
        ctl = tensordot(A, A_ket.conj(), axes=(0, 0))  # [b r] [b' r']
    else:
        A = A_bra.unfuse_legs(axes=0).transpose(axes=(0, 2, 3, 1))  # t [b r] s l
        Af = A if ht is None else ht @ A
        if hl is not None:
            Af = Af @ hl.T
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=0).transpose(axes=(0, 2, 3, 1))
        ctl = tensordot(Af, A_ket.conj(), axes=((0, 2, 3), (0, 2, 3)))  # [b r] [b' r']
    ctl = ctl.unfuse_legs(axes=(0, 1))  # b r b' r'
    ctl = ctl.swap_gate(axes=((0, 2), 3))  # b b' X r'
    ctl = ctl.fuse_legs(axes=((0, 2), (1, 3)))  # [b b'] [r r']
    return ctl  # [b b'] [r r']


def cor_bl(A_bra, hb=None, hl=None, A_ket=None):  # A = [t l] [b r] s
    """ bottom-left corner tensor """
    A = A_bra.unfuse_legs(axes=(0, 1))  # t l b r s
    if hb is None and hl is None:
        A = fuse_legs(A, axes=((0, 3), (1, 2, 4)))  # [t r] [b l s]
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((0, 3), (1, 2, 4)))
        cbl = tensordot(A, A_ket.conj(), axes=(1, 1))  # [t r] [t' r']
    else:
        A = fuse_legs(A, axes=(1, (0, 3), 4, 2))  # l [t r] s b
        Af = A if hl is None else hl @ A
        if hb is not None:
            Af = Af @ hb.T
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=(0, 1)).fuse_legs(axes=(1, (0, 3), 4, 2))
        cbl = tensordot(Af, A_ket.conj(), axes=((0, 2, 3), (0, 2, 3)))  # [t r] [t' r']
    cbl = cbl.unfuse_legs(axes=(0, 1))  # t r t' r'
    cbl = cbl.fuse_legs(axes=((1, 3), (0, 2)))  # [r r'] [t t']
    return cbl  # [r r'] [t t']


def cor_br(A_bra, hb=None, hr=None, A_ket=None):  # A = [t l] [b r] s
    """ bottom-right corner tensor """
    if hb is None and hr is None:
        A = fuse_legs(A_bra, axes=(0, (1, 2)))  # [t l] [[b r] s]
        A_ket = A if A_ket is None else fuse_legs(A_ket, axes=(0, (1, 2)))
        cbr = tensordot(A, A_ket.conj(), axes=(1, 1))  # [t l] [t' l']
    else:
        A = A_bra.unfuse_legs(axes=1).transpose(axes=(1, 0, 3, 2))  # b [t l] s r
        Af = A if hb is None else hb @ A
        if hr is not None:
            Af = Af @ hr.T
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=1).transpose(axes=(1, 0, 3, 2))
        cbr = tensordot(Af, A_ket.conj(), axes=((0, 2, 3), (0, 2, 3)))  # [t l] [t' l']
    cbr = cbr.unfuse_legs(axes=(0, 1))  # t l t' l'
    cbr = cbr.swap_gate(axes=((1, 3), 2))  # l l' X t'
    cbr = cbr.fuse_legs(axes=((0, 2), (1, 3)))  # [t t'] [l l']
    return cbr  # [t t'] [l l']


def cor_tr(A_bra, ht=None, hr=None, A_ket=None):  # A = [t l] [b r] s
    """ top-right corner tensor """
    A = A_bra.unfuse_legs(axes=(0, 1))  # t l b r s
    A = swap_gate(A, axes=(0, 1, 2, 3))  # t X l, b X r
    if ht is None and hr is None:
        A = fuse_legs(A, axes=((1, 2), (0, 3, 4)))  # [l b] [t r s]
        A_ket = A if A_ket is None else A_ket.unfuse_legs(axes=(0, 1)).swap_gate(axes=(0, 1, 2, 3)).fuse_legs(axes=((1, 2), (0, 3, 4)))
        ctr = tensordot(A, A_ket.conj(), axes=(1, 1))  # [l b] [l' b']
    else:
        A = fuse_legs(A, axes=(0, (1, 2), 4, 3))  # t [l b] s r
        Af = A if ht is None else ht @ A
        if hr is not None:
            Af = Af @ hr.T
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=(0, 1)).swap_gate(axes=(0, 1, 2, 3)).fuse_legs(axes=(0, (1, 2), 4, 3))
        ctr = tensordot(Af, A_ket.conj(), axes=((0, 2, 3), (0, 2, 3)))  # [l b] [l' b']
    ctr = ctr.unfuse_legs(axes=(0, 1))  # l b l' b'
    ctr = ctr.fuse_legs(axes=((0, 2), (1, 3)))  # [l l'] [b b']
    return ctr  # [l l'] [b b']


def edge_l(A_bra, hl=None, A_ket=None):  # A = [t l] [b r] s
    """ left edge tensor where left legs of double-layer A tensors can be contracted with hl """
    A = A_bra.unfuse_legs(axes=0)  # t l [b r] s
    if hl is None:
        A = A.fuse_legs(axes=((1, 3), (0, 2)))  # [l s] [t [b r]]
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=0).fuse_legs(axes=((1, 3), (0, 2)))
        egl = tensordot(A, A_ket.conj(), axes=(0, 0))  # [t [b r]] [t' [b' r']]
    else:
        A = A.fuse_legs(axes=(1, 3, (0, 2)))  # l s [t [b r]]
        hlA = hl @ A  # l' s [t [b r]]
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=0).fuse_legs(axes=(1, 3, (0, 2)))
        egl = tensordot(hlA, A_ket.conj(), axes=((0, 1), (0, 1)))  # [t [b r]] [t' [b' r']]
    egl = egl.unfuse_legs(axes=(0, 1))  # t [b r] t' [b' r']
    egl = egl.fuse_legs(axes=(1, 3, (0, 2)))  # [b r] [b' r'] [t t']
    egl = egl.unfuse_legs(axes=(0, 1))  # b r b' r' [t t']
    egl = egl.swap_gate(axes=((0, 2), 3))  # b b' X r'
    egl = egl.fuse_legs(axes=((0, 2), (1, 3), 4))  # [b b'] [r r'] [t t']
    return egl  # [b b'] [r r'] [t t']


def edge_t(A_bra, ht=None, A_ket=None):  # QA = [t l] [b r] s
    """ top edge tensor where top legs of double-layer A tensors can be contracted with ht """
    A = A_bra.unfuse_legs(axes=0)  # t l [b r] s
    A = A.swap_gate(axes=(0, 1))  # t X l
    if ht is None:
        A = A.fuse_legs(axes=((0, 3), (1, 2)))  # [t s] [l [b r]]
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=0).swap_gate(axes=(0, 1)).fuse_legs(axes=((0, 3), (1, 2)))
        egt = tensordot(A, A_ket.conj(), axes=(0, 0))  # [l [b r]] [l' [b' r']]
    else:
        A = A.fuse_legs(axes=(0, 3, (1, 2)))  # t s [l [b r]]
        htA = ht @ A  # t' s [l [b r]]
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=0).swap_gate(axes=(0, 1)).fuse_legs(axes=(0, 3, (1, 2)))
        egt = tensordot(htA, A_ket.conj(), axes=((0, 1), (0, 1)))  # [l [b r]] [l' [b' r']]
    egt = egt.unfuse_legs(axes=(0, 1))  # l [b r] l' [b' r']
    egt = egt.fuse_legs(axes=((0, 2), 1, 3))  # [l l'] [b r] [b' r']
    egt = egt.unfuse_legs(axes=(1, 2))  # [l l'] b r b' r'
    egt = egt.swap_gate(axes=((1, 3), 4))  # b b' X r'
    egt = egt.fuse_legs(axes=(0, (1, 3), (2, 4))) # [l l'] [b b'] [r r']
    return egt  # [l l'] [b b'] [r r']


def edge_r(A_bra, hr=None, A_ket=None):  # A = [t l] [b r] s
    """ right edge tensor where right legs of double-layer A tensors can be contracted with hr """
    A = A_bra.unfuse_legs(axes=1)  # [t l] b r s
    A = A.swap_gate(axes=(1, 2))  # b X r
    if hr is None:
        A = A.fuse_legs(axes=((2, 3), (0, 1)))  # [r s] [[t l] b]
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=1).swap_gate(axes=(1, 2)).fuse_legs(axes=((2, 3), (0, 1)))
        egr = tensordot(A, A_ket.conj(), axes=(0, 0))  # [[t l] b] [[t' l'] b']
    else:
        A = A.fuse_legs(axes=(2, 3, (0, 1)))  # r s [[t l] b]
        hrA = hr @ A  # r' s [[t l] b]
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=1).swap_gate(axes=(1, 2)).fuse_legs(axes=(2, 3, (0, 1)))
        egr = tensordot(hrA, A_ket.conj(), axes=((0, 1), (0, 1)))  # [[t l] b] [[t' l'] b']
    egr = egr.unfuse_legs(axes=(0, 1))  # [t l] b [t' l'] b'
    egr = egr.fuse_legs(axes=(0, 2, (1, 3)))  # [t l] [t' l'] [b b']
    egr = egr.unfuse_legs(axes=(0, 1))  # t l t' l' [b b']
    egr = egr.swap_gate(axes=((1, 3), 2))  # l l' X t'
    egr = egr.fuse_legs(axes=((0, 2), (1, 3), 4))  # [t t'] [l l'] [b b']
    return egr  # [t t'] [l l'] [b b']


def edge_b(A_bra, hb=None, A_ket=None):  # A = [t l] [b r] s;  hb = b' b
    """ bottom edge tensor where bottom legs of double-layer A tensors can be contracted with hb """
    A = A_bra.unfuse_legs(axes=1)  # [t l] b r s
    if hb is None:
        A = A.fuse_legs(axes=((1, 3), (0, 2)))  # [b s] [[t l] r]
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=1).fuse_legs(axes=((1, 3), (0, 2)))
        egb = tensordot(A, A_ket.conj(), axes=(0, 0))  # [[t l] r] [[t' l'] r']
    else:
        A = A.fuse_legs(axes=(1, 3, (0, 2)))  # b s [[t l] r]
        hbA = hb @ A  # b' s [[t l] r]
        A_ket= A if A_ket is None else A_ket.unfuse_legs(axes=1).fuse_legs(axes=(1, 3, (0, 2)))
        egb = tensordot(hbA, A_ket.conj(), axes=((0, 1), (0, 1)))  # [[t l] r] [[t' l'] r']
    egb = egb.unfuse_legs(axes=(0, 1))  # [t l] r [t' l'] r'
    egb = egb.fuse_legs(axes=((1, 3), 0, 2))  # [r r'] [t l] [t' l']
    egb = egb.unfuse_legs(axes=(1, 2))  # [r r'] t l t' l'
    egb = egb.swap_gate(axes=((2, 4), 3)) # l l' X t'
    egb = egb.fuse_legs(axes=(0, (1, 3), (2, 4)))  # [r r'] [t t'] [l l']
    return egb  # [r r'] [t t'] [l l']


def append_vec_tl(Ac, A, vectl, op=None, mode='old', in_b=(2, 1), out_a=(2, 3)):
    # A = [t l] [b r] s;  Ac = [t' l'] [b' r'] s;  vectl = x [l l'] [t t'] y
    """ Append the A and Ac tensors to the top-left vector """
    if A.config.fermionic:
        assert A.n == Ac.n == A.config.sym.zero(), "Sanity check; A, Ac should nor carry charge. "
    if op is not None:
        A = tensordot(A, op, axes=(2, 1))
    axes0 = tuple(ax for ax in range(vectl.ndim) if ax not in in_b)
    axes0 = (in_b[0], axes0, in_b[1])  # (2, (0, 3), 1), in_b == (t, l)
    vectl = vectl.fuse_legs(axes=axes0)  # [t t'] [x y] [l l']
    vectl = vectl.unfuse_legs(axes=(0, 2))  # t t' [x y] l l'
    vectl = vectl.swap_gate(axes=(1, (3, 4)))  # t' X l l'
    vectl = vectl.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [t l] [x y] [t' l']
    vectl = vectl.tensordot(Ac.conj(), axes=(2, 0))  # [t l] [x y] [b' r'] s
    vectl = A.tensordot(vectl, axes=((0, 2), (0, 3)))  # [b r] [x y] [b' r']
    vectl = vectl.unfuse_legs(axes=(0, 2))  # b r [x y] b' r'
    vectl = vectl.swap_gate(axes=((0, 3), 4))  # b b' X r'

    if mode == 'old':
        vectl = vectl.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [b b'] [x y] [r r']
        vectl = vectl.unfuse_legs(axes=1)  # [b b'] x y [r r']
        return vectl.transpose(axes=(1, 0, 2, 3))  # x [b b'] y [r r']

    if out_a == (2, 3):
        axes1 = ((0, 3), (1, 4))  # [b b'] [r r']
    elif out_a == (3, 2):
        axes1 = ((1, 4), (0, 3))  # [r r'] [b b']
    if mode == 'self-b':
        axes1, axes2 = axes1 + (2,), 2  # [] [] [x y] -> [] [] x y
    elif mode == 'b-self':
        axes1, axes2 = (2,) + axes1, 0  # [x y] [] [] -> x y [] []
    vectl = vectl.fuse_legs(axes=axes1)
    return vectl.unfuse_legs(axes=axes2)


def append_vec_br(Ac, A, vecbr, op=None, mode='old', in_b=(2, 1), out_a=(0, 1)):
    # A = [t l] [b r] s;  Ac = [t' l'] [b' r'] s;  vecbr = x [r r'] [b b'] y
    """ Append the A and Ac tensors to the bottom-right vector. """
    if A.config.fermionic:
        assert A.n == Ac.n == A.config.sym.zero(), "Sanity check; A, Ac should nor carry charge. "
    if op is not None:
        A = tensordot(A, op, axes=(2, 1))
    axes0 = tuple(ax for ax in range(vecbr.ndim) if ax not in in_b)
    axes0 = (in_b[0], axes0, in_b[1])  # (2, (0, 3), 1), in_b == (b, r)
    vecbr = vecbr.fuse_legs(axes=axes0)  # [b b'] [x y] [r r']
    vecbr = vecbr.unfuse_legs(axes=(0, 2))  # b b' [x y] r r'
    vecbr = vecbr.swap_gate(axes=((0, 1), 4))  # b b' X r'
    vecbr = vecbr.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [b r] [x y] [b' r']
    vecbr = vecbr.tensordot(Ac.conj(), axes=(2, 1))  # [b r] [x y] [t' l'] s
    vecbr = A.tensordot(vecbr, axes=((1, 2), (0, 3)))  # [t l] [x y] [t' l']
    vecbr = vecbr.unfuse_legs(axes=(0, 2))  # t l [x y] t' l'
    vecbr = vecbr.swap_gate(axes=((1, 4), 3))  # l l' X t'

    if mode == 'old':
        vecbr = vecbr.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [t t'] [x y] [l l']
        vecbr = vecbr.unfuse_legs(axes=1)  # [t t'] x y [l l']
        return vecbr.transpose(axes=(1, 0, 2, 3))  # x [t t'] y [l l']

    if out_a == (0, 1):
        axes1 = ((0, 3), (1, 4))  # [t t'] [l l']
    elif out_a == (1, 0):
        axes1 = ((1, 4), (0, 3))  # [l l'] [t t']
    if mode == 'self-b':
        axes1, axes2 = axes1 + (2,), 2  # [] [] [x y] -> [] [] x y
    elif mode == 'b-self':
        axes1, axes2 = (2,) + axes1, 0  # [x y] [] [] -> x y [] []
    vecbr = vecbr.fuse_legs(axes=axes1)
    return vecbr.unfuse_legs(axes=axes2)


def append_vec_tr(Ac, A, vectr, op=None, mode='old', in_b=(1, 2), out_a=(1, 2)):
    # A = [t l] [b r] s;  Ac = [t' l'] [b' r'] s;  vectr = x [t t'] [r r'] y
    """ Append the A and Ac tensors to the top-left vector """
    if A.config.fermionic:
        assert A.n == Ac.n == A.config.sym.zero(), "Sanity check; A, Ac should nor carry charge. "

    if op is not None:
        A = tensordot(A, op, axes=(2, 1))
        n_vec = A.config.sym.add_charges(vectr.n, op.n)
    else:
        n_vec = vectr.n
    # We have to assume that vactr can carry explicit charge;
    # We will swap with it for optimal swap placement.

    A = A.unfuse_legs(axes=(0, 1))  # t l b r s
    A = A.swap_gate(axes=(2, 4))  # b X s
    A = A.fuse_legs(axes=((0, 3), (1, 2), 4))  # [t r] [l b] s

    Ac = Ac.unfuse_legs(axes=(0, 1))  # t' l' b' r' s
    Ac = Ac.swap_gate(axes=(2, (0, 3)))  # b' X t' r'
    Ac = Ac.fuse_legs(axes=((0, 3), (1, 2), 4))  # [t' r'] [l' b'] s

    axes0 = tuple(ax for ax in range(vectr.ndim) if ax not in in_b)
    axes0 = (in_b[0], axes0, in_b[1])  # (1, (0, 3), 2), in_b == (t, r)
    vectr = vectr.fuse_legs(axes=axes0)  # [t t'] [x y] [r r']
    vectr = vectr.unfuse_legs(axes=(0, 2))  # t t' [x y] r r'
    vectr = vectr.swap_gate(axes=(1, 2))  # t' X [x y]
    vectr = vectr.swap_gate(axes=1, charge=n_vec)  # t' X [charge_vectr op.n]

    vectr = vectr.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [t r] [x y] [t' r']
    vectr = vectr.tensordot(Ac.conj(), axes=(2, 0))  # [t r] [x y] [l' b'] s
    vectr = A.tensordot(vectr, axes=((0, 2), (0, 3)))  # [l b] [x y] [l' b']
    vectr = vectr.unfuse_legs(axes=(0, 2))  # l b [x y] l' b'
    vectr = vectr.swap_gate(axes=(1, (3, 4)))  # b X l' b'

    if mode == 'old':
        vectr = vectr.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [l l'] [x y] [b b']
        vectr = vectr.unfuse_legs(axes=1)  # [l l'] x y [b b']
        return vectr.transpose(axes=(1, 0, 2, 3))  # x [l l'] y [b b']

    if out_a == (1, 2):
        axes1 = ((0, 3), (1, 4))  # [l l'] [b b']
    elif out_a == (2, 1):
        axes1 = ((1, 4), (0, 3))  # [b b'] [l l']
    if mode == 'self-b':
        axes1, axes2 = axes1 + (2,), 2  # [] [] [x y] -> [] [] x y
    elif mode == 'b-self':
        axes1, axes2 = (2,) + axes1, 0  # [x y] [] [] -> x y [] []
    vectr = vectr.fuse_legs(axes=axes1)
    return vectr.unfuse_legs(axes=axes2)


def append_vec_bl(Ac, A, vecbl, op=None, mode='old', in_b=(2, 1), out_a=(0, 3)):
    # A = [t l] [b r] s;  Ac = [t' l'] [b' r'] s;  vecbl = x [b b'] [l l'] y
    """ Append the A and Ac tensors to the top-left vector. """
    if A.config.fermionic:
        assert A.n == Ac.n == A.config.sym.zero(), "Sanity check; A, Ac should nor carry charge. "

    if op is not None:
        A = tensordot(A, op, axes=(2, 1))
    n_vec = vecbl.n  # We have to assume that vacbl can carry explicit charge;
    # We will swap with it for optimal swap placement.

    A = A.unfuse_legs(axes=(0, 1))  # t l b r s
    A = A.swap_gate(axes=(2, 4))  # b X s
    A = A.fuse_legs(axes=((2, 1), (3, 0), 4))  # [b l] [r t] s

    Ac = Ac.unfuse_legs(axes=(0, 1))  # t' l' b' r' s
    Ac = Ac.swap_gate(axes=(2, (0, 3)))  # b' X t' r'
    Ac = Ac.fuse_legs(axes=((2, 1), (3, 0), 4))  # [b' l'] [r' t'] s

    axes0 = tuple(ax for ax in range(vecbl.ndim) if ax not in in_b)
    axes0 = (in_b[1], axes0, in_b[0])  # (1, (0, 3), 2), in_b == (l, b)
    vecbl = vecbl.fuse_legs(axes=axes0)  # [b b'] [x y] [l l']
    vecbl = vecbl.unfuse_legs(axes=(0, 2))  # b b' [x y] l l'
    vecbl = vecbl.swap_gate(axes=(0, (1, 4)))  # b X b' l'
    vecbl = vecbl.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [b l] [x y] [b' l']
    vecbl = vecbl.tensordot(Ac.conj(), axes=(2, 0))  # [b l] [x y] [r' t'] s
    vecbl = A.tensordot(vecbl, axes=((0, 2), (0, 3)))  # [r t] [x y] [r' t']
    vecbl = vecbl.unfuse_legs(axes=(0, 2))  # r t [x y] r' t'
    vecbl = vecbl.swap_gate(axes=(2, 4))  #  [x y] X t'
    vecbl = vecbl.swap_gate(axes=4, charge=n_vec)  # t' X [charge_vec A]

    if mode == 'old':
        vecbl = vecbl.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [r r'] [x y] [t t']
        vecbl = vecbl.unfuse_legs(axes=1)  # [r r'] x y [t t']
        return vecbl.transpose(axes=(1, 0, 2, 3))  # x [r r'] y [t t']

    if out_a == (3, 0):
        axes1 = ((0, 3), (1, 4))  # [r r'] [t t']
    elif out_a == (0, 3):
        axes1 = ((1, 4), (0, 3))  # [t t'] [r r']
    if mode == 'self-b':
        axes1, axes2 = axes1 + (2,), 2  # [] [] [x y] -> [] [] x y
    elif mode == 'b-self':
        axes1, axes2 = (2,) + axes1, 0  # [x y] [] [] -> x y [] []
    vecbl = vecbl.fuse_legs(axes=axes1)
    return vecbl.unfuse_legs(axes=axes2)


def clear_projectors(sites, projectors, xrange, yrange):
    """ prepare projectors for sampling functions. """
    if not isinstance(projectors, dict) or all(isinstance(x, Tensor) for x in projectors.values()):
        projectors = {site: projectors for site in sites}  # spread projectors over sites
    if set(sites) != set(projectors.keys()):
        raise YastnError(f"Projectors not defined for some sites in xrange={xrange}, yrange={yrange}.")

    # change each list of projectors into keys and projectors
    projs_sites = {}
    for k, v in projectors.items():
        projs_sites[k] = dict(v) if isinstance(v, dict) else dict(enumerate(v))
        for l, pr in projs_sites[k].items():
            if pr.ndim == 1:  # vectors need conjugation
                if abs(pr.norm() - 1) > 1e-10:
                    raise YastnError("Local states to project on should be normalized.")
                projs_sites[k][l] = tensordot(pr, pr.conj(), axes=((), ()))
            elif pr.ndim == 2:
                if (pr.n != pr.config.sym.zero()) or abs(pr @ pr - pr).norm() > 1e-10:
                    raise YastnError("Matrix projectors should be projectors, P @ P == P.")
            elif pr.ndim == 4:
                pass
            else:
                raise YastnError("Projectors should consist of vectors (ndim=1) or matrices (ndim=2).")
    
    return projs_sites
