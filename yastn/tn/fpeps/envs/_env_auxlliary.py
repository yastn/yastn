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
from .... import tensordot, ones, Leg, eye, Tensor, YastnError, ncon
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
    return triv


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


def hair_t(A_bra, ht=None, hl=None, hr=None, A_ket=None):
    """ top hair tensor """
    if A_ket is None:
        A_ket = A_bra  # t l b r sa
    if hr is not None:
        A_ket = ncon([hr, A_ket], [(-3, 1), (-0, -1, -2, 1, -4)])
    if hl is not None:
        A_ket = ncon([hl, A_ket], [(-1, 1), (-0, 1, -2, -3, -4)])
    if ht is not None:
        A_ket = ncon([ht, A_ket], [(-0, 1), (1, -1, -2, -3, -4)])
    return tensordot(A_bra.conj(), A_ket, axes=((0, 1, 3, 4), (0, 1, 3, 4)))  # b' b


def hair_l(A_bra, ht=None, hl=None, hb=None, A_ket=None):
    """ left hair tensor """
    if A_ket is None:
        A_ket = A_bra  # t l b r sa
    if hb is not None:
        A_ket = ncon([hb, A_ket], [(-2, 1), (-0, -1, 1, -3, -4)])
    if hl is not None:
        A_ket = ncon([hl, A_ket], [(-1, 1), (-0, 1, -2, -3, -4)])
    if ht is not None:
        A_ket = ncon([ht, A_ket], [(-0, 1), (1, -1, -2, -3, -4)])
    return tensordot(A_bra.conj(), A_ket, axes=((0, 1, 2, 4), (0, 1, 2, 4)))  # r' r


def hair_b(A_bra, hl=None, hb=None, hr=None, A_ket=None):
    """ bottom hair tensor """
    if A_ket is None:
        A_ket = A_bra  # t l b r sa
    if hr is not None:
        A_ket = ncon([hr, A_ket], [(-3, 1), (-0, -1, -2, 1, -4)])
    if hb is not None:
        A_ket = ncon([hb, A_ket], [(-2, 1), (-0, -1, 1, -3, -4)])
    if hl is not None:
        A_ket = ncon([hl, A_ket], [(-1, 1), (-0, 1, -2, -3, -4)])
    return tensordot(A_bra.conj(), A_ket, axes=((1, 2, 3, 4), (1, 2, 3, 4)))  # t' t


def hair_r(A_bra, ht=None, hb=None, hr=None, A_ket=None):
    """ right hair tensor """
    if A_ket is None:
        A_ket = A_bra  # t l b r sa
    if hr is not None:
        A_ket = ncon([hr, A_ket], [(-3, 1), (-0, -1, -2, 1, -4)])
    if hb is not None:
        A_ket = ncon([hb, A_ket], [(-2, 1), (-0, -1, 1, -3, -4)])
    if ht is not None:
        A_ket = ncon([ht, A_ket], [(-0, 1), (1, -1, -2, -3, -4)])
    return tensordot(A_bra.conj(), A_ket, axes=((0, 2, 3, 4), (0, 2, 3, 4)))  # l' l


def cor_tl(A_bra, ht=None, hl=None, A_ket=None):
    """ top-left corner tensor """
    if A_ket is None:
        A_ket = A_bra  # t l b r sa
    if hl is not None:
        A_ket = ncon([hl, A_ket], [(-1, 1), (-0, 1, -2, -3, -4)])
    if ht is not None:
        A_ket = ncon([ht, A_ket], [(-0, 1), (1, -1, -2, -3, -4)])
    ctl = tensordot(A_bra, A_ket.conj(), axes=((0, 1, 4), (0, 1, 4)))  # b r b' r'
    ctl = ctl.swap_gate(axes=((0, 2), 3))  # b b' X r'
    return ctl.fuse_legs(axes=((0, 2), (1, 3)))  # [b b'] [r r']


def cor_bl(A_bra, hb=None, hl=None, A_ket=None):
    """ bottom-left corner tensor """
    if A_ket is None:
        A_ket = A_bra  # t l b r sa
    if hb is not None:
        A_ket = ncon([hb, A_ket], [(-2, 1), (-0, -1, 1, -3, -4)])
    if hl is not None:
        A_ket = ncon([hl, A_ket], [(-1, 1), (-0, 1, -2, -3, -4)])
    cbl = tensordot(A_bra, A_ket.conj(), axes=((1, 2, 4), (1, 2, 4)))  # t r t' r'
    return cbl.fuse_legs(axes=((1, 3), (0, 2)))  # [r r'] [t t']


def cor_br(A_bra, hb=None, hr=None, A_ket=None):
    """ bottom-right corner tensor """
    if A_ket is None:
        A_ket = A_bra  # t l b r sa
    if hr is not None:
        A_ket = ncon([hr, A_ket], [(-3, 1), (-0, -1, -2, 1, -4)])
    if hb is not None:
        A_ket = ncon([hb, A_ket], [(-2, 1), (-0, -1, 1, -3, -4)])
    cbr = tensordot(A_bra, A_ket.conj(), axes=((2, 3, 4), (2, 3, 4)))  # t l t' l'
    cbr = cbr.swap_gate(axes=((1, 3), 2))  # l l' X t'
    return cbr.fuse_legs(axes=((0, 2), (1, 3)))  # [t t'] [l l']


def cor_tr(A_bra, ht=None, hr=None, A_ket=None):
    """ top-right corner tensor """
    A_bra = A_bra.swap_gate(axes=(0, 1, 2, 3))  # t X l, b X r
    A_ket = A_bra if A_ket is None else A_ket.swap_gate(axes=(0, 1, 2, 3))  # t X l, b X r
    if hr is not None:
        A_ket = ncon([hr, A_ket], [(-3, 1), (-0, -1, -2, 1, -4)])
    if ht is not None:
        A_ket = ncon([ht, A_ket], [(-0, 1), (1, -1, -2, -3, -4)])
    ctr = tensordot(A_bra, A_ket.conj(), axes=((0, 3, 4), (0, 3, 4)))  # l b l' b'
    return ctr.fuse_legs(axes=((0, 2), (1, 3)))  # [l l'] [b b']


def edge_l(A_bra, hl=None, A_ket=None):
    """ left edge tensor where left legs of double-layer A tensors can be contracted with hl """
    if A_ket is None:
        A_ket = A_bra  # t l b r sa
    if hl is not None:
        A_ket = ncon([hl, A_ket], [(-1, 1), (-0, 1, -2, -3, -4)])
    egl = tensordot(A_bra, A_ket.conj(), axes=((1, 4), (1, 4)))  # t b r t' b' r'
    egl = egl.swap_gate(axes=((1, 4), 5))  # b b' X r'
    return egl.fuse_legs(axes=((1, 4), (2, 5), (0, 3)))  # [b b'] [r r'] [t t']


def edge_t(A_bra, ht=None, A_ket=None):
    """ top edge tensor where top legs of double-layer A tensors can be contracted with ht """
    A_bra = A_bra.swap_gate(axes=(0, 1))  # t X l
    A_ket = A_bra if A_ket is None else A_ket.swap_gate(axes=(0, 1))  # t X l
    if ht is not None:
        A_ket = ncon([ht, A_ket], [(-0, 1), (1, -1, -2, -3, -4)])
    egt = tensordot(A_bra, A_ket.conj(), axes=((0, 4), (0, 4)))  # l b r l' b' r'
    egt = egt.swap_gate(axes=((1, 4), 5))  # b b' X r'
    return egt.fuse_legs(axes=((0, 3), (1, 4), (2, 5))) # [l l'] [b b'] [r r']


def edge_r(A_bra, hr=None, A_ket=None):
    """ right edge tensor where right legs of double-layer A tensors can be contracted with hr """
    A_bra = A_bra.swap_gate(axes=(2, 3))  # b X r
    A_ket = A_bra if A_ket is None else A_ket.swap_gate(axes=(2, 3))  # b X r
    if hr is not None:
        A_ket = ncon([hr, A_ket], [(-3, 1), (-0, -1, -2, 1, -4)])
    egr = tensordot(A_bra, A_ket.conj(), axes=((3, 4), (3, 4)))  # t l b t' l' b'
    egr = egr.swap_gate(axes=((1, 4), 3))  # l l' X t'
    return egr.fuse_legs(axes=((0, 3), (1, 4), (2, 5)))  # [t t'] [l l'] [b b']


def edge_b(A_bra, hb=None, A_ket=None):  # A = [t l] [b r] s;  hb = b' b
    """ bottom edge tensor where bottom legs of double-layer A tensors can be contracted with hb """
    if A_ket is None:
        A_ket = A_bra  # t l b r sa
    if hb is not None:
        A_ket = ncon([hb, A_ket], [(-2, 1), (-0, -1, 1, -3, -4)])
    egb = tensordot(A_bra, A_ket.conj(), axes=((2, 4), (2, 4)))  # t l r t' l' r'
    egb = egb.swap_gate(axes=((1, 4), 3)) # l l' X t'
    return egb.fuse_legs(axes=((2, 5), (0, 3), (1, 4)))  # [r r'] [t t'] [l l']


def append_vec_tl(Ac, A, vectl, op=None, mode='old', in_b=(2, 1), out_a=(2, 3)):
    # A = t l b r s;  Ac = t' l' b' r' s;  vectl = x [l l'] [t t'] y (order in axes0)
    """ Append the A and Ac tensors to the top-left vector """
    if A.config.fermionic:
        assert A.n == Ac.n == A.config.sym.zero(), "Sanity check; A, Ac should nor carry charge. "
    if op is not None:
        A = tensordot(A, op, axes=(4, 1))
    axes0 = tuple(ax for ax in range(vectl.ndim) if ax not in in_b)
    axes0 = (in_b[0], axes0, in_b[1])  # (2, (0, 3), 1), in_b == (t, l)
    vectl = vectl.fuse_legs(axes=axes0)  # [t t'] [x y] [l l']
    vectl = vectl.unfuse_legs(axes=(0, 2))  # t t' [x y] l l'
    vectl = vectl.swap_gate(axes=(1, (3, 4)))  # t' X l l'
    vectl = tensordot(vectl, Ac.conj(), axes=((1, 4), (0, 1)))  # t [x y] l b' r' s
    vectl = tensordot(A, vectl, axes=((0, 1, 4), (0, 2, 5)))  # b r [x y] b' r'
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
    # A = t l b r s;  Ac = t' l' b' r' s;  vecbr = x [r r'] [b b'] y (order in axes0)
    """ Append the A and Ac tensors to the bottom-right vector. """
    if A.config.fermionic:
        assert A.n == Ac.n == A.config.sym.zero(), "Sanity check; A, Ac should nor carry charge. "
    if op is not None:
        A = tensordot(A, op, axes=(4, 1))
    axes0 = tuple(ax for ax in range(vecbr.ndim) if ax not in in_b)
    axes0 = (in_b[0], axes0, in_b[1])  # (2, (0, 3), 1), in_b == (b, r)
    vecbr = vecbr.fuse_legs(axes=axes0)  # [b b'] [x y] [r r']
    vecbr = vecbr.unfuse_legs(axes=(0, 2))  # b b' [x y] r r'
    vecbr = vecbr.swap_gate(axes=((0, 1), 4))  # b b' X r'
    vecbr = tensordot(vecbr, Ac.conj(), axes=((1, 4), (2, 3)))  # b [x y] r t' l' s
    vecbr = tensordot(A, vecbr, axes=((2, 3, 4), (0, 2, 5)))  # t l [x y] t' l'
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
        A = tensordot(A, op, axes=(4, 1))
        n_vec = A.config.sym.add_charges(vectr.n, op.n)
    else:
        n_vec = vectr.n
    # We have to assume that vactr can carry explicit charge;
    # We will swap with it for good swap placement.

    A = A.swap_gate(axes=(2, 4))  # b X s
    Ac = Ac.swap_gate(axes=(2, (0, 3)))  # b' X t' r'

    axes0 = tuple(ax for ax in range(vectr.ndim) if ax not in in_b)
    axes0 = (in_b[0], axes0, in_b[1])  # (1, (0, 3), 2), in_b == (t, r)
    vectr = vectr.fuse_legs(axes=axes0)  # [t t'] [x y] [r r']
    vectr = vectr.unfuse_legs(axes=(0, 2))  # t t' [x y] r r'
    vectr = vectr.swap_gate(axes=(1, 2))  # t' X [x y]
    vectr = vectr.swap_gate(axes=1, charge=n_vec)  # t' X [charge_vectr op.n]

    vectr = tensordot(vectr, Ac.conj(), axes=((1, 4), (0, 3)))  # t [x y] r l' b' s
    vectr = tensordot(A, vectr, axes=((0, 3, 4), (0, 2, 5)))  # l b [x y] l' b'
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
        A = tensordot(A, op, axes=(4, 1))
    n_vec = vecbl.n
    # We have to assume that vacbl can carry explicit charge;
    # We will swap with it for good swap placement.

    A = A.swap_gate(axes=(2, 4))  # b X s
    Ac = Ac.swap_gate(axes=(2, (0, 3)))  # b' X t' r'

    axes0 = tuple(ax for ax in range(vecbl.ndim) if ax not in in_b)
    axes0 = (in_b[1], axes0, in_b[0])  # (1, (0, 3), 2), in_b == (l, b)
    vecbl = vecbl.fuse_legs(axes=axes0)  # [b b'] [x y] [l l']
    vecbl = vecbl.unfuse_legs(axes=(0, 2))  # b b' [x y] l l'
    vecbl = vecbl.swap_gate(axes=(0, (1, 4)))  # b X b' l'

    vecbl = tensordot(vecbl, Ac.conj(), axes=((1, 4), (2, 1)))  # b [x y] l t' r' t' s
    vecbl = tensordot(A, vecbl, axes=((2, 1, 4), (0, 2, 5)))  # t r [x y] t' r'

    vecbl = vecbl.swap_gate(axes=(2, 3))  #  [x y] X t'
    vecbl = vecbl.swap_gate(axes=3, charge=n_vec)  # t' X [charge_vec A]

    if mode == 'old':
        vecbl = vecbl.fuse_legs(axes=((1, 4), 2, (0, 3)))  # [r r'] [x y] [t t']
        vecbl = vecbl.unfuse_legs(axes=1)  # [r r'] x y [t t']
        return vecbl.transpose(axes=(1, 0, 2, 3))  # x [r r'] y [t t']

    if out_a == (3, 0):
        axes1 = ((1, 4), (0, 3))  # [r r'] [t t']
    elif out_a == (0, 3):
        axes1 = ((0, 3), (1, 4))  # [t t'] [r r']
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
