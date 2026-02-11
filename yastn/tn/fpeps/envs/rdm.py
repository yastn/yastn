# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
from functools import reduce
import logging
from typing import Sequence, Union, TypeVar

from ._env_contractions import append_vec_tl, append_vec_br, append_vec_tr, append_vec_bl
from ._env_ctm import EnvCTM
from .._geometry import Site
from .._peps import Peps, Peps2Layers
from ....tensor import ncon, tensordot, Tensor, YastnError

log = logging.getLogger(__name__)
Scalar = TypeVar('Scalar')


# utility functions for corner contractions, leaving the physical indices uncontracted.
def _append_vec_tl_open(Ac, A, vectl):
    """
    Append the A and Ac tensors to the top-left vector with open and unswapped physical indices [s s'].
    A = t l b r s;  Ac = t' l' b' r' s';  vectl = x [l l'] [t t'] y
    """
    vectl = vectl.fuse_legs(axes=(2, (0, 3), 1))  # [t t'] [x y] [l l']
    vectl = vectl.unfuse_legs(axes=(0, 2))  # t t' [x y] l l'
    vectl = vectl.swap_gate(axes=(1, (3, 4)))  # t' X l l'
    vectl = tensordot(vectl, Ac.conj(), axes=((1, 4), (0, 1)))  # t [x y] l b' r' s'
    vectl = tensordot(A, vectl, axes=((0, 1), (0, 2)))  # b r s [x y] b' r' s'
    vectl = vectl.swap_gate(axes=((0, 4), 5))  # b b' X r'
    vectl = vectl.fuse_legs(axes=((0, 4), 3, (1, 5), (2, 6)))  # [b b'] [x y] [r r'] [s s']
    vectl = vectl.unfuse_legs(axes=1)  # [b b'] x y [r r'] [s s']
    vectl = vectl.transpose(axes=(1, 0, 2, 3, 4))  # x [b b'] y [r r'] [s s']
    return vectl


def _append_vec_br_open(Ac, A, vecbr):
    """
    Append the A and Ac tensors to the bottom-right vector with open and unswapped physical indices [s s'].
    A = t l b r s;  Ac = t' l' b' r' s';  vecbr = x [r r'] [b b'] y
    """
    vecbr = vecbr.fuse_legs(axes=(2, (0, 3), 1))  # [b b'] [x y] [r r']
    vecbr = vecbr.unfuse_legs(axes=(0, 2))  # b b' [x y] r r'
    vecbr = vecbr.swap_gate(axes=((0, 1), 4))  # b b' X r'
    vecbr = tensordot(vecbr, Ac.conj(), axes=((1, 4), (2, 3)))  # b [x y] r t' l' s'
    vecbr = tensordot(A, vecbr, axes=((2, 3), (0, 2)))  # t l s [x y] t' l' s'
    vecbr = vecbr.swap_gate(axes=((1, 5), 4))  # l l' X t'
    vecbr = vecbr.fuse_legs(axes=((0, 4), 3, (1, 5), (2, 6)))  # [t t'] [x y] [l l'] [s s']
    vecbr = vecbr.unfuse_legs(axes=1)  # [t t'] x y [l l'] [s s']
    vecbr = vecbr.transpose(axes=(1, 0, 2, 3, 4))  # x [t t'] y [l l'] [s s']
    return vecbr


def _append_vec_tr_open(Ac, A, vectr):
    """
    Append the A and Ac tensors to the top-right vector with open physical indices [s s'], swapped with t'.
    A = t l b r s;  Ac = t' l' b' r' s';  vectr = x [t t'] [r r'] y
    """
    vectr = vectr.fuse_legs(axes=(1, (0, 3), 2))  # [t t'] [x y] [r r']
    vectr = vectr.unfuse_legs(axes=(0, 2))  # t t' [x y] r r'
    vectr = tensordot(A, vectr, axes=((0, 3), (0, 3)))  # l b s t' [x y] r'
    vectr = vectr.swap_gate(axes=(3, (0, 2), 1, 5))  # t' X l s and r' X b
    Ac = Ac.swap_gate(axes=(0, (1, 4), 2, 3))  # t' X l' s' and b' X r'
    vectr = tensordot(vectr, Ac.conj(), axes=((3, 5), (0, 3)))  # l b s [x y] l' b' s'
    vectr = vectr.fuse_legs(axes=(3, (0, 4), (1, 5), (2, 6)))  # [x y] [l l'] [b b'] [s s']
    vectr = vectr.unfuse_legs(axes=0)  # x y [l l'] [b b'] [s s']
    vectr = vectr.transpose(axes=(0, 2, 1, 3, 4))  # x [l l'] y [b b'] [s s']
    return vectr


def _append_vec_bl_open(Ac, A, vecbl):
    """
    Append the A and Ac tensors to the bottom-left vector with open physical indices, swapped with b.
    A = t l b r s;  Ac = t' l' b' r' s';  vecbl = x [b b'] [l l'] y
    """
    vecbl = vecbl.fuse_legs(axes=(1, (0, 3), 2))  # [b b'] [x y] [l l']
    vecbl = vecbl.unfuse_legs(axes=(0, 2))  # b b' [x y] l l'
    Ac = Ac.swap_gate(axes=(0, 1, 2, 3))  # t' X l' and b' X r'
    vecbl = tensordot(vecbl, Ac.conj(), axes=((1, 4), (2, 1)))  # b [x y] l t' r' s'
    vecbl = vecbl.swap_gate(axes=(2, 3, 0, (4, 5)))  # l X t' and b X r' s'
    A = A.swap_gate(axes=(2, 4))  # b X s
    vecbl = tensordot(A, vecbl, axes=((2, 1), (0, 2)))  # t r s [x y] t' r' s'
    vecbl = vecbl.fuse_legs(axes=((3, (1, 5), (0, 4), (2, 6))))  # [x y] [r r'] [t t'] [s s']
    vecbl = vecbl.unfuse_legs(axes=0)  # x y [r r'] [t t'] [s s']
    vecbl = vecbl.transpose(axes=(0, 2, 1, 3, 4))  # x [r r'] y [t t'] [s s']
    return vecbl


# utility functions to regularize RDMs and log warnings in case of unphysical RDMs
def _normalize_and_regularize_rdm(rdm, order : str="interleaved", pos_def=False, who=None, verbosity=0, **kwargs):
    r"""
    Regularize reduced density matrix (RDM).

    Args:
        order: Index convention of RDM. `"interleaved"` for `[s0, s0', s1, s1', ...]`
        where s_i,s_i' is bra,ket pair. `"braket"` for `[s0,s1,...,s0',s1',...]` with all "bra" indices first
        followed by "ket" indices.
    """
    assert rdm.ndim % 2 == 0, "invalid rank of RDM"
    nsites = rdm.ndim // 2

    trace_axes=None
    conj_order=None
    if order=="interleaved":
        trace_order= (tuple(2*i for i in range(nsites)),tuple(2*i+1 for i in range(nsites)))
        conj_order= sum(zip(trace_order[1],trace_order[0]),())
    elif order=="braket":
        trace_order= (tuple(i for i in range(nsites)),tuple(i+nsites for i in range(nsites)))
        conj_order= trace_order[1]+trace_order[0]
    else:
        raise ValueError(f"order {order} not implemented.")

    rdm_norm= rdm.trace(axes=trace_order).to_number()
    # turn RDM into a matrix
    # rdm_asym = 0.5 * (rdm - rdm.conj().transpose(axes=conj_order))
    # rdm = 0.5 * (rdm + rdm.conj().transpose(axes=conj_order))

    # # given enforced symmetry of rdm, the trace has to be real
    # rdm_norm= rdm.trace(axes=trace_order).to_number().real
    # if verbosity > 0:
    #     log.info(f"{who} trace(rdm_sym) {rdm_norm} 2-norm(rdm_sym) {rdm.norm()} 2-norm(rdm_asym) {rdm_asym.norm()}")
    # if pos_def:
    #     # shift spectrum such that RDM is non-negative
    #     raise NotImplementedError()

    rdm = rdm/rdm_norm
    return rdm, rdm_norm


def trace_aux(rdm : Tensor, axis : int, swap : bool = False) -> Tensor:
    r"""
    This function assumes following index structure of reduced density matrix `rdm`:

        rdm[p0,p0',p1,p1',...]

    where p_i,p_i' is a bra,ket pair of indices.

    Trace out *auxiliary* physical index pair. Include swap gate if `swap`.

    Args:
        rdm: reduced density matrix
        axis: index of "bra" leg of leg pair to trace out
        swap: include swap gate when tracing

    Returns:
        reduced density matrix with auxiliary physical index pair traced out.

    TODO: also check that auxiliary leg has total dimension 1.
    """
    leg = rdm.get_legs(axes=axis)
    # check if a dummy leg is fused with the physical leg
    if leg.is_fused():
        rdm = rdm.unfuse_legs(axes=(axis, axis + 1))  # ... p d p' d' ...
        if swap:
            rdm = rdm.swap_gate(
                axes=(axis, axis + 1, axis + 2, axis + 3)
            )  # p X d & p' X d'
        rdm = rdm.trace(axes=(axis + 1, axis + 3))  # ... p p' ...
    return rdm


# def f_ordered(s0, s1) -> bool:
#     """Check if (s0, s1) appear in fermionic order.
#     The convention is consistent with the PEPS diagrams in https://arxiv.org/abs/0912.0646.
#     """
#     return s0[1] < s1[1] or (s0[1] == s1[1] and s0[0] <= s1[0])


def op_order(Oi, Oj, ordered, fermionic=True):
    """Preprocessing for the product of two fermionic operators Oi and Oj, with Oj acting first.
    A leg of odd-parity is added to connect the two operators, and a swap-gate is placed to
    be consistent with the predefined fermionic ordering.
    :param Oi: YASTN tensor
    :param Oj: YASTN tensor
    :param ordered: whether the product Oi Oj is consistent with the fermionic ordering
    """

    if not ordered:
        # if fermionic:
        #     Oi, Oj = -Oj, Oi
        # else:
        #     Oi, Oj = Oj, Oi

        if fermionic:
            Oi, Oj = -Oi, Oj

    # Add an auxiliary leg such that the total charges of Oi and Oj are zero, respectively
    Oi = Oi.add_leg(s=1)
    Oj = Oj.add_leg(s=-1)

    Oi = Oi.swap_gate(axes=(1, 2))
     #    |
     #  --+-----   |
     # |  |    |   |
     # |--Oi   ---Oj
     #    |       |

    return Oi, Oj


def rdm1x1(s0 : Site, psi : Peps, env : EnvCTM, **kwargs) -> tuple[Tensor, Scalar]:
    r"""
    Contract environment and on-site tensors of 1x1 patch centered at Site `s0`
    to reduced density matrix.

    TODO: Optionally symmetrize and make non-negative

    Args:
        s0: The site of the 1*1 reduced density matrix
        psi: Peps
        env: environment

    Returns:
        Reduced density matrix and its unnormalized trace
    """
    env0 = env[s0]
    psi_dl = Peps2Layers(psi)
    ten0 = psi_dl[s0]  # DoublePepsTensor

    vect = (env0.l @ env0.tl) @ (env0.t @ env0.tr)
    vecb = (env0.r @ env0.br) @ (env0.b @ env0.bl)

    tmp = _append_vec_tl_open(ten0.bra, ten0.ket, vect)
    res = vecb.tensordot(tmp, axes=((0, 1, 2, 3), (2, 3, 1, 0)))  # [s, s']

    rdm = res.unfuse_legs(axes=(0,))  # s s'

    rdm = trace_aux(rdm, 0, swap=False)

    # assert rdm.ndim == 2
    rdm, rdm_norm = _normalize_and_regularize_rdm(rdm, who=rdm1x1.__name__)

    return rdm, rdm_norm


def rdm1x2(s0 : Site, psi : Peps, env : EnvCTM, **kwargs) -> tuple[Tensor, Scalar]:
    r"""
    Contract environment and on-site tensors of 1x2 (horizontal) patch,
    with `s0` the leftmost Site, to reduced density matrix::

        C T  T  C
        T s0 s1 T
        C T  T  C

    The index convention for reduced density matrix is `[s0, s0', s1, s1']`,
    where s_i,s_i' is bra,ket pair.

    TODO: Optionally symmetrize and make non-negative

    Args:
        s0: The site of the 1x2 reduced density matrix
        psi: Peps
        env: environment

    Returns:
        Reduced density matrix and its unnormalized trace
    """
    s1 = psi.nn_site(s0, "r")

    env0, env1 = env[s0], env[s1]
    # psi_dl = Peps2Layers(psi)
    psi_dl = env.psi
    ten0, ten1 = psi_dl[s0], psi_dl[s1]  # DoublePepsTensor

    vecl = (env0.bl @ env0.l) @ (env0.tl @ env0.t)
    vecr = (env1.tr @ env1.r) @ (env1.br @ env1.b)

    tmp0 = _append_vec_tl_open(ten0.bra, ten0.ket, vecl)  # x [b b'] y [r r'] [s s']
    tmp0 = tmp0.unfuse_legs(axes=(1, 4))  # x b b' y [r r'] s s'
    tmp0 = tmp0.swap_gate(axes=(1, (5, 6)))  # b X s s'
    tmp0 = tmp0.fuse_legs(axes=(0, (1, 2), 3, 4, (5, 6)))  # x [b b'] y [r r'] [s s']
    tmp0 = env0.b.tensordot(tmp0, axes=((2, 1), (0, 1)))

    tmp1 = _append_vec_br_open(ten1.bra, ten1.ket, vecr)  # x [t t'] y [l l'] [s s']
    tmp1 = tmp1.unfuse_legs(axes=(1, 4))  # x t t' y [l l'] s s'
    tmp1 = tmp1.swap_gate(axes=(2, (5, 6)))  # t' X s s'
    tmp1 = tmp1.fuse_legs(axes=(0, (1, 2), 3, 4, (5, 6)))  # x [t t'] y [l l'] [s s']
    tmp1 = env1.t.tensordot(tmp1, axes=((2, 1), (0, 1)))

    res = tmp0.tensordot(tmp1, axes=((0, 1, 2), (1, 0, 2)))  # [s0 s0'] [s1 s1']

    rdm = res.unfuse_legs (axes=(0, 1))  # s0 s0' s1 s1'

    # We pick a convention such that the aux. leg of the
    # left or top site needs to be swapped with the physical legs
    rdm = trace_aux(rdm, 0, swap=True)
    rdm = trace_aux(rdm, 2, swap=False)
    rdm, rdm_norm = _normalize_and_regularize_rdm(rdm, who=rdm1x2.__name__)

    return rdm, rdm_norm


def rdm2x1(s0 : Site, psi : Peps, env : EnvCTM, **kwargs) -> tuple[Tensor, Scalar]:
    r"""
    Contract environment and on-site tensors of 2x1 (vertical) patch,
    with `s0` the top-most Site, to reduced density matrix::

        C T  C
        T s0 T
        T s1 T
        C T  C

    The index convention for reduced density matrix is `[s0, s0', s1, s1']`,
    where s_i,s_i' is bra,ket pair.

    TODO: Optionally symmetrize and make non-negative

    Args:
        s0: The site of the 2x1 reduced density matrix
        psi: Peps
        env: environment

    Returns:
        Reduced density matrix and its unnormalized trace
    """
    s1 = psi.nn_site(s0, "b")

    env0, env1 = env[s0], env[s1]
    # psi_dl = Peps2Layers(psi)
    psi_dl = env.psi
    ten0, ten1 = psi_dl[s0], psi_dl[s1]  # DoublePepsTensor

    vect = (env0.l @ env0.tl) @ (env0.t @ env0.tr)
    vecb = (env1.r @ env1.br) @ (env1.b @ env1.bl)

    tmp0 = _append_vec_tl_open(ten0.bra, ten0.ket, vect)  # x [b b'] y [r r'] [s s']
    tmp0 = tmp0.unfuse_legs(axes=(1, 3, 4))  # x b b' y r r' s s'
    tmp0 = tmp0.swap_gate(axes=(5, (6, 7)))  # r' X s s'
    tmp0 = tmp0.swap_gate(axes=(2, (6, 7)))  # b' X s s'
    tmp0 = tmp0.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [b b'] y [r r'] [s s']
    tmp0 = env0.r.tensordot(tmp0, axes=((0, 1), (2, 3)))

    tmp1 = _append_vec_br_open(ten1.bra, ten1.ket, vecb)  # x [t t'] y [l l'] [s s']
    tmp1 = tmp1.unfuse_legs(axes=(1, 3, 4))  # x t t' y l l' s s'
    tmp1 = tmp1.swap_gate(axes=(2, (6, 7)))  # t' X s s'
    tmp1 = tmp1.swap_gate(axes=(4, (6, 7)))  # l X s s'
    tmp1 = tmp1.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [b b'] y [r r'] [s s']
    tmp1 = env1.l.tensordot(tmp1, axes=((0, 1), (2, 3)))

    res = tmp0.tensordot(tmp1, axes=((0, 1, 2), (1, 0, 2)))  # [s0 s0'] [s1 s1']

    rdm = res.unfuse_legs(axes=(0, 1))  # s0 s0' s1 s1'
    rdm = trace_aux(rdm, 0, swap=True)
    rdm = trace_aux(rdm, 2, swap=False)
    rdm, rdm_norm= _normalize_and_regularize_rdm(rdm, who=rdm2x1.__name__)

    return rdm, rdm_norm


# def rdm2x2(s0 : Site, psi : Peps, env : EnvCTM, **kwargs) -> tuple[Tensor, Scalar]:
#     r"""
#     Contract environment and on-site tensors of 2x2 patch,
#     with `s0` the upper-left Site, to reduced density matrix::

#         C T  T  C
#         T s0 s1 T
#         T s2 s3 T
#         C T  T  C

#     The index convention for reduced density matrix is `[s0, s0', ..., s3, s3']`,
#     where s_i,s_i' is bra,ket pair.

#     TODO: Optionally symmetrize and make non-negative

#     Args:
#         s0: The site of the 2x2 reduced density matrix
#         psi: Peps
#         env: environment

#     Returns:
#         Reduced density matrix and its unnormalized trace
#     """
#     s1, s2, s3 = psi.nn_site(s0, "r"), psi.nn_site(s0, "b"), psi.nn_site(s0, "br")
#     env0, env1, env2, env3 = env[s0], env[s1], env[s2], env[s3]

#     psi_dl = Peps2Layers(psi)
#     ten0, ten1, ten2, ten3 = (
#         psi_dl[s0],
#         psi_dl[s1],
#         psi_dl[s2],
#         psi_dl[s3],
#     )  # DoublePepsTensor

#     vectl = (env0.l @ env0.tl) @ env0.t
#     vectr = (env1.t @ env1.tr) @ env1.r
#     vecbl = (env2.b @ env2.bl) @ env2.l
#     vecbr = (env3.r @ env3.br) @ env3.b

#     tmp0 = _append_vec_tl_open(ten0.bra, ten0.ket, vectl)  # x [b b'] y [r r'] [s s']
#     tmp0 = tmp0.unfuse_legs(axes=(1, 3, 4))  # x b b' y r r' s s'
#     tmp0 = tmp0.swap_gate(axes=(1, (6, 7)))  # b X s s'
#     tmp0 = tmp0.swap_gate(axes=(5, (6, 7)))  # r' X s s'
#     tmp0 = tmp0.fuse_legs(
#         axes=(0, (1, 2), 3, (4, 5), (6, 7))
#     )  # x [b b'] y [r r'] [s0 s0']

#     tmp1 = _append_vec_tr_open(ten1.bra, ten1.ket, vectr)  # x [l l'] y [b b'] [s s']
#     tmp1 = tmp1.unfuse_legs(axes=(1, 3, 4))  # x l l' y b b' s s'
#     tmp1 = tmp1.swap_gate(axes=(2, (6, 7)))  # l' X s s'
#     tmp1 = tmp1.fuse_legs(
#         axes=(0, (1, 2), 3, (4, 5), (6, 7))
#     )  # x [l l'] y [b b'] [s1 s1']

#     tmp2 = _append_vec_bl_open(ten2.bra, ten2.ket, vecbl)  # x [r r'] y [t t'] [s s']
#     tmp2 = tmp2.unfuse_legs(axes=(1, 3, 4))  # x r r' y t t' s s'
#     tmp2 = tmp2.swap_gate(axes=(1, (6, 7)))  # r X s s'
#     tmp2 = tmp2.fuse_legs(
#         axes=(0, (1, 2), 3, (4, 5), (6, 7))
#     )  # x [r r'] y [t t'] [s2 s2']

#     tmp3 = _append_vec_br_open(ten3.bra, ten3.ket, vecbr)  # x [t t'] y [l l'] [s s']
#     tmp3 = tmp3.unfuse_legs(axes=(1, 3, 4))  # x t t' y l l' s s'
#     tmp3 = tmp3.swap_gate(axes=(2, (6, 7)))  # t' X s s'
#     tmp3 = tmp3.swap_gate(axes=(4, (6, 7)))  # l X s s'
#     tmp3 = tmp3.fuse_legs(
#         axes=(0, (1, 2), 3, (4, 5), (6, 7))
#     )  # x [t t'] y [l l'] [s3 s3']

#     res = tmp0.tensordot(
#         tmp1, axes=((2, 3), (0, 1))
#     )  # x0 [b0 b0'] [s0 s0'] y1 [b1 b1'] [s2 s2']
#     res = res.tensordot(
#         tmp2, axes=((0, 1), (2, 3))
#     )  # [s0 s0'] y1 [b1 b1'] [s2 s2'] x2 [r2 r2'] [s1 s1']
#     res = res.tensordot(
#         tmp3, axes=((1, 2, 4, 5), (0, 1, 2, 3))
#     )  # [s0 s0'] [s2 s2'] [s1 s1'] [s3 s3']
#     rdm = res.unfuse_legs(axes=(0, 1, 2, 3))  # s0 s0' s2 s2' s1 s1' s3 s3'
#     rdm = trace_aux(rdm, 0, swap=True)
#     rdm = trace_aux(rdm, 2, swap=False)
#     rdm = trace_aux(rdm, 4, swap=True)
#     rdm = trace_aux(rdm, 6, swap=False)

#     rdm = rdm.transpose(axes=(0, 1, 4, 5, 2, 3, 6, 7)) # s0 s0' s1 s1' s2 s2' s3 s3'
#     rdm, rdm_norm= _normalize_and_regularize_rdm(rdm, who=rdm2x2.__name__)

#     return rdm, rdm_norm

def rdm2x2(s0 : Site, psi : Peps, env : EnvCTM, **kwargs) -> tuple[Tensor, Scalar]:
    r"""
    Contract environment and on-site tensors of 2x2 patch,
    with `s0` the upper-left Site, to reduced density matrix::

        C T  T  C
        T s0 s2 T
        T s1 s3 T
        C T  T  C

    The index convention for reduced density matrix is `[s0, s0', ..., s3, s3']`,
    where s_i,s_i' is bra,ket pair.

    TODO: Optionally symmetrize and make non-negative

    Args:
        s0: The site of the 2x2 reduced density matrix
        psi: Peps
        env: environment

    Returns:
        Reduced density matrix and its unnormalized trace
    """
    s1, s2, s3 = psi.nn_site(s0, "b"), psi.nn_site(s0, "r"), psi.nn_site(s0, "br")
    env0, env1, env2, env3 = env[s0], env[s1], env[s2], env[s3]

    psi_dl = Peps2Layers(psi)
    ten0, ten1, ten2, ten3 = (
        psi_dl[s0],
        psi_dl[s1],
        psi_dl[s2],
        psi_dl[s3],
    )  # DoublePepsTensor

    vectl = (env0.l @ env0.tl) @ env0.t
    vectr = (env2.t @ env2.tr) @ env2.r
    vecbl = (env1.b @ env1.bl) @ env1.l
    vecbr = (env3.r @ env3.br) @ env3.b

    tmp0 = _append_vec_tl_open(ten0.bra, ten0.ket, vectl)  # x [b b'] y [r r'] [s s']
    tmp0 = tmp0.unfuse_legs(axes=(1, 3, 4))  # x b b' y r r' s s'
    tmp0 = tmp0.swap_gate(axes=(1, (6, 7)))  # b X s s'
    tmp0 = tmp0.swap_gate(axes=(5, (6, 7)))  # r' X s s'
    tmp0 = tmp0.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [b b'] y [r r'] [s0 s0']

    tmp2 = _append_vec_tr_open(ten2.bra, ten2.ket, vectr)  # x [l l'] y [b b'] [s s']
    tmp2 = tmp2.unfuse_legs(axes=(1, 3, 4))  # x l l' y b b' s s'
    tmp2 = tmp2.swap_gate(axes=(2, (6, 7)))  # l' X s s'
    tmp2 = tmp2.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [l l'] y [b b'] [s1 s1']

    tmp1 = _append_vec_bl_open(ten1.bra, ten1.ket, vecbl)  # x [r r'] y [t t'] [s s']
    tmp1 = tmp1.unfuse_legs(axes=(1, 3, 4))  # x r r' y t t' s s'
    tmp1 = tmp1.swap_gate(axes=(1, (6, 7)))  # r X s s'
    tmp1 = tmp1.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [r r'] y [t t'] [s2 s2']

    tmp3 = _append_vec_br_open(ten3.bra, ten3.ket, vecbr)  # x [t t'] y [l l'] [s s']
    tmp3 = tmp3.unfuse_legs(axes=(1, 3, 4))  # x t t' y l l' s s'
    tmp3 = tmp3.swap_gate(axes=(2, (6, 7)))  # t' X s s'
    tmp3 = tmp3.swap_gate(axes=(4, (6, 7)))  # l X s s'
    tmp3 = tmp3.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [t t'] y [l l'] [s3 s3']

    res = tmp0.tensordot(
        tmp2, axes=((2, 3), (0, 1))
    )  # x0 [b0 b0'] [s0 s0'] y1 [b1 b1'] [s2 s2']
    res = res.tensordot(
        tmp1, axes=((0, 1), (2, 3))
    )  # [s0 s0'] y1 [b1 b1'] [s2 s2'] x2 [r2 r2'] [s1 s1']
    res = res.tensordot(
        tmp3, axes=((1, 2, 4, 5), (0, 1, 2, 3))
    )  # [s0 s0'] [s2 s2'] [s1 s1'] [s3 s3']
    rdm = res.unfuse_legs(axes=(0, 1, 2, 3))  # s0 s0' s2 s2' s1 s1' s3 s3'
    rdm = trace_aux(rdm, 0, swap=True)
    rdm = trace_aux(rdm, 2, swap=False)
    rdm = trace_aux(rdm, 4, swap=True)
    rdm = trace_aux(rdm, 6, swap=False)

    rdm = rdm.transpose(axes=(0, 1, 4, 5, 2, 3, 6, 7)) # s0 s0' s1 s1' s2 s2' s3 s3'
    rdm, rdm_norm= _normalize_and_regularize_rdm(rdm, who=rdm2x2.__name__)
    return rdm, rdm_norm

def rdm2x2_diagonal(s0 : Site, psi : Peps, env : EnvCTM, **kwargs) -> tuple[Tensor, Scalar]:
    r"""
    Contract environment and on-site tensors of 2x2 patch,
    with `s0` the upper-left Site, to reduced density matrix::

        C T  T  C
        T s0 x  T
        T x  s3 T
        C T  T  C

    The index convention for reduced density matrix is `[s0, s0', s3, s3']`,
    where s_i,s_i' is bra,ket pair.

    TODO: Optionally symmetrize and make non-negative

    Args:
        s0: The site of the 2x2 reduced density matrix
        psi: Peps
        env: environment

    Returns:
        Reduced density matrix and its unnormalized trace
    """
    s1, s2, s3 = psi.nn_site(s0, "b"), psi.nn_site(s0, "r"), psi.nn_site(s0, "br")
    env0, env1, env2, env3 = env[s0], env[s1], env[s2], env[s3]

    psi_dl = Peps2Layers(psi)
    ten0, ten1, ten2, ten3 = (
        psi_dl[s0],
        psi_dl[s1],
        psi_dl[s2],
        psi_dl[s3],
    )  # DoublePepsTensor

    vectl = (env0.l @ env0.tl) @ env0.t
    vecbl = (env1.b @ env1.bl) @ env1.l
    vectr = (env2.t @ env2.tr) @ env2.r
    vecbr = (env3.r @ env3.br) @ env3.b

    tmp0 = _append_vec_tl_open(ten0.bra, ten0.ket, vectl)  # x [b b'] y [r r'] [s s']
    tmp0 = tmp0.unfuse_legs(axes=(1, 3, 4))  # x b b' y r r' s s'
    tmp0 = tmp0.swap_gate(axes=(1, (6, 7)))  # b X s s'
    tmp0 = tmp0.swap_gate(axes=(5, (6, 7)))  # r' X s s'
    tmp0 = tmp0.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [b b'] y [r r'] [s0 s0']

    tmp1 = append_vec_tr(ten2.bra, ten2.ket, vectr) # x [l l'] y [b b']
    tmp2 = append_vec_bl(ten1.bra, ten1.ket, vecbl)  # x [r r'] y [t t']

    tmp3 = _append_vec_br_open(ten3.bra, ten3.ket, vecbr)  # x [t t'] y [l l'] [s s']
    tmp3 = tmp3.unfuse_legs(axes=(1, 3, 4))  # x t t' y l l' s s'
    tmp3 = tmp3.swap_gate(axes=(2, (6, 7)))  # t' X s s'
    tmp3 = tmp3.swap_gate(axes=(4, (6, 7)))  # l X s s'
    tmp3 = tmp3.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [t t'] y [l l'] [s3 s3']

    res = tmp0.tensordot(
        tmp1, axes=((2, 3), (0, 1))
    )  # x0 [b0 b0'] [s0 s0'] y1 [b1 b1']
    res = res.tensordot(
        tmp2, axes=((0, 1), (2, 3))
    )  # [s0 s0'] y1 [b1 b1'] x2 [r2 r2']
    res = res.tensordot(
        tmp3, axes=((1, 2, 3, 4), (0, 1, 2, 3))
    )  # [s0 s0'] [s3 s3']

    rdm = res.unfuse_legs(axes=(0, 1))  # s0 s0' s3 s3'
    rdm = trace_aux(rdm, 0, swap=True)
    rdm = trace_aux(rdm, 2, swap=False)
    rdm, rdm_norm= _normalize_and_regularize_rdm(rdm, who=rdm2x2_diagonal.__name__)

    return rdm, rdm_norm

def rdm2x2_anti_diagonal(s0 : Site, psi : Peps, env : EnvCTM, **kwargs) -> tuple[Tensor, Scalar]:
    r"""
    Contract environment and on-site tensors of 2x2 patch,
    with `s0` the upper-left Site, to reduced density matrix::

        C T  T  C
        T x  s2 T
        T s1 x  T
        C T  T  C

    The index convention for reduced density matrix is `[s1, s1', s2, s2']`,
    where s_i,s_i' is bra,ket pair.

    TODO: Optionally symmetrize and make non-negative

    Args:
        s0: The site of the 2x2 reduced density matrix
        psi: Peps
        env: environment

    Returns:
        Reduced density matrix and its unnormalized trace
    """
    s1, s2, s3 = psi.nn_site(s0, "b"), psi.nn_site(s0, "r"), psi.nn_site(s0, "br")
    env0, env1, env2, env3 = env[s0], env[s1], env[s2], env[s3]

    psi_dl = Peps2Layers(psi)
    ten0, ten1, ten2, ten3 = (
        psi_dl[s0],
        psi_dl[s1],
        psi_dl[s2],
        psi_dl[s3],
    )  # DoublePepsTensor

    vectl = (env0.l @ env0.tl) @ env0.t
    vecbl = (env1.b @ env1.bl) @ env1.l
    vectr = (env2.t @ env2.tr) @ env2.r
    vecbr = (env3.r @ env3.br) @ env3.b

    tmp0 = append_vec_tl(ten0.bra, ten0.ket, vectl)  # x [b b'] y [r r']

    tmp1 = _append_vec_tr_open(ten2.bra, ten2.ket, vectr)  # x [l l'] y [b b'] [s s']
    tmp1 = tmp1.unfuse_legs(axes=(1, 3, 4))  # x l l' y b b' s s'
    tmp1 = tmp1.swap_gate(axes=(2, (6, 7)))  # l' X s s'
    tmp1 = tmp1.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [l l'] y [b b'] [s2 s2']

    tmp2 = _append_vec_bl_open(ten1.bra, ten1.ket, vecbl)  # x [r r'] y [t t'] [s s']
    tmp2 = tmp2.unfuse_legs(axes=(1, 3, 4))  # x r r' y t t' s s'
    tmp2 = tmp2.swap_gate(axes=(1, (6, 7)))  # r X s s'
    tmp2 = tmp2.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [r r'] y [t t'] [s1 s1']

    tmp3 = append_vec_br(ten3.bra, ten3.ket, vecbr)  # x [t t'] y [l l']

    res1 = tmp0.tensordot(
        tmp1, axes=((2, 3), (0, 1))
    )  # x0 [b0 b0'] y1 [b1 b1'] [s2 s2']
    res2 = tmp2.tensordot(
        tmp3, axes=((0, 1), (2, 3))
        ) # y2 [t t'] [s1 s1'] x3 [t t']

    res = res1.tensordot(res2, axes=((0, 1, 2, 3), (0, 1, 3, 4))) # [s2 s2'] [s1 s1']

    rdm = res.unfuse_legs(axes=(0, 1))  # s2 s2' s1 s1'
    rdm = trace_aux(rdm, 0, swap=False)
    rdm = trace_aux(rdm, 2, swap=True)

    rdm = rdm.transpose(axes=(2, 3, 0, 1)) # s1 s1' s2 s2'
    rdm, rdm_norm= _normalize_and_regularize_rdm(rdm, who=rdm2x2_anti_diagonal.__name__)

    return rdm, rdm_norm


def measure_rdm_1site(s0 : Site, psi : Peps, env : EnvCTM, op : Union[Tensor, Sequence[Tensor]])->Union[Scalar, Sequence[Scalar]]:
    """
    Measure one or more observables on 1x1 patch centered at site `s0`.

    Args:
        s0: site
        psi: PEPS wavefunction
        env: CTM environment
        op: one or more observables

    Returns:
        expectation value or a list of expectations values of provided `op`.
    """
    rdm, norm = rdm1x1(s0, psi, env)  # s s'
    # norm = rdm.trace(axes=(0, 1)).to_number()

    if isinstance(op,tuple) or isinstance(op,list):
        return [ncon([_op, rdm], ((1, 2), (2, 1))).to_number() for _op in op]
    return ncon([op, rdm], ((1, 2), (2, 1))).to_number() #/ norm


def measure_rdm_nn(s0 : Site, dirn : str, psi : Peps, env : EnvCTM, op : Union[Sequence[Tensor], Sequence[Sequence[Tensor]]])->Union[Scalar, Sequence[Scalar]]:
    """
    Measure one or more observables on 1x2 or 2x1 patch with site `s0`
    being leftmost or topmost respectively.

    Observables are expected to be one or more pairs of operators/Tensors, i.e.::

        op = (Tensor, Tensor)

        or

        op = [(Tensor,Tensor), (Tensor,Tensor), ...]

    with first operator acting always on site `s0`. Operators are applied from "right-to-left",
    i.e. the second operator `op[1]` is applied first, which might matter for fermionic operators.

    Args:
        s0: site
        dirn: 'h' for horizontal 1x2 patch (see :func:`rdm1x2`) and 'v' for vertical 2x1 patch (see :func:`rdm2x1`)
        psi: PEPS wavefunction
        env: CTM environment
        op: one or more observables

    Returns:
        Expectation value or a list of expectations values of provided `op`.
    """
    if dirn in ("h", "lr"):
        rdm, norm = rdm1x2(s0, psi, env)  # s0 s0' s1 s1'
        s1 = psi.nn_site(s0, "r")
    elif dirn in ("v", "tb"):
        rdm, norm = rdm2x1(s0, psi, env)  # s0 s0' s1 s1'
        s1 = psi.nn_site(s0, "b")
    # norm = rdm.trace(axes=((0, 2), (1, 3))).to_number()

    ncon_order = ((1, 2, 5), (3, 4, 5), (2, 1, 4, 3))
    def _eval_op(O0, O1):
        ordered = env.f_ordered(s0, s1)
        fermionic = True if (O0.n[0] and O1.n[0]) else False
        O0, O1 = op_order(O0, O1, ordered, fermionic)
        return ncon([O0, O1, rdm], ncon_order).to_number()

    if isinstance(op[0],Tensor) and isinstance(op[1],Tensor):
        return _eval_op(op[0],op[1])
    return [ _eval_op(_op[0],_op[1]) for _op in op ]

def measure_rdm_diag(s0 : Site, dirn : str, psi : Peps, env : EnvCTM, op : Union[Sequence[Tensor], Sequence[Sequence[Tensor]]])->Union[Scalar, Sequence[Scalar]]:
    """
    Measure one or more observables on 2x2 diag or 2x2 anti_diag patch with site `s0`
    being topleft.

    Observables are expected to be one or more pairs of operators/Tensors, i.e.::

        op = (Tensor, Tensor)

        or

        op = [(Tensor,Tensor), (Tensor,Tensor), ...]

    with first operator acting always on site `s0`. Operators are applied from "right-to-left",
    i.e. the second operator `op[1]` is applied first, which might matter for fermionic operators.

    Args:
        s0: topleft site
        dirn: 'diag' for diagonal patch (see :func:`rdm2x2_diagonal`) and 'anti_diag' for anti_diagonal patch (see :func:`rdm2x2_anti_diagonal`)
        psi: PEPS wavefunction
        env: CTM environment
        op: one or more observables

    Returns:
        Expectation value or a list of expectations values of provided `op`.
    """
    if dirn == "diag":
        rdm, norm = rdm2x2_diagonal(s0, psi, env)  # s0 s0' s3 s3'
        s1 = psi.nn_site(s0, "br")
    elif dirn == "anti_diag":
        rdm, norm = rdm2x2_anti_diagonal(s0, psi, env)  # s1 s1' s2 s2'
        s0 = psi.nn_site(s0, "b")
        s1 = psi.nn_site(s0, "tr")
    # norm = rdm.trace(axes=((0, 2), (1, 3))).to_number()

    ncon_order = ((1, 2, 5), (3, 4, 5), (2, 1, 4, 3))
    def _eval_op(O0, O1):
        ordered = env.f_ordered(s0, s1)
        fermionic = True if (O0.n[0] and O1.n[0]) else False
        O0, O1 = op_order(O0, O1, ordered, fermionic)
        return ncon([O0, O1, rdm], ncon_order).to_number()

    if isinstance(op[0],Tensor) and isinstance(op[1],Tensor):
        return _eval_op(op[0],op[1])
    return [ _eval_op(_op[0],_op[1]) for _op in op ]

def measure_rdm_2x2(s0 : Site, psi : Peps, env : EnvCTM, op : Union[Sequence[Tensor], Sequence[Sequence[Tensor]]]) -> Union[Scalar, Sequence[Scalar]]:
    r"""
    Measure one or more observables on 2x2 patch with site `s0` being upper-left site.

    Observables are expected to be one or more quadruples of operators/Tensors, i.e.::

        op = (Tensor, Tensor, Tensor, Tensor)

        or

        op = [(Tensor, Tensor, Tensor, Tensor), (Tensor, Tensor, Tensor, Tensor), ...]

    with first operator acting on site `s0`, second on site `s1`, etc. according to :func:`rdm2x2`
    index ordering convention.

    Args:
        s0: site
        psi: PEPS wavefunction
        env: CTM environment
        op: one or more observables

    Returns:
        Expectation value or a list of expectations values of provided `op`.
    """
    rdm, norm = rdm2x2(s0, psi, env)  # s0 s0' s1 s1' s2 s2' s3 s3'
    s1 = psi.nn_site(s0, "b")
    s2 = psi.nn_site(s0, "r")
    s3 = psi.nn_site(s2, "b")
    sym = psi.config.sym

    # def _eval_op(O0, O1, O2, O3):
    #     return ncon([O0, O1, O2, O3, rdm], ncon_order).to_number()

    def _eval_op(O0, O1, O2, O3):
        Os = [O0, O1, O2, O3]
        charges = [O.n for O in Os]
        charge_sum = reduce(sym.add_charges, charges)
        if charge_sum != sym.zero():
            raise YastnError("Non-zero parity charges within the 2x2 measument window!")
        non_zero_charges = 0
        for c in charges:
            non_zero_charges += int(c != sym.zero())

        if non_zero_charges == 0:
            # no auxiliary leg presented
            ncon_order = ((1, 2), (3, 4), (5, 6), (7, 8), (2, 1, 4, 3, 6, 5, 8, 7))
            return ncon([O0, O1, O2, O3, rdm], ncon_order).to_number()

        elif non_zero_charges == 2:
            # one auxiliary leg connecting two ops with non-zero charges
            if O0.n != sym.zero() and O1.n != sym.zero():
                O0, O1 = op_order(O0, O1, True, True)
                O1 = O1.swap_gate(axes=(1,2))

                ncon_order = ((1, 2, -1), (3, 4, -2), (5, 6), (7, 8), (2, 1, 4, 3, 6, 5, 8, 7))
                res = ncon([O0, O1, O2, O3, rdm], ncon_order)
                return res.trace(axes=(0,1)).to_number()

            elif O0.n != sym.zero() and O2.n != sym.zero():
                O0, O2 = op_order(O0, O2, True, True)

                ncon_order = ((1, 2, -1), (3, 4), (5, 6, -2), (7, 8), (2, 1, 4, 3, 6, 5, 8, 7))
                res = ncon([O0, O1, O2, O3, rdm], ncon_order)
                return res.trace(axes=(0,1)).to_number()

            elif O0.n != sym.zero() and O3.n[0] != sym.zero():
                O0, O3 = op_order(O0, O3, True, True)
                ncon_order = ((1, 2, -1), (3, 4), (5, 6), (7, 8, -2), (2, 1, 4, 3, 6, 5, 8, 7))
                res = ncon([O0, O1, O2, O3, rdm], ncon_order)
                return res.trace(axes=(0,1)).to_number()

            elif O1.n != sym.zero() and O2.n != sym.zero():
                O1, O2 = op_order(O1, O2, True, True)
                ncon_order = ((1, 2), (3, 4, -1), (5, 6, -2), (7, 8), (2, 1, 4, 3, 6, 5, 8, 7))
                res = ncon([O0, O1, O2, O3, rdm], ncon_order)
                return res.trace(axes=(0,1)).to_number()

            elif O1.n != sym.zero() and O3.n != sym.zero():
                O1, O3 = op_order(O1, O3, True, True)
                ncon_order = ((1, 2), (3, 4, -1), (5, 6), (7, 8, -2), (2, 1, 4, 3, 6, 5, 8, 7))
                res = ncon([O0, O1, O2, O3, rdm], ncon_order)
                return res.trace(axes=(0,1)).to_number()

            elif O2.n != sym.zero() and O3.n != sym.zero():
                O2, O3 = op_order(O2, O3, True, True)
                O2 = O2.swap_gate(axes=(0, 2))
                ncon_order = ((1, 2), (3, 4), (5, 6, -1), (7, 8, -2), (2, 1, 4, 3, 6, 5, 8, 7))
                res = ncon([O0, O1, O2, O3, rdm], ncon_order)
                return res.trace(axes=(0,1)).to_number()

        elif non_zero_charges == 4:
            # either O0 + O1 == 0 or O0 + O4==0
            if sym.add_charges(O0.n, O1.n) == sym.zero():
                # connecting O0 with O1
                O0, O1 = op_order(O0, O1, True, True)
                O2, O3 = op_order(O2, O3, True, True)
                O1 = O1.swap_gate(axes=(1, 2))
                O2 = O2.swap_gate(axes=(0, 2))
                ncon_order = ((1, 2, -1), (3, 4, -2), (5, 6, -3), (7, 8, -4), (2, 1, 4, 3, 6, 5, 8, 7))
                res = ncon([O0, O1, O2, O3, rdm], ncon_order)
            elif sym.add_charges(O0.n, O3.n) == sym.zero():
                O0, O3 = op_order(O0, O3, True, True)
                O1, O2 = op_order(O1, O2, True, True)
                ncon_order = ((1, 2, -1), (3, 4, -3), (5, 6, -4), (7, 8, -2), (2, 1, 4, 3, 6, 5, 8, 7))
                res = ncon([O0, O1, O2, O3, rdm], ncon_order)
                res = res.swap_gate(axes=(0, 2))

            return res.trace(axes=((0, 2), (1, 3))).to_number()



    if len(op)==4 and all([isinstance(_op, Tensor) for _op in op]):
        return _eval_op(*tuple(op))
    return [_eval_op(*tuple(_op)) for _op in op]
