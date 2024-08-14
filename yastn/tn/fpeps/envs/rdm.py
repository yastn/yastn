from .... import ncon
from .. import Site, Peps, Peps2Layers, EnvCTM
from .... import Tensor
from typing import Sequence, Union, TypeVar
Scalar = TypeVar('Scalar')
# from yastn import Tensor, tensordot
# from yastn.tn.fpeps.envs._env_auxlliary import append_vec_tl, append_vec_tr, append_vec_bl, append_vec_br

# utility functions for corner contractions, leaving the physical indices uncontracted.
def append_vec_tl_open(
    A, Ac, vectl
):  # A = [t l] [b r] s;  Ac = [t' l'] [b' r'] s';  vectl = x [l l'] [t t'] y
    """Append the A and Ac tensors to the top-left vector with open and unswapped physical indices [s s']"""
    vectl = vectl.fuse_legs(axes=(2, (0, 3), 1))  # [t t'] [x y] [l l']
    vectl = vectl.unfuse_legs(axes=(0, 2))  # t t' [x y] l l'
    vectl = vectl.swap_gate(axes=(1, (3, 4)))  # t' X l l'
    vectl = vectl.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [t l] [x y] [t' l']
    vectl = vectl.tensordot(Ac.conj(), axes=(2, 0))  # [t l] [x y] [b' r'] s'
    vectl = A.tensordot(vectl, axes=(0, 0))  # [b r] s [x y] [b' r'] s'
    vectl = vectl.fuse_legs(axes=(0, 2, 3, (1, 4)))  # [b r] [x y] [b' r'] [s s']
    vectl = vectl.unfuse_legs(axes=(0, 2))  # b r [x y] b' r' [s s']
    vectl = vectl.swap_gate(axes=((0, 3), 4))  # b b' X r'
    vectl = vectl.fuse_legs(axes=((0, 3), 2, (1, 4), 5))  # [b b'] [x y] [r r'] [s s']
    vectl = vectl.unfuse_legs(axes=1)  # [b b'] x y [r r'] [s s']
    vectl = vectl.transpose(axes=(1, 0, 2, 3, 4))  # x [b b'] y [r r'] [s s']
    return vectl


def append_vec_br_open(
    A, Ac, vecbr
):  # A = [t l] [b r] s;  Ac = [t' l'] [b' r'] s';  vecbr = x [r r'] [b b'] y
    """Append the A and Ac tensors to the bottom-right vector with open and unswapped physical indices [s s']."""
    vecbr = vecbr.fuse_legs(axes=(2, (0, 3), 1))  # [b b'] [x y] [r r']
    vecbr = vecbr.unfuse_legs(axes=(0, 2))  # b b' [x y] r r'
    vecbr = vecbr.swap_gate(axes=((0, 1), 4))  # b b' X r'
    vecbr = vecbr.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [b r] [x y] [b' r']
    vecbr = vecbr.tensordot(Ac.conj(), axes=(2, 1))  # [b r] [x y] [t' l'] s'
    vecbr = A.tensordot(vecbr, axes=(1, 0))  # [t l] s [x y] [t' l'] s'
    vecbr = vecbr.fuse_legs(axes=(0, 2, 3, (1, 4)))  # [t l] [x y] [t' l'] [s s']
    vecbr = vecbr.unfuse_legs(axes=(0, 2))  # t l [x y] t' l' [s s']
    vecbr = vecbr.swap_gate(axes=((1, 4), 3))  # l l' X t'
    vecbr = vecbr.fuse_legs(axes=((0, 3), 2, (1, 4), 5))  # [t t'] [x y] [l l'] [s s']
    vecbr = vecbr.unfuse_legs(axes=1)  # [t t'] x y [l l'] [s s']
    vecbr = vecbr.transpose(axes=(1, 0, 2, 3, 4))  # x [t t'] y [l l'] [s s']
    return vecbr


def append_vec_tr_open(
    A, Ac, vectr
):  # A = [t l] [b r] s;  Ac = [t' l'] [b' r'] s';  vectr = x [t t'] [r r'] y
    """Append the A and Ac tensors to the top-right vector with open physical indices [s s'],
    swapped with t'"""
    vectr = vectr.fuse_legs(axes=(1, (0, 3), 2))  # [t t'] [x y] [r r']
    vectr = vectr.unfuse_legs(axes=(0, 2))  # t t' [x y] r r'
    vectr = vectr.fuse_legs(axes=((0, 3), 2, (1, 4)))  # [t r] [x y] [t' r']
    A = A.unfuse_legs(axes=(0, 1))  # t l b r s
    A = A.fuse_legs(axes=((0, 3), (1, 2), 4))  # [t r] [l b] s
    vectr = vectr.tensordot(A, axes=(0, 0))  # [x y] [t' r'] [l b] s
    vectr = vectr.unfuse_legs(axes=(1, 2))  # [x y] t' r' l b s
    vectr = vectr.swap_gate(axes=(1, (3, 5), 2, 4))  # t' X l s and r' X b
    vectr = vectr.fuse_legs(axes=(0, (1, 2), (3, 4), 5))  # [x,y] [t' r'] [l b] s

    Ac = Ac.unfuse_legs(axes=(0, 1))  # t' l' b' r' s'
    Ac = Ac.swap_gate(axes=(0, (1, 4), 2, 3))  # t' X l' s' and b' X r'
    Ac = Ac.fuse_legs(axes=((0, 3), (1, 2), 4))  # [t' r'] [l' b'] s'
    vectr = vectr.tensordot(Ac.conj(), axes=(1, 0))  # [x y] [l b] s [l' b'] s'
    vectr = vectr.unfuse_legs(axes=(0, 1, 3))  # x y l b s l' b' s'
    vectr = vectr.fuse_legs(
        axes=(0, (2, 5), 1, (3, 6), (4, 7))
    )  # x [l l'] y [b b'] [s s']

    return vectr


def append_vec_bl_open(
    A, Ac, vecbl
):  # A = [t l] [b r] s;  Ac = [t' l'] [b' r'] s';  vecbl = x [b b'] [l l'] y
    """Append the A and Ac tensors to the bottom-left vector with open physical indices,
    swapped with b"""
    vecbl = vecbl.fuse_legs(axes=(1, (0, 3), 2))  # [b b'] [x y] [l l']
    vecbl = vecbl.unfuse_legs(axes=(0, 2))  # b b' [x y] l l'
    vecbl = vecbl.fuse_legs(axes=((3, 0), 2, (4, 1)))  # [l b] [x y] [l' b']

    Ac = Ac.unfuse_legs(axes=(0, 1))  # t' l' b' r' s'
    Ac = Ac.swap_gate(axes=(0, 1, 2, 3))  # t' X l' and b' X r'
    Ac = Ac.fuse_legs(axes=((0, 3), (1, 2), 4))  # [t' r'] [l' b'] s'

    vecbl = vecbl.tensordot(Ac.conj(), axes=(2, 1))  # [l b] [x y] [t' r'] s'
    vecbl = vecbl.unfuse_legs(axes=(0, 2))  # l b [x y] t' r' s'
    vecbl = vecbl.swap_gate(axes=(0, 3, 1, (4, 5)))  # l X t' and b X r' s'
    vecbl = vecbl.fuse_legs(axes=((0, 1), 2, (3, 4), 5))  # [l b] [x y] [t' r'] s'

    A = A.unfuse_legs(axes=(0, 1))  # t l b r s
    A = A.swap_gate(axes=(2, 4))  # b X s
    A = A.fuse_legs(axes=((0, 3), (1, 2), 4))  # [t r] [l b] s
    vecbl = vecbl.tensordot(A, axes=(0, 1))  # [x y] [t' r'] s' [t r] s
    vecbl = vecbl.unfuse_legs(axes=(0, 1, 3))  # x y t' r' s' t r s

    vecbl = vecbl.fuse_legs(
        axes=((0, (6, 3), 1, (5, 2), (7, 4)))
    )  # x [r r'] y [t t'] [s s']
    return vecbl


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
        if fermionic:
            Oi, Oj = -Oj, Oi
        else:
            Oi, Oj = Oj, Oi

    # Add an auxiliary leg such that the total charges of Oi and Oj are zero, respectively
    Oi = Oi.add_leg(s=1)
    Oj = Oj.add_leg(s=-1)

    Oi = Oi.swap_gate(axes=(1, 2))
    # Oi = Oi.swap_gate(axes=(0, 2))

    return Oi, Oj


def rdm1x1(s0 : Site, psi : Peps, env : EnvCTM) -> tuple[Tensor, Scalar]:
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

    tmp = append_vec_tl_open(ten0.top, ten0.btm, vect)
    res = vecb.tensordot(tmp, axes=((0, 1, 2, 3), (2, 3, 1, 0)))  # [s, s']

    rdm = res.unfuse_legs(axes=(0,))  # s s'

    # check if a dummy leg is fused with the physical leg
    if rdm.get_legs(0).is_fused():
        rdm = rdm.unfuse_legs(axes=(0, 1))  # p d p' d'
        rdm = rdm.trace(axes=(1, 3))  # p p'

    # assert rdm.ndim == 2
    rdm_norm = rdm.trace(axes=(0, 1)).to_number()
    rdm = rdm / rdm_norm

    return rdm, rdm_norm


def rdm1x2(s0 : Site, psi : Peps, env : EnvCTM) -> tuple[Tensor, Scalar]:
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

    tmp0 = append_vec_tl_open(ten0.top, ten0.btm, vecl)  # x [b b'] y [r r'] [s s']
    tmp0 = tmp0.unfuse_legs(axes=(1, 4))  # x b b' y [r r'] s s'
    tmp0 = tmp0.swap_gate(axes=(1, (5, 6)))  # b X s s'
    tmp0 = tmp0.fuse_legs(axes=(0, (1, 2), 3, 4, (5, 6)))  # x [b b'] y [r r'] [s s']
    tmp0 = env0.b.tensordot(tmp0, axes=((2, 1), (0, 1)))

    tmp1 = append_vec_br_open(ten1.top, ten1.btm, vecr)  # x [t t'] y [l l'] [s s']
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
    rdm_norm = rdm.trace(axes=((0, 2), (1, 3))).to_number()
    rdm = rdm / rdm_norm

    return rdm, rdm_norm


def rdm2x1(s0 : Site, psi : Peps, env : EnvCTM) -> tuple[Tensor, Scalar]:
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

    tmp0 = append_vec_tl_open(ten0.top, ten0.btm, vect)  # x [b b'] y [r r'] [s s']
    tmp0 = tmp0.unfuse_legs(axes=(1, 3, 4))  # x b b' y r r' s s'
    tmp0 = tmp0.swap_gate(axes=(5, (6, 7)))  # r' X s s'
    tmp0 = tmp0.swap_gate(axes=(2, (6, 7)))  # b' X s s'
    tmp0 = tmp0.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [b b'] y [r r'] [s s']
    tmp0 = env0.r.tensordot(tmp0, axes=((0, 1), (2, 3)))

    tmp1 = append_vec_br_open(ten1.top, ten1.btm, vecb)  # x [t t'] y [l l'] [s s']
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

    rdm_norm = rdm.trace(axes=((0, 2), (1, 3))).to_number()
    rdm = rdm / rdm_norm

    return rdm, rdm_norm


def rdm2x2(s0, psi, env):
    r"""
    :param s0: The upper-left site of the 2*2 reduced density matrix
    :param psi: Peps
    :param env: environment
    :type site: yastn.tn.fpeps._geometry.Site
    :type psi: yastn.tn.fpeps.Peps
    :type env: yastn.tn.fpeps.EnVCTM
    """
    s1, s2, s3 = psi.nn_site(s0, "r"), psi.nn_site(s0, "b"), psi.nn_site(s0, "br")
    env0, env1, env2, env3 = env[s0], env[s1], env[s2], env[s3]

    psi_dl = Peps2Layers(psi)
    ten0, ten1, ten2, ten3 = (
        psi_dl[s0],
        psi_dl[s1],
        psi_dl[s2],
        psi_dl[s3],
    )  # DoublePepsTensor

    vectl = (env0.l @ env0.tl) @ env0.t
    vectr = (env1.t @ env1.tr) @ env1.r
    vecbl = (env2.b @ env2.bl) @ env2.l
    vecbr = (env3.r @ env3.br) @ env3.b

    tmp0 = append_vec_tl_open(ten0.top, ten0.btm, vectl)  # x [b b'] y [r r'] [s s']
    tmp0 = tmp0.unfuse_legs(axes=(1, 3, 4))  # x b b' y r r' s s'
    tmp0 = tmp0.swap_gate(axes=(1, (6, 7)))  # b X s s'
    tmp0 = tmp0.swap_gate(axes=(5, (6, 7)))  # r' X s s'
    tmp0 = tmp0.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [b b'] y [r r'] [s0 s0']

    tmp1 = append_vec_tr_open(ten1.top, ten1.btm, vectr)  # x [l l'] y [b b'] [s s']
    tmp1 = tmp1.unfuse_legs(axes=(1, 3, 4))  # x l l' y b b' s s'
    tmp1 = tmp1.swap_gate(axes=(2, (6, 7)))  # l' X s s'
    tmp1 = tmp1.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [l l'] y [b b'] [s1 s1']

    tmp2 = append_vec_bl_open(ten2.top, ten2.btm, vecbl)  # x [r r'] y [t t'] [s s']
    tmp2 = tmp2.unfuse_legs(axes=(1, 3, 4))  # x r r' y t t' s s'
    tmp2 = tmp2.swap_gate(axes=(1, (6, 7)))  # r X s s'
    tmp2 = tmp2.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [r r'] y [t t'] [s2 s2']

    tmp3 = append_vec_br_open(ten3.top, ten3.btm, vecbr)  # x [t t'] y [l l'] [s s']
    tmp3 = tmp3.unfuse_legs(axes=(1, 3, 4))  # x t t' y l l' s s'
    tmp3 = tmp3.swap_gate(axes=(2, (6, 7)))  # t' X s s'
    tmp3 = tmp3.swap_gate(axes=(4, (6, 7)))  # l X s s'
    tmp3 = tmp3.fuse_legs(
        axes=(0, (1, 2), 3, (4, 5), (6, 7))
    )  # x [t t'] y [l l'] [s3 s3']

    res = tmp0.tensordot(
        tmp1, axes=((2, 3), (0, 1))
    )  # x0 [b0 b0'] [s0 s0'] y1 [b1 b1'] [s1 s1']
    res = res.tensordot(
        tmp2, axes=((0, 1), (2, 3))
    )  # [s0 s0'] y1 [b1 b1'] [s1 s1'] x2 [r2 r2'] [s2 s2']
    res = res.tensordot(
        tmp3, axes=((1, 2, 4, 5), (0, 1, 2, 3))
    )  # [s0 s0'] [s1 s1'] [s2 s2'] [s3 s3']
    rdm = res.unfuse_legs(axes=(0, 1, 2, 3))  # s0 s0' s1 s1' s2 s2' s3 s3'
    rdm = trace_aux(rdm, 0, swap=True)
    rdm = trace_aux(rdm, 2, swap=False)
    rdm = trace_aux(rdm, 4, swap=True)
    rdm = trace_aux(rdm, 6, swap=False)

    rdm_norm = rdm.trace(axes=((0, 2, 4, 6), (1, 3, 5, 7))).to_number()
    rdm = rdm / rdm_norm

    return rdm


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
    if dirn == "h":
        rdm, norm = rdm1x2(s0, psi, env)  # s0 s0' s1 s1'
        s1 = psi.nn_site(s0, "r")
    elif dirn == "v":
        rdm, norm = rdm2x1(s0, psi, env)  # s0 s0' s1 s1'
        s1 = psi.nn_site(s0, "b")
    # norm = rdm.trace(axes=((0, 2), (1, 3))).to_number()

    ncon_order = ((1, 2, 5), (3, 4, 5), (2, 1, 4, 3))
    def _eval_op(O0, O1):
        ordered = env.f_ordered(s0, s1)
        fermionic = True if (O0.n[0] and O1.n[0]) else False
        O0, O1 = op_order(O0, O1, ordered, fermionic)
        return ncon([O0, O1, rdm], ncon_order).to_number() / norm
    
    if isinstance(op[0],Tensor) and isinstance(op[1],Tensor):
        return _eval_op(op[0],op[1])
    return [ _eval_op(_op[0],_op[1]) for _op in op ]


def measure_rdm_2x2(s0, psi, env, op_list):
    r"""
    Parameters
    ----------
    s0: site
    s0 -- s1
    |     |
    s2 -- s3

    op_list: [O0, O1, O2, O3] bosonic operators
    """
    rdm = rdm2x2(s0, psi, env)  # s0 s0' s1 s1'
    s1 = psi.nn_site(s0, "r")
    s2 = psi.nn_site(s0, "b")
    s3 = psi.nn_site(s2, "r")

    # ordered = f_ordered(s0, s1)
    # fermionic = True if (O0.n[0] and O1.n[0]) else False
    # O0, O1 = op_order(O0, O1, ordered, fermionic)
    O0, O1, O2, O3 = tuple(op_list)
    norm = rdm.trace(axes=((0, 2, 4, 6), (1, 3, 5, 7))).to_number()
    ncon_order = ((1, 2), (3, 4), (5, 6), (7, 8), (2, 1, 4, 3, 6, 5, 8, 7))
    return ncon([O0, O1, O2, O3, rdm], ncon_order).to_number() / norm


def test_1site(psi, env, op):
    for s0 in psi.sites():
        val0 = measure_rdm_1site(s0, psi, env, op)
        val1 = env.measure_1site(op, s0)
        print(s0, val0.item(), val1.item())
        assert torch.isclose(val0, val1)


def test_nn(psi, env, op0, op1):
    for s0 in psi.sites():
        val0 = measure_rdm_nn(s0, "h", psi, env, op0, op1)
        s1 = psi.nn_site(s0, "r")
        bond = (s0, s1)
        val1 = env.measure_nn(op0, op1, bond)
        print(s0, val0.item(), val1.item())
        assert torch.isclose(val0, val1)

        val0 = measure_rdm_nn(s0, "v", psi, env, op0, op1)
        s1 = psi.nn_site(s0, "b")
        bond = (s0, s1)
        val1 = env.measure_nn(op0, op1, bond)
        print(s0, val0.item(), val1.item())
        assert torch.isclose(val0, val1)


def test_2x2(psi, env, op_list):
    for s0 in psi.sites():
        val0 = measure_rdm_2x2(s0, psi, env, op_list)

        s1 = psi.nn_site(s0, "r")
        s2 = psi.nn_site(s0, "b")
        s3 = psi.nn_site(s2, "r")
        sites = [s0, s1, s2, s3]
        val1 = env.measure_2x2(*op_list, sites=sites)
        assert torch.isclose(val0, val1)
