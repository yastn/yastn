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
from itertools import pairwise
from typing import NamedTuple
from ... import eye, tensordot, YastnError


class Gate(NamedTuple):
    r"""
    Gate to be applied on Peps state.

    `G` contains operators to be applied on respective `sites`.
    Operators have virtual legs connecting them, forming an MPO.

    The convention of legs is (ket, bra, virtual_0, virtual_1) -- i.e., the first two legs are always physical (operator) legs.
    For one site, there are no virtual legs.
    For two or more sites, the first and last elements of G have one virtual leg (3 in total).
    For three sites or more, the middle elements of `G` have two virtual legs connecting, respectively, to preceding and following gates.

    Note that this is a different convention than `yastn.mps.MPO`.
    """
    G : tuple = None
    sites : tuple = None


def match_ancilla(ten, G, dirn=None):
    """
    Kronecker product and fusion of local gate with identity for ancilla.

    Identity is read from the ancilla leg of the tensor.
    Can perform a swap gate of the auxiliary operator leg (if present) with an ancilla.
    """
    leg = ten.get_legs(axes=-1)

    if not leg.is_fused():
        return G

    _, leg = leg.unfuse_leg()  # unfuse to get ancilla leg
    one = eye(config=ten.config, legs=[leg, leg.conj()], isdiag=False)
    Gnew = tensordot(G, one, axes=((), ()))

    swap = dirn and dirn in 'tl'

    if G.ndim == 2:
        return Gnew.fuse_legs(axes=((0, 2), (1, 3)))
    elif G.ndim == 3:
        if dirn and dirn in 'tl':
            Gnew = Gnew.swap_gate(axes=(2, 3))
        return Gnew.fuse_legs(axes=((0, 3), (1, 4), 2))
    elif G.ndim == 4:
        if dirn and dirn[0] in 'tl':
            Gnew = Gnew.swap_gate(axes=(2, 4))
        if dirn and dirn[1] in 'tl':
            Gnew = Gnew.swap_gate(axes=(3, 4))
        return Gnew.fuse_legs(axes=((0, 4), (1, 5), 2, 3))


def apply_gate_onsite(ten, G, dirn=None):
    """
    Applies operator to the physical leg of (ket) PEPS tensor.

    Operator with auxiliary leg should have dirn in 'l', 'r', 't', 'b', indicating
    the fusion of the auxiliary leg with the corresponding virtual tensor leg and
    application of a proper swap gate.
    For a local operator with no auxiliary index, dirn should be None.
    """
    G = match_ancilla(ten, G, dirn=dirn)
    tmp = tensordot(ten, G, axes=(2, 1)) # [t l] [b r] [s a] c
    if not dirn:
        return tmp

    fuse_one = False
    if len(dirn) == 2:
        tmp = tmp.fuse_legs(axes=(0, 1, (2, 3), 4), mode='meta')
        fuse_one = True

    for dd in dirn[::-1]:
        if dd == 't':
            tmp = tmp.unfuse_legs(axes=1)  # [t l] b r [s a] c
            tmp = tmp.fuse_legs(axes=(0, (1, 4), 2, 3))  # [t l] [b c] r [s a]
            tmp = tmp.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [[b c] r] [s a]
        if dd == 'b':
            tmp = tmp.unfuse_legs(axes=0)  # t l [b r] [s a] c
            tmp = tmp.swap_gate(axes=(1, 4))
            tmp = tmp.fuse_legs(axes=((0, 4), 1, 2, 3))  # [t c] l [b r] [s a]
            tmp = tmp.fuse_legs(axes=((0, 1), 2, 3))  # [[t c] l] [b r] [s a]
        if dd == 'l':
            tmp = tmp.unfuse_legs(axes=1)  # [t l] b r [s a] c
            tmp = tmp.swap_gate(axes=(1, 4))
            tmp = tmp.fuse_legs(axes=(0, 1, (2, 4), 3))  # [t l] b [r c] [s a]
            tmp = tmp.fuse_legs(axes=(0, (1, 2), 3))  # [t l] [b [r c]] [s a]
        if dd == 'r':
            tmp = tmp.unfuse_legs(axes=0)  # t l [b r] [s a] c
            tmp = tmp.fuse_legs(axes=(0, (1, 4), 2, 3))  # t [l c] [b r] [s a]
            tmp = tmp.fuse_legs(axes=((0, 1), 2, 3))  # [t [l c]] [b r] [s a]
        if fuse_one:
            fuse_one = False
            tmp = tmp.unfuse_legs(axes=2)
    return tmp
    # raise YastnError("dirn should be equal to 'l', 'r', 't', 'b', or None")


def apply_bond_tensors(Q0f, Q1f, M0, M1, dirn):
    """
    Combine unitaries in Q0f, Q1f with optimized M0, M1 to form new peps tensors.
    """
    if dirn == "h":
        ten0 = Q0f @ M0  # [[[[t l] sa] b] r
        ten0 = ten0.unfuse_legs(axes=0)  # [[t l] sa] b r
        ten0 = ten0.fuse_legs(axes=(0, (1, 2)))  # [[t l] sa] [b r]
        ten0 = ten0.unfuse_legs(axes=0)  # [t l] sa [b r]
        ten0 = ten0.transpose(axes=(0, 2, 1))  # [t l] [b r] sa

        ten1 = M1 @ Q1f  # l [t [[b r] s]]
        ten1 = ten1.unfuse_legs(axes=1)  # l t [[b r] sa]
        ten1 = ten1.fuse_legs(axes=((1, 0), 2))  # [t l] [[b r] sa]
        ten1 = ten1.unfuse_legs(axes=1)  # [t l] [b r] sa
    else:  # dirn == "v":
        ten0 = Q0f @ M0  # [[[t l] sa] r] b
        ten0 = ten0.unfuse_legs(axes=0)  # [[t l] sa] r b
        ten0 = ten0.fuse_legs(axes=(0, (2, 1)))  # [[t l] sa] [b r]
        ten0 = ten0.unfuse_legs(axes=0)  # [t l] sa [b r]
        ten0 = ten0.transpose(axes=(0, 2, 1))  # [t l] [b r] sa

        ten1 = M1 @ Q1f  # t [l [[b r] sa]]
        ten1 = ten1.unfuse_legs(axes=1)  # t l [[b r] sa]
        ten1 = ten1.fuse_legs(axes=((0, 1), 2))  # [t l] [[b r] sa]
        ten1 = ten1.unfuse_legs(axes=1)  # [t l] [b r] sa
    return ten0, ten1


def gate_product_operator(O0, O1, l_ordered=True, f_ordered=True, merge=False):
    """
    Takes two ndim=2 local operators O0 O1 to be applied
    on two sites with O1 acting first (relevant for fermionic operators).
    Adds a connecting leg with a swap_gate consistent with fermionic order.
    Orders output to match lattice order (canonical l_order is 'lr' and 'tb').

    If merge, returns equivalent of ncon([O0, O1], [(-0, -2), (-1, -3)]),
    with proper operator order and swap gate applied.
    """
    s = -1 if l_ordered else 1
    O0 = O0.add_leg(s=s, axis=2)
    O1 = O1.add_leg(s=-s, axis=2)

    if f_ordered:
        O0 = O0.swap_gate(axes=(1, 2))
    else:
        O1 = O1.swap_gate(axes=(0, 2))

    G0, G1 = (O0, O1) if l_ordered else (O1, O0)

    if merge:
        return tensordot(G0, G1, axes=(2, 2)).transpose(axes=(0, 2, 1, 3))
    return G0, G1


def fkron(A, B, sites=(0, 1), merge=True):
    """
    Fermionic kron; auxiliary function for gate generation.
    Returns a Kronecker product of two local operators, A and B,
    including swap-gate (fermionic string) to handle fermionic operators.

    Calculate A_0 B_1 for sites==(0, 1), and A_1 B_0 for sites==(1, 0),
    i.e., operator B acts first (relevant when operators are fermionically non-trivial).

    Site 0 is assumed to be first in both lattice and fermionic orders.

    If merge, returns equivalent of ncon([O0, O1], [(-0, -2), (-1, -3)]),
    with proper operator order and swap gate applied;
    O0 corresponds to the operator acting on site 0.
    """
    if sites not in ((0, 1), (1, 0)):
        raise YastnError("sites should be equal to (0, 1) or (1, 0)")
    order = (sites == (0, 1))
    return gate_product_operator(A, B, order, order, merge)


def gate_fix_order(G0, G1, l_ordered=True, f_ordered=True):
    """
    Modifies two gate tensors,
    that were generated consistently with lattice and fermionic orders,
    making them consistent with provided orders.

    l_ordered and f_ordered typically coincide;
    they do not coincide in a special case of
    cylindric lattice geometry across the periodic boundary.
    """
    if not f_ordered:
        G0 = G0.swap_gate(axes=(1, 2))
        G1 = G1.swap_gate(axes=(0, 2))
    if not l_ordered:
        G0, G1 = G1, G0
    return G0, G1


def gate_from_mpo(op):
    G = []
    G.append(op[op.first].remove_leg(axis=0).transpose(axes=(0, 2, 1)))
    for n in op.sweep(to='last', df=1):
        G.append(op[n].transpose(axes=(1, 3, 0, 2)))
    G[-1] = G[-1].remove_leg(axis=-1)
    return G
