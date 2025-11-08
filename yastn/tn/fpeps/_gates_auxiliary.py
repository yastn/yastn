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
from typing import NamedTuple
from ...tensor import tensordot, YastnError
from ...initialize import eye


class Gate(NamedTuple):
    r"""
    Gate to be applied on Peps state.

    `G` contains operators for respective `sites`.

    Operator can be given in the form of an MPO (:class:`yastn.tn.mps.MpsMpoOBC`) of the same length as the number of provided `sites`.
    Sites should form a continuous path in the two-dimensional PEPS lattice.
    The fermionic order of MPO should be linear, with the first MPO site being first in the fermionic order, irrespective of the provided `sites`.

    For a two-site operator acting on sites beyond nearest neighbor, it can be provided as `G` with two elements, and `sites` forming a path between the end sites where the provided elements of `G` will act.

    It is also possible to provide `G` as a list of tensots.
    In this case, the convention of legs is (ket, bra, virtual_0, virtual_1) -- i.e., the first two legs are always physical (operator) legs.
    For one site, there are no virtual legs.
    For two or more sites, the first and last elements of G have one virtual leg (3 in total).
    For three sites or more, the middle elements of `G` have two virtual legs connecting, respectively, to the preceding and following elements of `G`.
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
    tmp = tensordot(ten, G, axes=(4, 1))  # t l b r [s a] c
    if not dirn:
        return tmp

    fuse_one = False
    if len(dirn) == 2:
        tmp = tmp.fuse_legs(axes=(0, 1, 2, 3, (4, 5), 6), mode='meta')
        fuse_one = True

    for dd in dirn[::-1]:
        if dd == 't':
            tmp = tmp.fuse_legs(axes=(0, 1, (2, 5), 3, 4))  # t l [b c] r [s a]
        if dd == 'b':
            tmp = tmp.swap_gate(axes=(1, 5))  # l X c
            tmp = tmp.fuse_legs(axes=((0, 5), 1, 2, 3, 4))  # [t c] l b r [s a]
        if dd == 'l':
            tmp = tmp.swap_gate(axes=(2, 5))  # b X c
            tmp = tmp.fuse_legs(axes=(0, 1, 2, (3, 5), 4))  # t l b [r c] [s a]
        if dd == 'r':
            tmp = tmp.fuse_legs(axes=(0, (1, 5), 2, 3, 4))  # t [l c] b r [s a]
        if fuse_one:
            fuse_one = False
            tmp = tmp.unfuse_legs(axes=4)
    return tmp
    # raise YastnError("dirn should be equal to 'l', 'r', 't', 'b', or None")


def fkron(A, B, sites=(0, 1), merge=True):
    """
    Fermionic kron; auxiliary function for gate generation.
    Returns a Kronecker product of two local operators, A and B,
    including swap-gate (fermionic string) to handle fermionic operators.

    Calculate A_0 B_1 for sites==(0, 1), and A_1 B_0 for sites==(1, 0),
    i.e., operator B acts first (relevant when operators are fermionically non-trivial).

    Site 0 is assumed to be first in fermionic orders.

    If merge, returns equivalent of
    ncon([A, B], [(-0, -1), (-2, -3)]) for sites==(0, 1), and
    ncon([A, B], [(-2, -3), (-0, -1)]) for sites==(1, 0),
    with proper operator order and swap gate applied.::

           1     3
           |     |
        ┌──┴─────┴──┐
        |           |
        └──┬─────┬──┘
           |     |
           0     2

    """

    if sites not in ((0, 1), (1, 0)):
        raise YastnError("Sites should be equal to (0, 1) or (1, 0).")
    ordered = (sites == (0, 1))

    s = -1 if ordered else 1
    A = A.add_leg(s=s, axis=2)
    B = B.add_leg(s=-s, axis=2)

    if ordered:
        A = A.swap_gate(axes=(1, 2))
    else:
        B = B.swap_gate(axes=(0, 2))

    G0, G1 = (A, B) if ordered else (B, A)

    if merge:
        return tensordot(G0, G1, axes=(2, 2))
    return G0, G1


def gate_fix_swap_gate(G0, G1, dirn, f_ordered):
    """
    Modifies two gate tensors, that were generated consistently with fermionic order 0->1.
    Apply swap_gates to make them consistent with provided orders.

    Lattice order (dirn=='lr' or 'tb') and f_ordered typically coincide.
    They do not coincide in a special case of cylindric geometry across the periodic boundary.
    """
    if dirn in ['rl', 'bt']:
        G0 = G0.swap_gate(axes=(1, -1))
        G1 = G1.swap_gate(axes=(0, 2))
    if f_ordered ^ (dirn in ['lr', 'tb']):  # for cylinder
        G1 = G1.swap_gate(axes=(2, 2))
    return G0, G1


def gate_from_mpo(op):
    G = [op.factor * op[op.first].remove_leg(axis=0).transpose(axes=(0, 2, 1))]
    for n in op.sweep(to='last', df=1):
        G.append(op[n].transpose(axes=(1, 3, 0, 2)))
    G[-1] = G[-1].remove_leg(axis=-1)
    return G


def fill_eye_in_gate(peps, G, sites):
    g0, g1 = G
    G = [g0]
    leg = g0.get_legs(axes=2)
    vb = eye(g0.config, legs=(leg.conj(), leg), isdiag=False)
    for site in sites[1:-1]:
        leg = peps[site].get_legs(axes=-1)
        if leg.is_fused():  # unfuse to get system leg
            leg, _ = leg.unfuse_leg()
        vp = eye(g0.config, legs=(leg, leg.conj()), isdiag=False)
        ten = vp.tensordot(vb, axes=((), ()))
        ten = ten.swap_gate(axes=(1, 2))
        G.append(ten)
    G.append(g1)
    return G
