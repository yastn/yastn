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

from ._geometry import Lattice
from ...initialize import eye
from ...tensor import tensordot, YastnError, Tensor


class Gate(NamedTuple):
    r"""
    Gate to be applied on Peps state.

    `G` contains operators for respective `sites`.

    Operator can be given in the form of an MPO (:class:`yastn.tn.mps.MpsMpoOBC`) of the same length as the number of provided `sites`.
    Sites should form a continuous path in the two-dimensional PEPS lattice.
    The fermionic order of MPO should be linear, with the first MPO site being first in the fermionic order, irrespective of the provided `sites`.

    For a two-site operator acting on sites beyond nearest neighbor, it can be provided as `G` with two elements, and `sites` forming a path between the end sites where the provided elements of `G` will act.

    It is also possible to provide `G` as a list of tensors.
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
    if G is None:
        return G

    leg = ten.get_legs(axes=-1)
    legG = G.get_legs(axes=1)
    if leg.hf.tree == legG.hf.tree:
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


# How the bond of a gate tensor meets the PEPS tensor it is absorbed into, by the role of the
# site in a step of the gate's path ('t', 'b', 'l', 'r': the site is the top, bottom, left or
# right end of the lattice bond, see :meth:`SquareLattice.nn_bond_dirn`).  The bond is fused
# into one virtual leg (0 = top, 1 = left, 2 = bottom, 3 = right) and, on the way there, crosses
# the leg given second, if any.
BOND_FUSION = {'t': (2, None), 'b': (0, 1), 'l': (3, 2), 'r': (1, None)}


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

    for dd in dirn[::-1]:  # the outgoing bond first, the incoming one waiting in the physical leg
        leg, crossed = BOND_FUSION[dd]
        if crossed is not None:
            tmp = tmp.swap_gate(axes=(crossed, 5))
        tmp = tmp.fuse_legs(axes=tuple((ax, 5) if ax == leg else ax for ax in range(5)))
        if fuse_one:
            fuse_one = False
            tmp = tmp.unfuse_legs(axes=4)
    return tmp
    # raise YastnError("dirn should be equal to 'l', 'r', 't', 'b', or None")


def ordering_swaps(dirn, f_ordered):
    """
    Swap gates that adapt a step of a gate, built as if it ran forward -- its first
    tensor left of or above the second, and first in the fermionic order -- to a step
    in direction ``dirn`` (:meth:`SquareLattice.nn_bond_dirn`) whose two sites are in
    fermionic order ``f_ordered``.

    Returns ``(tensor, leg)`` pairs, each a swap of the bond joining the two tensors
    with ``leg`` of ``tensor``: ``tensor`` is 0 for the first, which the bond leaves,
    or 1 for the second, which it enters; ``leg`` is ``'out'`` or ``'in'`` for a
    physical leg, or ``'bond'`` for the bond itself, i.e. its parity.  Lattice and
    fermionic order disagree only across the periodic boundary of a cylinder.
    """
    swaps = []
    if dirn in ('rl', 'bt'):  # the step runs against the lattice order
        swaps += [(0, 'in'), (1, 'out')]
    if f_ordered ^ (dirn in ('lr', 'tb')):  # only across the periodic boundary of a cylinder
        swaps.append((1, 'bond'))
    return swaps


def ordering_swap_axes(tensor, leg):
    """Axes of the swap gate realizing an :func:`ordering_swaps` entry on a gate tensor
    with legs ``(out, in, bonds...)``: the joining bond is the last leg of the first
    tensor and the third of the second."""
    bond = -1 if tensor == 0 else 2  # -1: outgoing bond (right side), 2: incoming bond (left side)
    return (bond if leg == 'bond' else {'out': 0, 'in': 1}[leg], bond)


def gate_fix_swap_gate(G0, G1, dirn, f_ordered):
    """
    Modifies two gate tensors, that were generated consistently with fermionic order 0->1,
    to make them consistent with the step ``dirn`` and the fermionic order ``f_ordered``;
    see :func:`ordering_swaps`.

    The bond ``v`` leaves ``G0`` on its right side and enters ``G1`` on its left side.
    In a forward step, ``'lr'`` (or ``'tb'``, rotated), this matches the lattice and no
    swap gate is needed::

        out0              out1
         │                 │
         G0 ───── v ────── G1
         │                 │
        in0               in1

    In a backward step, ``'rl'`` (or ``'bt'``, rotated), ``G1`` sits on the other side.
    The bond passes below ``G0``, crossing ``in0``, and over ``G1``, crossing ``out1``::

                    out1
            ┌────────┼────────┐
            │        │        │             out0
            └── v ── G1       │              │
                     │        │              G0 ── v ──┐
                    in1       │              │         │
                              └──────────────┼─────────┘
                                            in0

    Across the periodic boundary of a cylinder, the swap gate of ``v`` with itself, i.e.
    its parity, is applied to ``G1``.
    """
    G = [G0, G1]
    for tensor, leg in ordering_swaps(dirn, f_ordered):
        G[tensor] = G[tensor].swap_gate(axes=ordering_swap_axes(tensor, leg))
    return G[0], G[1]


def gate_from_mpo(op):
    G = [op.factor * op[op.first].remove_leg(axis=0).transpose(axes=(0, 2, 1))]
    for n in op.sweep(to='last', df=1):
        G.append(op[n].transpose(axes=(1, 3, 0, 2)))
    G[-1] = G[-1].remove_leg(axis=-1)
    return G


def system_leg(ten):
    """Physical leg of a PEPS tensor, without the ancilla of a purification."""
    leg = ten.get_legs(axes=-1)
    return leg.unfuse_leg()[0] if leg.is_fused() else leg


def fill_eye_in_gate(peps, G, sites):
    g0, g1 = G
    G = [g0]
    leg = g0.get_legs(axes=2)
    try:
        vb = eye(g0.config, legs=(leg.conj(), leg), isdiag=False)
    except YastnError as exc:
        if "not a result of outer_product" not in str(exc):
            raise
        # A compact sum-of-products gate has a direct-sum auxiliary leg.
        # Its block history need not be an outer product; the propagating
        # identity depends only on its charge sectors and dimensions.
        leg = leg.drop_history()
        vb = eye(g0.config, legs=(leg.conj(), leg), isdiag=False)
    for site in sites[1:-1]:
        leg = system_leg(peps[site])
        vp = eye(g0.config, legs=(leg, leg.conj()), isdiag=False)
        ten = vp.tensordot(vb, axes=((), ()))
        ten = ten.swap_gate(axes=(1, 2))
        G.append(ten)
    G.append(g1)
    return G


def clear_projectors(sites, projectors):
    """ prepare projectors for sampling functions. """
    if isinstance(projectors, Lattice):
        projectors = projectors.shallow_copy()
    elif isinstance(projectors, dict) and not all(isinstance(x, Tensor) for x in projectors.values()):
        projectors = projectors.copy()
    else:
        projectors = {site: projectors.copy() for site in sites}

    try:
        for k in sites: projectors[k]
    except KeyError:
        raise YastnError(f"Projectors not defined for some sites.")

    # change each list of projectors into keys and projectors
    for k, v in projectors.items():
        projectors[k] = dict(v) if isinstance(v, dict) else dict(enumerate(v))
        for l, pr in projectors[k].items():
            if pr.ndim == 1:  # vectors need conjugation
                if abs(pr.norm() - 1) > 1e-10:
                    raise YastnError("Local states to project on should be normalized.")
                projectors[k][l] = tensordot(pr, pr.conj(), axes=((), ()))
            elif pr.ndim == 2:
                if (pr.n != pr.config.sym.zero()) or abs(pr @ pr - pr).norm() > 1e-10:
                    raise YastnError("Matrix projectors should be projectors, P @ P == P.")
            elif pr.ndim == 4:
                pass
            else:
                raise YastnError("Projectors should consist of vectors (ndim=1) or matrices (ndim=2).")

    return projectors


def clear_operator_input(op, sites):
    if isinstance(op, Lattice):
        op_dict = op.shallow_copy()
    elif isinstance(op, dict):
        op_dict = op.copy()
    else:
        op_dict = {site: op for site in sites}

    try:
        for k in sites: op_dict[k]
    except KeyError:
        raise YastnError(f"Operators not defined for some sites.")

    for k, v in op_dict.items():
        if isinstance(v, dict):
            op_dict[k] = {(i,): vi for i, vi in v.items()}
        elif isinstance(v, Tensor):
            op_dict[k] = {(): v}
        else: # is iterable
            op_dict[k] = {(i,): vi for i, vi in enumerate(v)}
    return op_dict
