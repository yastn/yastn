# Copyright 2026 The YASTN Authors. All Rights Reserved.
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
""" Support for einsum and ncon. """
from __future__ import annotations
from functools import lru_cache

from ._auxiliary import _clear_axes, _flatten
from ._contractions import tensordot, trace, swap_gate
from ._tests import YastnError

__all__ = ['ncon', 'einsum']


def einsum(subscripts, *operands, order=None, swap=None) -> 'Tensor':
    r"""
    Execute series of tensor contractions.

    Covering trace, tensordot (including outer products), and transpose.
    Follows notation of :meth:`np.einsum` as close as possible.

    Parameters
    ----------
    subscripts: str

    operands: Sequence[yastn.Tensor]

    order: str
        Specify order in which repeated indices from subscipt are contracted.
        By default it follows alphabetic order.

    Example
    -------

    ::

        yastn.einsum('*ij,jh->ih', t1, t2)

        # matrix-matrix multiplication, where the first matrix is conjugated.
        # Equivalent to

        t1.conj() @ t2

        yastn.einsum('ab,al,bm->lm', t1, t2, t3, order='ba')

        # Contract along `b` first, and `a` second.
    """
    if not isinstance(subscripts, str):
        raise YastnError('The first argument should be a string.')

    subscripts = subscripts.replace(' ', '')

    tmp = subscripts.split('->')
    if len(tmp) == 1:
        sin, sout = tmp[0], ''
    elif len(tmp) == 2:
        sin, sout = tmp
    else:
        raise YastnError('Subscript should have at most one separator ->')

    alphabet1 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    alphabet2 = alphabet1 + ',*'
    if any(v not in alphabet1 for v in sout) or \
       any(v not in alphabet2 for v in sin):
        raise YastnError('Only alphabetic characters can be used to index legs.')

    conjs = [1 if '*' in ss else 0 for ss in sin.split(',')]
    sin = sin.replace('*', '')

    if sout == '':
        for v in sin.replace(',', ''):
            if sin.count(v) == 1:
                sout += v
    elif len(sout) != len(set(sout)):
        raise YastnError('Repeated index after ->')

    if order is None:
        order = []
        for v in sin.replace(',', ''):
            if sin.count(v) > 1:
                order.append(v)
        order = ''.join(sorted(order))
    din = {v: i + 1 for i, v in enumerate(order)}
    dout = {v: -i for i, v in enumerate(sout)}
    d = {**din, **dout}
    d[','] = 0

    if any(v not in d for v in sin):
        raise YastnError('Order does not cover all contracted indices.')
    inds = [tuple(d[v] for v in ss) for ss in sin.split(',')]
    if swap is not None:
        swap = [tuple(d[v] for v in ss) for ss in swap.split(',')]

    ts = list(operands)
    return ncon(ts, inds, conjs=conjs, swap=swap)


def ncon(ts, inds, conjs=None, order=None, swap=None) -> 'Tensor':
    r"""
    Execute series of tensor contractions.

    Parameters
    ----------
    ts: Sequence[yastn.Tensor]
        list of tensors to be contracted.

    inds: Sequence[Sequence[int]]
        each inner tuple labels legs of respective tensor with integers.
        Positive values label legs to be contracted,
        with pairs of legs to be contracted denoted by the same integer label.
        Non-positive numbers label legs of the resulting tensor, in reversed order,
        i.e. -0 for the first outgoing leg, -1 for the second, -2 for the third, etc.

    swap: Sequence[Sequence[int]]
        Sequence of two-element tuples identifying pairs of legs where swap gate is applied.

    conjs: Sequence[int]
        For each tensor in ``ts`` contains either ``0`` or ``1``.
        If the value is ``1``, the tensor is conjugated.

    order: Sequence[int]
        Order in which legs, marked by positive indices in inds, are contracted.
        If None, the legs are contracted following an ascending indices order.
        The default is None.

    Note
    ----
    :meth:`yastn.ncon` and :meth:`yastn.einsum` differ only by syntax.

    Example
    -------

    ::

        # matrix-matrix multiplication where the first matrix is conjugated

        yastn.ncon([a, b], ((-0, 1), (1, -1)), conjs=(1, 0))

        # outer product

        yastn.ncon([a, b], ((-0, -2), (-1, -3)))
    """
    if len(ts) != len(inds):
        raise YastnError('Number of tensors and indices do not match.')
    for tensor, ind in zip(ts, inds):
        if tensor.ndim != len(ind):
            raise YastnError('Number of legs of one of the tensors do not match provided indices.')
    #
    ts = dict(enumerate(ts))
    #
    if conjs is not None:
        for t, to_conj in enumerate(conjs):
            if to_conj:
                ts[t] = ts[t].conj()
    #
    inds = tuple(_clear_axes(*inds))
    if order is not None:
        order = tuple(order)
    swap = tuple(_clear_axes(*swap)) if swap is not None else ()
    #
    commands = _meta_ncon(inds, order, swap)
    #
    ts = _execute_commands(ts, commands)
    assert len(ts) == 1, "Sanity check"
    return ts.popitem()[1]


def _execute_commands(ts, commands):
    for command in commands:
        if command[0] == 'tensordot':
            tout, (t1, t2), axes = command[1:]
            ts[tout] = tensordot(ts.pop(t1), ts.pop(t2), axes=axes)
        elif command[0] == 'swap_gate':
            tout, tin, axes = command[1:]
            ts[tout] = swap_gate(ts.pop(tin), axes=axes)
        elif command[0] == 'parity_sign':
            # Correction for jump-move on potentially parity-odd tensor.
            # If the jumped tensor has odd parity, apply (-1)^{n_d}
            # as a fermionic string on the partner leg.
            jumped_ten, d_ten, d_legs = command[1:]
            jumped = ts[jumped_ten]
            fss = jumped.config.fermionic
            if fss:
                nsym = jumped.config.sym.NSYM
                fss_tuple = (True,) * nsym if fss is True else fss
                n = jumped.n
                charge = tuple(n[k] % 2 if fss_tuple[k] else 0 for k in range(nsym))
                if any(charge):
                    ts[d_ten] = swap_gate(ts[d_ten], axes=d_legs, charge=charge)
        elif command[0] == 'trace':
            tout, tin, axes = command[1:]
            ts[tout] = trace(ts.pop(tin), axes=axes)
        else:
            assert command[0] == 'transpose', "Sanity check"
            tout, tin, axes = command[1:]
            ts[tout] = ts.pop(tin).transpose(axes=axes)
    return ts


@lru_cache(maxsize=1024)
def _meta_ncon(inds, order, swap):
    r""" Turning information in ``inds`` and ``conjs`` into list of contraction commands. """
    if not all(-256 < x < 256 for x in _flatten(inds)):
        raise YastnError('Ncon requires indices to be between -256 and 256.')

    if order is not None:
        if len(order) != len(set(order)) or not all(o > 0 for o in order):
            raise YastnError("Order should be a list of positive ints with no repetitions.")
        if not set(o for o in _flatten(inds) if o > 0) == set(order):
            raise YastnError("Positive ints in ins and order should match.")
        reorder = {o: k for k, o in enumerate(order, start=1)}
        inds = [[reorder[o] if o > 0 else o for o in xx] for xx in inds]
        swap = [[reorder[o] if o > 0 else o for o in xx] for xx in swap]
    #
    edges = [[ind, ten, leg] for ten, el in enumerate(inds) for leg, ind in enumerate(el)]
    #
    swaps = []
    if any(len(sw) != 2 for sw in swap):
        raise YastnError("swap should be a sequence of pairs.")
    for ind1, ind2 in swap:
        sw1 = [[ten, leg] for ind, ten, leg in edges if ind == ind1]
        sw2 = [[ten, leg] for ind, ten, leg in edges if ind == ind2]
        if len(sw1) not in [1, 2] or len(sw2) not in [1, 2]:
            raise YastnError("Indices of the legs to swap do not match inds.")
        swaps.append([sw1, sw2])
    #
    edges.append([512, 512, 512])  # this will mark the end of contractions.
    for edge in edges:  # modify outgoing indices for sorting
        if edge[0] <= 0:
            edge[0] = -edge[0] + 1024
    #
    edges = sorted(edges, reverse=True)
    #
    nlegs = {k: len(v) for k, v in enumerate(inds)}
    ten_out = max(nlegs)
    #
    commands, dot_done = [], False
    #
    axes1, axes2 = [], []
    while edges[-1][0] != 512:  # tensordot two tensors, or trace one tensor; 512 is cutoff marking end of contractions
        ind1, ten1, leg1 = edges.pop()
        ind2, ten2, leg2 = edges.pop()
        if ind1 != ind2:
            raise YastnError('Indices of legs to contract do not match.')
        if ten1 > ten2:
            ten1, ten2 = ten2, ten1
            leg1, leg2 = leg2, leg1
        axes1.append(leg1)
        axes2.append(leg2)
        if edges[-1][0] == 512 or (edges[-1][1], edges[-2][1]) not in [(ten1, ten2), (ten2, ten1)]:
            # first collect swaps
            swap_now, swap_later = [], []
            for sw12 in swaps:
                sw_now = _swap_on_tensor(*sw12)
                swap_now.append(sw_now) if sw_now else swap_later.append(sw12)
            swap_tensors = {}
            for ten_swap, axes_swap in swap_now:
                if ten_swap in swap_tensors:
                    swap_tensors[ten_swap].extend(axes_swap)
                else:
                    swap_tensors[ten_swap] = list(axes_swap)
            for ten_swap, axes_swap in swap_tensors.items():
                commands.append(('swap_gate', ten_swap, ten_swap, tuple(axes_swap)))
            swaps = swap_later
            #
            # resolve swaps on contracted legs via jump-moves
            tas = [[ten1, ax] for ax in axes1] + [[ten2, ax] for ax in axes2]
            if any(any(ta in sw12[0] or ta in sw12[1] for ta in tas) for sw12 in swaps):
                new_cmds, swaps = _resolve_bad_swaps(
                    swaps, edges, nlegs, ten1, ten2, axes1, axes2)
                commands.extend(new_cmds)
            #
            # contract
            if ten1 == ten2: # trace
                if dot_done:
                    raise YastnError("Likely inefficient order of contractions. Do all traces before tensordot. " +
                                     "Call all axes connecting two tensors one after another.")
                commands.append(('trace', ten1, ten1, (tuple(axes1), tuple(axes2))))
                axes12 = axes1 + axes2
                nlegs[ten1] -= len(axes12)
                _shift_edges_(edges, ten1, ten1, dax=lambda x: -sum(ax < x for ax in axes12))
                _shift_swaps_(swaps, ten1, ten1, dax=lambda x: -sum(ax < x for ax in axes12))
            else:  # tensordot
                dot_done = True
                ten_out += 1
                commands.append(('tensordot', ten_out, (ten1, ten2), (tuple(axes1), tuple(axes2))))
                nlegs[ten1] -= len(axes1)
                nlegs[ten2] -= len(axes2)
                _shift_edges_(edges, ten1, ten_out, dax=lambda x: -sum(ax < x for ax in axes1))
                _shift_edges_(edges, ten2, ten_out, dax=lambda x: nlegs[ten1] - sum(ax < x for ax in axes2))
                _shift_swaps_(swaps, ten1, ten_out, dax=lambda x: -sum(ax < x for ax in axes1))
                _shift_swaps_(swaps, ten2, ten_out, dax=lambda x: nlegs[ten1] - sum(ax < x for ax in axes2))
                nlegs[ten_out] = nlegs.pop(ten1) + nlegs.pop(ten2)
            axes1, axes2 = [], []
    #
    edges.pop()  # eliminate cutoff element
    #
    remaining = list(nlegs.keys())
    ten1 = remaining[0]
    for ten2 in remaining[1:]:  # tensordot
        ten_out += 1
        commands.append(('tensordot', ten_out, (ten1, ten2), ((), ())))
        _shift_edges_(edges, ten1, ten_out, dax=lambda x: 0)
        _shift_edges_(edges, ten2, ten_out, dax=lambda x: nlegs[ten1])
        _shift_swaps_(swaps, ten1, ten_out, dax=lambda x: 0)
        _shift_swaps_(swaps, ten2, ten_out, dax=lambda x: nlegs[ten1])
        nlegs[ten_out] = nlegs.pop(ten1) + nlegs.pop(ten2)
        ten1 = ten_out
    #
    if len(edges) != len(set(ind for ind, _, _ in edges)):
        raise YastnError("Repeated non-positive (outgoing) index is ambiguous.")
    #
    # final swaps
    axes_swap = tuple(ax for sw12 in swaps for ax in _swap_on_tensor(*sw12)[1])
    if axes_swap:
        commands.append(('swap_gate', ten_out, ten_out, axes_swap))
    #
    # final order for transpose
    axes = tuple(leg for _, _, leg in sorted(edges))
    if axes != tuple(range(len(axes))):
        commands.append(('transpose', ten_out, ten_out, axes))
    #
    return tuple(commands)


def _resolve_bad_swaps(swaps, edges, nlegs, ten1, ten2, axes1, axes2):
    r"""
    Resolve swap gates that sit on legs about to be contracted.

    Called before contracting ``ten1`` and ``ten2`` along ``axes1``/``axes2``.
    A swap is "bad" if it touches any of these contracted legs, because those
    legs will disappear after the contraction. This function moves every bad
    swap away from the contracted legs using jump-moves, emitting equivalent
    ``swap_gate`` and ``parity_sign`` commands that act on surviving legs.

    Data structures
    ---------------
    Internally, each "side" of a swap is a ``frozenset`` of ``(tensor, leg)``
    tuples — the endpoints sharing one edge.  A swap key is a canonical
    ``(tuple, tuple)`` pair derived from the two sides.  The Z2 swap set
    ``z2`` is a Python ``set`` of such keys; toggling is
    ``symmetric_difference_update``.

    An ``edge_of`` dict maps ``(tensor, leg)`` to its side (frozenset),
    giving O(1) edge lookup instead of scanning the ``edges`` list.

    Jump-move identity
    ------------------
    For a tensor T with legs {l_1, ..., l_n} and total charge n_T,
    define parity P_T = n_T mod 2 (componentwise on fermionic charge sectors).
    A swap between leg l_i of T and an external edge d can be replaced by::

        swap(l_i, d)  =  (-1)^{P_T * n_d}  *  prod_{j != i} swap(l_j, d)

    where n_d is the charge flowing through edge d.  Two identical swaps cancel
    (Z2 structure), so toggling the same swap twice is a no-op.

    For P_T = 0 (parity-even) the prefactor is 1.
    For P_T = 1 (parity-odd) the prefactor depends on the charge of d,
    which is only known at execution time.  A ``parity_sign`` command is
    emitted so that ``_execute_commands`` can apply the correction.

    Resolution strategy
    -------------------
    Bad swaps entering this function always have one side on the contracted
    edge and the other side on a third-party edge (both endpoints != ten1,
    ten2), because same-tensor swaps were already absorbed by the caller.

    The resolution loop applies one jump per iteration:

    **Step 1** — Find a third-party tensor C where every other leg of C
    either Z2-cancels an existing swap or shares a tensor with the partner
    side (same-tensor swap).  The jump fully absorbs all resulting swaps.
    Tensors with >= 2 distinct legs in bad swaps are tried first, as they
    are more likely to produce Z2 cancellations.

    **Step 2** — No suitable third-party tensor found.  Jump directly on one
    of the contracting tensors (ten1 or ten2) from its contracted leg.
    Since that tensor's own identity appears in both the new swap sides
    and the partner side, all resulting swaps are same-tensor swaps.
    Step 2 never creates new bad swaps.

    Each Step 1 iteration reduces the bad swap count by at least 1 (the
    jumped swap is removed, and all new swaps are either cancelled or
    absorbed).  Step 2 also removes exactly 1 bad swap.  The loop
    terminates in at most ``len(swaps)`` iterations.

    Parameters
    ----------
    swaps : list
        List of swap pairs ``[side_A, side_B]`` that were not absorbed as
        same-tensor swaps by the caller.  Each side is a list of
        ``[tensor_id, leg_index]`` entries sharing the same edge.

    edges : list
        Mutable list of ``[edge_index, tensor_id, leg_index]`` triples
        tracking which legs share each edge.  Modified in place by
        ``_shift_edges_`` after each contraction.

    nlegs : dict
        ``{tensor_id: number_of_legs}`` for every tensor currently alive.

    ten1, ten2 : int
        Tensor ids being contracted.

    axes1, axes2 : list
        Legs of ten1 and ten2 to be contracted (paired by position).

    Returns
    -------
    commands : list
        Sequence of ``('swap_gate', ...)`` and ``('parity_sign', ...)``
        commands to be executed before the contraction.

    remaining_swaps : list
        Swaps still in z2 after resolution (none of them bad).
    """
    contracted = frozenset((ten1, ax) for ax in axes1) | frozenset((ten2, ax) for ax in axes2)

    # Build edge lookup: (tensor, leg) -> sorted tuple of (tensor, leg) endpoints.
    edge_of = {}
    for ax1, ax2 in zip(axes1, axes2):
        edge = tuple(sorted(((ten1, ax1), (ten2, ax2))))
        edge_of[(ten1, ax1)] = edge
        edge_of[(ten2, ax2)] = edge
    by_ind = {}
    for ind, t, l in edges:
        if ind != 512:
            by_ind.setdefault(ind, []).append((t, l))
    for endpoints in by_ind.values():
        edge = tuple(sorted(endpoints))
        for tl in endpoints:
            edge_of[tl] = edge

    def _canonical(edge_a, edge_b):
        r"""Canonical key for an unordered swap of two edges (sorted tuples)."""
        return (edge_a, edge_b) if edge_a <= edge_b else (edge_b, edge_a)

    def _same_tensor(edge_a, edge_b):
        r"""If edges share a common tensor, return (tensor, (leg_a, leg_b)); else None."""
        tens_a = {t for t, _ in edge_a}
        tens_b = {t for t, _ in edge_b}
        common = tens_a & tens_b
        if common:
            t = min(common)
            la = next(l for tt, l in edge_a if tt == t)
            lb = next(l for tt, l in edge_b if tt == t)
            return t, tuple(sorted((la, lb)))
        return None

    # Build Z2 swap set from input swaps.
    z2 = set()
    for sw12 in swaps:
        edge_a = tuple(sorted(tuple(x) for x in sw12[0]))
        edge_b = tuple(sorted(tuple(x) for x in sw12[1]))
        z2.symmetric_difference_update({_canonical(edge_a, edge_b)})

    commands = []

    def toggle(edge_a, edge_b):
        r"""Toggle a swap in the Z2 set.  Add if absent, remove if present."""
        z2.symmetric_difference_update({_canonical(edge_a, edge_b)})

    def collect_same_tensor():
        r"""Absorb all same-tensor swaps from z2 into swap_gate commands."""
        by_tensor = {}
        for key in list(z2):
            result = _same_tensor(key[0], key[1])
            if result:
                z2.discard(key)
                ten_s, axes_s = result
                by_tensor.setdefault(ten_s, []).extend(axes_s)
        for ten_s, ax_s in by_tensor.items():
            if ax_s:
                commands.append(('swap_gate', ten_s, ten_s, tuple(ax_s)))

    def jump(tid, leg_xs, partner):
        r"""Replace swap(leg_xs, partner) with swaps on all other legs of tid."""
        if type(leg_xs) is int:
            leg_xs = (leg_xs,)
        for l in range(nlegs[tid]):
            if l not in leg_xs:
                toggle(edge_of[(tid, l)], partner)
        d_ten, d_leg = partner[0]
        commands.append(('parity_sign', tid, d_ten, (d_leg,)))

    def get_ax_to_jump(ax_dict):
        r"""Get the leg of a tensor with minimal bad swaps."""
        min_cross, ax = min((len(keys), ax) for ax, keys in ax_dict.items())

        if min_cross == len(axes1): # No need to apply step-1
            return None
        return ax

    def get_third_party_tensor():
        # Collect third-party tensor appearances in bad swaps.
        bad = [key for key in z2
               if any(tl in contracted for tl in key[0])
               or any(tl in contracted for tl in key[1])]
        if not bad:
            return None

        # tp[C] = list of (key, partner_edge, leg) entries.
        tp = {}
        for key in bad:
            for si in (0, 1):
                for t, l in key[si]:
                    if t != ten1 and t != ten2:
                        if t not in tp:
                            tp[t] = {}
                        tp[t].setdefault(l, []).append((key, key[1 - si]))
        return tp

    for _iter in range(256):
        tp = get_third_party_tensor()
        if not tp:
            break
        ready_for_step2 = True
        # Step-1: jump on a third-party tensor
        for C in tp:
            ax = get_ax_to_jump(tp[C])
            if ax is not None:
                _key, partner = tp[C][ax][0]
                z2.discard(_key)
                jump(C, ax, partner)
                collect_same_tensor()
                ready_for_step2 = False
                break

        if ready_for_step2:
            break

            # all_absorbed = True
            # for l in range(nlegs[C]):
            #     if l != leg_x:
            #         new_edge = edge_of[(C, l)]
            #         if _canonical(new_edge, partner) in z2:
            #             continue  # will Z2-cancel
            #         if _same_tensor(new_edge, partner):
            #             continue  # same-tensor swap
            #         all_absorbed = False
            #         break

    # Step-2: jump on a contracting tensor from its contracted legs.
    # Each remaining bad swap has a third-party edge crossing all contracted
    # legs.  Group by unique third-party edge: discard all bad swaps for that
    # edge, then do one jump from the contracted legs.
    tp = get_third_party_tensor()
    if nlegs[ten1] - len(axes1) <= nlegs[ten2] - len(axes2):
        t, ls = ten1, axes1
    else:
        t, ls = ten2, axes2
    if tp:
        seen_edges = set()
        for C in tp:
            for ax in tp[C]:
                edge = edge_of[(C, ax)]
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                for _key, _ in tp[C][ax]:
                    z2.discard(_key)
                jump(t, ls, edge)
                collect_same_tensor()

    # Convert back to list format for the caller (_shift_swaps_ mutates).
    remaining = [[[list(tl) for tl in key[0]],
                  [list(tl) for tl in key[1]]] for key in z2]
    return commands, remaining


def _shift_edges_(edges, ten_old, ten_new, dax):
    for edge in edges:
        if edge[1] == ten_old:
            edge[1] = ten_new
            edge[2] += dax(edge[2])


def _shift_swaps_(swaps, ten_old, ten_new, dax):
    for sw12 in swaps:
        for sws in sw12:
            for sw in sws:
                if sw[0] == ten_old:
                    sw[0] = ten_new
                    sw[1] += dax(sw[1])


def _swap_on_tensor(sw1, sw2):
    tens1 = [x[0] for x in sw1]
    tens2 = [x[0] for x in sw2]
    ten = set(tens1) & set(tens2)
    if ten:
        ten = ten.pop()
        i1 = tens1.index(ten)
        i2 = tens2.index(ten)
        return ten, tuple(sorted((sw1[i1][1], sw2[i2][1])))
    return False
