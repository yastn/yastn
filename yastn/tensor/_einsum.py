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
        Specify order in which repeated indices from subscript are contracted.
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
        order = ''.join(sorted(set(v for v in sin.replace(',', '') if sin.count(v) > 1)))
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
            charge = ts[jumped_ten].n
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
    r"""
    Plan a sequence of contraction commands from index notation.

    This is a pure-metadata planner: it inspects only index labels and tensor
    IDs, never tensor data.  The result is an ``@lru_cache``-d tuple of
    commands executed later by ``_execute_commands``.

    Index encoding
    --------------
    * Positive indices label legs to be contracted (matching pairs).
    * Non-positive indices label outgoing legs of the result.
    * ``order`` remaps positive indices so that index 1 is contracted first.

    Edge list
    ---------
    ``edges`` is a mutable list of ``[ind, tensor_id, leg_index]`` triples,
    sorted **descending**.  ``edges.pop()`` yields the lowest index (next to
    contract).  A sentinel ``[512, 512, 512]`` separates contracted edges
    (ind < 512) from outgoing edges (ind > 1024, remapped from non-positive).

    Main loop
    ---------
    Each iteration pops a matched pair of edges and batches consecutive
    edges between the same tensor pair.  Before contracting, same-tensor
    swaps are collected and bad swaps (touching contracted legs) are
    resolved via ``_resolve_bad_swaps``.  The contraction is one of:

    * **Tensordot** (``ten1 != ten2``): merge two tensors.
    * **Trace** (``ten1 == ten2``): contract self-loop legs.
    * **Unresolved fallback**: when ``_resolve_bad_swaps`` cannot move all
      bad swaps away (partial crossing), contract only the crossed axes
      and re-insert the rest as trace edges on the merged tensor.

    Deferred trace
    --------------
    The jump-move identity is a Z2 tautology for self-loop edges, so
    ``_resolve_bad_swaps`` cannot resolve bad swaps on trace legs.  When
    bad swaps remain after resolution, the trace is deferred: its edges
    are re-inserted with index = max external-edge index of the bad swaps.
    Once the relevant third-party tensors merge into the trace tensor
    (turning cross-tensor swaps into same-tensor swaps), the trace
    executes.  If the merge chain is indirect, the trace re-defers with
    a progressively higher index until resolved.

    Post-loop
    ---------
    Disconnected tensors are combined via outer products.  Remaining
    same-tensor swaps are applied, and output legs are transposed to the
    requested order.
    """
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
    commands = []

    def _apply_trace_split(ten, imm_axes1, imm_axes2, deferred_pairs):
        r"""Trace immediate pairs on ten and reinsert deferred pairs as trace edges."""
        if deferred_pairs:
            defer_ind = min((e[0] for e in edges
                             if e[1] == ten and 0 < e[0] < 512), default=511)
            for npair, (ax1, ax2, ind) in enumerate(deferred_pairs, start=1):
                d = ind if ind is not None else defer_ind + npair * 0.01
                edges.append([d, ten, ax1])
                edges.append([d, ten, ax2])
        if imm_axes1:
            commands.append(('trace', ten, ten, (tuple(imm_axes1), tuple(imm_axes2))))
            axes12 = imm_axes1 + imm_axes2
            nlegs[ten] -= len(axes12)
            _shift_edges_(edges, ten, ten, dax=lambda x: -sum(ax < x for ax in axes12))
            _shift_swaps_(swaps, ten, ten, dax=lambda x: -sum(ax < x for ax in axes12))
        if deferred_pairs:
            edges.sort(reverse=True)
    #
    axes1, axes2, inds_for_axes = [], [], []
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
        inds_for_axes.append(ind1)
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
            unresolved = False
            if any(any(ta in sw12[0] or ta in sw12[1] for ta in tas) for sw12 in swaps):
                new_cmds, swaps, unresolved = _resolve_bad_swaps(
                    swaps, edges, nlegs, ten1, ten2, axes1, axes2)
                commands.extend(new_cmds)
            #
            # contract
            if unresolved:
                # Partial crossing: defer crossed legs as trace pairs and
                # contract the rest in one pass.  _resolve_bad_swaps already
                # minimizes this deferred set.
                deferred = sorted(unresolved)
                deferred_set = set(deferred)
                if ten1 == ten2:
                    imm_axes1 = [axes1[k] for k in range(len(axes1)) if k not in deferred_set]
                    imm_axes2 = [axes2[k] for k in range(len(axes2)) if k not in deferred_set]
                    deferred_pairs = [(axes1[k], axes2[k], None) for k in deferred]
                    _apply_trace_split(ten1, imm_axes1, imm_axes2, deferred_pairs)
                    axes1, axes2, inds_for_axes = [], [], []
                    continue
                to_contract = [k for k in range(len(axes1)) if k not in deferred_set]
                axes1_c = [axes1[k] for k in to_contract]
                axes2_c = [axes2[k] for k in to_contract]
                ten_out += 1
                commands.append(('tensordot', ten_out, (ten1, ten2),
                                 (tuple(axes1_c), tuple(axes2_c))))
                nlegs[ten1] -= len(axes1_c)
                nlegs[ten2] -= len(axes2_c)
                nlegs_ten1_rem = nlegs[ten1]
                _shift_edges_(edges, ten1, ten_out,
                              dax=lambda x: -sum(ax < x for ax in axes1_c))
                _shift_edges_(edges, ten2, ten_out,
                              dax=lambda x: nlegs_ten1_rem - sum(ax < x for ax in axes2_c))
                _shift_swaps_(swaps, ten1, ten_out,
                              dax=lambda x: -sum(ax < x for ax in axes1_c))
                _shift_swaps_(swaps, ten2, ten_out,
                              dax=lambda x: nlegs_ten1_rem - sum(ax < x for ax in axes2_c))
                nlegs[ten_out] = nlegs.pop(ten1) + nlegs.pop(ten2)
                # Re-insert deferred crossed axes as trace pairs on ten_out.
                for k in deferred:
                    new_ax1 = axes1[k] - sum(ax < axes1[k] for ax in axes1_c)
                    new_ax2 = nlegs_ten1_rem + axes2[k] - sum(ax < axes2[k] for ax in axes2_c)
                    edges.append([inds_for_axes[k], ten_out, new_ax1])
                    edges.append([inds_for_axes[k], ten_out, new_ax2])
                edges.sort(reverse=True)
            elif ten1 == ten2:  # trace
                # If _resolve_bad_swaps saw any bad trace pairs, it must have
                # reported them via `unresolved` already. Reaching this branch
                # means every current trace pair is safe to execute immediately.
                _apply_trace_split(ten1, axes1, axes2, [])
            else:  # tensordot
                ten_out += 1
                commands.append(('tensordot', ten_out, (ten1, ten2), (tuple(axes1), tuple(axes2))))
                nlegs[ten1] -= len(axes1)
                nlegs[ten2] -= len(axes2)
                _shift_edges_(edges, ten1, ten_out, dax=lambda x: -sum(ax < x for ax in axes1))
                _shift_edges_(edges, ten2, ten_out, dax=lambda x: nlegs[ten1] - sum(ax < x for ax in axes2))
                _shift_swaps_(swaps, ten1, ten_out, dax=lambda x: -sum(ax < x for ax in axes1))
                _shift_swaps_(swaps, ten2, ten_out, dax=lambda x: nlegs[ten1] - sum(ax < x for ax in axes2))
                nlegs[ten_out] = nlegs.pop(ten1) + nlegs.pop(ten2)
            axes1, axes2, inds_for_axes = [], [], []
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
    axes_swap = tuple(ax for sw12 in swaps for sw in [_swap_on_tensor(*sw12)] if sw for ax in sw[1])
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
    Internally, each edge gets a stable integer ``edge_id``. Two maps are
    maintained:

    * ``edge_endpoints[edge_id] -> tuple[(tensor, leg), ...]``
    * ``leg_to_edge[(tensor, leg)] -> edge_id``

    A swap key is a canonical ``(edge_id_a, edge_id_b)`` pair with
    ``edge_id_a <= edge_id_b``. The Z2 swap set ``z2`` stores these keys;
    ``toggle`` adds a key if absent or removes it if present.

    Newly added keys are tracked in ``_newly_added`` so that
    ``collect_same_tensor`` only inspects recent additions rather than
    scanning all of ``z2``.

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

    Strategy
    --------------------------
    Starting from a third-party tensor C, find the leg L with the most bad-swaps,
    and apply the jump to make all contracted legs cross that leg.
    We then apply jump to move L to the other side of ten1 (or ten2) [eliminate-move].
    The choice of ten1 vs ten2 is arbitrary, but we pick the one that generates fewer swap gates.
    Repeat eliminate-move for the legs of C that cross all contracted legs.

    For the legs of C that still cross contracted legs partially, we resolve it by going to the other tensor
    that share the leg with C and repeat the procedure above. The propagation is done in a DFS manner,
    and terminates when either all bad swaps associated to the current tensor are resolved, or we encounter an uncontracted leg.

    If the direct DFS/jump strategy stalls and some bad swaps still touch the contracted legs, a fallback resolver
    is invoked. The fallback first looks for a third-party edge that crosses all currently active contracted legs
    and removes it with the standard step-2 jump. If only partial crossings remain, it defers the non-crossed
    contracted axes, shrinking the active contraction set until the remaining bad swaps become resolvable. For
    traces, this fallback simply marks the touched trace pairs for deferral.


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
        Swaps still in z2 after resolution (none of them bad when
        ``unresolved`` is ``False``).

    deferred : False or frozenset
        ``False`` if all bad swaps were resolved.  Otherwise a
        ``frozenset`` of axis indices (into ``axes1``/``axes2``) that
        should be deferred as trace pairs.  The caller should contract
        the remaining axes and defer these.
    """
    deferred_indices = set()

    # Build edge registries:
    # - edge_endpoints: edge_id -> sorted tuple of (tensor, leg)
    # - leg_to_edge: (tensor, leg) -> edge_id
    edge_endpoints = {}
    leg_to_edge = {}
    edge_to_id = {}
    next_edge_id = 0

    def register_edge(endpoints):
        r"""Register/lookup an edge from endpoints and return its edge_id."""
        nonlocal next_edge_id
        edge = tuple(sorted(endpoints))
        edge_id = edge_to_id.get(edge)
        if edge_id is None:
            edge_id = next_edge_id
            next_edge_id += 1
            edge_to_id[edge] = edge_id
            edge_endpoints[edge_id] = edge
        for tl in edge:
            leg_to_edge[tl] = edge_id
        return edge_id

    contracted_edges_by_axis = []
    for ax1, ax2 in zip(axes1, axes2):
        contracted_edges_by_axis.append(register_edge(((ten1, ax1), (ten2, ax2))))

    by_ind = {}
    for ind, t, l in edges:
        if ind != 512:
            by_ind.setdefault(ind, []).append((t, l))
    for endpoints in by_ind.values():
        register_edge(endpoints)

    contracted_edges = set(contracted_edges_by_axis)

    def _canonical(edge_a, edge_b):
        r"""Canonical key for an unordered swap of two edge IDs."""
        return (edge_a, edge_b) if edge_a <= edge_b else (edge_b, edge_a)

    def _same_tensor(edge_a, edge_b):
        r"""If edges share a common tensor, return (tensor, (leg_a, leg_b)); else None."""
        eps_a = edge_endpoints[edge_a]
        eps_b = edge_endpoints[edge_b]
        tens_a = {t for t, _ in eps_a}
        tens_b = {t for t, _ in eps_b}
        common = tens_a & tens_b
        if common:
            t = min(common)
            la = next(l for tt, l in eps_a if tt == t)
            lb = next(l for tt, l in eps_b if tt == t)
            return t, tuple(sorted((la, lb)))
        return None

    def side_to_edge(side):
        r"""Map one swap side (list of legs on the same edge) to edge_id."""
        edge_ids = {leg_to_edge[tuple(tl)] for tl in side}
        if len(edge_ids) != 1:
            raise YastnError("Inconsistent edge encoding in swap.")
        return edge_ids.pop()

    # Build Z2 swap set from input swaps.
    z2 = set()
    for sw12 in swaps:
        edge_a = side_to_edge(sw12[0])
        edge_b = side_to_edge(sw12[1])
        z2.symmetric_difference_update({_canonical(edge_a, edge_b)})

    commands = []
    _newly_added = []  # keys added by toggle, consumed by collect_same_tensor

    def get_pivot_edge(ax_dict):
        r"""Get the leg of a tensor with maximal bad swaps."""
        max_cross, ax = max((len(keys), ax) for ax, keys in ax_dict.items())
        return max_cross, ax

    def get_crossed_axes(contracted_edge, ax_dict):
        r"""Get all the axes of a tensor that are crossed by a contracted edge."""
        crossed = set()
        for ax, keys in ax_dict.items():
            if any(contracted_edge in key for key, _ in keys):
                crossed.add(ax)
        return crossed

    def get_third_party_tensor():
        r"""Return {tensor: {leg: [(key, partner), ...]}} for non-contracted tensors in bad swaps."""
        bad = [key for key in z2
               if key[0] in contracted_edges or key[1] in contracted_edges]
        if not bad:
            return None

        tp = {}
        for key in bad:
            for si in (0, 1):
                for t, l in edge_endpoints[key[si]]:
                    if t != ten1 and t != ten2:
                        if t not in tp:
                            tp[t] = {}
                        tp[t].setdefault(l, []).append((key, key[1 - si]))
        return tp

    def _is_bad_key(key):
        r"""Check if a z2 key belongs to the bad-swap view tracked by tp."""
        return key[0] in contracted_edges or key[1] in contracted_edges

    def _tp_add_key(tp, key):
        r"""Insert one z2 key into tp."""
        if not _is_bad_key(key):
            return
        for edge, partner in ((key[0], key[1]), (key[1], key[0])):
            for t, l in edge_endpoints[edge]:
                if t != ten1 and t != ten2:
                    tp.setdefault(t, {}).setdefault(l, []).append((key, partner))

    def _tp_remove_key(tp, key):
        r"""Remove one z2 key from tp."""
        if not _is_bad_key(key):
            return
        for edge, partner in ((key[0], key[1]), (key[1], key[0])):
            for t, l in edge_endpoints[edge]:
                if t != ten1 and t != ten2 and t in tp and l in tp[t]:
                    tp[t][l] = [item for item in tp[t][l] if item != (key, partner)]
                    if not tp[t][l]:
                        del tp[t][l]
                    if not tp[t]:
                        del tp[t]

    def _tp_apply_deltas(tp, deltas):
        r"""Apply a sequence of ('add'|'remove', key) updates to tp."""
        for op, key in deltas:
            if op == 'add':
                _tp_add_key(tp, key)
            else:
                _tp_remove_key(tp, key)

    def toggle_with_delta(edge_a, edge_b):
        r"""Toggle a swap in z2 and return the corresponding delta for tp."""
        key = _canonical(edge_a, edge_b)
        if key in z2:
            z2.discard(key)
            return 'remove', key
        z2.add(key)
        _newly_added.append(key)
        return 'add', key

    def jump_with_deltas(tid, leg_xs, partner):
        r"""Apply jump, consuming swap(leg_xs, partner), and return z2 deltas for tp."""
        if isinstance(leg_xs, int):
            leg_xs = (leg_xs,)
        skip = set(leg_xs)
        deltas = []
        seen_removed = set()
        for l in skip:
            key = _canonical(leg_to_edge[(tid, l)], partner)
            if key in seen_removed:
                continue
            seen_removed.add(key)
            if key in z2:
                z2.discard(key)
                deltas.append(('remove', key))
        for l in range(nlegs[tid]):
            if l not in skip:
                deltas.append(toggle_with_delta(leg_to_edge[(tid, l)], partner))
        d_ten, d_leg = edge_endpoints[partner][0]
        commands.append(('parity_sign', tid, d_ten, (d_leg,)))
        return deltas

    def collect_same_tensor_with_tp(tp):
        r"""Absorb same-tensor swaps and keep tp synchronized with the surviving z2 state."""
        by_tensor = {}
        while _newly_added:
            key = _newly_added.pop()
            if key not in z2:
                continue  # Z2-cancelled since it was added
            result = _same_tensor(key[0], key[1])
            if result:
                z2.discard(key)
                _tp_remove_key(tp, key)
                ten_s, axes_s = result
                by_tensor.setdefault(ten_s, []).extend(axes_s)
        for ten_s, ax_s in by_tensor.items():
            commands.append(('swap_gate', ten_s, ten_s, tuple(ax_s)))

    def get_step2_jump_target():
        r"""Return the tensor/legs used for step-2 jumps, or (None, ()) for traces."""
        if ten1 == ten2:
            return None, ()
        if nlegs[ten1] - len(axes1) <= nlegs[ten2] - len(axes2):
            return ten1, [axes1[k] for k in range(len(axes1)) if k not in deferred_indices]
        return ten2, [axes2[k] for k in range(len(axes2)) if k not in deferred_indices]

    def get_crossed_active_indices(entries, active):
        r"""Return active contracted-axis indices touched by bad swaps in entries."""
        crossed = set()
        for _key, _partner in entries:
            for side_edge in _key:
                for k in active:
                    if side_edge == contracted_edges_by_axis[k]:
                        crossed.add(k)
        return crossed

    def run_fallback_resolver():
        r"""Fallback to full-crossing step-2 or deferred axes if the main pass stalls."""
        while True:
            tp = get_third_party_tensor()
            if not tp:
                return

            active = set(range(len(axes1))) - deferred_indices
            if not active:
                return
            n_active = len(active)

            if ten1 == ten2:
                touched = set()
                for C in tp:
                    for ax in tp[C]:
                        touched |= get_crossed_active_indices(tp[C][ax], active)
                if touched:
                    deferred_indices.update(touched)
                    for k in touched:
                        contracted_edges.discard(contracted_edges_by_axis[k])
                return

            first_full = None
            best_partial = None
            seen_edges = set()
            touched = set()
            for C in tp:
                for ax in tp[C]:
                    edge = leg_to_edge[(C, ax)]
                    if edge in seen_edges:
                        continue
                    seen_edges.add(edge)
                    crossed = get_crossed_active_indices(tp[C][ax], active)
                    touched |= crossed
                    if len(crossed) == n_active:
                        if first_full is None:
                            first_full = (C, ax, edge)
                    elif best_partial is None or len(crossed) > len(best_partial):
                        best_partial = frozenset(crossed)

            if first_full is not None:
                t, ls = get_step2_jump_target()
                if t is None:
                    return
                _, _, edge = first_full
                deltas = jump_with_deltas(t, ls, edge)
                _tp_apply_deltas(tp, deltas)
                collect_same_tensor_with_tp(tp)
                continue

            if best_partial:
                to_defer_now = active - set(best_partial)
            else:
                to_defer_now = touched

            if not to_defer_now:
                return

            deferred_indices.update(to_defer_now)
            for k in to_defer_now:
                contracted_edges.discard(contracted_edges_by_axis[k])

    def resolve_tensor_dfs(tp, C, ax):
        r"""Main resolution path: normalize one tensor and recursively push leftovers outward."""
        num_cross = len(tp[C][ax])
        if num_cross < len(axes1):
            for k in range(len(axes1)):
                if C not in tp:
                    break
                crossed_ax = get_crossed_axes(contracted_edges_by_axis[k], tp[C])
                if ax not in crossed_ax:
                    # apply jump-move to make the contracted edge cross the pivot leg
                    deltas = jump_with_deltas(C, tuple(crossed_ax), contracted_edges_by_axis[k])
                    _tp_apply_deltas(tp, deltas)
                    collect_same_tensor_with_tp(tp)

        # C is completely resolved
        if C not in tp:
            return

        # Once one edge of C crosses every active contracted leg, remove it with step-2.
        t, ls = get_step2_jump_target()

        num_cross = len(tp[C][ax])
        edge = leg_to_edge[(C, ax)]
        while t is not None and num_cross == len(axes1):
            deltas = jump_with_deltas(t, ls, edge)
            _tp_apply_deltas(tp, deltas)
            collect_same_tensor_with_tp(tp)
            if C not in tp:
                break
            else:
                num_cross, ax = get_pivot_edge(tp[C])
                edge = leg_to_edge[(C, ax)]

        # C is completely resolved
        if C not in tp:
            return

        # Otherwise, propagate the bad swaps to the neighboring tensor (Depth-first-search)
        for bad_ax in tuple(tp[C]): # static snapshot since tp is mutated by the loop
            if C not in tp:
                return
            elif bad_ax not in tp[C]:
                continue

            num_cross = len(tp[C][bad_ax])
            if num_cross < len(axes1) and num_cross > 0:
                bad_edge = leg_to_edge[(C, bad_ax)]
                eps = edge_endpoints[bad_edge]
                if len(eps) < 2: # can't propagate further due to the open edge
                    continue
                other_side = eps[0] if (C, bad_ax) != eps[0] else eps[1]
                resolve_tensor_dfs(tp, *other_side)

    def run_primary_resolver():
        r"""Drive the DFS-based resolver from each third-party tensor snapshot."""
        tp = get_third_party_tensor()
        if not tp:
            return

        # Iterate over a stable snapshot because local tp updates can remove tensors on the fly.
        for C in tuple(tp):
            if C not in tp:
                continue
            _, pivot_ax = get_pivot_edge(tp[C])
            resolve_tensor_dfs(tp, C, pivot_ax)

    # Phase 1: prefer the direct DFS/jump strategy that tries to eliminate all bad swaps.
    run_primary_resolver()

    # Phase 2: if any bad swaps remain on contracted legs, fall back to defer-on-partial logic.
    if get_third_party_tensor():
        print("Warning: ncon fallback resolver activated. This may indicate a suboptimal contraction path or a pathological case for the main resolver.")
        run_fallback_resolver()


    # Convert back to list format for the caller (_shift_swaps_ mutates).
    remaining = [[[list(tl) for tl in edge_endpoints[key[0]]],
                  [list(tl) for tl in edge_endpoints[key[1]]]]
                 for key in sorted(z2)]
    if deferred_indices:
        return commands, remaining, frozenset(deferred_indices)
    return commands, remaining, False


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
