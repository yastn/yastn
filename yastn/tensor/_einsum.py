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

from functools import lru_cache, partial
from itertools import product
from typing import TYPE_CHECKING

from ._auxiliary import _clear_axes, _flatten, _unpack_axes
from ._contractions import apply_mask, tensordot, trace, swap_gate
from ._tests import YastnError

__all__ = ['ncon', 'einsum']

if TYPE_CHECKING:
    from . import Tensor

def einsum(subscripts, *operands, order=None, swap=None) -> 'Tensor':
    r"""
    Execute a series of tensor contractions.

    This covers trace, tensordot (including outer products), and transpose operations.
    It follows the notation of :meth:`np.einsum` as closely as possible.

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


def ncon(ts, inds, conjs=None, order=None, swap=None, release_cuda_cache=False,
         oom_retry=False) -> 'Tensor':
    r"""
    Execute a series of tensor contractions.

    Parameters
    ----------
    ts: Sequence[yastn.Tensor]
        list of tensors to be contracted.

    oom_retry: bool
        If ``True`` (torch/CUDA backends), each contraction op is retried once
        after :func:`torch.cuda.empty_cache` when it raises a CUDA out-of-memory
        error. This reclaims reserved-but-unallocated (fragmented) cache back to
        the driver so a large contiguous allocation can succeed. The blocking
        ``empty_cache`` only runs on the rare OOM path; the common path is
        untouched. Default: ``False``.

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
    ts = _execute_commands(ts, commands, release_cuda_cache=release_cuda_cache,
                           oom_retry=oom_retry)
    assert len(ts) == 1, "Sanity check. Contact developers."
    return ts.popitem()[1]


def _fermionic_components(config):
    """Indices of the charge components whose parities enter swap gates."""
    if config.fermionic is True:
        return tuple(range(config.sym.NSYM))
    return tuple(i for i, f in enumerate(config.fermionic) if f)


def _restrict_parity(a, axis, pv):
    """Return a with leg `axis` restricted to sectors whose fermionic components have parities pv, or None if empty."""
    from ..initialize import eye  # deferred: yastn.initialize imports the tensor package
    leg = a.get_legs(axis)
    fc = _fermionic_components(a.config)
    if not any(all(t[i] % 2 == p for i, p in zip(fc, pv)) for t in leg.t):
        return None
    m = eye(config=a.config, legs=leg)
    for i, p in zip(fc, pv):
        unit = tuple(int(j == i) for j in range(a.config.sym.NSYM))  # charge 1 on component i
        string = m.swap_gate(axes=(0,), charge=unit)
        m = (m + string) / 2 if p == 0 else (m - string) / 2
    return apply_mask(m, a, axes=axis)


def _add_parity_pair(r, pv):
    """Append a charge-neutral pair of one-dimensional legs (aux, aux') carrying the fermionic parities pv."""
    t = [0] * r.config.sym.NSYM
    for i, p in zip(_fermionic_components(r.config), pv):
        t[i] = p
    return r.add_leg(axis=-1, s=1, t=tuple(t)).add_leg(axis=-1, s=-1, t=tuple(t))


def _contract_psplit(contract, a, paxes):
    """contract(a) with the fermionic parities of a's legs `paxes` recorded in trailing (aux, aux') pairs."""
    nf = len(_fermionic_components(a.config))
    out = None
    for pvs in product(product((0, 1), repeat=nf), repeat=len(paxes)):
        aa = a
        for ax, pv in zip(paxes, pvs):
            aa = _restrict_parity(aa, ax, pv)
            if aa is None:
                break
        if aa is None:
            continue
        r = contract(aa)
        for pv in pvs:
            r = _add_parity_pair(r, pv)
        out = r if out is None else out + r
    if out is None:  # every sector empty: fall back to plain contraction, pad zero pairs
        r = contract(a)
        for _ in paxes:
            r = _add_parity_pair(r, (0,) * nf)
        out = r
    return out


# TODO? Import from backend
def _run_op_oom_retry(fn, oom_retry, retries=1):
    r"""Run ``fn()``; on a CUDA out-of-memory error, release torch's cached-but-
    unused (fragmented) memory back to the driver and retry, up to ``retries``
    extra times.

    Operands must be captured by ``fn`` as live locals so they survive the
    ``empty_cache`` (which only frees fully-unused segments) and are available
    for the retry. Non-OOM errors — and OOM on the final attempt — propagate.
    ``oom_retry=False`` is a transparent pass-through (no torch import), keeping
    non-CUDA backends unaffected.

    ``empty_cache`` runs *after* the ``except`` block returns, not inside it: on
    leaving the handler the interpreter drops its active-exception state, so the
    traceback that pinned the failed attempt's frames (and their intermediate
    CUDA tensors) is released and those blocks are freed by refcounting. No
    ``gc.collect()`` is needed — nothing here is held in a reference cycle — so
    we avoid a full-heap sweep on the (rare) OOM path.
    """
    if not oom_retry:
        return fn()
    import torch
    for attempt in range(retries + 1):
        try:
            return fn()
        except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError
            if attempt == retries or 'out of memory' not in str(e).lower():
                raise
        # Outside the handler: exception (+ its traceback) is gone, failed
        # intermediates freed. Reclaim them and the fragmented free pool.
        torch.cuda.empty_cache()


def _execute_commands(ts, commands, release_cuda_cache=False, oom_retry=False):
    for command in commands:
        if command[0] in ('tensordot', 'tensordot_psplit'):
            tout, (t1, t2), axes, *paxes = command[1:]  # paxes: [legs whose parity is recorded] for _psplit
            a, b = ts.pop(t1), ts.pop(t2)
            op = partial(tensordot, b=b, axes=axes)
            ts[tout] = _run_op_oom_retry(lambda: _contract_psplit(op, a, paxes[0]) if paxes else op(a), oom_retry)
            if release_cuda_cache:
                import torch
                torch.cuda.empty_cache()
        elif command[0] in ('trace', 'trace_psplit'):
            tout, tin, axes, *paxes = command[1:]
            a = ts.pop(tin)
            op = partial(trace, axes=axes)
            ts[tout] = _run_op_oom_retry(lambda: _contract_psplit(op, a, paxes[0]) if paxes else op(a), oom_retry)
        elif command[0] == 'swap_gate':
            tout, tin, axes = command[1:]
            a = ts.pop(tin)
            ts[tout] = _run_op_oom_retry(lambda: swap_gate(a, axes=axes), oom_retry)
        elif command[0] == 'parity_sign':
            # Correction for jump-move on potentially parity-odd tensor.
            # If the jumped tensor has odd parity, apply (-1)^{n_d}
            # as a fermionic string on the partner leg.
            jumped_ten, d_ten, d_legs = command[1:]
            charge = ts[jumped_ten].n
            if any(charge):
                a = ts[d_ten]
                ts[d_ten] = _run_op_oom_retry(
                    lambda: swap_gate(a, axes=d_legs, charge=charge), oom_retry)
        else:
            assert command[0] == 'transpose', "Sanity check"
            tout, tin, axes = command[1:]
            a = ts.pop(tin)
            ts[tout] = _run_op_oom_retry(lambda: a.transpose(axes=axes), oom_retry)
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
    edges between the same tensor pair into one tensordot (``ten1 != ten2``)
    or trace (``ten1 == ten2``).  Before it, same-tensor swaps are applied and
    bad swaps (touching contracted legs) are resolved exactly by
    ``_resolve_bad_swaps``, with jump moves or parity gadgets.  ``aux_pairs``
    tracks the live gadget pairs ``[ten, leg_aux, leg_aux']``, traced as soon
    as no swap touches them (``trace_free_aux``).  The post-loop takes outer
    products of disconnected tensors, applies the remaining swaps and
    transposes the output legs.  See ``docs/source/tensor/_einsum.rst``.
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
    aux_pairs = []  # live gadget pairs [ten, leg_aux, leg_aux'] still present

    def collect_same_tensor():
        nonlocal swaps
        swap_now, swap_later = [], []
        for sw12 in swaps:
            sw_now = _swap_on_tensor(*sw12)
            swap_now.append(sw_now) if sw_now else swap_later.append(sw12)
        swap_tensors = {}
        for ten_swap, axes_swap in swap_now:
            swap_tensors.setdefault(ten_swap, []).extend(axes_swap)
        for ten_swap, axes_swap in swap_tensors.items():
            commands.append(('swap_gate', ten_swap, ten_swap, tuple(axes_swap)))
        swaps = swap_later

    def shift(ten_old, ten_new, dax):
        """Move the legs of ten_old onto ten_new, leg l -> l + dax(l), in edges, swaps and gadget pairs."""
        _shift_edges_(edges, ten_old, ten_new, dax)
        _shift_swaps_(swaps, ten_old, ten_new, dax)
        _shift_aux_(aux_pairs, ten_old, ten_new, dax)

    def trace_free_aux():
        """Trace every gadget pair no swap touches any more."""
        nonlocal aux_pairs
        touched = {(sw[0], sw[1]) for sw12 in swaps for sws in sw12 for sw in sws}
        by_ten = {}
        for ap in aux_pairs:
            if (ap[0], ap[1]) not in touched and (ap[0], ap[2]) not in touched:
                by_ten.setdefault(ap[0], []).append(ap)
        for ten, aps in by_ten.items():
            ax1 = tuple(ap[1] for ap in aps)
            ax2 = tuple(ap[2] for ap in aps)
            commands.append(('trace', ten, ten, (ax1, ax2)))
            axes12 = ax1 + ax2
            nlegs[ten] -= len(axes12)
            aux_pairs = [ap for ap in aux_pairs if ap not in aps]
            shift(ten, ten, lambda x, axes12=axes12: -sum(ax < x for ax in axes12))

    def attach_gadgets(ten_res, base, nsplit):
        """Replace marker sides [[ten_res, -1-2j], [ten_res, -2-2j]] by real aux legs."""
        for j in range(nsplit):
            la, lb = base + 2 * j, base + 2 * j + 1
            for sw12 in swaps:
                for sws in sw12:
                    for sw in sws:
                        if sw[0] == ten_res and sw[1] == -1 - 2 * j:
                            sw[1] = la
                        elif sw[0] == ten_res and sw[1] == -2 - 2 * j:
                            sw[1] = lb
            aux_pairs.append([ten_res, la, lb])
        nlegs[ten_res] += 2 * nsplit
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
            collect_same_tensor()
            tas = [[ten1, ax] for ax in axes1] + [[ten2, ax] for ax in axes2]
            psplit = []
            if any(any(ta in sw12[0] or ta in sw12[1] for ta in tas) for sw12 in swaps):
                new_cmds, swaps, psplit = _resolve_bad_swaps(
                    swaps, edges, nlegs, aux_pairs, ten1, ten2, axes1, axes2)
                commands.extend(new_cmds)
                collect_same_tensor()
            # a step needing gadgets is '*_psplit' and lists the legs of ten1 whose parity is recorded
            kind = '_psplit' if psplit else ''
            paxes = (tuple(axes1[k] for k in psplit),) if psplit else ()
            if ten1 == ten2:  # trace
                commands.append(('trace' + kind, ten1, ten1, (tuple(axes1), tuple(axes2))) + paxes)
                axes12 = axes1 + axes2
                nlegs[ten1] -= len(axes12)
                shift(ten1, ten1, lambda x, axes12=axes12: -sum(ax < x for ax in axes12))
                attach_gadgets(ten1, nlegs[ten1], len(psplit))
            else:  # tensordot
                ten_out += 1
                commands.append(('tensordot' + kind, ten_out, (ten1, ten2), (tuple(axes1), tuple(axes2))) + paxes)
                nlegs[ten1] -= len(axes1)
                nlegs[ten2] -= len(axes2)
                n1 = nlegs[ten1]
                shift(ten1, ten_out, lambda x, a=tuple(axes1): -sum(ax < x for ax in a))
                shift(ten2, ten_out, lambda x, a=tuple(axes2), n1=n1: n1 - sum(ax < x for ax in a))
                nlegs[ten_out] = nlegs.pop(ten1) + nlegs.pop(ten2)
                attach_gadgets(ten_out, nlegs[ten_out], len(psplit))
            trace_free_aux()
            axes1, axes2 = [], []
    #
    edges.pop()  # eliminate cutoff element
    #
    remaining = list(nlegs.keys())
    ten1 = remaining[0]
    for ten2 in remaining[1:]:  # tensordot
        ten_out += 1
        commands.append(('tensordot', ten_out, (ten1, ten2), ((), ())))
        shift(ten1, ten_out, lambda x: 0)
        shift(ten2, ten_out, lambda x: nlegs[ten1])
        nlegs[ten_out] = nlegs.pop(ten1) + nlegs.pop(ten2)
        ten1 = ten_out
    #
    if len(edges) != len(set(ind for ind, _, _ in edges)):
        raise YastnError("Repeated non-positive (outgoing) index is ambiguous.")
    #
    collect_same_tensor()
    assert not swaps, "all swaps must be same-tensor on the final tensor"
    trace_free_aux()
    assert not aux_pairs
    #
    # final order for transpose
    axes = tuple(leg for _, _, leg in sorted(edges))
    if axes != tuple(range(len(axes))):
        commands.append(('transpose', ten_out, ten_out, axes))
    #
    return tuple(commands)


def ncon_prefilter(ts_meta, inds, nsym):
    r"""
    Predict which blocks of each input tensor contribute to the ncon result.

    Uses iterative pairwise edge-based filtering: for each pair of tensors
    sharing contracted indices, find which blocks have matching charges on
    those axes.  Intersect surviving block sets per tensor across all edges,
    then repeat until convergence (trimming one tensor can cascade).

    **Skip**: returns ``None`` when no blocks survive (contraction is zero).

    **Trim**: returns a dict mapping each tensor id to a ``frozenset`` of
    needed block indices (``None`` = all blocks needed).

    Parameters
    ----------
    ts_meta : dict[int, tuple]
        ``{tensor_pos: (struct_t, ndim_n, trans, mfs)}`` for each input tensor.
        Keys must be the positional tensor indices ``0, 1, ..., len(inds) - 1``
        matching the order of ``inds``.

        leg_first: ``struct_t`` is the **nested** block-charge sequence in
        native order, ``struct_t[block_idx][native_leg]`` -> charge tuple
        (length ``nsym``); ``len(struct_t)`` is the number of blocks.  Build it
        from ``get_blocks(sym, struct).t`` (e.g.
        ``tuple(tuple(map(tuple, blk)) for blk in bl.t.tolist())``).
        ``ndim_n`` is the number of native dimensions, ``trans`` is the
        user-to-native axis permutation (``None`` = identity), and ``mfs``
        records meta-fused user-axis structure (may be omitted for unfused
        inputs).
    inds : tuple[tuple[int, ...], ...]
        ncon index notation.  Positive labels = contracted (matching pairs),
        non-positive labels = output legs.
    nsym : int
        Number of symmetry charges (``config.sym.NSYM``).

    Returns
    -------
    None
        If the contraction produces an empty (zero) tensor.
    dict[int, frozenset | None]
        Mapping ``{tensor_id: needed_block_indices}``.
        ``None`` means all blocks are needed (no trimming).
    """
    if nsym == 0:
        return {tid: None for tid in ts_meta}

    if len(ts_meta) != len(inds):
        raise YastnError("ts_meta and inds must describe the same number of tensors.")
    expected_tids = tuple(range(len(inds)))
    if set(ts_meta.keys()) != set(expected_tids):
        raise YastnError("ts_meta keys must be positional tensor indices 0..len(inds)-1.")
    tids = expected_tids

    # --- Extract pairwise contracted edges and traces from index notation ---
    # Group contracted axes by tensor pair for stronger filtering.
    pair_axes = {}   # (tid_a, tid_b) -> [(uax_a, uax_b), ...]
    trace_axes = {}  # tid -> [(uax1, uax2), ...]

    label_locs = {}  # positive_label -> (tensor_id, user_axis)
    for i, tid in enumerate(tids):
        for ax, label in enumerate(inds[i]):
            if label > 0:
                if label in label_locs:
                    prev_tid, prev_ax = label_locs.pop(label)
                    if prev_tid == tid:
                        trace_axes.setdefault(tid, []).append((prev_ax, ax))
                    else:
                        key = (prev_tid, tid) if prev_tid < tid else (tid, prev_tid)
                        axes = (prev_ax, ax) if prev_tid < tid else (ax, prev_ax)
                        pair_axes.setdefault(key, []).append(axes)
                else:
                    label_locs[label] = (tid, ax)

    if not pair_axes and not trace_axes:
        return {tid: None for tid in ts_meta}

    # --- Helpers ---
    def native_axes(tid, uax):
        meta = ts_meta[tid]
        if len(meta) == 4:
            _, ndim_n, trans, mfs = meta
        else:
            _, ndim_n, trans = meta
            n_user = len(trans) if trans else ndim_n
            mfs = tuple((1,) for _ in range(n_user))
        native, = _unpack_axes(mfs, (uax,))
        return tuple(trans[ax] for ax in native) if trans else native

    def block_charge_key(struct_t, block_idx, native_axes):
        # leg_first: struct_t[block_idx][na] is already a per-leg charge tuple.
        t = struct_t[block_idx]
        return tuple(
            tuple(t[na] for na in axes)
            for axes in native_axes
        )

    # --- Surviving block indices per tensor ---
    surviving = {tid: set(range(len(meta[0]))) for tid, meta in ts_meta.items()}

    converged = False
    while not converged:
        converged = True

        # Pairwise contracted edges (grouped by tensor pair)
        for (tid_a, tid_b), ax_pairs in pair_axes.items():
            st_a = ts_meta[tid_a][0]
            st_b = ts_meta[tid_b][0]
            naxes_a = tuple(native_axes(tid_a, ua) for ua, _ in ax_pairs)
            naxes_b = tuple(native_axes(tid_b, ub) for _, ub in ax_pairs)

            grp_a = {}
            for i in surviving[tid_a]:
                grp_a.setdefault(block_charge_key(st_a, i, naxes_a), []).append(i)
            grp_b = {}
            for i in surviving[tid_b]:
                grp_b.setdefault(block_charge_key(st_b, i, naxes_b), []).append(i)

            common = set(grp_a) & set(grp_b)

            new_a = set()
            new_b = set()
            for c in common:
                new_a.update(grp_a[c])
                new_b.update(grp_b[c])

            if not new_a or not new_b:
                return None

            if len(new_a) < len(surviving[tid_a]):
                surviving[tid_a] = new_a
                converged = False
            if len(new_b) < len(surviving[tid_b]):
                surviving[tid_b] = new_b
                converged = False

        # Traces (same positive label appears twice on one tensor)
        for tid, tax_pairs in trace_axes.items():
            st = ts_meta[tid][0]
            nat_pairs = [(native_axes(tid, u1), native_axes(tid, u2)) for u1, u2 in tax_pairs]
            new_surv = set()
            for i in surviving[tid]:
                t = st[i]
                if all(
                    len(axes1) == len(axes2) and all(
                        t[na1] == t[na2]
                        for na1, na2 in zip(axes1, axes2)
                    )
                    for axes1, axes2 in nat_pairs
                ):
                    new_surv.add(i)

            if not new_surv:
                return None
            if len(new_surv) < len(surviving[tid]):
                surviving[tid] = new_surv
                converged = False

    # --- Build result ---
    result = {}
    for tid in ts_meta:
        n_total = len(ts_meta[tid][0])
        result[tid] = None if len(surviving[tid]) == n_total else frozenset(surviving[tid])
    return result


def _resolve_bad_swaps(swaps, edges, nlegs, aux_pairs, ten1, ten2, axes1, axes2):
    r"""
    Resolve swap gates that sit on legs about to be contracted, exactly.

    Called before contracting ``ten1`` and ``ten2`` along ``axes1``/``axes2``
    (``ten1 == ten2`` for a trace).  A swap is "bad" if it touches any of the
    contracted legs.  ``edges`` is the mutable ``[ind, tensor_id, leg]`` list
    of the planner and ``nlegs`` maps each live tensor to its number of legs;
    both describe the network before the contraction.  ``aux_pairs`` lists
    live parity-gadget pairs ``[ten, leg_aux, leg_aux']``; each pair is
    registered as a self-loop edge so swaps touching it are tracked.

    Rows of bad swaps are grouped into classes by the cut test
    (``coboundary``); the largest class (for a trace, the class of the empty
    row) is emptied by row and column jumps, and every other row gets a parity
    gadget.  The algebra is described in ``docs/source/tensor/_einsum.rst``.

    Returns
    -------
    commands : list
        ``('swap_gate', ...)`` and ``('parity_sign', ...)`` commands to run
        before the contraction.
    remaining_swaps : list
        Swaps still in the Z2 set after resolution, in the format expected by
        ``_shift_swaps_``.  For a gadget row ``j`` the side that was the
        contracted leg is the marker ``[[ten1, -1-2j], [ten1, -2-2j]]``,
        which the planner's ``attach_gadgets`` turns into the real ``(aux,
        aux')`` legs of the result.
    psplit : list
        Sorted positions ``k`` (into ``axes1``) whose contraction needs a
        parity gadget.
    """
    K = len(axes1)
    is_trace = ten1 == ten2
    edge_endpoints, leg_to_edge, edge_to_id = {}, {}, {}

    def register_edge(endpoints):
        edge = tuple(sorted(endpoints))
        eid = edge_to_id.get(edge)
        if eid is None:
            eid = len(edge_to_id)
            edge_to_id[edge] = eid
            edge_endpoints[eid] = edge
        for tl in edge:
            leg_to_edge[tl] = eid
        return eid

    contracted = [register_edge(((ten1, a1), (ten2, a2))) for a1, a2 in zip(axes1, axes2)]
    by_ind = {}
    for ind, t, l in edges:
        if ind != 512:
            by_ind.setdefault(ind, []).append((t, l))
    for eps in by_ind.values():
        register_edge(eps)
    for ten, la, lb in aux_pairs:
        register_edge(((ten, la), (ten, lb)))

    def canon(a, b):
        return (a, b) if a <= b else (b, a)

    def side_to_edge(side):
        eids = {leg_to_edge[tuple(tl)] for tl in side}
        if len(eids) != 1:
            raise YastnError("Inconsistent edge encoding in swap.")
        return eids.pop()

    z2 = set()
    for sw12 in swaps:
        z2.symmetric_difference_update({canon(side_to_edge(sw12[0]), side_to_edge(sw12[1]))})

    def third_party(eid):
        return all(t not in (ten1, ten2) for t, _ in edge_endpoints[eid])

    commands = []

    def same_tensor_cleanup():
        by_tensor = {}
        for key in list(z2):
            ea, eb = key
            tens_a = {t for t, _ in edge_endpoints[ea]}
            tens_b = {t for t, _ in edge_endpoints[eb]}
            common = tens_a & tens_b
            if common:
                t = min(common)
                la = next(l for tt, l in edge_endpoints[ea] if tt == t)
                lb = next(l for tt, l in edge_endpoints[eb] if tt == t)
                z2.discard(key)
                by_tensor.setdefault(t, []).extend(sorted((la, lb)))
        for t, ax in by_tensor.items():
            commands.append(('swap_gate', t, t, tuple(ax)))

    def flip(tid, partner):
        """Toggle (l, partner) for every distinct edge l of tid; sign on partner's first endpoint."""
        seen = set()
        for l in range(nlegs[tid]):
            eid = leg_to_edge[(tid, l)]
            if eid in seen:      # self-loop / gadget pair: toggled twice -> no-op
                seen.discard(eid)
                continue
            seen.add(eid)
        for eid in seen:
            z2.symmetric_difference_update({canon(eid, partner)})
        d_ten, d_leg = edge_endpoints[partner][0]
        commands.append(('parity_sign', tid, d_ten, (d_leg,)))

    def rows():
        return [frozenset(L for key in z2 if contracted[k] in key
                          for L in key if L != contracted[k] and third_party(L)) for k in range(K)]

    def coboundary(D):
        """Set of tensors to flip realizing D as a cut of H, or None when D is not a cut."""
        adj = {}
        for eid, eps in edge_endpoints.items():
            if not third_party(eid):
                continue
            tens = [t for t, _ in eps]
            if len(tens) == 1:                       # open leg: private fixed vertex
                tens = [tens[0], ('inf', eid)]
            w = 1 if eid in D else 0
            a, b = tens
            adj.setdefault(a, []).append((b, w))
            adj.setdefault(b, []).append((a, w))
        color, flips = {}, set()
        for root in adj:
            if root in color:
                continue
            color[root] = 0
            comp, stack = [root], [root]
            while stack:
                u = stack.pop()
                for v, w in adj[u]:
                    c = color[u] ^ w
                    if v in color:
                        if color[v] != c:
                            return None
                    else:
                        color[v] = c
                        comp.append(v)
                        stack.append(v)
            fixed = [v for v in comp if isinstance(v, tuple) and v[0] == 'inf']
            ones = [v for v in comp if color[v] == 1]
            if fixed:
                fc = {color[v] for v in fixed}
                if len(fc) > 1:
                    return None
                if fc == {1}:
                    ones = [v for v in comp if color[v] == 0]
            elif len(ones) > len(comp) - len(ones):
                ones = [v for v in comp if color[v] == 0]
            flips.update(v for v in ones if not (isinstance(v, tuple) and v[0] == 'inf'))
        return flips

    Y = rows()
    # class assignment: reference row for each class; trace: reference is the empty row.
    classes = []  # list of [ref_row_set, [k...]]
    if is_trace:
        classes.append([frozenset(), []])
    for k in range(K):
        placed = False
        for cl in classes:
            if cl[0] is None:
                continue
            F = coboundary(Y[k] ^ cl[0])
            if F is not None:
                cl[1].append((k, F))
                placed = True
                break
        if not placed:
            if is_trace:
                classes.append([None, [(k, None)]])     # unresolvable trace row
            else:
                classes.append([Y[k], [(k, frozenset())]])
    if is_trace:
        keep = classes[0]
    else:
        keep = max(classes, key=lambda cl: len(cl[1]))
    psplit = sorted(k for cl in classes if cl is not keep for k, _ in cl[1])
    # row flips bring every kept row onto the reference row
    for k, F in keep[1]:
        for T in F:
            flip(T, contracted[k])
    same_tensor_cleanup()
    # column flips remove the common reference row
    if not is_trace and keep[1]:
        if nlegs[ten1] - K <= nlegs[ten2] - K:
            t = ten1
        else:
            t = ten2
        for L in sorted(keep[0]):
            flip(t, L)
        same_tensor_cleanup()
    Y = rows()
    for k in range(K):
        assert (k in psplit) or not Y[k], f"row {k} not resolved: {Y[k]}"
    # gadgets: (e_k, L) -> (aux_j, L) marker sides
    marker = {}
    for j, k in enumerate(psplit):
        e = contracted[k]
        marker[e] = [[ten1, -1 - 2 * j], [ten1, -2 - 2 * j]]
        for key in list(z2):
            if e in key:
                L = key[0] if key[1] == e else key[1]
                assert L != e and third_party(L)
    remaining = []
    for key in sorted(z2):
        sides = []
        for e in key:
            sides.append(marker[e] if e in marker else [list(tl) for tl in edge_endpoints[e]])
        remaining.append(sides)
    return commands, remaining, psplit


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


def _shift_aux_(aux_pairs, ten_old, ten_new, dax):
    for ap in aux_pairs:
        if ap[0] == ten_old:
            ap[0] = ten_new
            ap[1] += dax(ap[1])
            ap[2] += dax(ap[2])


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
