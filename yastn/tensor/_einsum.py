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

    alphabet1 = 'ABCDEFGHIJKLMNOPQRSTUWXYZabcdefghijklmnopqrstuvwxyz'
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
    while edges[-1][0] != 512:  # tensordot two tensors, or trace one tensor; 512 is cutoff marking end of truncation
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

            # contract
            if ten1 == ten2: # trace
                if dot_done:
                    raise YastnError("Likely inefficient order of contractions. Do all traces before tensordot. " +
                                     "Call all axes connecting two tensors one after another.")
                commands.append(('trace', ten1, ten1, (tuple(axes1), tuple(axes2))))
                axes12 = axes1 + axes2
                nlegs[ten1] -= len(axes12)
                for edge in edges:
                    if edge[1] == ten1:
                        edge[2] += -sum(ax < edge[2] for ax in axes12)
                for sw12 in swaps:
                    for sws in sw12:
                        for sw in sws:
                            if sw[0] == ten1:
                                sw[1] += -sum(ax < sw[1] for ax in axes12)
            else:  # tensordot
                dot_done = True
                ten_out += 1
                commands.append(('tensordot', ten_out, (ten1, ten2), (tuple(axes1), tuple(axes2))))
                nlegs[ten1] -= len(axes1)
                nlegs[ten2] -= len(axes2)
                for edge in edges:
                    if edge[1] == ten1:
                        edge[1] = ten_out
                        edge[2] += -sum(ax < edge[2] for ax in axes1)
                    elif edge[1] == ten2:
                        edge[1] = ten_out
                        edge[2] += nlegs[ten1] - sum(ax < edge[2] for ax in axes2)
                for sw12 in swaps:
                    for sws in sw12:
                        for sw in sws:
                            if sw[0] == ten1:
                                sw[0] = ten_out
                                sw[1] += -sum(ax < sw[1] for ax in axes1)
                            elif sw[0] == ten2:
                                sw[0] = ten_out
                                sw[1] += nlegs[ten1] - sum(ax < sw[1] for ax in axes2)
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
        nlegs[ten1] -= len(axes1)
        nlegs[ten2] -= len(axes2)
        for edge in edges:
            if edge[1] == ten1:
                edge[1] = ten_out
            if edge[1] == ten2:
                edge[1] = ten_out
                edge[2] += nlegs[ten1]
        for sw12 in swaps:
            for sws in sw12:
                for sw in sws:
                    if sw[0] == ten1:
                        sw[0] = ten_out
                    if sw[0] == ten2:
                        sw[0] = ten_out
                        sw[1] += nlegs[ten1]
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
