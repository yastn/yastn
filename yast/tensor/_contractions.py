""" Contractions of yast tensors """

import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _common_rows, _common_keys, _tarray, _Darray
from ._auxliary import YastError, _check, _test_configs_match, _test_fusions_match
from ._merging import _merge_to_matrix, _unmerge_matrix

__all__ = ['tensordot', 'vdot', 'trace', 'swap_gate', 'ncon']


def tensordot(a, b, axes, conj=(0, 0)):
    r"""
    Compute tensor dot product of two tensor along specified axes.

    Outgoing legs are ordered such that first ones are the remaining legs of the first tensor in the original order,
    and than those of the second tensor.

    Parameters
    ----------
    a, b: Tensors to contract

    axes: tuple
        legs of both tensors (for each it is specified by int or tuple of ints)
        e.g. axes=(0, 3) to contract 0th leg of a with 3rd leg of b
                axes=((0, 3), (1, 2)) to contract legs 0 and 3 of a with 1 and 2 of b, respectivly.

    conj: tuple
        shows which tensor to conjugate: (0, 0), (0, 1), (1, 0), (1, 1).
        Defult is (0, 0), i.e. no tensor is conjugated

    Returns
    -------
    tansor: Tensor
    """
    _test_configs_match(a, b)
    la_con, lb_con = _clear_axes(*axes)  # contracted meta legs
    la_out = tuple(ii for ii in range(a.mlegs) if ii not in la_con)  # outgoing meta legs
    lb_out = tuple(ii for ii in range(b.mlegs) if ii not in lb_con)  # outgoing meta legs

    axes_a = _unpack_axes(a, la_out, la_con)  # actual legs of a; tuple of two tuples
    axes_b = _unpack_axes(b, lb_con, lb_out)  # actual legs of b; tuple of two tuples

    naxes_a = tuple(np.array(x, dtype=np.intp) for x in axes_a)
    naxes_b = tuple(np.array(x, dtype=np.intp) for x in axes_b)

    conja, conjb = (1 - 2 * conj[0]), (1 - 2 * conj[1])
    mconj = (-conja * conjb)

    if _check["signatures_match"] and not all(a.struct.s[i1] == mconj * b.struct.s[i2] for i1, i2 in zip(axes_a[1], axes_b[0])):
        # if a.isdiag:  # if tensor is diagonal, than freely changes the signature by a factor of -1
        #     a.flip_signature(inplace=True)
        # elif b.isdiag:
        #     b.flip_signature(inplace=True)
        # elif _check["signatures_match"]:
        raise YastError('Signs do not match')

    c_n = np.array(a.struct.n + b.struct.n, dtype=int).reshape(1, 2, a.config.sym.NSYM)
    c_s = np.array([conja, conjb], dtype=int)
    c_n = a.config.sym.fuse(c_n, c_s, 1)[0]

    ind_a, ind_b = _common_rows(_tarray(a)[:, naxes_a[1], :], _tarray(b)[:, naxes_b[0], :])
    s_eff_a, s_eff_b = (conja, -conja), (conjb, -conjb)

    Am, ls_l, ls_ac, ua_l, ua_r = _merge_to_matrix(a, axes_a, s_eff_a, ind_a, sort_r=True)
    Bm, ls_bc, ls_r, ub_l, ub_r = _merge_to_matrix(b, axes_b, s_eff_b, ind_b)

    meta_dot = tuple((al + br, al + ar, bl + br) for al, ar, bl, br in zip(ua_l, ua_r, ub_l, ub_r))

    if _check["consistency"] and not (ua_r == ub_l and ls_ac.match(ls_bc)):
        raise YastError('Something went wrong in matching the indices of the two tensors')

    c_s = tuple(conja * a.struct.s[i1] for i1 in axes_a[0]) + tuple(conjb * b.struct.s[i2] for i2 in axes_b[1])
    c_meta_fusion = [a.meta_fusion[ii] for ii in la_out] + [b.meta_fusion[ii] for ii in lb_out]
    c = a.__class__(config=a.config, s=c_s, n=c_n, meta_fusion=c_meta_fusion)

    c.A = c.config.backend.dot(Am, Bm, conj, meta_dot)
    _unmerge_matrix(c, ls_l, ls_r)
    return c


def vdot(a, b, conj=(1, 0)):
    r"""
    Compute scalar product x = <a|b> of two tensors. a is conjugated by default.

    Parameters
    ----------
    a, b: Tensor

    conj: tuple
        shows which tensor to conjugate: (0, 0), (0, 1), (1, 0), (1, 1).
        Defult is (1, 0), i.e. tensor a is conjugated.

    Returns
    -------
    x: number
    """
    _test_configs_match(a, b)
    _test_fusions_match(a, b)
    conja, conjb = (1 - 2 * conj[0]), (1 - 2 * conj[1])

    if not ((conja * conjb == -1 and a.struct.s == b.struct.s)
            or (conja * conjb == 1 and all(s1 == -s2 for s1, s2 in zip(a.struct.s, b.struct.s)))):
        raise YastError('Signs do not match')

    c_n = np.array(a.struct.n + b.struct.n, dtype=int).reshape(1, 2, a.config.sym.NSYM)
    c_s = np.array([conja, conjb], dtype=int)
    c_n = a.config.sym.fuse(c_n, c_s, 1)

    k12, _, _ = _common_keys(a.A, b.A)
    if (len(k12) > 0) and np.all(c_n == 0):
        return a.config.backend.vdot(a.A, b.A, conj, k12)
    return a.zero_of_dtype()


def trace(a, axes=(0, 1)):
    """
    Compute trace of legs specified by axes.

    Parameters
    ----------
        axes: tuple
        Legs to be traced out, e.g axes=(0, 1); or axes=((2, 3, 4), (0, 1, 5))

    Returns
    -------
        tensor: Tensor
    """
    lin1, lin2 = _clear_axes(*axes)  # contracted legs
    lin12 = lin1 + lin2
    lout = tuple(ii for ii in range(a.mlegs) if ii not in lin12)
    in1, in2, out = _unpack_axes(a, lin1, lin2, lout)

    if len(in1) != len(in2) or len(lin1) != len(lin2):
        raise YastError('Number of axis to trace should be the same')
    if len(in1) == 0:
        return a

    order = in1 + in2 + out
    ain1 = np.array(in1, dtype=np.intp)
    ain2 = np.array(in2, dtype=np.intp)
    aout = np.array(out, dtype=np.intp)

    if not all(a.struct.s[i1] == -a.struct.s[i2] for i1, i2 in zip(in1, in2)):
        raise YastError('Signs do not match')

    c_s = tuple(a.struct.s[i3] for i3 in out)
    c_meta_fusion=tuple(a.meta_fusion[ii] for ii in lout)
    c = a.__class__(config=a.config, s=c_s, n=a.struct.n, meta_fusion=c_meta_fusion)

    tset, Dset = _tarray(a), _Darray(a)
    lt = len(tset)
    t1 = tset[:, ain1, :].reshape(lt, -1)
    t2 = tset[:, ain2, :].reshape(lt, -1)
    to = tset[:, aout, :].reshape(lt, -1)
    D1 = Dset[:, ain1]
    D2 = Dset[:, ain2]
    D3 = Dset[:, aout]
    pD1 = np.prod(D1, axis=1).reshape(lt, 1)
    pD2 = np.prod(D2, axis=1).reshape(lt, 1)
    ind = (np.all(t1 == t2, axis=1)).nonzero()[0]
    Drsh = np.hstack([pD1, pD2, D3])
    if not np.all(D1[ind] == D2[ind]):
        raise YastError('Not all bond dimensions of the traced legs match')
    meta = [(tuple(to[n]), tuple(tset[n].flat), tuple(Drsh[n])) for n in ind]
    c.A = c.config.backend.trace(a.A, order, meta)
    c.update_struct()
    return c


def swap_gate(a, axes, inplace=False):
    """
    Return tensor after application of the swap gate.

    Multiply the block with odd charges on swaped legs by -1.
    If one of the provided axes is -1, then swap with the charge n.

    Parameters
    ----------
    axes: tuple
        two groups of legs to be swaped

    Returns
    -------
    tensor : Tensor
    """
    try:
        fss = a.config.sym.FERMIONIC  # fermionic symmetry sectors
    except AttributeError:
        return a
    if any(fss):
        c = a if inplace else a.clone()
        tset = _tarray(c)
        l1, l2 = _clear_axes(*axes)  # swaped groups of legs
        if len(set(l1) & set(l2)) > 0:
            raise YastError('Cannot sweep the same index')
        if l2 == (-1,):
            l1, l2 = l2, l1
        if l1 == (-1,):
            l2, = _unpack_axes(a, l2)
            t1 = c.n
        else:
            l1, l2 = _unpack_axes(a, l1, l2)
            al1 = np.array(l1, dtype=np.intp)
            t1 = np.sum(tset[:, al1, :], axis=1)
        al2 = np.array(l2, dtype=np.intp)
        t2 = np.sum(tset[:, al2, :], axis=1)
        tp = np.sum(t1 * t2, axis=1) % 2 == 1
        for ind, odd in zip(a.struct.t, tp):
            if odd:
                c.A[ind] = -c.A[ind]
        return c
    return a


def ncon(ts, inds, conjs=None):
    """Execute series of tensor contractions"""
    if len(ts) != len(inds):
        raise YastError('Wrong number of tensors')
    for ii, ind in enumerate(inds):
        if ts[ii].get_ndim() != len(ind):
            raise YastError('Wrong number of legs in %02d-th tensors.' % ii)

    ts = dict(enumerate(ts))
    cutoff = 512
    cutoff2 = 2 * cutoff
    edges = [(order, leg, ten) if order >= 0 else (-order + cutoff2, leg, ten)
             for ten, el in enumerate(inds) for leg, order in enumerate(el)]

    edges.append((cutoff, cutoff, cutoff))
    conjs = [0] * len(inds) if conjs is None else list(conjs)
    edges = sorted(edges, reverse=True, key=lambda x: x[0])  # order of contraction with info on tensor and axis

    order1, leg1, ten1 = edges.pop()
    ax1, ax2 = [], []
    while order1 != cutoff:  # tensordot two tensors; or trace one tensor
        order2, leg2, ten2 = edges.pop()
        if order1 != order2:
            raise YastError('Contracted legs do not match')
        if ten1 < ten2:
            (t1, t2) = (ten1, ten2)
            ax1.append(leg1)
            ax2.append(leg2)
        else:
            (t1, t2) = (ten2, ten1)
            ax1.append(leg2)
            ax2.append(leg1)
        if edges[-1][0] == cutoff or min(edges[-1][2], edges[-2][2]) != t1 or max(edges[-1][2], edges[-2][2]) != t2:
            # execute contraction
            if t1 == t2:  # trace
                ts[t1] = ts[t1].trace(axes=(ax1, ax2))
                ax12 = ax1 + ax2
                for ii, (order, leg, ten) in enumerate(edges):
                    if ten == t1:
                        edges[ii] = (order, leg - sum(ii < leg for ii in ax12), ten)
            else:  # tensordot
                ts[t1] = ts[t1].tensordot(ts[t2], axes=(ax1, ax2), conj=(conjs[t1], conjs[t2]))
                conjs[t1], conjs[t2] = 0, 0
                del ts[t2]
                lt1 = sum(ii[2] == t1 for ii in edges)  # legs of t1
                for ii, (order, leg, ten) in enumerate(edges):
                    if ten == t1:
                        edges[ii] = (order, leg - sum(ii < leg for ii in ax1), ten)
                    elif ten == t2:
                        edges[ii] = (order, lt1 + leg - sum(ii < leg for ii in ax2), t1)
            ax1, ax2 = [], []
        order1, leg1, ten1 = edges.pop()

    if edges:
        while len(ts) > 1:
            edges = sorted(edges, key=lambda x: x[2])
            t1 = edges[0][2]
            t2 = [key for key in ts.keys() if key != t1][0]
            ts[t1] = ts[t1].tensordot(ts[t2], axes=((), ()), conj=(conjs[t1], conjs[t2]))
            conjs[t1], conjs[t2] = 0, 0
            lt1 = sum(ii[2] == t1 for ii in edges)
            for ii, (order, leg, ten) in enumerate(edges):
                if ten == t2:
                    edges[ii] = (order, leg + lt1, t1)
            del ts[t2]
        order = [ed[1] for ed in sorted(edges)]
        _, result = ts.popitem()
        return result.transpose(axes=order, inplace=True)
    it = iter(ts.values())
    result = next(it)
    for num in it:
        result.A[()] = result.A[()] * num.A[()]
    return result
