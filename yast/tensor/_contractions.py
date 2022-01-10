""" Contractions of yast tensors """
from functools import lru_cache
from itertools import groupby, product
import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _common_rows, _struct, _flatten
from ._tests import YastError, _test_configs_match, _test_axes_match
from ._merging import _merge_to_matrix, _unmerge_matrix, _flip_hf
from ._merging import _masks_for_tensordot, _masks_for_vdot, _masks_for_trace, _masks_for_axes


__all__ = ['tensordot', 'vdot', 'trace', 'swap_gate', 'ncon', 'einsum', 'broadcast', 'mask']


def __matmul__(a, b):
    """
    Compute tensor dot product, contracting the last axis of a with the first axis of b.

    Shorthand for yast.tensordot(a, b, axes=(a.ndim - 1, 0)).

    Returns
    -------
    tansor: Tensor
    """
    return tensordot(a, b, axes=(a.ndim - 1, 0))


def tensordot(a, b, axes, conj=(0, 0), policy=None):
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
        Defult is (0, 0), i.e. neither tensor is conjugated

    policy: str
        method of executing contraction.
        `merge` is merging blocks into effective 2d matrices before executing matrix multiplication
        (typically peferable for many small blocks).
        `direct` is performing multiplication block by block
        (might be preferable for tensors with fewer legs, or contracting over single axis).
        `hybrid` (default) switches between those methods using simple heuristic.

    Returns
    -------
    tansor: Tensor
    """
    in_a, in_b = _clear_axes(*axes)  # contracted meta legs
    conja, conjb = (1 - 2 * conj[0]), (1 - 2 * conj[1])
    needs_mask, (nin_a, nin_b) = _test_axes_match(a, b, sgn=-conja * conjb, axes=(in_a, in_b))

    if b.isdiag:
        return _tensordot_diag(a, b, in_a, (-1,), conj)
    if a.isdiag:
        return _tensordot_diag(b, a, in_b, (0,), conj[::-1])

    _test_configs_match(a, b)
    nout_a = tuple(ii for ii in range(a.ndim_n) if ii not in nin_a)  # outgoing native legs
    nout_b = tuple(ii for ii in range(b.ndim_n) if ii not in nin_b)  # outgoing native legs

    c_s = tuple(conja * a.struct.s[i1] for i1 in nout_a) + tuple(conjb * b.struct.s[i2] for i2 in nout_b)
    c_n = np.array(a.struct.n + b.struct.n, dtype=int).reshape((1, 2, a.config.sym.NSYM))
    c_n = tuple(a.config.sym.fuse(c_n, (conja, conjb), 1)[0])
    c_mfs = [a.meta_fusion[ii] for ii in range(a.ndim) if ii not in in_a]
    c_mfs += [b.meta_fusion[ii] for ii in range(b.ndim) if ii not in in_b]
    c_hfs = [a.hard_fusion[ii] for ii in nout_a] if conj[0] == 0 else \
            [_flip_hf(a.hard_fusion[ii]) for ii in nout_a]
    c_hfs += [b.hard_fusion[ii] for ii in nout_b] if conj[1] == 0 else \
             [_flip_hf(b.hard_fusion[ii]) for ii in nout_b]

    if a.config.force_tensordot is not None:
        policy = a.config.force_tensordot
    elif policy is None:
        policy = a.config.default_tensordot

    if policy == 'merge' or (policy == 'hybrid' and len(nin_a) != 1 and a.config.sym.NSYM > 0):
        t_a = np.array(a.struct.t, dtype=int).reshape((len(a.struct.t), len(a.struct.s), len(a.struct.n)))
        t_b = np.array(b.struct.t, dtype=int).reshape((len(b.struct.t), len(b.struct.s), len(b.struct.n)))
        ind_a, ind_b = _common_rows(t_a[:, nin_a, :], t_b[:, nin_b, :])
        s_eff_a, s_eff_b = (conja, -conja), (conjb, -conjb)

        Am, ls_l, ls_ac, ua_l, ua_r = _merge_to_matrix(a, (nout_a, nin_a), s_eff_a, ind_a, sort_r=True)
        Bm, ls_bc, ls_r, ub_l, ub_r = _merge_to_matrix(b, (nin_b, nout_b), s_eff_b, ind_b)

        meta_dot = tuple((al + br, al + ar, bl + br) for al, ar, bl, br in zip(ua_l, ua_r, ub_l, ub_r))

        if needs_mask:
            msk_a, msk_b = _masks_for_tensordot(a.config, a.struct, a.hard_fusion, nin_a, ls_ac,
                                                        b.struct, b.hard_fusion, nin_b, ls_bc)
            Am = {ul + ur: Am[ul + ur][:, msk_a[ur]] for ul, ur in zip(ua_l, ua_r)}
            Bm = {ul + ur: Bm[ul + ur][msk_b[ul], :] for ul, ur in zip(ub_l, ub_r)}
        elif ua_r != ub_l or ls_ac != ls_bc:
            raise YastError('Bond dimensions do not match.')

        c = a.__class__(config=a.config, s=c_s, n=c_n, meta_fusion=c_mfs, hard_fusion=c_hfs)
        c.A = c.config.backend.dot(Am, Bm, conj, meta_dot)
        _unmerge_matrix(c, ls_l, ls_r)
    elif policy in ('hybrid', 'direct'):
        meta, c_t, c_D = _meta_tensordot_nomerge(a.struct, b.struct, nout_a, nin_a, nin_b, nout_b)
        c_struct = _struct(t=c_t, D=c_D, s=c_s, n=c_n)
        c = a.__class__(config=a.config, isdiag=a.isdiag,
                        meta_fusion=c_mfs, hard_fusion=c_hfs, struct=c_struct)
        oA = tuple(nout_a + nin_a)
        oB = tuple(nin_b + nout_b)
        if needs_mask:
            ma, mb = _masks_for_axes(a.config, a.struct, a.hard_fusion, nin_a, b.struct, b.hard_fusion, nin_b, meta)
            c.A = a.config.backend.dot_nomerge_masks(a.A, b.A, conj, oA, oB, meta, ma, mb)
        else:
            if any(Da[1] != Db[0] for _, _, _, Da, Db, _, _ in meta):
                raise YastError('Bond dimensions do not match.')

            c.A = a.config.backend.dot_nomerge(a.A, b.A, conj, oA, oB, meta)
    else:
        raise YastError("Unknown policy for tensordot. policy should be in ('hybrid', 'direct', 'merge').")
    return c


def _tensordot_diag(a, b, in_a, destination, conj): # (-1,)
    """ executes broadcast and then transpose into order expected by tensordot. """
    if len(in_a) == 1:
        c = a.broadcast(b, axis=in_a[0], conj=conj)
        c.moveaxis(source=in_a, destination=destination, inplace=True)
        return c
    if len(in_a) == 2:
        c = a.broadcast(b, axis=in_a[0], conj=conj)
        return c.trace(axes=in_a)
    raise YastError('Outer product with diagonal tensor not supported. Use yast.diag() first.')  # len(in_a) == 0


@lru_cache(maxsize=1024)
def _meta_tensordot_nomerge(a_struct, b_struct, nout_a, nin_a, nin_b, nout_b):
    """ meta information for backend, and new tensor structure for tensordot_nomerge """
    nsym = len(a_struct.n)
    a_ndim, b_ndim = len(a_struct.s), len(b_struct.s)

    ta = np.array(a_struct.t, dtype=int).reshape((len(a_struct.t), a_ndim, nsym))
    tb = np.array(b_struct.t, dtype=int).reshape((len(b_struct.t), b_ndim, nsym))
    Da = np.array(a_struct.D, dtype=int).reshape((len(a_struct.D), a_ndim))
    Db = np.array(b_struct.D, dtype=int).reshape((len(b_struct.D), b_ndim))

    ta_con = ta[:, nin_a, :]
    tb_con = tb[:, nin_b, :]
    ta_out = ta[:, nout_a, :]
    tb_out = tb[:, nout_b, :]

    Da_con = np.prod(Da[:, nin_a], axis=1)
    Db_con = np.prod(Db[:, nin_b], axis=1)
    Da_out = Da[:, nout_a]
    Db_out = Db[:, nout_b]
    Da_pro = np.prod(Da_out, axis=1)
    Db_pro = np.prod(Db_out, axis=1)

    block_a = [(tuple(t1.flat), tuple(t2.flat), tuple(t3.flat), D1, tuple(D2), D3)
            for t1, t2, t3, D1, D2, D3 in zip(ta_con, ta_out, ta, Da_con, Da_out, Da_pro)]
    block_a = groupby(sorted(block_a, key=lambda x: x[0]), key=lambda x: x[0])

    block_b = [(tuple(t1.flat), tuple(t2.flat), tuple(t3.flat), D1, tuple(D2), D3)
            for t1, t2, t3, D1, D2, D3 in zip(tb_con, tb_out, tb, Db_con, Db_out, Db_pro)]
    block_b = groupby(sorted(block_b, key=lambda x: x[0]), key=lambda x: x[0])

    meta = []
    try:
        tta, ga = next(block_a)
        ttb, gb = next(block_b)
        while True:
            if tta == ttb:
                for ta, tb in product(ga, gb):
                    meta.append((ta[2], tb[2], ta[1] + tb[1], (ta[5], ta[3]), (tb[3], tb[5]), ta[4] + tb[4], ta[0]))
                tta, ga = next(block_a)
                ttb, gb = next(block_b)
            elif tta < ttb:
                tta, ga = next(block_a)
            elif tta > ttb:
                ttb, gb = next(block_b)
    except StopIteration:
        pass

    meta = tuple(sorted(meta, key=lambda x: x[2]))
    if len(nin_a) == 1:
        c_t = tuple(mm[2] for mm in meta)
        c_D = tuple(mm[5] for mm in meta)
    else:
        if len(meta) > 0:
            ctD = tuple((kk, next(mm)[5]) for kk, mm in groupby(meta, key=lambda x: x[2]))
            c_t, c_D = zip(*ctD)
        else:
            c_t, c_D = tuple(), tuple()
    return meta, c_t, c_D


def broadcast(a, b, axis, conj=(0, 0)):
    r"""
    Compute tensor dot product of tensor a with diagonal tensor b.

    Legs of resulting tensor are ordered in the same way as those of tensor a.
    Produce diagonal tensor if both are diagonal.

    Parameters
    ----------
    a, b: Tensors
        b is a diagonal tensor

    axis: int
        leg of non-diagonal tensor to be multiplied by the diagonal one.

    conj: tuple
        shows which tensor to conjugate: (0, 0), (0, 1), (1, 0), (1, 1)
    """
    _test_configs_match(a, b)
    axis = _broadcast_input(axis, a.meta_fusion, b.isdiag)
    if a.hard_fusion[axis].tree != (1,):
        raise YastError('First tensor`s leg specified by axis cannot be fused.')

    conja = (1 - 2 * conj[0])
    c_hf = a.hard_fusion if conja == 1 else tuple(_flip_hf(x) for x in a.hard_fusion)
    meta, c_struct = _meta_broadcast(a.config, a.struct, b.struct, axis, conja)

    c = a.__class__(config=a.config, isdiag=a.isdiag,
                    meta_fusion=a.meta_fusion, hard_fusion=c_hf, struct=c_struct)
    a_ndim, axis = (1, 0) if a.isdiag else (a.ndim_n, axis)
    c.A = a.config.backend.dot_diag(a.A, b.A, conj, meta, axis, a_ndim)
    return c


def _broadcast_input(axis, mf, isdiag):
    if not isdiag:
        raise YastError('Second tensor should be diagonal.')
    if not isinstance(axis, int):
        raise YastError('axis should be an int.')
    axis = axis % len(mf)
    if mf[axis] != (1,):
        raise YastError('First tensor`s leg specified by axis cannot be fused.')
    axis = sum(mf[ii][0] for ii in range(axis))  # unpack
    return axis


@lru_cache(maxsize=1024)
def _meta_broadcast(config, a_struct, b_struct, axis, conja):
    """ meta information for backend, and new tensor structure for brodcast """
    nsym = config.sym.NSYM
    ind_ta = tuple(x[axis * nsym: (axis + 1) * nsym] for x in a_struct.t)
    ind_tb = tuple(x[:nsym] for x in b_struct.t)
    meta = tuple((ta, ia + ia, Da) for ta, ia, Da in zip(a_struct.t, ind_ta, a_struct.D) if ia in ind_tb)

    c_n = np.array(a_struct.n, dtype=int).reshape((1, 1, -1))
    c_n = tuple(config.sym.fuse(c_n, (1,), conja)[0])
    c_s = a_struct.s if conja == 1 else tuple(-x for x in a_struct.s)

    Db = dict(zip(b_struct.t, b_struct.D))
    if any(Da[axis] != Db[tb][0] for _, tb, Da in meta):
        raise YastError("Bond dimensions do not match.")
    c_t = tuple(ta for ta, _, _ in meta)
    c_D = tuple(Da for _, _, Da in meta)
    c_struct = _struct(t=c_t, D=c_D, s=c_s, n=c_n)
    meta = tuple((ta, tb) for ta, tb, _ in meta)
    return meta, c_struct


def mask(a, b, axis=0):
    r"""
    Apply mask given by nonzero elements of diagonal tensor b on specified axis of tensor a.

    Legs of resulting tensor are ordered in the same way as those of tensor a.
    Bond dimensions of specified axis of a are truncated according to the mask.
    Produce diagonal tensor if both are diagonal.

    Parameters
    ----------
    a, b: Tensors
        b is a diagonal tensor

    axis: int
        leg of tensor a where the mask is applied.
    """
    _test_configs_match(a, b)
    axis = _broadcast_input(axis, a.meta_fusion, b.isdiag)
    if a.hard_fusion[axis].tree != (1,):
        raise YastError('First tensor`s leg specified by axis cannot be fused.')

    Dbnew = tuple(b.config.backend.count_nonzero(b.A[tb]) for tb in b.struct.t)
    meta, c_struct = _meta_mask(a.struct, a.isdiag, b.struct, Dbnew, axis)

    c = a.__class__(config=a.config, isdiag=a.isdiag,
                    meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=c_struct)
    a_ndim, axis = (1, 0) if a.isdiag else (a.ndim_n, axis)
    c.A = a.config.backend.mask_diag(a.A, b.A, meta, axis, a_ndim)
    return c


@lru_cache(maxsize=1024)
def _meta_mask(a_struct, a_isdiag, b_struct, Dbnew, axis):
    """ meta information for backend, and new tensor structure for mask."""
    nsym = len(a_struct.n)
    ind_tb = tuple(x[:nsym] for x, d in zip(b_struct.t, Dbnew) if d > 0)
    ind_ta = tuple(x[axis * nsym: (axis + 1) * nsym] for x in a_struct.t)
    meta = tuple((ta, ia + ia, Da) for ta, ia, Da in zip(a_struct.t, ind_ta, a_struct.D) if ia in ind_tb)

    Db = dict(zip(b_struct.t, b_struct.D))
    if any(Da[axis] != Db[tb][0] for _, tb, Da in meta):
        raise YastError('Bond dimensions do not match.')
    c_t = tuple(ta for ta, _, _ in meta)
    Db = dict(zip(b_struct.t, Dbnew))
    if a_isdiag:
        c_D = tuple((Db[tb], Db[tb] ) for _, tb, _ in meta)
    else:
        c_D = tuple(Da[:axis] + (Db[tb],) + Da[axis + 1:] for _, tb, Da in meta)
    c_struct = a_struct._replace(t=c_t, D=c_D)
    meta = tuple((ta, tb) for ta, tb, _ in meta)
    return meta, c_struct


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
    conja, conjb = (1 - 2 * conj[0]), (1 - 2 * conj[1])
    needs_mask, _ = _test_axes_match(a, b, sgn=-conja * conjb)

    c_n = np.array(a.struct.n + b.struct.n, dtype=int).reshape((1, 2, a.config.sym.NSYM))
    c_n = a.config.sym.fuse(c_n, (conja, conjb), 1)

    meta = _vdot_meta(a.struct, b.struct)
    if len(meta) > 0 and np.all(c_n == 0):
        meta, Da, Db = zip(*meta)
        if needs_mask:
            sla, slb = _masks_for_vdot(a.config, a.struct, a.hard_fusion, b.struct, b.hard_fusion, meta)
            Aa = {t: a.A[t][sla[t]] for t in meta}
            Ab = {t: b.A[t][slb[t]] for t in meta}
        else:
            if Da != Db:
                raise YastError('Bond dimensions do not match.')
            Aa, Ab = a.A, b.A
        return a.config.backend.vdot(Aa, Ab, conj, meta)
    return a.zero_of_dtype()


def _vdot_meta(a_struct, b_struct):
    """ instruction for backend of vdot and """
    ia, ib, meta = 0, 0, []
    while ia < len(a_struct.t) and ib < len(b_struct.t):
        ta, Da = a_struct.t[ia], a_struct.D[ia]
        tb, Db = b_struct.t[ib], b_struct.D[ib]
        if ta == tb:
            meta.append((ta, Da, Db))
            ia += 1
            ib += 1
        elif ta < tb:
            ia += 1
        else:
            ib += 1
    return meta


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
    if len(set(lin1) & set(lin2)) > 0:
        raise YastError('The same axis in axes[0] and axes[1].')
    needs_mask, (in1, in2) = _test_axes_match(a, a, sgn=-1, axes=(lin1, lin2))

    if len(in1) == 0:
        return a

    order = in1 + in2
    out = tuple(i for i in range(a.ndim_n) if i not in order)
    order = order + out
    c_mfs = tuple(a.meta_fusion[i] for i in range(a.ndim) if i not in lin1 + lin2)
    c_hfs = tuple(a.hard_fusion[ii] for ii in out)

    if a.isdiag:
        # if needs_mask: raise YastError('Should not have happend')
        c_struct = _struct(s=(), n=a.struct.n, t=((),), D=((),))
        c = a.__class__(config=a.config, struct=c_struct, meta_fusion=c_mfs, hard_fusion=c_hfs)
        c.A = {(): c.config.backend.sum_elements(a.A)}
        return c

    meta, c_struct, t12, D1, D2 = _trace_meta(a.struct, in1, in2, out)
    c = a.__class__(config=a.config, meta_fusion=c_mfs, hard_fusion=c_hfs, struct=c_struct)
    if needs_mask:
        msk12 = _masks_for_trace(a.config, t12, D1, D2, a.hard_fusion, in1, in2)
        c.A = c.config.backend.trace_with_mask(a.A, order, meta, msk12)
    else:
        if D1 != D2:
            raise YastError('Bond dimensions do not match.')
        c.A = c.config.backend.trace(a.A, order, meta)
    return c


@lru_cache(maxsize=1024)
def _trace_meta(struct, in1, in2, out):
    """ meta-information for backend and struct of traced tensor. """
    lt = len(struct.t)
    tset = np.array(struct.t, dtype=int).reshape((lt, len(struct.s), len(struct.n)))
    Dset = np.array(struct.D, dtype=int).reshape((lt, len(struct.s)))
    t1 = tset[:, in1, :].reshape(lt, -1)
    t2 = tset[:, in2, :].reshape(lt, -1)
    to = tset[:, out, :].reshape(lt, -1)
    D1 = Dset[:, in1]
    D2 = Dset[:, in2]
    D3 = Dset[:, out]
    pD1 = np.prod(D1, axis=1).reshape(lt, 1)
    pD2 = np.prod(D2, axis=1).reshape(lt, 1)
    ind = (np.all(t1 == t2, axis=1)).nonzero()[0]
    Drsh = np.hstack([pD1, pD2, D3])
    t12 = tuple(tuple(t.flat) for t in t1[ind])
    D1 = tuple(tuple(x.flat) for x in D1[ind])
    D2 = tuple(tuple(x.flat) for x in D2[ind])
    meta = [(tuple(to[n]), tuple(tset[n].flat), tuple(Drsh[n]), tt) for n, tt in zip(ind, t12)]
    meta = tuple(sorted(meta, key=lambda x: x[0]))
    newtD = [(m[0], m[2][2:]) for m in meta]
    newtD = tuple(k for k, _ in groupby(newtD))
    if len(newtD) > 0:
        newt, newD = zip(*newtD)
    else:
        newt, newD = (), ()
    c_s = tuple(struct.s[i] for i in out)
    c_struct = _struct(t=newt, D=newD, s=c_s, n=struct.n)
    return meta, c_struct, t12, D1, D2


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
    if not a.config.fermionic:
        return a
    fss = (True,) * len(a.struct.n) if a.config.fermionic is True else a.config.fermionic
    axes = tuple(_clear_axes(*axes))  # swapped groups of legs
    tp = _swap_gate_meta(a.struct.t, a.struct.n, a.meta_fusion, a.ndim_n, axes, fss)
    if inplace:
        for ts, odd in zip(a.struct.t, tp):
            if odd:
                a.A[ts] = -a.A[ts]
        return a
    c = a.__class__(config=a.config, isdiag=a.isdiag, meta_fusion=a.meta_fusion, hard_fusion=a.hard_fusion, struct=a.struct)
    c.A = {ts: -a.A[ts] if odd else a.config.backend.clone(a.A[ts]) for ts, odd in zip(a.struct.t, tp)}
    return c


@lru_cache(maxsize=1024)
def _swap_gate_meta(t, n, mf, ndim, axes, fss):
    """ calculate which blocks to negate. """
    axes = _unpack_axes(mf, *axes)
    tset = np.array(t, dtype=int).reshape((len(t), ndim, len(n)))
    iaxes = iter(axes)
    tp = np.zeros(len(t), dtype=int)

    if len(axes) % 2 == 1:
        raise YastError('Odd number of elements in axes. Elements of axes should come in pairs.')
    for l1, l2 in zip(*(iaxes, iaxes)):
        if len(set(l1) & set(l2)) > 0:
            raise YastError('Cannot swap the same index.')
        t1 = np.sum(tset[:, l1, :], axis=1) % 2
        t2 = np.sum(tset[:, l2, :], axis=1) % 2
        tp += np.sum(t1[:, fss] * t2[:, fss], axis=1)
    return tuple(tp % 2)


def einsum(subscripts, *operants, order='Alphabetic'):
    """
    Execute series of tensor contractions.

    Covering trace, tensordot (including outter product), and transpose.
    Follows notation of `np.einsum` as close as possible.

    Parameters
    ----------
    subscripts: str

    *operants: tensors to be contracted

    order: str
    Specify order in which repeated indices from subscipt are contracted.
    By default, follows alphabetic order.

    Example
    -------
    yast.einsum('*ij,jh->ih', t1, t2)  # matrix-matrix multiplication, where first matrix is conjugated.
    Equivalent to t1.conj() @ t2

    yast.einsum('ab,al,bm->lm', t1, t2, t3, order='ba')
    Contract along b first, and a second.

    Returns
    -------
    tensor : Tensor
    """
    pass


def ncon(ts, inds, conjs=None):
    """
    Execute series of tensor contractions.

    Parameters
    ----------
    ts: list
        list of tensors to be contracted

    inds: tuple of tuples of ints (or list of lists of ints)
        each inner tuple marks axis of respectiv tensor with ints.
        Positive values mark legs to be contracted,
        where two axis to be contracted have the same number.
        Legs are contracted in order of increasing number.
        Non-positive numbers mark legs of resulting tensor, in reversed order.

    conjs: tuple of ints
        For each tensor in ts, it contains
        1 if the tensor should be conjugated and 0 otherwise.

    Example
    ------
    yast.ncon([a, b], ((-0, 1), (1, -1)), conjs=(1, 0)) is
    matrix-matrix multiplication where first matrix is conjugated.
    yast.ncon([a, b], ((-0, -2), (-1, -3))) is outter product.

    Returns
    -------
    tensor : Tensor
    """
    if len(ts) != len(inds):
        raise YastError('Number of tensors and indices do not match.')
    for ii, ind in enumerate(inds):
        if ts[ii].ndim != len(ind):
            raise YastError('Number of legs of one of the tensors do not match provided indices.')

    inds = tuple(_clear_axes(*inds))
    if conjs is not None:
        conjs = tuple(conjs)

    meta_trace, meta_dot, meta_transpose = _ncom_meta(inds, conjs)
    ts = dict(enumerate(ts))
    for command in meta_trace:
        t, axes = command
        ts[t] = trace(ts[t], axes=axes)
    for command in meta_dot:
        (t1, t2), axes, conj = command
        ts[t1] = tensordot(ts[t1], ts[t2], axes=axes, conj=conj)
        del ts[t2]
    t, axes, to_conj = meta_transpose
    if axes is not None:
        ts[t].transpose(axes=axes, inplace=True)
    return ts[t].conj() if to_conj else ts[t]


@lru_cache(maxsize=1024)
def _ncom_meta(inds, conjs):
    """ turning information in inds and conjs into list of contraction commands """
    if not all( -256 < x < 256 for x in _flatten(inds)):
        raise YastError('ncon requires indices to be between -256 and 256.')

    edges = [(order, leg, ten) if order > 0 else (-order + 1024, leg, ten)
             for ten, el in enumerate(inds) for leg, order in enumerate(el)]
    edges.append((512, 512, 512)) # this will mark the end of contractions.
    conjs = [0] * len(inds) if conjs is None else list(conjs)

    # order of contraction with info on tensor and axis
    edges = sorted(edges, reverse=True, key=lambda x: x[0])
    return _consume_edges(edges, conjs)


def _consume_edges(edges, conjs):
    """ consumes edges to generate order of contractions. """
    eliminated, ntensors = [], len(conjs)
    meta_trace, meta_dot = [], []
    order1, leg1, ten1 = edges.pop()
    ax1, ax2 = [], []
    while order1 != 512:  # tensordot two tensors, or trace one tensor; 512 is cutoff marking end of truncation
        order2, leg2, ten2 = edges.pop()
        if order1 != order2:
            raise YastError('Indices of legs to contract do not match.')
        t1, t2, leg1, leg2 = (ten1, ten2, leg1, leg2) if ten1 < ten2 else (ten2, ten1, leg2, leg1)
        ax1.append(leg1)
        ax2.append(leg2)
        if edges[-1][0] == 512 or min(edges[-1][2], edges[-2][2]) != t1 or max(edges[-1][2], edges[-2][2]) != t2:
            # execute contraction
            if t1 == t2:  # trace
                if len(meta_dot) > 0:
                    raise YastError("Likely inefficient order of contractions. Do all traces before tensordot. " +
                        "Call all axes connecting two tensors one after another.")
                meta_trace.append((t1, (tuple(ax1), tuple(ax2))))
                ax12 = ax1 + ax2
                for ii, (order, leg, ten) in enumerate(edges):
                    if ten == t1:
                        edges[ii] = (order, leg - sum(i < leg for i in ax12), ten)
            else:  # tensordot (tensor numbers, axes, conj)
                meta_dot.append(((t1, t2), (tuple(ax1), tuple(ax2)), (conjs[t1], conjs[t2])))
                eliminated.append(t2)
                conjs[t1], conjs[t2] = 0, 0
                lt1 = sum(ii[2] == t1 for ii in edges)  # legs of t1
                for ii, (order, leg, ten) in enumerate(edges):
                    if ten == t1:
                        edges[ii] = (order, leg - sum(i < leg for i in ax1), ten)
                    elif ten == t2:
                        edges[ii] = (order, lt1 + leg - sum(i < leg for i in ax2), t1)
            ax1, ax2 = [], []
        order1, leg1, ten1 = edges.pop()

    remaining = [i for i in range(ntensors) if i not in eliminated]
    t1 = remaining[0]
    for t2 in remaining[1:]:
        meta_dot.append(((t1, t2), ((), ()), (conjs[t1], conjs[t2])))
        eliminated.append(t2)
        conjs[t1], conjs[t2] = 0, 0
        lt1 = sum(tt == t1 for _, _, tt in edges)
        for ii, (order, leg, ten) in enumerate(edges):
            if ten == t2:
                edges[ii] = (order, leg + lt1, t1)
    unique_out = tuple(ed[0] for ed in edges)
    if len(unique_out) != len(set(unique_out)):
        raise YastError("Repeated non-positive (outgoing) index is ambiguous.")
    axes = tuple(ed[1] for ed in sorted(edges)) # final order for transpose
    if axes == tuple(range(len(axes))):
        axes = None
    meta_transpose = (t1, axes, conjs[t1])
    return tuple(meta_trace), tuple(meta_dot), meta_transpose
