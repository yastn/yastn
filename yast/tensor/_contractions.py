""" Contractions of yast tensors """
from functools import lru_cache
from itertools import groupby, product
import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _struct, _flatten
from ._tests import YastError, _test_configs_match, _test_axes_match
from ._merging import _merge_to_matrix, _meta_unfuse_legdec, _unmerge
from ._merging import _masks_for_tensordot, _masks_for_vdot, _masks_for_trace


__all__ = ['tensordot', 'vdot', 'trace', 'swap_gate', 'ncon', 'einsum', 'broadcast', 'apply_mask']


def __matmul__(a, b):
    """
    The ``@`` operator computes tensor dot product, contracting the last axis
    of `a` with the first axis of `b`.

    Shorthand for yast.tensordot(a, b, axes=(a.ndim - 1, 0)).

    Returns
    -------
    tensor: Tensor
    """
    return tensordot(a, b, axes=(a.ndim - 1, 0))


def tensordot(a, b, axes, conj=(0, 0)):
    r"""
    Compute tensor dot product of two tensor along specified axes.

    Outgoing legs are ordered such that first ones are the remaining legs 
    of the first tensor in the original order, and than those 
    of the second tensor.

    Parameters
    ----------
    a, b: Tensors to contract

    axes: tuple
        legs of both tensors (for each it is specified by int or tuple of ints)
        e.g. axes=(0, 3) to contract 0th leg of `a` with 3rd leg of `b`
        axes=((0, 3), (1, 2)) to contract legs 0 and 3 of `a` with 1 and 2 of `b`, respectively.

    conj: tuple
        shows which tensors to conjugate: (0, 0), (0, 1), (1, 0), or (1, 1).
        Default is (0, 0), i.e. neither tensor is conjugated

    policy: str
        method of executing contraction (reserved for future extensions)

    Returns
    -------
    tensor: Tensor
    """
    if conj[0]:
         a = a.conj()
    if conj[1]:
         b = b.conj()

    in_a, in_b = _clear_axes(*axes)  # contracted meta legs
    needs_mask, (nin_a, nin_b) = _test_axes_match(a, b, sgn=-1, axes=(in_a, in_b))
    if a.isdiag:
        return _tensordot_diag(a, b, in_b, destination=(0,))
    if b.isdiag:
        return _tensordot_diag(b, a, in_a, destination=(-1,))

    _test_configs_match(a, b)
    nout_a = tuple(ii for ii in range(a.ndim_n) if ii not in nin_a)  # outgoing native legs
    nout_b = tuple(ii for ii in range(b.ndim_n) if ii not in nin_b)  # outgoing native legs

    s_c = tuple(a.struct.s[i1] for i1 in nout_a) + tuple(b.struct.s[i2] for i2 in nout_b)
    mfs_c = tuple(a.mfs[ii] for ii in range(a.ndim) if ii not in in_a) + tuple(b.mfs[ii] for ii in range(b.ndim) if ii not in in_b)
    hfs_c = tuple(a.hfs[ii] for ii in nout_a) + tuple(b.hfs[ii] for ii in nout_b)

    ind_a, ind_b = _common_inds(a.struct.t, b.struct.t, nin_a, nin_b, a.ndim_n, b.ndim_n, a.config.sym.NSYM)

    data_a, struct_a, ls_l, ls_ac = _merge_to_matrix(a, (nout_a, nin_a), ind_a)
    data_b, struct_b, ls_bc, ls_r = _merge_to_matrix(b, (nin_b, nout_b), ind_b)

    meta_dot, struct_c = _meta_tensordot(a.config, struct_a, struct_b)
    Dsize = struct_c.sl[-1][1] if len(struct_c.sl) > 0 else 0

    if needs_mask:
        msk_a, msk_b = _masks_for_tensordot(a.config, a.struct, a.hfs, nin_a, ls_ac, b.struct, b.hfs, nin_b, ls_bc)
        data = a.config.backend.dot_with_mask(data_a, data_b, meta_dot, Dsize, msk_a, msk_b)
    else:
        if ls_ac != ls_bc:
            raise YastError('Bond dimensions do not match.')
        data = a.config.backend.dot(data_a, data_b, meta_dot, Dsize)
    meta_unmerge, struct_c = _meta_unfuse_legdec(a.config, struct_c, [ls_l, ls_r], s_c)
    data = _unmerge(a.config, data, meta_unmerge)
    return a._replace(data=data, struct=struct_c, mfs=mfs_c, hfs=hfs_c)


@lru_cache(maxsize=1024)
def _common_inds(t_a, t_b, nin_a, nin_b, ndimn_a, ndimn_b, nsym): 
    """ Return row indices of nparray a that are in b, and vice versa.  Outputs tuples."""
    t_a = np.array(t_a, dtype=int).reshape((len(t_a), ndimn_a, nsym))
    t_b = np.array(t_b, dtype=int).reshape((len(t_b), ndimn_b, nsym))
    la = [tuple(x.flat) for x in t_a[:, nin_a, :]]
    lb = [tuple(x.flat) for x in t_b[:, nin_b, :]]
    sa = set(la)
    sb = set(lb)
    ia = tuple(ii for ii, el in enumerate(la) if el in sb)
    ib = tuple(ii for ii, el in enumerate(lb) if el in sa)
    if len(ia) == len(la):
        ia = None
    if len(ib) == len(lb):
        ib = None
    return ia, ib


@lru_cache(maxsize=1024)
def _meta_tensordot(config, struct_a, struct_b):
    nsym = len(struct_a.n)
    n_c = np.array(struct_a.n + struct_b.n, dtype=int).reshape((1, 2, nsym))
    n_c = tuple(config.sym.fuse(n_c, (1, 1), 1)[0])
    struct_a_resorted = iter(sorted(((t[nsym:], t, D, sl) for t, D, sl in zip(struct_a.t, struct_a.D, struct_a.sl))))
    struct_b_resorted = ((t[:nsym], t, D, sl) for t, D, sl in zip(struct_b.t, struct_b.D, struct_b.sl))
    meta = []
    try:
        tar, ta, Da, sla = next(struct_a_resorted)
        tbl, tb, Db, slb = next(struct_b_resorted)
        while True:
            if tar == tbl:
                meta.append((ta[:nsym] + tb[nsym:], (Da[0], Db[1]), sla, Da, slb, Db, tar, tbl))
                tar, ta, Da, sla = next(struct_a_resorted)
                tbl, tb, Db, slb = next(struct_b_resorted)
            elif tar < tbl:
                tar, ta, Da, sla = next(struct_a_resorted)
            elif tar > tbl:
                tbl, tb, Db, slb = next(struct_b_resorted)
    except StopIteration:
        pass
    meta = sorted(meta)
    t_c = tuple(x[0] for x in meta)
    D_c = tuple(x[1] for x in meta)
    Dp_c = tuple(D[0] * D[1] for D in D_c)
    sl_c = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(Dp_c), Dp_c))
    meta = tuple((sl, *mt[1:]) for sl, mt in zip(sl_c, meta))
    s_c = (struct_a.s[0], struct_b.s[1])
    struct_c = _struct(s=s_c, n=n_c, t=t_c, D=D_c, Dp=Dp_c, sl=sl_c)
    return meta, struct_c


def _tensordot_diag(a, b, in_b, destination):
    """ executes broadcast and then transpose into order expected by tensordot. """
    if len(in_b) == 1:
        c = a.broadcast(b, axis=in_b[0])
        return c.moveaxis(source=in_b, destination=destination)
    if len(in_b) == 2:
        c = a.broadcast(b, axis=in_b[0])
        return c.trace(axes=in_b)
    raise YastError('Outer product with diagonal tensor not supported. Use yast.diag() first.')  # len(in_a) == 0


def broadcast(a, *args, axis=0):
    r"""
    Compute tensor dot product of diagonal tensor a with tensor b.

    Legs of the resulting tensor are ordered in the same way as those of tensor b.
    Produce diagonal tensor if both are diagonal.

    Parameters
    ----------
    a, b: Tensors
        a is a diagonal tensor to be broadcasted

    axis: int
        leg of tensor b to be multiplied by the diagonal tensor a.

    conj: tuple
        shows which tensor to conjugate: (0, 0), (0, 1), (1, 0), (1, 1)
    """
    axes = (axis,) if not hasattr(axis, '__iter__') else axis
    if len(axes) != len(args):
        raise YastError("There should be exactly one axis for each tensor to be projected.")
    results = []
    for b, axis in zip(args, axes):
        _test_configs_match(a, b)
        axis = _broadcast_input(axis, b.mfs, a.isdiag)
        if b.hfs[axis].tree != (1,):
            raise YastError('Second tensor`s leg specified by axis cannot be fused.')

        meta, struct = _meta_broadcast(b.struct, a.struct, axis)

        if b.isdiag:
            b_ndim, axis = (1, 0)
            meta = tuple((sln, slb, Db[0], sla) for sln, slb, Db, sla in meta)
        else:
            b_ndim = b.ndim_n
        Dsize = struct.sl[-1][1] if len(struct.sl) > 0 else 0
        data = b.config.backend.dot_diag(a._data, b._data, meta, Dsize, axis, b_ndim)
        results.append(b._replace(struct=struct, data=data))
    return results.pop() if len(results) == 1 else results


def _broadcast_input(axis, mf, isdiag):
    if not isdiag:
        raise YastError('First tensor should be diagonal.')
    axis = axis % len(mf)
    if mf[axis] != (1,):
        raise YastError('Second tensor`s leg specified by axis cannot be fused.')
    axis = sum(mf[ii][0] for ii in range(axis))  # unpack
    return axis


@lru_cache(maxsize=1024)
def _meta_broadcast(b_struct, a_struct, axis):
    """ meta information for backend, and new tensor structure for brodcast """
    nsym = len(a_struct.n)
    ind_tb = tuple(x[axis * nsym: (axis + 1) * nsym] for x in b_struct.t)
    ind_ta = tuple(x[:nsym] for x in a_struct.t)
    sl_a = dict(zip(ind_ta, a_struct.sl))

    meta = tuple((tb, slb, Db, Dbp, sl_a[ib]) for tb, slb, Db, Dbp, ib in \
                 zip(b_struct.t, b_struct.sl, b_struct.D, b_struct.Dp, ind_tb) if ib in ind_ta)

    if any(Db[axis] != sla[1] - sla[0] for _, _, Db, _, sla in meta):
        raise YastError("Bond dimensions do not match.")

    if len(meta) < len(b_struct.t):
        c_t = tuple(mt[0] for mt in meta)
        c_D = tuple(mt[2] for mt in meta)
        c_Dp = tuple(mt[3] for mt in meta)
        c_sl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(c_Dp), c_Dp))
        c_struct = b_struct._replace(t=c_t, D=c_D, Dp=c_Dp, sl=c_sl)
    else:
        c_struct = b_struct

    meta = tuple((sln, slb, Db, sla) for (_, slb, Db, _, sla), sln in zip(meta, c_struct.sl))
    return meta, c_struct


def apply_mask(a, *args, axis=0):
    r"""
    Apply mask given by nonzero elements of diagonal tensor a on specified axis of tensor b.
    Can provide arbitrary number of tensors b, in which case axis is a list of corresponding length.

    Legs of resulting tensor are ordered in the same way as those of tensor b.
    Bond dimensions of specified axis of b are truncated according to the mask a.
    Produce diagonal tensor if both are diagonal.

    Parameters
    ----------
    a, b: Tensors
        a is a diagonal tensor

    axis: int or tuple of ints
        leg of tensor a where the mask is applied.
    """
    axes = (axis,) if not hasattr(axis, '__iter__') else axis
    if len(axes) != len(args):
        raise YastError("There should be exactly one axis for each tensor to be projected.")
    results = []
    for b, axis in zip(args, axes):
        _test_configs_match(a, b)
        axis = _broadcast_input(axis, b.mfs, a.isdiag)
        if b.hfs[axis].tree != (1,):
            raise YastError('Second tensor`s leg specified by axis cannot be fused.')

        Dbnew = tuple(a.config.backend.count_nonzero(a._data[slice(*sl)]) for sl in a.struct.sl)
        meta, struct = _meta_mask(b.struct, b.isdiag, a.struct, Dbnew, axis)
        Dsize = struct.sl[-1][1] if len(struct.sl) > 0 else 0

        if b.isdiag:
            b_ndim, axis = (1, 0)
            meta = tuple((sln, sla, Da[0], slb) for sln, sla, Da, slb in meta)
        else:
            b_ndim = b.ndim_n
        data = b.config.backend.mask_diag(b._data, a._data, meta, Dsize, axis, b_ndim)
        results.append(b._replace(struct=struct, data=data))
    return results.pop() if len(results) == 1 else results


@lru_cache(maxsize=1024)
def _meta_mask(a_struct, a_isdiag, b_struct, Dbnew, axis):
    """ meta information for backend, and new tensor structure for mask."""
    nsym = len(a_struct.n)
    ind_tb = {x[:nsym]: (sl, d) for x, d, sl in zip(b_struct.t, Dbnew, b_struct.sl) if d > 0}
    ind_ta = tuple(x[axis * nsym: (axis + 1) * nsym] for x in a_struct.t)

    meta = tuple((ta, sla, Da, *ind_tb[ia]) for ta, sla, Da, ia in \
                zip(a_struct.t, a_struct.sl, a_struct.D, ind_ta) if ia in ind_tb)

    if any(Da[axis] != slb[1] - slb[0] for _, _, Da, slb, _ in meta):
        raise YastError("Bond dimensions do not match.")

    # mt = (ta, sla, Da, slb, Db)
    c_t = tuple(mt[0] for mt in meta)
    if a_isdiag:
        c_D = tuple((mt[4], mt[4]) for mt in meta)
        c_Dp = tuple(x[0] for x in c_D)
    else:
        c_D = tuple(mt[2][:axis] + (mt[4],) + mt[2][axis + 1:] for mt in meta)
        c_Dp = tuple(np.prod(c_D, axis=1))
    c_sl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(c_Dp), c_Dp))
    c_struct = a_struct._replace(t=c_t, D=c_D, Dp=c_Dp, sl=c_sl)
    meta = tuple((sln, sla, Da, slb) for (_, sla, Da, slb, _), sln in zip(meta, c_struct.sl))
    return meta, c_struct


def vdot(a, b, conj=(1, 0)):
    r"""
    Compute scalar product :math:`x = \langle a|b \rangle` of two tensors.

    Parameters
    ----------
    a, b: Tensor

    conj: tuple(int)
        shows which tensor to conjugate: (0, 0), (0, 1), (1, 0), or (1, 1).
        Default is (1, 0), i.e. tensor `a` is conjugated.

    Returns
    -------
    x: scalar
    """
    _test_configs_match(a, b)
    if conj[0] == 1:
         a = a.conj()
    if conj[1] == 1:
         b = b.conj()
    needs_mask, _ = _test_axes_match(a, b, sgn=-1)
    if a.struct.t == b.struct.t:
        Adata, Bdata = a._data, b._data
        struct_a, struct_b = a.struct, b.struct
    else:
        ia, ib, ta, Da, Dpa, inter_sla,  tb, Db, Dpb, inter_slb,  = 0, 0, [], [], [], [], [], [], [], []
        while ia < len(a.struct.t) and ib < len(b.struct.t):
            if a.struct.t[ia] == b.struct.t[ib]:
                ta.append(a.struct.t[ia])
                tb.append(b.struct.t[ib])
                Da.append(a.struct.D[ia])
                Db.append(b.struct.D[ib])
                Dpa.append(a.struct.Dp[ia])
                Dpb.append(b.struct.Dp[ib])
                inter_sla.append(a.struct.sl[ia])
                inter_slb.append(b.struct.sl[ib])
                ia += 1
                ib += 1
            elif a.struct.t[ia] < b.struct.t[ib]:
                ia += 1
            else:
                ib += 1
        sla = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(Dpa), Dpa))
        slb = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(Dpb), Dpb))
        struct_a = a.struct._replace(t=tuple(ta), D=tuple(Da), Dp=tuple(Dpa), sl=sla)
        struct_b = b.struct._replace(t=tuple(tb), D=tuple(Db), Dp=tuple(Dpb), sl=slb)
        Adata = a.config.backend.apply_slice(a._data, sla, inter_sla)
        Bdata = b.config.backend.apply_slice(b._data, slb, inter_slb)
    if needs_mask:
        msk_a, msk_b, struct_a, struct_b = _masks_for_vdot(a.config, struct_a, a.hfs, struct_b, b.hfs)
        Adata = Adata[msk_a]
        Bdata = Bdata[msk_b]
    if struct_a.D != struct_b.D:
        raise YastError('Bond dimensions do not match.')

    c_n = np.array(a.struct.n + b.struct.n, dtype=int).reshape((1, 2, a.config.sym.NSYM))
    c_n = a.config.sym.fuse(c_n, (1, 1), 1)
    if len(struct_a.D) > 0 and np.all(c_n == 0):
        return a.config.backend.vdot(Adata, Bdata)
    return a.zero_of_dtype()


def trace(a, axes=(0, 1)):
    """
    Compute trace of legs specified by axes.

    Parameters
    ----------
        axes: tuple(int) or tuple(tuple(int))
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
    mfs = tuple(a.mfs[i] for i in range(a.ndim) if i not in lin1 + lin2)
    hfs = tuple(a.hfs[ii] for ii in out)

    if a.isdiag:
        # if needs_mask: raise YastError('Should not have happend')
        struct = a.struct._replace(s=(), diag=False, t=((),), D=((),), Dp=(1,), sl=((0, 1),))
        data = a.config.backend.sum_elements(a._data)
        return a._replace(struct=struct, mfs=mfs, hfs=hfs, isdiag=False, data=data)

    meta, struct, tcon, D1, D2 = _trace_meta(a.struct, in1, in2, out)
    Dsize = struct.sl[-1][1] if len(struct.sl) > 0 else 0
    if needs_mask:
        msk12 = _masks_for_trace(a.config, tcon, D1, D2, a.hfs, in1, in2)
        data = a.config.backend.trace_with_mask(a._data, order, meta, Dsize, tcon, msk12)
    else:
        if D1 != D2:
            raise YastError('Bond dimensions do not match.')
        data = a.config.backend.trace(a._data, order, meta, Dsize)
    return a._replace(mfs=mfs, hfs=hfs, struct=struct, data=data)



@lru_cache(maxsize=1024)
def _trace_meta(struct, in1, in2, out):
    """ meta-information for backend and struct of traced tensor. """
    lt = len(struct.t)
    tset = np.array(struct.t, dtype=int).reshape((lt, len(struct.s), len(struct.n)))
    Dset = np.array(struct.D, dtype=int).reshape((lt, len(struct.s)))
    t1 = tset[:, in1, :].reshape(lt, -1)
    t2 = tset[:, in2, :].reshape(lt, -1)
    tn = tset[:, out, :].reshape(lt, -1)
    D1 = Dset[:, in1]
    D2 = Dset[:, in2]
    Dn = Dset[:, out]
    Dnp = np.prod(Dn, axis=1, dtype=int)
    pD1 = np.prod(D1, axis=1, dtype=int).reshape(lt, 1)
    pD2 = np.prod(D2, axis=1, dtype=int).reshape(lt, 1)
    ind = (np.all(t1 == t2, axis=1)).nonzero()[0]
    Drsh = np.hstack([pD1, pD2, Dn])
    t12 = tuple(tuple(t.flat) for t in t1[ind])
    tn = tuple(tuple(x.flat) for x in tn[ind])
    D1 = tuple(tuple(x.flat) for x in D1[ind])
    D2 = tuple(tuple(x.flat) for x in D2[ind])
    Dn = tuple(tuple(x.flat) for x in Dn[ind])
    Dnp = Dnp[ind]
    slo = tuple(struct.sl[n] for n in ind)
    Do = tuple(struct.D[n] for n in ind)
    Drsh = tuple(tuple(x.flat) for x in Drsh[ind])

    meta = tuple(sorted(zip(tn, Dn, Dnp, t12, slo, Do, Drsh), key=lambda x: x[0]))

    low, high = 0, 0
    c_t, c_D, c_Dp, c_sl, meta2, tcon = [], [], [], [], [], []
    for t, group in groupby(meta, key=lambda x: x[0]):
        c_t.append(t)
        mt = next(group)
        c_D.append(mt[1])
        c_Dp.append(mt[2])
        high = low + mt[2]
        sl = (low, high)
        low = high
        c_sl.append(sl)
        tcon.append(mt[3])
        meta2.append((sl, *mt[4:]))
        for mt in group:
            tcon.append(mt[3])
            meta2.append((sl, *mt[4:]))
    c_s = tuple(struct.s[i] for i in out)
    c_struct = _struct(s=c_s, n=struct.n, t=tuple(c_t), D=tuple(c_D), Dp=tuple(c_Dp), sl=tuple(c_sl))
    return tuple(meta2), c_struct, tuple(tcon), D1, D2



def swap_gate(a, axes):
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
    tp = _swap_gate_meta(a.struct.t, a.struct.n, a.mfs, a.ndim_n, axes, fss)
    c = a.clone()
    for sl, odd in zip(c.struct.sl, tp):
        if odd:
            c._data[slice(*sl)] = -1 * c._data[slice(*sl)]
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


def einsum(subscripts, *operands, order='Alphabetic'):
    """
    Execute series of tensor contractions.

    Covering trace, tensordot (including outer products), and transpose.
    Follows notation of `np.einsum` as close as possible.

    Parameters
    ----------
    subscripts: str

    operands: tensors to be contracted

    order: str
        Specify order in which repeated indices from subscipt are contracted.
        By default, follows alphabetic order.

    Example
    -------
    
    ::
    
        yast.einsum('*ij,jh->ih', t1, t2) 
        
        # matrix-matrix multiplication, where the first matrix is conjugated.
        # Equivalent to 

        t1.conj() @ t2

        yast.einsum('ab,al,bm->lm', t1, t2, t3, order='ba')
        
        # Contract along b first, and a second.

    Returns
    -------
    tensor : Tensor
    """
    if not isinstance(subscripts, str):
        raise YastError('The first argument should be a string.')

    subscripts = subscripts.replace(' ', '')

    tmp = subscripts.split('->')
    if len(tmp) == 1:
        sin, sout = tmp[0], ''
    elif len(tmp) == 2:
        sin, sout = tmp
    else:
        raise YastError('Subscript should have at most one separator ->')

    alphabet1 = 'ABCDEFGHIJKLMNOPQRSTUWXYZabcdefghijklmnopqrstuvwxyz'
    alphabet2 = alphabet1 + ',*'
    if any(v not in alphabet1 for v in sout) or \
       any(v not in alphabet2 for v in sin):
        raise YastError('Only alphabetic characters can be used to index legs.')

    conjs = [1 if '*' in ss else 0 for ss in sin.split(',')]
    sin = sin.replace('*', '')

    if sout == '':
        for v in sin.replace(',', ''):
            if sin.count(v) == 1:
                sout += v
    elif len(sout) != len(set(sout)):
        raise YastError('Repeated index after ->')

    if order in ('Alphabetic', 'alphabetic'):
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
        raise YastError('order does not cover all contracted indices')
    inds = [tuple(d[v] for v in ss) for ss in sin.split(',')]
    ts = list(operands)
    return ncon(ts, inds, conjs)


def ncon(ts, inds, conjs=None):
    """
    Execute series of tensor contractions.

    Parameters
    ----------
    ts: list
        list of tensors to be contracted

    inds: tuple of tuples of ints (or list of lists of ints)
        each inner tuple labels axes of respective tensor with ints.
        Positive values labels legs to be contracted,
        with pairs of axes to be contracted denoted by the same integer label.
        Legs are contracted in the order of ascending integer value.
        Non-positive numbers label the legs of resulting tensor, in reversed order.

    conjs: tuple of ints
        For each tensor in ts, it contains
        1 if the tensor should be conjugated and 0 otherwise.

    Example
    -------

    ::
        
        # matrix-matrix multiplication where the first matrix is conjugated
        
        yast.ncon([a, b], ((-0, 1), (1, -1)), conjs=(1, 0))
        
        # outer product
    
        yast.ncon([a, b], ((-0, -2), (-1, -3))) 

    Returns
    -------
    tensor : Tensor
    """
    if len(ts) != len(inds):
        raise YastError('Number of tensors and indices do not match.')
    for tensor, ind in zip(ts, inds):
        if tensor.ndim != len(ind):
            raise YastError('Number of legs of one of the tensors do not match provided indices.')

    inds = tuple(_clear_axes(*inds))
    if conjs is not None:
        conjs = tuple(conjs)

    meta_trace, meta_dot, meta_transpose = _ncon_meta(inds, conjs)
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
        ts[t] = ts[t].transpose(axes=axes)
    return ts[t].conj() if to_conj else ts[t]


@lru_cache(maxsize=1024)
def _ncon_meta(inds, conjs):
    """ turning information in inds and conjs into list of contraction commands """
    if not all(-256 < x < 256 for x in _flatten(inds)):
        raise YastError('ncon requires indices to be between -256 and 256.')

    edges = [[order, leg, ten] if order > 0 else [-order + 1024, leg, ten]
             for ten, el in enumerate(inds) for leg, order in enumerate(el)]
    edges.append([512, 512, 512])  # this will mark the end of contractions.
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
                for edge in edges:  # edge = (order, leg, tensor)
                    edge[1] -= sum(i < edge[1] for i in ax12) if edge[2] == t1 else 0
            else:  # tensordot (tensor numbers, axes, conj)
                meta_dot.append(((t1, t2), (tuple(ax1), tuple(ax2)), (conjs[t1], conjs[t2])))
                eliminated.append(t2)
                conjs[t1], conjs[t2] = 0, 0
                lt1 = sum(ii[2] == t1 for ii in edges)  # legs of t1
                for edge in edges:  # edge = (order, leg, tensor)
                    if edge[2] == t1:
                        edge[1] = edge[1] - sum(i < edge[1] for i in ax1)
                    elif edge[2] == t2:
                        edge[1:] = edge[1] + lt1 - sum(i < edge[1] for i in ax2), t1
            ax1, ax2 = [], []
        order1, leg1, ten1 = edges.pop()

    remaining = [i for i in range(ntensors) if i not in eliminated]
    t1 = remaining[0]
    for t2 in remaining[1:]:
        meta_dot.append(((t1, t2), ((), ()), (conjs[t1], conjs[t2])))
        eliminated.append(t2)
        conjs[t1], conjs[t2] = 0, 0
        lt1 = sum(tt == t1 for _, _, tt in edges)
        for edge in edges:  # edge = (order, leg, tensor)
            if edge[2] == t2:
                edge[1:] = edge[1] + lt1, t1
    unique_out = tuple(ed[0] for ed in edges)
    if len(unique_out) != len(set(unique_out)):
        raise YastError("Repeated non-positive (outgoing) index is ambiguous.")
    axes = tuple(ed[1] for ed in sorted(edges))  # final order for transpose
    if axes == tuple(range(len(axes))):
        axes = None
    meta_transpose = (t1, axes, conjs[t1])
    return tuple(meta_trace), tuple(meta_dot), meta_transpose


    # if policy in ('hybrid', 'direct'):
    #     meta, c_t, c_D, c_Dp, c_sl, tcon = _meta_tensordot_nomerge(a.struct, b.struct, nout_a, nin_a, nin_b, nout_b)
    #     c_struct = _struct(t=c_t, D=c_D, Dp=c_Dp, sl=c_sl, s=c_s, n=c_n)
    #     Dsize = c_sl[-1][1] if len(c_sl) > 0 else 0
    #     oA = tuple(nout_a + nin_a)
    #     oB = tuple(nin_b + nout_b)
    #     if needs_mask:
    #         ma, mb = _masks_for_axes(a.config, a.struct, a.hfs, nin_a, b.struct, b.hfs, nin_b, tcon)
    #         data = a.config.backend.dot_nomerge_masks(a._data, b._data, conj, oA, oB, meta, Dsize, tcon, ma, mb)
    #     else:
    #         if any(mt[3][1] != mt[6][0] for mt in meta):
    #             raise YastError('Bond dimensions do not match.')
    #         data = a.config.backend.dot_nomerge(a._data, b._data, conj, oA, oB, meta, Dsize)
    #     return a._replace(mfs=c_mfs, hfs=c_hfs, struct=c_struct, data=data)
    # raise YastError("Unknown policy for tensordot. policy should be in ('hybrid', 'direct', 'merge').")

    # @lru_cache(maxsize=1024)
    # def _meta_tensordot_nomerge(a_struct, b_struct, nout_a, nin_a, nin_b, nout_b):
    # """ meta information for backend, and new tensor structure for tensordot_nomerge """
    # nsym = len(a_struct.n)
    # a_ndim, b_ndim = len(a_struct.s), len(b_struct.s)

    # ta = np.array(a_struct.t, dtype=int).reshape((len(a_struct.t), a_ndim, nsym))
    # tb = np.array(b_struct.t, dtype=int).reshape((len(b_struct.t), b_ndim, nsym))
    # Da = np.array(a_struct.D, dtype=int).reshape((len(a_struct.D), a_ndim))
    # Db = np.array(b_struct.D, dtype=int).reshape((len(b_struct.D), b_ndim))

    # ta_con = tuple(tuple(t.flat) for t in ta[:, nin_a, :])
    # tb_con = tuple(tuple(t.flat) for t in tb[:, nin_b, :])
    # ta_out = tuple(tuple(t.flat) for t in ta[:, nout_a, :])
    # tb_out = tuple(tuple(t.flat) for t in tb[:, nout_b, :])

    # Da_pcon = np.prod(Da[:, nin_a], axis=1, dtype=int)
    # Db_pcon = np.prod(Db[:, nin_b], axis=1, dtype=int)
    # Da_out = tuple(tuple(D.flat) for D in Da[:, nout_a])
    # Db_out = tuple(tuple(D.flat) for D in Db[:, nout_b])
    # Da_pout = np.prod(Da_out, axis=1, dtype=int)
    # Db_pout = np.prod(Db_out, axis=1, dtype=int)

    # block_a = [x for x in zip(ta_con, ta_out, a_struct.sl, a_struct.D, Da_pcon, Da_out, Da_pout)]
    # block_a = groupby(sorted(block_a, key=lambda x: x[0]), key=lambda x: x[0])

    # block_b = [x for x in zip(tb_con, tb_out, b_struct.sl, b_struct.D, Db_pcon, Db_out, Db_pout)]
    # block_b = groupby(sorted(block_b, key=lambda x: x[0]), key=lambda x: x[0])

    # meta = []
    # try:
    #     tta, ga = next(block_a)
    #     ttb, gb = next(block_b)
    #     while True:
    #         if tta == ttb:
    #             for ta, tb in product(ga, gb):
    #                 meta.append((ta[1] + tb[1], ta[2], ta[3], (ta[6], ta[4]), tb[2], tb[3], (tb[4], tb[6]), ta[5] + tb[5], ta[6] * tb[6], ta[0]))
    #             tta, ga = next(block_a)
    #             ttb, gb = next(block_b)
    #         elif tta < ttb:
    #             tta, ga = next(block_a)
    #         elif tta > ttb:
    #             ttb, gb = next(block_b)
    # except StopIteration:
    #     pass

    # meta = tuple(sorted(meta, key=lambda x: x[0]))
    # tcon = tuple(mm[9] for mm in meta)
    # if len(nin_a) == 1:
    #     c_t = tuple(mm[0] for mm in meta)
    #     c_D = tuple(mm[7] for mm in meta)
    #     c_Dp = tuple(mm[8] for mm in meta)
    #     c_sl = tuple((stop - dp, stop) for stop, dp in zip(np.cumsum(c_Dp), c_Dp))
    #     meta = tuple((sl, *mt[1:7]) for sl, mt in zip(c_sl, meta))
    #     return meta, c_t, c_D, c_Dp, c_sl, tcon
    # if len(meta) > 0:
    #     low, high = 0, 0
    #     c_t, c_D, c_Dp, c_sl, meta2 = [], [], [], [], []
    #     for t, group in groupby(meta, key=lambda x: x[0]):
    #         c_t.append(t)
    #         mt = next(group)
    #         c_D.append(mt[7])
    #         c_Dp.append(mt[8])
    #         high = low + mt[8]
    #         sl = (low, high)
    #         low = high
    #         c_sl.append(sl)
    #         meta2.append((sl, *mt[1:7]))
    #         for mt in group:
    #             meta2.append((sl, *mt[1:7]))
    #     return tuple(meta2), tuple(c_t), tuple(c_D), tuple(c_Dp), tuple(c_sl), tcon
    # return meta, (), (), (), (), tcon