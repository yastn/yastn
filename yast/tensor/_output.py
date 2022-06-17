""" methods outputing data from yast tensor. """

import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _mf_to_ntree, _struct, _flatten
from ._tests import YastError
from ..sym import sym_none
from ._legs import Leg, leg_union, _leg_fusions_need_mask
from ._merging import _embed_tensor


__all__ = ['compress_to_1d', 'save_to_dict', 'save_to_hdf5', 'requires_grad']


def save_to_dict(a):
    r"""
    Export YAST tensor to dictionary.

    Returns
    -------
    a: yast.Tensor
        tensor to export
    d: dict
        dictionary containing all the information needed to recreate the tensor.
    """
    _d = a.config.backend.to_numpy(a._data)
    hfs = [hf._asdict() for hf in a.hfs]
    return {'_d': _d, 's': a.struct.s, 'n': a.struct.n,
            't': a.struct.t, 'D': a.struct.D, 'isdiag': a.isdiag,
            'mfs': a.mfs, 'hfs': hfs,
            'SYM_ID': a.config.sym.SYM_ID, 'fermionic': a.config.fermionic}


def save_to_hdf5(a, file, path):
    """
    Export tensor into hdf5 type file.

    Parameters
    ----------
    a : yast.Tensor
        tensor to export
    
    TODO
    """
    _d = a.config.backend.to_numpy(a._data)
    hfs = [tuple(hf) for hf in a.hfs]
    file.create_dataset(path+'/isdiag', data=[int(a.isdiag)])
    file.create_group(path+'/mfs/'+str(a.mfs))
    file.create_group(path+'/hfs/'+str(hfs))
    file.create_dataset(path+'/n', data=a.struct.n)
    file.create_dataset(path+'/s', data=a.struct.s)
    file.create_dataset(path+'/ts', data=a.struct.t)
    file.create_dataset(path+'/Ds', data=a.struct.D)
    file.create_dataset(path+'/matrix', data=_d)


def compress_to_1d(a, meta=None):
    """
    Store each block within 1D array in contiguous manner 
    (without cloning the data if not necessary) and 
    create metadata allowing re-creation of the original tensor.

    Parameters
    ----------
        a: yast.Tensor
            tensor to export
        meta: dict
            If not None, uses this extra information to fill-in extra zero blocks
            not present in the tensor but allowed by the symmetry.
            Raises error if tensor has some blocks which are not included in meta or otherwise
            meta does not match the tensor.

    Returns
    -------
    tensor (type derived from backend)
        1D array with tensor data
    dict
        metadata with structure of the symmetric tensor
    """
    if meta is None:
        meta = {'struct': a.struct, 'hfs': a.hfs,
                'mfs': a.mfs}
        return a._data, meta
    # else:
    if a.struct.s != meta['struct'].s:
        raise YastError("Tensor has different signature than metadata.")
    if a.struct.n != meta['struct'].n:
        raise YastError("Tensor has different tensor charge than metadata.")
    if a.isdiag != meta['struct'].diag:
        raise YastError("Tensor has different diagonality than metadata.")
    if a.mfs != meta['mfs'] or a.hfs != meta['hfs']:
        raise YastError("Tensor has different leg fusion structure than metadata.")
    Dsize = meta['struct'].sl[-1][1] if len(meta['struct'].sl) > 0 else 0

    if a.struct == meta['struct']:
        return a._data, meta
    # else: embed filling in missing zero blocks
    ia, im, meta_merge = 0, 0, []
    while ia < len(a.struct.t):
        if a.struct.t[ia] < meta['struct'].t[im] or im >= len(meta['struct'].t):
            raise YastError("Tensor has blocks that do not appear in meta.")
        if a.struct.t[ia] == meta['struct'].t[im]:
            meta_merge.append((meta['struct'].sl[im], a.struct.sl[ia]))
            ia += 1
            im += 1
        else: #a.struct.t[ia] > meta['struct'].t[im]:
            im += 1
    data = a.config.backend.embed_slc(a._data, meta_merge, Dsize)
    return data, meta


############################
#    output information    #
############################


def show_properties(a):
    """
    Print basic properties of the tensor:
        * it's symmetry
        * signature
        * total charge
        * whether it is a diagonal tensor
        * meta/logical rank - treating fused legs as single leg
        * native rank
        * total dimension of all existing charge sectors for each leg, treating fused legs as single leg
        * total dimension of all existing charge sectors for native leg
        * number of non-empty blocks
        * total number of elements across all non-empty blocks
        * fusion tree for each leg
        * metadata for fused legs
    """
    print("Symmetry    :", a.config.sym.SYM_ID)
    print("signature   :", a.struct.s)  # signature
    print("charge      :", a.struct.n)  # total charge of tensor
    print("isdiag      :", a.isdiag)
    print("dim meta    :", a.ndim)  # number of meta legs
    print("dim native  :", a.ndim_n)  # number of native legs
    print("shape meta  :", a.get_shape(native=False))
    print("shape native:", a.get_shape(native=True))
    print("no. blocks  :", len(a.struct.t))  # number of blocks
    print("size        :", a.size)  # total number of elements in all blocks
    mfs = {i: _mf_to_ntree(mf) for i, mf in enumerate(a.mfs)}
    print("meta fusion :", mfs)  # encoding meta fusion tree for each leg
    hfs = {i: _mf_to_ntree(hf.tree) for i, hf in enumerate(a.hfs)}
    print("hard fusion :", hfs, "\n")  # encoding info on hard fusion for each leg


def __str__(a):
    legs = a.get_legs()
    ts = tuple(leg.t for leg in legs)
    Ds = tuple(leg.D for leg in legs)
    s = f"{a.config.sym.SYM_ID} s= {a.struct.s} n= {a.struct.n}\n"
    s += f"leg charges  : {ts}\n"
    s += f"dimensions   : {Ds}"
    return s


def requires_grad(a):
    """
    Returns
    -------
    bool : bool
            ``True`` if any of the blocks of the tensor has autograd enabled
    """
    return a.config.backend.requires_grad(a._data)


def print_blocks_shape(a):
    """
    Print shapes of blocks as a sequence of block's charge followed by its shape
    """
    for t, D in zip(a.struct.t, a.struct.D):
        print(f"{t} {D}")


def is_complex(a):
    """
    Returns
    -------
    bool : bool
        ``True`` if all of the blocks of the tensor are complex
    """
    return a.config.backend.is_complex(a._data)


def get_tensor_charge(a):
    """
    Returns
    -------
    int or tuple[int]
        see :attr:`yast.Tensor.n`
    """
    return a.struct.n


def get_signature(a, native=False):
    """
    Returns
    -------
    tuple[int]
        Tensor signature, equivalent to :attr:`yast.Tensor.s`.
        If native, returns the signature of tensors's native legs, see :attr:`yast.Tensor.s_n`.
    """
    return a.s_n if native else a.s


def get_rank(a, native=False):
    """
    Returns
    -------
    int
        Tensor rank equivalent to :attr:`yast.Tensor.ndim`.
        If native, the native rank of the tensor is returned, see :attr:`yast.Tensor.ndim_n`.
    """
    return a.ndim_n if native else a.ndim


def get_blocks_charge(a):
    """
    Returns
    -------
    t : tuple(tuple(int))
        Charges of all native blocks. In case of product of abelian symmetries,
        for each block the individual symmetry charges are flattened into a single tuple.
    """
    return a.struct.t


def get_blocks_shape(a):
    """
    Returns
    -------
    D : tuple(tuple(int))
        Shapes of all native blocks.
    """
    return a.struct.D


def get_shape(a, axis=None, native=False):
    r"""
    Return effective bond dimensions as sum of dimensions along sectors for each leg.

    Parameters
    ----------
    axis : int or tuple[int]
        indices of legs; If axis is ``None`` returns for all legs (default).

    Returns
    -------
    shape : int or tuple[int]
        effective bond dimensions of legs specified by axes
    """
    if axis is None:
        axis = tuple(n for n in range(a.ndim_n if native else a.ndim))
    if isinstance(axis, int):
        return sum(a.get_legs(axis, native=native).D)
    return tuple(sum(leg.D) for leg in a.get_legs(axis, native=native))


def get_dtype(a):
    """
    Returns
    -------
    dtype : dtype
        Returns data ``dtype``.
    """
    return a.config.backend.get_dtype(a._data)


def __getitem__(a, key):
    """
    Parameters
    ----------
    key : tuple[int] or tuple[tuple[int]]
        charges of the block.

    Returns
    -------
    tensor-like
        The type of the returned tensor depends on the backend, for example 
        :class:`numpy.ndarray` or :class:`torch.Tensor`.
        In case of diagonal tensor, returns 1D array.
    """
    key = tuple(_flatten(key))
    try:
        ind = a.struct.t.index(key)
    except ValueError as exc:
        raise YastError('tensor does not have block specify by key') from exc
    x = a._data[slice(*a.struct.sl[ind])]

    # TODO this should be reshape called from backend ?
    return x if a.isdiag else x.reshape(a.struct.D[ind])


##################################################
#    output tensors info - advanced structure    #
##################################################

def get_leg_fusion(a, axes=None):
    """
    .. deprecated::
        to inspect Legs of the tensor, use :meth:`yast.Tensor.get_legs`.

    Fusion trees for meta legs.

    Parameters
    ----------
    axes : Int or tuple of ints
        indices of legs; If axes is None returns all (default).
    """
    if axes is None:
        return {'meta': a.mfs, 'hard': a.hfs}
    if isinstance(axes, int):
        return a.mfs(axes)
    return {'meta': tuple(a.mfs(n) for n in axes), 'hard': tuple(a.hfs(n) for n in axes)}


def get_legs(a, axis=None, native=False):
    r"""
    Return a leg or a set of legs of a Tensor.

    Parameters
    ----------
    axis : int or tuple[int] or None
        indices of legs to retrieve. If ``None`` return list with all legs.

    native : bool
        consider native legs if ``True``; otherwise returns fused legs (default).

    Returns
    -------
        Leg if axis is `int`, otherwise tuple[Leg].
    """
    legs = []
    tset = np.array(a.struct.t, dtype=int).reshape((len(a.struct.t), len(a.struct.s), len(a.struct.n)))
    Dset = np.array(a.struct.D, dtype=int).reshape((len(a.struct.D), len(a.struct.s)))
    if axis is None:
        axis = tuple(range(a.ndim)) if not native else tuple(range(a.ndim_n))
    axes, = _clear_axes(axis)
    for ax in axes:
        legs_ax = []
        if not native:
            mf = a.mfs[ax]
            nax, = _unpack_axes(a.mfs, (ax,))
        else:
            nax = (ax,)
        for i in nax:
            tseta = tset[:, i, :].reshape(len(tset), a.config.sym.NSYM)
            Dseta = Dset[:, i].reshape(-1)
            tDn = {tuple(tn.flat): Dn for tn, Dn in zip(tseta, Dseta)}
            t, D = tuple(tDn.keys()), tuple(tDn.values())
            legs_ax.append(Leg(a.config, s=a.struct.s[i], t=t, D=D, legs=(a.hfs[i],)))
        if not native and mf[0] > 1:
            tseta = tset[:, nax, :].reshape(len(tset), len(nax) * a.config.sym.NSYM)
            Dseta = np.prod(Dset[:, nax], axis=1, dtype=int)
            tDn = {tuple(tn.flat): Dn for tn, Dn in zip(tseta, Dseta)}
            t = tuple(sorted(tDn.keys()))
            D = tuple(tDn[x] for x in t)
            legs.append(Leg(a.config.sym, s=legs_ax[0].s, t=t, D=D, fusion=mf, legs=tuple(legs_ax), _verified=True))
        else:
            legs.append(legs_ax.pop())
    return tuple(legs) if hasattr(axis, '__iter__') else legs.pop()


def get_leg_structure(a, axis, native=False):
    r"""
    .. deprecated::
        to inspect Legs of the tensor, use :meth:`yast.Tensor.get_legs`.

    Find all charges and the corresponding bond dimension for n-th leg.

    Parameters
    ----------
    axis : int
        Index of a leg.

    native : bool
        consider native legs if True; otherwise meta/fused legs (default).

    Returns
    -------
        tDn : dict of {charge of the sector: size of the sector}
    """
    axis, = _clear_axes(axis)
    if not native:
        axis, = _unpack_axes(a.mfs, axis)
    tset = np.array(a.struct.t, dtype=int).reshape((len(a.struct.t), len(a.struct.s), len(a.struct.n)))
    Dset = np.array(a.struct.D, dtype=int).reshape((len(a.struct.D), len(a.struct.s)))
    tset = tset[:, axis, :]
    Dset = Dset[:, axis]
    tset = tset.reshape(len(tset), len(axis) * a.config.sym.NSYM)
    Dset = np.prod(Dset, axis=1, dtype=int) if len(axis) > 1 else Dset.reshape(-1)

    tDn = {tuple(tn.flat): Dn for tn, Dn in zip(tset, Dset)}
    return tDn


def get_leg_charges_and_dims(a, native=False):
    """ 
    .. deprecated::
        to inspect Legs of the tensor, use :meth:`yast.Tensor.get_legs`.

    Collect information about charges and dimensions on all legs into two lists. 
    """
    _tmp = [a.get_leg_structure(n, native=native) for n in range(a.ndim_n if native else a.ndim)]
    _tmp = [{k: lst[k] for k in sorted(lst)} for lst in _tmp]
    ts_and_Ds= tuple(zip(*[tuple(zip(*lst.items())) for lst in _tmp]))
    if len(ts_and_Ds) < 1:
        return (), ()
    ts, Ds = ts_and_Ds
    return ts, Ds


############################
#   Down-casting tensors   #
############################

def to_dense(a, legs=None, native=False, reverse=False):
    r"""
    Create dense tensor corresponding to the symmetric tensor.

    Blocks are ordered according to increasing charges on each leg.
    It is possible to supply a list of additional charge sectors to be included by explictly
    specifying `legs`. These legs should be consistent with current structure of the tensor. 
    This allows to fill in extra zero blocks.

    Parameters
    ----------
    legs : dict[int, yast.Leg]
        specify extra charge sectors on the legs by adding desired :class:`yast.Leg`
        under legs's index into dictionary.

    native: bool
        output native tensor (ignoring fusion of legs).

    reverse: bool
        reverse the order in which blocks are sorted. Default order is ascending in
        values of block's charges.

    Returns
    -------
    out : tensor
        The type of the returned tensor depends on the backend, i.e. ``numpy.ndarray`` or ``torch.tensor``.
    """
    c = a.to_nonsymmetric(legs, native, reverse)
    x = c.config.backend.clone(c._data)
    x = c.config.backend.diag_create(x) if c.isdiag else x.reshape(c.struct.D[0])
    return x


def to_numpy(a, legs=None, native=False, reverse=False):
    r"""
    Create dense :class:`numpy.ndarray`` corresponding to the symmetric tensor. 
    See :func:`yast.to_dense`.

    Returns
    -------
    numpy.ndarray
        dense NumPy array equivalent to symmetric tensor
    """
    return a.config.backend.to_numpy(a.to_dense(legs, native, reverse))


def to_raw_tensor(a):
    """
    If the symmetric tensor has just a single non-empty block, return raw tensor representing
    that block.

    Returns
    -------
    out : tensor-like
        The type of the returned tensor depends on the backend, i.e. ``numpy.ndarray`` or ``torch.tensor``.
    """
    if len(a.struct.D) == 1:
        return a._data.reshape(a.struct.D[0])
    raise YastError('Only tensor with a single block can be converted to raw tensor')


def to_nonsymmetric(a, legs=None, native=False, reverse=False):
    r"""
    Create equivalent ``yast.Tensor`` with no explict symmetry. All blocks of the original
    tensor are accummulated into a single block.

    Blocks are ordered according to increasing charges on each leg.
    It is possible to supply a list of additional charge sectors to be included by explictly
    specifying `legs`. These legs should be consistent with current structure of the tensor. 
    This allows to fill in extra zero blocks.

    .. note::
        YAST structure is redundant since resulting tensor is effectively just
        a single dense block. To obtain this single dense block directly, use :meth:`yast.Tensor.to_dense`.

    Parameters
    ----------
    legs : dict[int, yast.Leg]
        specify extra charge sectors on the legs by adding desired :class:`yast.Leg`
        under legs's index into dictionary.

    native: bool
        output native tensor (ignoring meta-fusion of legs).

    reverse: bool
        reverse the order in which blocks are sorted. Default order is ascending in
        values of block's charges.

    Returns
    -------
    yast.Tensor
        tensor with no explicit symmetry
    """
    config_dense = a.config._replace(sym=sym_none)

    ndim = a.ndim_n if native else a.ndim
    legs_a = list(a.get_legs(range(ndim), native=native))
    if legs is not None:
        if any((n < 0) or (n >= ndim) for n in legs.keys()):
            raise YastError('Specified leg out of ndim')
        legs_new = {n: leg_union(legs_a[n], leg) for n, leg in legs.items()}
        if any(_leg_fusions_need_mask(leg, legs_a[n]) for n, leg in legs_new.items()):
            a = _embed_tensor(a, legs_a, legs_new)  # mask needed
        for n, leg in legs_new.items():
            legs_a[n] = leg

    Dtot = tuple(sum(leg.D) for leg in legs_a)
    tD, step = [], -1 if reverse else 1
    for n, leg in enumerate(legs_a):
        Dlow, tDn = 0, {}
        for tn, Dn in zip(leg.t[::step], leg.D[::step]):
            Dhigh = Dlow + Dn
            tDn[tn] = (Dlow, Dhigh)
            Dlow = Dhigh
        tD.append(tDn)
    axes = tuple((n,) for n in range(ndim))
    if not native:
        axes = tuple(_unpack_axes(a.mfs, *axes))
    meta = []
    tset = np.array(a.struct.t, dtype=int).reshape((len(a.struct.t), len(a.struct.s), len(a.struct.n)))
    for t_sl, tt in zip(a.struct.sl, tset):
        meta.append((slice(*t_sl), tuple(tD[n][tuple(tt[ax, :].flat)] for n, ax in enumerate(axes))))
    if a.isdiag:
        Dtot = Dtot[:1]
        meta = [(sl, D[:1]) for sl, D in meta]

    c_s = a.get_signature(native)
    c_t = ((),)
    c_D = (Dtot,) if not a.isdiag else (Dtot + Dtot,)
    Dp = np.prod(Dtot, dtype=int)
    c_Dp = (Dp,)
    c_sl = ((0, Dp),)
    c_struct = _struct(s=c_s, n=(), diag=a.isdiag, t=c_t, D=c_D, Dp=c_Dp, sl=c_sl)
    data = a.config.backend.merge_to_dense(a._data, Dtot, meta)
    return a._replace(config=config_dense, struct=c_struct, data=data, mfs=None, hfs=None)


# def to_nonsymmetric(a, legs=None, native=False, reverse=False):
#     r"""
#     Create equivalent ``yast.Tensor`` with no explict symmetry. All blocks of the original
#     tensor are accummulated into a single block.

#     Blocks are ordered according to increasing charges on each leg.
#     It is possible to supply a list of additional charge sectors with dimensions to be included.
#     (should be consistent with the tensor). This allows to fill in some explicit zero blocks.

#     .. note::
#         yast structure can be redundant since resulting tensor is effectively just
#         a single dense block. If that's the case, use :meth:`yast.Tensor.to_dense`.

#     Parameters
#     ----------
#     legs : dict
#         {n: Leg} specify charges and dimensions to include on some legs (indicated by keys n).

#     native: bool
#         output native tensor (ignoring meta-fusion of legs).

#     reverse: bool
#         reverse the order in which blocks are sorted. Default order is ascending in
#         values of block's charges.

#     Returns
#     -------
#     out : yast.Tensor
#         returned tensor does not use any symmetry
#     """
#     config_dense = a.config._replace(sym=sym_none)
#     a = a.embed(legs=legs)
#     a_legs = a.get_legs(native=native)

#     Dtot = tuple(sum(leg.D) for leg in a_legs)
#     tD, step = [], -1 if reverse else 1
#     for n, leg in enumerate(a_legs):
#         Dlow, tDn = 0, {}
#         for tn, Dn in zip(leg.t[::step], leg.D[::step]):
#             Dhigh = Dlow + Dn
#             tDn[tn] = (Dlow, Dhigh)
#             Dlow = Dhigh
#         tD.append(tDn)
#     if native:
#         axes = tuple((n,) for n in range(a.ndim_n))
#     else:
#         axes = tuple((n,) for n in range(a.ndim))
#         axes = tuple(_unpack_axes(a.mfs, *axes))
#     meta = []
#     tset = np.array(a.struct.t, dtype=int).reshape((len(a.struct.t), len(a.struct.s), len(a.struct.n)))
#     for t_sl, tt in zip(a.struct.sl, tset):
#         meta.append((slice(*t_sl), tuple(tD[n][tuple(tt[ax, :].flat)] for n, ax in enumerate(axes))))
#     if a.isdiag:
#         Dtot = Dtot[:1]
#         meta = [(sl, D[:1]) for sl, D in meta]

#     c_s = a.get_signature(native)
#     c_t = ((),)
#     c_D = (Dtot,) if not a.isdiag else (Dtot + Dtot,)
#     Dp = np.prod(Dtot, dtype=int)
#     c_Dp = (Dp,)
#     c_sl = ((0, Dp),)
#     c_struct = _struct(s=c_s, n=(), diag=a.isdiag, t=c_t, D=c_D, Dp=c_Dp, sl=c_sl)
#     data = a.config.backend.merge_to_dense(a._data, Dtot, meta)
#     return a._replace(config=config_dense, struct=c_struct, data=data, mfs=None, hfs=None)


def zero_of_dtype(a):
    """ Return zero scalar of the instance specified by backend and dtype. """
    return a.config.backend.zeros((), dtype=a.yast_dtype, device=a.device)


def to_number(a, part=None):
    r"""
    Assuming the symmetric tensor has just a single non-empty block of total dimension one,
    return this element as a scalar.

    For empty tensor return 0.

    .. note::
        This operation preserves autograd.

    Parameters
    ----------
    part : str
        if 'real' return real part only

    Returns
    -------
    out : scalar
        the type of the scalar is given by the backend.
    """
    size = a.size
    if size == 1:
        x = a.config.backend.first_element(a._data)
    elif size == 0:
        x = a.zero_of_dtype()
    else:
        raise YastError('Only single-element (symmetric) Tensor can be converted to scalar')
    return a.config.backend.real(x) if part == 'real' else x


def item(a):
    """
    Assuming the symmetric tensor has just a single non-empty block of total dimension one,
    return this element as standard Python scalar.

    For empty tensor, returns 0.

    Returns
    -------
    out : scalar
    """
    size = a.size
    if size == 1:
        return a.config.backend.item(a._data)
    if size == 0:
        return 0
    raise YastError("Only single-element (symmetric) Tensor can be converted to scalar")
