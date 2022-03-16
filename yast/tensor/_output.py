""" methods outputing data from yast tensor. """

import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _mf_to_ntree, _struct, _flatten
from ._tests import YastError
from ..sym import sym_none
from ._legs import Leg


__all__ = ['compress_to_1d', 'save_to_dict', 'save_to_hdf5',
            'leg_structures_for_dense', 'requires_grad']


def save_to_dict(a):
    r"""
    Export relevant information about tensor to dictionary. Such dictionary can be then
    stored in file using i.e. numpy.save

    Returns
    -------
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
    ADD DESCRIPTION
    """
    _d = a.config.backend.to_numpy(a._data)
    file.create_dataset(path+'/isdiag', data=[int(a.isdiag)])
    file.create_group(path+'/meta/'+str(a.mfs))
    file.create_dataset(path+'/n', data=a.struct.n)
    file.create_dataset(path+'/s', data=a.struct.s)
    file.create_dataset(path+'/ts', data=a.struct.t)
    file.create_dataset(path+'/Ds', data=a.struct.D)
    file.create_dataset(path+'/matrix', data=_d)


def compress_to_1d(a, meta=None):
    """
    Store each block as 1D array within r1d in contiguous manner (do not clone the data if not necceaary);
    outputs meta-information to reconstruct the original tensor

    Parameters
    ----------
        meta: dict
            If not None, uses this metainformation to merge into 1d structure,
            filling-in zeros if tensor does not have some blocks.
            Raise error if tensor has some blocks which are not included in meta or otherwise
            meta does not match the tensor.
    """
    if meta is None:
        meta = {'struct': a.struct, 'isdiag': a.isdiag, 'hfs': a.hfs,
                'mfs': a.mfs}
        return a._data, meta
    # else:
    if a.struct.s != meta['struct'].s:
        raise YastError("Tensor has different signature than metadata.")
    if a.struct.n != meta['struct'].n:
        raise YastError("Tensor has different tensor charge than metadata.")
    if a.isdiag != meta['isdiag']:
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
        elif a.struct.t[ia] == meta['struct'].t[im]:
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
    ts, Ds = a.get_leg_charges_and_dims(native=False)
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
    n : int or tuple(int)
        see :attr:`yast.Tensor.n`
    """
    return a.struct.n


def get_signature(a, native=False):
    """
    Returns
    -------
    s : tuple
        Tensor signature, equivalent to :attr:`yast.Tensor.s`.
        If native, returns the signature of tensors's native legs, see :attr:`yast.Tensor.s_n`.
    """
    return a.s_n if native else a.s


def get_rank(a, native=False):
    """
    Returns
    -------
    n : int
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


def get_shape(a, axes=None, native=False):
    r"""
    Return total bond dimension, sum of dimensions along sectors, of meta legs.

    Parameters
    ----------
    axes : Int or tuple of ints
        indices of legs; If axes is None returns all (default).

    Returns
    -------
    shape : Int or tuple(int)
        shapes of legs specified by axes
    """
    if axes is None:
        axes = tuple(n for n in range(a.ndim_n if native else a.ndim))
    if isinstance(axes, int):
        return sum(a.get_leg_structure(axes, native=native).values())
    return tuple(sum(a.get_leg_structure(ii, native=native).values()) for ii in axes)


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
    key : tuple(int)
        charges of the block

    Returns
    -------
    out : tensor
        The type of the returned tensor depends on the backend, i.e. ``numpy.ndarray`` or ``torch.tensor``.
        Output 1d array for diagonal tensor, outherwise reshape into n-dim array.
    """
    key = tuple(_flatten(key))
    try:
        ind = a.struct.t.index(key)
    except ValueError:
        raise YastError('tensor does not have block specify by key')
    x = a._data[slice(*a.struct.sl[ind])]
    
    return x if a.isdiag else x.reshape(a.struct.D[ind])


##################################################
#    output tensors info - advanced structure    #
##################################################

def get_leg_fusion(a, axes=None):
    """
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


def get_leg_structure2(a, axis, native=False):
    r"""
    Find all charges and the corresponding bond dimension for n-th leg.

    Parameters
    ----------
    axis : int
        Index of a leg.

    native : bool
        consider native legs if True; otherwise meta/fused legs (default).

    Returns
    -------
        tDn : dict of {tn: Dn}
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
    for tn, Dn in zip(tset, Dset):
        if tDn[tuple(tn.flat)] != Dn:
            raise YastError('Inconsistend bond dimension of charge.')
    t, D = tuple(tDn.keys()), tuple(tDn.values())
    return Leg(s=a.s[axis[0]], t=t, D=D)


def get_leg_structure(a, axis, native=False):
    r"""
    Find all charges and the corresponding bond dimension for n-th leg.

    Parameters
    ----------
    axis : int
        Index of a leg.

    native : bool
        consider native legs if True; otherwise meta/fused legs (default).

    Returns
    -------
        tDn : dict of {tn: Dn}
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
    for tn, Dn in zip(tset, Dset):
        if tDn[tuple(tn.flat)] != Dn:
            raise YastError('Inconsistend bond dimension of charge.')
    return tDn


def get_leg_charges_and_dims(a, native=False):
    """ collect information about charges and dimensions on all legs into two lists. """
    _tmp = [a.get_leg_structure(n, native=native) for n in range(a.ndim_n if native else a.ndim)]
    _tmp = [{k: lst[k] for k in sorted(lst)} for lst in _tmp]
    ts, Ds = tuple(zip(*[tuple(zip(*lst.items())) for lst in _tmp]))
    return ts, Ds


def leg_structures_for_dense(tensors=(), native=False, leg_structures=None):
    r"""
    Combine and output charges and bond dimensions from legs of provided tensors.
    Auxliary function to ```tensor.to_dense``` and ```tensor.to_numpy```,
    to create dense tensors with consistent dimensions (charge sectors)

    Rises exception if there are some inconsistencies in bond dimensions.

    Parameters
    ----------
    tensors : list
        [reference_tensor, {leg of reference_tensor: leg of tensor to be made dense}]
        If dict not present, assumes {n: n for n in reference_tensor.ndim_n}

    native: bool
        output data for native tensor (neglecting meta fusions).
    """
    lss = {}
    itensors = iter(tensors)
    a = next(itensors, None)
    while a is not None:
        b = next(itensors, None)
        if isinstance(b, dict):
            for la, lo in b.items():
                if lo in lss:
                    lss[lo].append(a.get_leg_structure(la, native=native))
                else:
                    lss[lo] = [a.get_leg_structure(la, native=native)]
            a = next(itensors, None)
        else:
            for n in range(a.ndim_n if native else a.ndim):
                if n in lss:
                    lss[n].append(a.get_leg_structure(n, native=native))
                else:
                    lss[n] = [a.get_leg_structure(n, native=native)]
            a = b

    if leg_structures is not None:
        for n, ls in leg_structures.items():
            if n in lss:
                lss[n].append(ls)
            else:
                lss[n] = [ls]

    for lo in lss:
        lss[lo] = leg_structures_union(*lss[lo])
    return lss


def leg_structures_union(*args):
    """
    Makes a union of leg structures {t: D} specified in args.

    Raise error if there are inconsistencies.
    """
    ls_out = {}
    len_t = -1
    for tD in args:
        for t, D in tD.items():
            if (t in ls_out) and ls_out[t] != D:
                raise YastError(f'Bond dimensions for charge {t} are inconsistent.')
            ls_out[t] = D
            if len(t) != len_t and len_t >= 0:
                raise YastError('Inconsistent charge structure. Likely mixing merged and native legs.')
            len_t = len(t)
    return ls_out


############################
#   Down-casting tensors   #
############################

def to_dense(a, leg_structures=None, native=False, reverse=False):
    r"""
    Create dense tensor corresponding to the symmetric tensor.

    Blocks are ordered according to increasing charges on each leg.
    It is possible to supply a list of additional charge sectors with dimensions to be included.
    (should be consistent with the tensor). This allows to fill in some explicit zero blocks.

    Parameters
    ----------
    leg_structures : dict
        {n: {tn: Dn}} specify charges and dimensions to include on some legs (indicated by keys n).

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
    c = a.to_nonsymmetric(leg_structures, native, reverse)
    x = c.config.backend.clone(c._data)
    x = c.config.backend.diag_create(x) if c.isdiag else x.reshape(c.struct.D[0])
    return x


def to_numpy(a, leg_structures=None, native=False, reverse=False):
    r"""
    Create full ``numpy.ndarray`` corresponding to the symmetric tensor. See :func:`yast.to_dense`

    Returns
    -------
    out : numpy.ndarray
        NumPy array equivalent to symmetric tensor
    """
    return a.config.backend.to_numpy(a.to_dense(leg_structures, native, reverse))


def to_raw_tensor(a):
    """
    If the symmetric tensor has just a single non-empty block, return raw tensor representing
    that block.

    Returns
    -------
    out : tensor
        The type of the returned tensor depends on the backend, i.e. ``numpy.ndarray`` or ``torch.tensor``.
    """
    if len(a.struct.D) == 1:
        return a._data.reshape(a.struct.D[0])
    raise YastError('Only tensor with a single block can be converted to raw tensor')


def to_nonsymmetric(a, leg_structures=None, native=False, reverse=False):
    r"""
    Create equivalent ``yast.Tensor`` with no explict symmetry. All blocks of the original
    tensor are accummulated into a single block.

    Blocks are ordered according to increasing charges on each leg.
    It is possible to supply a list of additional charge sectors with dimensions to be included.
    (should be consistent with the tensor). This allows to fill in some explicit zero blocks.

    .. note::
        yast structure can be redundant since resulting tensor is effectively just
        a single dense block. If that's the case, use :meth:`yast.Tensor.to_dense`.

    Parameters
    ----------
    leg_structures : dict
        {n: {tn: Dn}} specify charges and dimensions to include on some legs (indicated by keys n).

    native: bool
        output native tensor (ignoring fusion of legs).

    reverse: bool
        reverse the order in which blocks are sorted. Default order is ascending in
        values of block's charges.

    Returns
    -------
    out : Tensor
        the config of returned tensor does not use any symmetry
    """
    config_dense = a.config._replace(sym=sym_none)

    ndim = a.ndim_n if native else a.ndim
    tD = [a.get_leg_structure(n, native=native) for n in range(ndim)]
    if leg_structures is not None:
        for n, tDn in leg_structures.items():
            if (n < 0) or (n >= ndim):
                raise YastError('Specified leg out of ndim')
            tD[n] = leg_structures_union(tD[n], tDn)
    Dtot = tuple(sum(tDn.values()) for tDn in tD)
    for tDn in tD:
        tns = sorted(tDn.keys(), reverse=reverse)
        Dlow = 0
        for tn in tns:
            Dhigh = Dlow + tDn[tn]
            tDn[tn] = (Dlow, Dhigh)
            Dlow = Dhigh
    axes = tuple((n,) for n in range(ndim))
    if not native:
        axes = tuple(_unpack_axes(a.mfs, *axes))
    meta = []
    tset = np.array(a.struct.t, dtype=int).reshape((len(a.struct.t), len(a.struct.s), len(a.struct.n)))
    for t_sl, tt in zip(a.struct.sl, tset):
        meta.append((slice(*t_sl), tuple(tD[n][tuple(tt[m, :].flat)] for n, m in enumerate(axes))))
    if a.isdiag:
        Dtot = Dtot[:1]
        meta = [(sl, D[:1]) for sl, D in meta]

    c_s = a.get_signature(native)
    c_t = ((),)
    c_D = (Dtot,) if not a.isdiag else (Dtot + Dtot,)
    Dp = np.prod(Dtot, dtype=int)
    c_Dp = (Dp,)
    c_sl = ((0, Dp),)
    c_struct = _struct(t=c_t, D=c_D, s=c_s, n=(), Dp=c_Dp, sl=c_sl)
    data = a.config.backend.merge_to_dense(a._data, Dtot, meta)
    return a._replace(config=config_dense, struct=c_struct, data=data, mfs=None, hfs=None)


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
        raise YastError('Specified bond dimensions inconsistent with tensor.')
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
    raise YastError("only single-element (symmetric) Tensor can be converted to scalar")
