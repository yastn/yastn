""" methods outputing data from yast tensor. """

import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _tarray, _Darray, _mf_to_ntree
from ._tests import YastError
from ..sym import sym_none


__all__ = ['compress_to_1d', 'export_to_dict', 'export_to_hdf5',
            'leg_structures_for_dense', 'requires_grad']


def export_to_dict(a):
    r"""
    Export relevant information about tensor to dictionary that can be saved using numpy.save

    Returns
    -------
    d: dict
        dictionary containing all the information needed to recreate the tensor.
    """
    AA = {ind: a.config.backend.to_numpy(a.A[ind]) for ind in a.A}
    out = {'A': AA, 's': a.struct.s, 'n': a.struct.n, 'isdiag': a.isdiag,
            'meta_fusion': a.meta_fusion, 'hard_fusion': a.hard_fusion}
    return out


def export_to_hdf5(a, file, path):
    """
    Export tensor into hdf5 type file.

    Parameters
    ----------
    ADD DESCRIPTION
    """
    vec, _ = a.compress_to_1d()
    file.create_dataset(path+'/isdiag', data=[int(a.isdiag)])
    file.create_group(path+'/meta/'+str(a.meta_fusion))
    file.create_dataset(path+'/n', data=a.struct.n)
    file.create_dataset(path+'/s', data=a.struct.s)
    file.create_dataset(path+'/ts', data=a.struct.t)
    file.create_dataset(path+'/Ds', data=a.struct.D)
    file.create_dataset(path+'/matrix', data=vec)


def compress_to_1d(a, meta=None):
    """
    Store each block as 1D array within r1d in contiguous manner; outputs meta-information to reconstruct the original tensor

    Parameters
    ----------
        meta: dict
            If not None, uses this metainformation to merge into 1d structure,
            filling-in zeros if tensor does not have some blocks.
            Raise error if tensor has some blocks which are not included in meta or otherwise meta does not match the tensor.
    """
    if meta is None:
        D_rsh = _Darray(a)[:, 0] if a.isdiag else np.prod(_Darray(a), axis=1)
        aD_rsh = np.cumsum(D_rsh)
        D_tot = np.sum(D_rsh)
        meta_new = (((),), (D_tot,))
        # meta_merge = ((tn, to, Dslc, Drsh), ...)
        meta_merge = tuple(((), t, ((aD - D, aD),), D) for t, D, aD in zip(a.struct.t, D_rsh, aD_rsh))
        # (told, tnew, Dslc, Dnew)
        DD = tuple((x[0],) for x in a.struct.D) if a.isdiag else a.struct.D
        meta_unmerge = tuple(((), t, (aD - D, aD), Dnew) for t, D, aD, Dnew in zip(a.struct.t, D_rsh, aD_rsh, DD))
        meta = {'s': a.struct.s, 'n': a.struct.n, 'isdiag': a.isdiag, 'hard_fusion': a.hard_fusion,
                'meta_fusion': a.meta_fusion, 'meta_unmerge': meta_unmerge, 'meta_merge': meta_merge}
    else:
        if a.struct.s != meta['s'] or a.struct.n != meta['n'] or a.isdiag != meta['isdiag'] \
            or a.meta_fusion != meta['meta_fusion'] or a.hard_fusion != meta['hard_fusion']:
            raise YastError("Tensor structure does not match provided metadata.")
        meta_merge = meta['meta_merge']
        D_tot = meta_merge[-1][2][0][1]
        meta_new = (((),), (D_tot,))
        if len(a.A) != sum(ind in a.A for (_, ind, _, _) in meta_merge):
            raise YastError("Tensor has blocks that do not appear in meta.")

    order = (0,) if a.isdiag else tuple(range(a.ndim_n))
    A = a.config.backend.merge_blocks(a.A, order, meta_new, meta_merge, a.config.device)
    return A[()], meta


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
    print("no. blocks  :", len(a.A))  # number of blocks
    print("size        :", a.size)  # total number of elements in all blocks
    mfs = {i: _mf_to_ntree(mf) for i, mf in enumerate(a.meta_fusion)}
    print("meta fusion :", mfs)  # encoding meta fusion tree for each leg
    hfs = {i: _mf_to_ntree(hf.tree) for i, hf in enumerate(a.hard_fusion)}
    print("hard fusion :", hfs, "\n")  # encoding info on hard fusion for each leg


def __str__(a):
    # return str(a.A)
    ts, Ds = a.get_leg_charges_and_dims(native=False)
    s = f"{a.config.sym.SYM_ID} s= {a.struct.s} n= {a.struct.n}\n"
    # s += f"charges      : {a.ts}\n"
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
    return a.config.backend.requires_grad(a.A)


def print_blocks_shape(a):
    """
    Print shapes of blocks as a sequence of block's charge followed by its shape 
    """
    for ind, x in a.A.items():
        print(f"{ind} {a.config.backend.get_shape(x)}")


def is_complex(a):
    """ 
    Returns
    -------
    bool : bool
        ``True`` if all of the blocks of the tensor are complex 
    """
    return all(a.config.backend.is_complex(x) for x in a.A.values())


def get_tensor_charge(a):
    """ 
    Returns
    -------
    n : int or tuple(int)
        total charge of the tensor. In case of direct product of abelian symmetries, total
        charges for each symmetry are returned in a tuple 
    """
    return a.struct.n


def get_signature(a, native=False):
    """ 
    Returns
    -------
    s : tuple
        Tensor signature. If native, returns the original signature.
        Otherwise the signature of the first native leg for each leg obtained by fusion is returned.
    """
    return a.s_n if native else a.s


def get_rank(a, native=False):
    """ 
    Returns
    -------
    n : int
        Tensor rank. If native is ``True``, the rank of the tensor with all legs unfused is returned. 
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
        axes = tuple(n for n in range(a.nlegs if native else a.mlegs))
    if isinstance(axes, int):
        return sum(a.get_leg_structure(axes, native=native).values())
    return tuple(sum(a.get_leg_structure(ii, native=native).values()) for ii in axes)


def unique_dtype(a):
    """ 
    Returns
    -------
    dtype : dtype or bool
        Returns common ``dtype`` if all blocks have the same type. Otherwise, returns False.
    """
    return a.config.backend.unique_dtype(a)


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
    """
    return a.A[key]


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
        return {'meta': a.meta_fusion, 'hard': a.hard_fusion}
    if isinstance(axes, int):
        return a.meta_fusion(axes)
    return {'meta': tuple(a.meta_fusion(n) for n in axes), 'hard': tuple(a.hard_fusion(n) for n in axes)}


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
        axis, = _unpack_axes(a.meta_fusion, axis)
    tset, Dset = _tarray(a), _Darray(a)
    tset = tset[:, axis, :]
    Dset = Dset[:, axis]
    tset = tset.reshape(len(tset), len(axis) * a.config.sym.NSYM)
    Dset = np.prod(Dset, axis=1) if len(axis) > 1 else Dset.reshape(-1)

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
    Create full tensor corresponding to the symmetric tensor.

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
    x = c.A[()] if not c.isdiag else c.config.backend.diag_create(c.A[()])
    return c.config.backend.clone(x)


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
    if len(a.A) == 1:
        key = next(iter(a.A))
        return a.A[key]
    raise YastError('Only tensor with a single block can be converted to raw tensor')


def to_nonsymmetric(a, leg_structures=None, native=False, reverse=False):
    r"""
    Create equivalent ``yast.Tensor`` with no explict symmetry (equivalent to single dense tensor).

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
    config_dense = a.config._replace(sym=sym_none)
    news = a.get_signature(native)
    c = a.__class__(config=config_dense, s=news, n=None, isdiag=a.isdiag)

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
        axes = tuple(_unpack_axes(a.meta_fusion, *axes))
    meta = []
    tset = _tarray(a)
    for tind, tt in zip(a.struct.t, tset):
        meta.append((tind, tuple(tD[n][tuple(tt[m, :].flat)] for n, m in enumerate(axes))))
    if a.isdiag:
        Dtot = Dtot[:1]
        meta = [(t, D[:1]) for t, D in meta]
    c.A[()] = a.config.backend.merge_to_dense(a.A, Dtot, meta, a.config.device)
    c.update_struct()
    return c


def zero_of_dtype(a):
    """ Return zero scalar of the instance specified by backend and dtype. """
    return a.config.backend.dtype_scalar(0, device=a.config.device)


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
        x = a.config.backend.first_element(next(iter(a.A.values())))
    elif size == 0:
        x = a.zero_of_dtype()  # is there a better solution for torch autograd?
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
        return a.config.backend.item(next(iter(a.A.values())))
    if size == 0:
        return 0
    raise YastError("only single-element (symmetric) Tensor can be converted to scalar")
