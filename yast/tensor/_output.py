import numpy as np
from ._auxliary import _clear_axes, _unpack_axes, _tarray, _Darray, _tDarrays
from ._testing import YastError, _check
from ..sym import sym_none


__all__ = ['export_to_dict','compress_to_1d']


#######################
#    export tensor    #
#######################

def export_to_dict(a):
    r"""
    Export relevant information about tensor to dictionary --  it can be saved using numpy.save

    Returns
    -------
    d: dict
        dictionary containing all the information needed to recreate the tensor.
    """
    AA = {ind: a.config.backend.to_numpy(a.A[ind]) for ind in a.A}
    if a.isdiag:
        AA = {t: np.diag(x) for t, x in AA.items()}
    out = {'A': AA, 's': tuple(a.s), 'n': tuple(a.n), 'isdiag': a.isdiag, 'meta_fusion': a.meta_fusion}
    return out

def compress_to_1d(a, meta=None):
    """
    Store each block as 1D array within r1d in contiguous manner; outputs meta-information to reconstruct the original tensor

    Parameters
    ----------
        meta: dict
            If not None, uses this metainformation to merge into 1d structure (filling-in zeros if tensor does not have some blocks).
            Raise error, if tensor has some blocks which are not included in meta; or otherwise meta does not match the tensor.
    """
    if meta is None:
        D_rsh = np.prod(_Darray(a), axis=1)
        aD_rsh = np.cumsum(D_rsh)
        D_tot = np.sum(D_rsh)
        meta_new = (((), D_tot),)
        # meta_merge = ((tn, Ds, to, Do), ...)
        meta_merge = tuple(((), (aD - D, aD), t, (D,)) for t, D, aD in zip(a.struct.t, D_rsh, aD_rsh))
        # (told, tnew, Dsl, Dnew)
        meta_unmerge = tuple((told, tnew, Dsl, Dnew) for (told, Dsl, tnew, _), Dnew in zip(meta_merge, a.struct.D))
        meta = {'s': tuple(a.s), 'n': tuple(a.n), 'isdiag': a.isdiag,
                'meta_fusion': a.meta_fusion, 'meta_unmerge': meta_unmerge, 'meta_merge': meta_merge}
    else:
        if tuple(a.s) != meta['s'] or tuple(a.n) != meta['n'] or a.isdiag != meta['isdiag'] or a.meta_fusion != meta['meta_fusion']:
            raise YastError("Tensor do not match provided metadata.")
        meta_merge = meta['meta_merge']
        D_tot = meta_merge[-1][1][1]
        meta_new = (((), D_tot),)
        if len(a.A) != sum(ind in a.A for (_, _, ind, _) in meta_merge):
            raise YastError("Tensor has blocks that do not appear in meta.")

    A = a.config.backend.merge_one_leg(a.A, 0, tuple(range(a.nlegs)),
                                            meta_new, meta_merge, a.config.dtype, a.config.device)
    return A[()], meta

############################
#    output information    #
############################

def show_properties(a):
    """ Display basic properties of the tensor. """
    print("Symmetry    :", a.config.sym.name)
    print("signature   :", a.s)  # signature
    print("charge      :", a.n)  # total charge of tensor
    print("isdiag      :", a.isdiag)
    print("dim meta    :", a.mlegs)  # number of meta legs
    print("dim native  :", a.nlegs)  # number of native legs
    print("shape meta  :", a.get_shape(native=False))
    print("shape native:", a.get_shape(native=True))
    print("no. blocks  :", len(a.A))  # number of blocks
    print("size        :", a.get_size())  # total number of elements in all blocks
    print("meta fusion :", a.meta_fusion, "\n")  # encoding meta fusion tree for each leg

def __str__(a):
    # return str(a.A)
    ts, Ds= a.get_leg_charges_and_dims(native=False)
    s = f"{a.config.sym.name} s= {a.s} n= {a.n}\n"
    # s+=f"charges      : {a.ts}\n"
    s += f"leg charges  : {ts}\n"
    s += f"dimensions   : {Ds}"
    return s

def print_blocks(a):
    """ print shapes of blocks """
    for ind, x in a.A.items():
        print(f"{ind} {a.config.backend.get_shape(x)}")

def is_complex(a):
    """ Returns True if all blocks are complex. """
    return all([a.config.backend.is_complex(x) for x in a.A.values()])

def get_size(a):
    """ Total number of elements in the tensor. """
    return sum(np.prod(_Darray(a), axis=1))

def get_tensor_charge(a):
    """ Global charge of the tensor. """
    return tuple(a.n)

def get_signature(a, native=False):
    """ Tensor signatures. If not native, returns the signature of the first leg in each group."""
    if native:
        return tuple(a.s)
    pn = tuple((n,) for n in range(a.mlegs)) if a.mlegs > 0 else ()
    un = tuple(_unpack_axes(a, *pn))
    return tuple(a.s[p[0]] for p in un)

def get_blocks_charges(a):
    """ Charges of all native blocks. """
    return a.struct.t

def get_blocks_shapes(a):
    """ Shapes fo all native blocks. """
    return a.struct.D

def get_leg_fusion(a, axes=None):
    """
    Fusion trees for meta legs.

    Parameters
    ----------
    axes : Int or tuple of ints
        indices of legs; If axes is None returns all (default).
    """
    if axes is None:
        return a.meta_fusion
    if isinstance(axes, int):
        return a.meta_fusion(axes)
    return tuple(a.meta_fusion(n) for n in axes)

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
        axis, = _unpack_axes(a, axis)
    tset, Dset = _tDarrays(a)
    tset = tset[:, axis, :]
    Dset = Dset[:, axis]
    tset = tset.reshape(len(tset), len(axis) * a.config.sym.nsym)
    Dset = np.prod(Dset, axis=1) if len(axis) > 1 else Dset.reshape(-1)

    tDn = {tuple(tn.flat): Dn for tn, Dn in zip(tset, Dset)}
    if _check["consistency"]:
        for tn, Dn in zip(tset, Dset):
            if tDn[tuple(tn.flat)] != Dn:
                raise YastError('Inconsistend bond dimension of charge.')
    return tDn

def get_leg_charges_and_dims(a, native=False):
    """ collect information about charges and dimensions on all legs into two lists. """
    _tmp = [a.get_leg_structure(n, native=native) for n in range(a.get_ndim(native))]
    ts, Ds = tuple(zip(*[tuple(zip(*lst.items())) for lst in _tmp]))
    return ts, Ds

def get_shape(a, axes=None, native=False):
    r"""
    Return total bond dimension of meta legs.

    Parameters
    ----------
    axes : Int or tuple of ints
        indices of legs; If axes is None returns all (default).

    Returns
    -------
    shape : Int or tuple of ints
        shapes of legs specified by axes
    """
    if axes is None:
        axes = tuple(n for n in range(a.nlegs if native else a.mlegs))
    if isinstance(axes, int):
        return sum(a.get_leg_structure(axes, native=native).values())
    return tuple(sum(a.get_leg_structure(ii, native=native).values()) for ii in axes)

def get_ndim(a, native=False):
    """ Number of: meta legs if not native else native legs. """
    return a.nlegs if native else a.mlegs

def __getitem__(a, key):
    """ Returns block based on its charges. """
    return a.A[key]

#########################
#    output tensors     #
#########################

def to_dense(a, leg_structures=None, native=False):
    r"""
    Create full tensor corresponding to the symmetric tensor.

    Blockes are ordered according to increasing charges on each leg.
    It is possible to supply a list of charges and bond dimensions to be included
    (should be consistent with the tensor). This allows to fill in some zero blocks.

    Parameters
    ----------
    leg_structures : dict
        {n: {tn: Dn}} specify charges and dimensions to include on some legs (indicated by keys n).

    native: bool
        output native tensor (neglecting meta fusions).

    Returns
    -------
    out : tensor of the type used by backend
    """
    nlegs = a.get_ndim(native=native)
    tD = [a.get_leg_structure(n, native=native) for n in range(nlegs)]
    if leg_structures is not None:
        for n, tDn in leg_structures.items():
            if (n < 0) or (n >= nlegs):
                raise YastError('Specified leg out of ndim')
            for tn, Dn in tDn.items():
                if (tn in tD[n]) and tD[n][tn] != Dn:
                    raise YastError('Specified bond dimensions inconsistent with tensor.')
                tD[n][tn] = Dn
    Dtot = [sum(tDn.values()) for tDn in tD]
    for tDn in tD:
        tns = sorted(tDn.keys())
        Dlow = 0
        for tn in tns:
            Dhigh = Dlow + tDn[tn]
            tDn[tn] = (Dlow, Dhigh)
            Dlow = Dhigh
    axes = tuple((n,) for n in range(nlegs))
    if not native:
        axes = tuple(_unpack_axes(a, *axes))
    meta = []
    tset = _tarray(a)
    for tind, tt in zip(a.struct.t, tset):
        meta.append((tind, tuple(tD[n][tuple(tt[m, :].flat)] for n, m in enumerate(axes))))
    return a.config.backend.merge_to_dense(a.A, Dtot, meta, a.config.dtype, a.config.device)

def to_numpy(a, leg_structures=None, native=False):
    r"""
    Create full nparray corresponding to the symmetric tensor. See `yast.to_dense`
    """
    return a.config.backend.to_numpy(a.to_dense(leg_structures, native))

def to_raw_tensor(a):
    """
    For tensor with a single block, return raw tensor representing that block.
    """
    if len(a.A) == 1:
        key = next(iter(a.A))
        return a.A[key]
    raise YastError('Only tensor with a single block can be converted to raw tensor')

def to_nonsymmetric(a, leg_structures=None, native=False):
    r"""
    Create full tensor corresponding to the symmetric tensor. Output it as yast tensor with no symmetry.

    Blockes are ordered according to increasing charges on each leg.
    It is possible to supply a list of charges and bond dimensions to be included
    (should be consistent with the tensor). This allows to fill in some zero blocks.

    Parameters
    ----------
    leg_structures : dict
        {n: {tn: Dn}} specify charges and dimensions to include on some legs (indicated by keys n).

    native: bool
        output native tensor (neglecting meta fusions).

    Returns
    -------
    out : tensor of the type used by backend
    """
    config_dense = a.config._replace(sym=sym_none)
    news = a.get_signature(native)
    T = a.__class__(config=config_dense, s=news, n=None, isdiag=a.isdiag)
    T.set_block(val=a.to_dense(leg_structures, native))
    return T

#########################
#    output numbers     #
#########################

def zero_of_dtype(a):
    return a.config.backend.zero_scalar(dtype=a.config.dtype, device=a.config.device)

def to_number(a):
    """
    Return an element of the size-one tensor as a scalar of the same type as the
    type use by backend.

    For empty tensor, returns 0
    """
    size = a.get_size()
    if size == 1:
        return a.config.backend.first_element(next(iter(a.A.values())))
    if size == 0:
        return a.zero_of_dtype()
        # is there a better solution for torch autograd?
    raise YastError('Specified bond dimensions inconsistent with tensor.')

def item(a):
    """
    Return an element of the size-one tensor as a standard Python scalar.

    For empty tensor, returns 0
    """
    size = a.get_size()
    if size == 1:
        return a.config.backend.item(next(iter(a.A.values())))
    if size == 0:
        return 0
    raise YastError("only single-element (symmetric) Tensor can be converted to scalar")
