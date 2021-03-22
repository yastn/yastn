""" Methods creating a new yast tensor """
from itertools import chain, repeat, accumulate
import numpy as np
from .core import Tensor, YastError
from ._auxliary import _clear_axes, _unpack_axes

__all__ = ['rand', 'randR', 'zeros', 'ones', 'eye', 'import_from_dict', 'decompress_from_1d', 'match_legs', 'block']


def rand(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with the random numbers in [-1, 1] and type specified in config.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal

    Returns
    -------
    tensor : tensor
        a random instance of a tensor
    """
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, **kwargs)
    a.fill_tensor(t=t, D=D, val='rand')
    return a


def randR(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with real random numbers in [-1, 1].

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal

    Returns
    -------
    tensor : tensor
        a random instance of a tensor
    """
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, **kwargs)
    a.fill_tensor(t=t, D=D, val='randR')
    return a


def zeros(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with zeros.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions
    isdiag : bool
        makes tensor diagonal

    Returns
    -------
    tensor : tensor
        an instance of a tensor filled with zeros
    """
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, **kwargs)
    a.fill_tensor(t=t, D=D, val='zeros')
    return a


def ones(config=None, s=(), n=None, t=(), D=(), isdiag=False, **kwargs):
    r"""
    Initialize tensor with all possible blocks filled with ones.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    s : tuple
        a signature of tensor
    n : int
        total charge
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions

    Returns
    -------
    tensor : tensor
        an instance of a tensor filled with ones
    """
    a = Tensor(config=config, s=s, n=n, isdiag=isdiag, **kwargs)
    a.fill_tensor(t=t, D=D, val='ones')
    return a


def eye(config=None, t=(), D=(), **kwargs):
    r"""
    Initialize diagonal tensor with all possible blocks filled with ones.

    Initialize tensor and call :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    t : list
        a list of charges for each leg, see :meth:`Tensor.fill_tensor` for description.
    D : list
        a list of corresponding bond dimensions

    Returns
    -------
    tensor : tensor
        an instance of diagonal tensor filled with ones
    """
    a = Tensor(config=config, isdiag=True, **kwargs)
    a.fill_tensor(t=t, D=D, val='ones')
    return a


def import_from_dict(config=None, d=None):
    """
    Generate tensor based on information in dictionary d.

    Parameters
    ----------
    config: module
            configuration with backend, symmetry, etc.

    d : dict
        information about tensor stored with :meth:`Tensor.to_dict`
    """
    if d is not None:
        a = Tensor(config=config, **d)
        for ind in d['A']:
            a.set_block(ts=ind, Ds=d['A'][ind].shape, val=d['A'][ind])
        return a
    raise YastError("Dictionary d is required.")


def decompress_from_1d(r1d, config, meta):
    """
    Generate tensor based on information in dictionary d and 1D array
    r1d containing the serialized blocks

    Parameters
    ----------
    config: module
            configuration with backend, symmetry, etc.

    d : dict
        information about tensor stored with :meth:`Tensor.to_dict`
    """
    a = Tensor(config=config, **meta)
    A = {(): r1d}
    a.A = a.config.backend.unmerge_one_leg(A, 0, meta['meta_unmerge'])
    a.update_struct()
    return a


def match_legs(tensors=None, legs=None, conjs=None, val='ones', n=None, isdiag=False):
    r"""
    Initialize tensor matching legs of existing tensors, so that it can be contracted with those tensors.

    Finds all matching symmetry sectors and their bond dimensions and passes it to :meth:`Tensor.fill_tensor`.

    Parameters
    ----------
    tensors: list
        list of tensors -- they should not be diagonal to properly identify signature.
    legs: list
        and their corresponding legs to match
    conjs: list
        if tensors are entering dot as conjugated
    val: str
        'randR', 'rand', 'ones', 'zeros'
    """
    t, D, s, lf = [], [], [], []
    if conjs is None:
        conjs = (0,) * len(tensors)
    for nf, te, cc in zip(legs, tensors, conjs):
        lf.append(te.meta_fusion[nf])
        un, = _unpack_axes(te, (nf,))
        for nn in un:
            tdn = te.get_leg_structure(nn, native=True)
            t.append(tuple(tdn.keys()))
            D.append(tuple(tdn.values()))
            s.append(te.s[nn] * (2 * cc - 1))
    a = Tensor(config=tensors[0].config, s=s, n=n, isdiag=isdiag, meta_fusion=lf)
    a.fill_tensor(t=t, D=D, val=val)
    return a


def block(tensors, common_legs=None):
    """ Assemble new tensor by blocking a set of tensors.

        Parameters
        ----------
        tensors : dict
            dictionary of tensors {(x,y,...): tensor at position x,y,.. in the new, blocked super-tensor}.
            Length of tuple should be equall to tensor.ndim - len(common_legs)

        common_legs : list
            Legs that are not blocked
            (equivalently on common legs all tensors have the same position in the supertensor, and those positions are not given in tensors)

    """
    out_s, = ((),) if common_legs is None else _clear_axes(common_legs)
    tn0 = next(iter(tensors.values()))  # first tensor; used to initialize new objects and retrive common values
    out_b = tuple((ii,) for ii in range(tn0.mlegs) if ii not in out_s)
    pos = list(_clear_axes(*tensors))
    lind = tn0.mlegs - len(out_s)
    for ind in pos:
        if len(ind) != lind:
            raise YastError('Wrong number of coordinates encoded in tensors.keys()')

    out_s, =  _unpack_axes(tn0, out_s)
    u_b = tuple(_unpack_axes(tn0, *out_b))
    out_b = tuple(chain(*u_b))
    pos = tuple(tuple(chain.from_iterable(repeat(x, len(u)) for x, u in zip(ind, u_b))) for ind in pos)

    for ind, tn in tensors.items():
        ind, = _clear_axes(ind)
        if tn.nlegs != tn0.nlegs or tn.meta_fusion != tn0.meta_fusion or\
           not np.all(tn.s == tn0.s) or not np.all(tn.n == tn0.n) or\
           tn.isdiag != tn0.isdiag :
            raise YastError('Ndims, signatures, total charges or fusion trees of blocked tensors are inconsistent.')

    posa = np.ones((len(pos), tn0.nlegs), dtype=int)
    posa[:, np.array(out_b, dtype=np.intp)] = np.array(pos, dtype=int).reshape(len(pos), -1)

    tDs = []  # {leg: {charge: {position: D, 'D' : Dtotal}}}
    for n in range(tn0.nlegs):
        tDl = {}
        for tn, pp in zip(tensors.values(), posa):
            tDn = tn.get_leg_structure(n, native=True)
            for t, D in tDn.items():
                if t in tDl:
                    if (pp[n] in tDl[t]) and (tDl[t][pp[n]] != D):
                        raise YastError('Dimensions of blocked tensors are not consistent')
                    tDl[t][pp[n]] = D
                else:
                    tDl[t] = {pp[n]: D}
        for t, pD in tDl.items():
            ps = sorted(pD.keys())
            Ds = [pD[p] for p in ps]
            tDl[t] = {p: (aD-D, aD)  for p, D, aD in zip(ps, Ds, accumulate(Ds))}
            tDl[t]['Dtot'] = sum(Ds)
        tDs.append(tDl)

    # all unique blocks
    # meta_new = {tind: Dtot};  #meta_block = [(tind, pos, Dslc)]
    meta_new, meta_block = {}, []
    for pind, pa in zip(tensors, posa):
        a = tensors[pind]
        tset = a._tarray()
        for tind, t in zip(a.struct.t, tset):
            if tind not in meta_new:
                meta_new[tind] = tuple(tDs[n][tuple(t[n].flat)]['Dtot'] for n in range(a.nlegs))
            meta_block.append((tind, pind, tuple(tDs[n][tuple(t[n].flat)][pa[n]] for n in range(a.nlegs))))
    meta_new = tuple((ts, Ds) for ts, Ds in meta_new.items())

    c = Tensor(config=a.config, s=a.s, isdiag=a.isdiag, n=a.n, meta_fusion=tn0.meta_fusion)
    c.A = c.config.backend.merge_super_blocks(tensors, meta_new, meta_block, a.config.dtype, c.config.device)
    c.update_struct()
    return c
