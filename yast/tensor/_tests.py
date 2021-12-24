""" Testing and controls. """
import numpy as np
from ._auxliary import _flatten, _tarray

__all__ = ['are_independent', 'is_consistent']


class YastError(Exception):
    """Errors cought by checks in yast."""


def _test_configs_match(a, b):
    """Check if config's of two tensors allow for performing operations mixing them. """
    if a.config.device != b.config.device:
        raise YastError('Devices of the two tensors do not match.')
    if a.config.sym.SYM_ID != b.config.sym.SYM_ID:
        raise YastError('Two tensors have different symmetries.')
    if a.config.fermionic != b.config.fermionic:
        raise YastError('Two tensors have different assigment of fermionic statistics.')
    if a.config.backend.BACKEND_ID != b.config.backend.BACKEND_ID:
        raise YastError('Two tensors have different backends.')


def _test_tensors_match(a, b):
    if a.struct.s != b.struct.s or a.struct.n != b.struct.n:
        raise YastError('Tensor signatures do not match')
    if a.meta_fusion != b.meta_fusion:
        raise YastError('Fusion trees do not match')
    # if a.hard_fusion != b.hard_fusion:
    #    raise YastError('Hard fusions of the two tensors do not match')


def _test_fusions_match(a, b):
    if a.meta_fusion != b.meta_fusion:
        raise YastError('Fusion trees do not match')


def _test_all_axes(a, axes, native=False):
    axes = tuple(_flatten(axes))
    ndim = a.ndim_n if native else a.ndim
    if ndim != len(axes):
        raise YastError('Wrong number of axis indices in axes')
    if sorted(set(axes)) != list(range(ndim)):
        raise YastError('Repeated axis index in axes')


def _test_hard_fusion_match(hf1, hf2, mconj):
    if hf1.t != hf2.t or hf1.D != hf2.D or hf1.tree != hf2.tree:
        raise YastError('Hard fusions do not match')
    if (mconj == 1 and hf1.s != hf2.s) or (mconj == -1 and hf1.s != hf2.ms):
        raise YastError('Hard fusions do not match. Singnature problem.')


def are_independent(a, b):
    """
    Test if all elements of two yast tensors are independent objects in memory.
    """
    test = []
    test.append(not a is b)
    test.append(not a.A is b.A)
    for key in a.A.keys():
        if key in b.A:
            test.append(a.config.backend.is_independent(a.A[key], b.A[key]))
    return all(test)


def is_consistent(a):
    """
    Test is yast tensor is does not contain inconsistent structures

    Check that:
    1) tset and Dset correspond to A
    2) tset follow symmetry rule f(s@t)==n
    3) block dimensions are consistent (this requires config.test=True)
    """
    for ind, D in zip(a.struct.t, a.struct.D):
        assert ind in a.A, 'index in struct.t not in dict A'
        d = a.config.backend.get_shape(a.A[ind])
        if a.isdiag:
            d = d + d
        assert d == D, 'block dimensions do not match struct.D'
    assert len(a.struct.t) == len(a.A), 'length of struct.t do not match dict A'
    assert len(a.struct.D) == len(a.A), 'length of struct.D do not match dict A'

    tset = _tarray(a)
    sa = np.array(a.struct.s, dtype=int)
    na = np.array(a.struct.n, dtype=int)
    assert np.all(a.config.sym.fuse(tset, sa, 1) == na), 'charges of some block do not satisfy symmetry condition'
    for n in range(a.ndim_n):
        a.get_leg_structure(n, native=True)
    device = {a.config.backend.get_device(x) for x in a.A.values()}
    for s, hf in zip(a.struct.s, a.hard_fusion):
        assert s == hf.s[0]
        assert s == -hf.ms[0]
    assert len(device) <= 1, 'not all blocks reside on the same device'
    if len(device) == 1:
        assert device.pop().startswith(a.config.device), 'device of blocks inconsistent with config.device'
    return True


def _get_tD_legs(struct):
    """ different views on struct.t and struct.D """
    tset = np.array(struct.t, dtype=int).reshape((len(struct.t), len(struct.s), len(struct.n)))
    Dset = np.array(struct.D, dtype=int).reshape((len(struct.t), len(struct.s)))
    tD_legs = [sorted(set((tuple(t.flat), D) for t, D in zip(tset[:, n, :], Dset[:, n]))) for n in range(len(struct.s))]
    tD_dict = [dict(tD) for tD in tD_legs]
    if any(len(x) != len(y) for x, y in zip(tD_legs, tD_dict)):
        raise YastError('Bond dimensions related to some charge are not consistent.')
    tlegs = [tuple(tD.keys()) for tD in tD_dict]
    Dlegs= [tuple(tD.values()) for tD in tD_dict]
    return tlegs, Dlegs, tD_dict, tset, Dset
