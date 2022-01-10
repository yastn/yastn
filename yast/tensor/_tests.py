""" Testing and controls. """
import numpy as np
from ._auxliary import _flatten, _unpack_axes

__all__ = ['are_independent', 'is_consistent']


class YastError(Exception):
    """Errors cought by checks in yast."""


def _test_configs_match(a, b):
    """Check if config's of two tensors allow for performing operations mixing them. """
    if a.config.device != b.config.device:
        raise YastError('Devices of the two tensors do not match.')
    if a.config.sym.SYM_ID != b.config.sym.SYM_ID:
        raise YastError('Two tensors have different symmetry rules.')
    if a.config.fermionic != b.config.fermionic:
        raise YastError('Two tensors have different assigment of fermionic statistics.')
    if a.config.backend.BACKEND_ID != b.config.backend.BACKEND_ID:
        raise YastError('Two tensors have different backends.')


def _test_axes_match(a, b, sgn=1, axes=None):
    """
    Test if legs of a in axes[0] and legs ob b in axes[1] have matching signature and fusion structures.
    sgn in (-1, 1) is the sign of required match between signatures.

    Return meta-unfused axes and information if hard-fusion mask will be required.
    """

    if axes is None:
        if a.ndim != b.ndim:
            raise YastError('Tensors have different number of legs.')
        axes = (tuple(range(a.ndim)), tuple(range(b.ndim)))
        uaxes = (tuple(range(a.ndim_n)), tuple(range(b.ndim_n)))
    else:
        if axes is not None and len(axes[0]) != len(axes[1]):
            raise YastError('axes[0] and axes[1] indicated different number of legs.')
        if len(set(axes[0])) != len(axes[0]) or (len(set(axes[1])) != len(axes[1])):
            raise YastError('Repeated axis in axes[0] or axes[1].')
        if len(set(axes[0]) - set(range(a.ndim))) > 0 or len(set(axes[1]) - set(range(b.ndim))) > 0:
            raise YastError('Axis outside of tensor ndim.')
        ua, = _unpack_axes(a.meta_fusion, axes[0])
        ub, = _unpack_axes(b.meta_fusion, axes[1])
        uaxes = (ua, ub)

    if not all(a.struct.s[i1] == sgn * b.struct.s[i2] for i1, i2 in zip(*uaxes)):
        raise YastError('Signatures do not match.')

    if any(a.meta_fusion[i1] != b.meta_fusion[i2] for i1, i2 in zip(*axes)):
        raise YastError('Indicated axes of two tensors have different number of meta-fused legs or sub-fusions order.')

    needs_mask = False  # for hard-fused legs
    for i1, i2 in zip(*uaxes):
        if a.hard_fusion[i1].tree != b.hard_fusion[i2].tree:
            raise YastError('Indicated axes of two tensors have different number of hard-fused legs or sub-fusions order.')
        if any(s1 != sgn * s2 for s1, s2 in zip(a.hard_fusion[i1].s, b.hard_fusion[i2].s)):
            raise YastError('Signatures of hard-fused legs do not match.')
        if a.hard_fusion[i1].t != b.hard_fusion[i2].t or a.hard_fusion[i1].D != b.hard_fusion[i2].D:
            needs_mask = True
    return needs_mask, uaxes


def _test_axes_all(a, axes, native=False):
    axes = tuple(_flatten(axes))
    ndim = a.ndim_n if native else a.ndim
    if ndim != len(axes) or sorted(set(axes)) != list(range(ndim)):
        raise YastError('Provided axes do not match tensor ndim.')


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

    for i in range(len(a.struct.t) - 1):
        assert a.struct.t[i] < a.struct.t[i + 1]

    tset = np.array(a.struct.t, dtype=int).reshape((len(a.struct.t), len(a.struct.s), len(a.struct.n)))
    sa = np.array(a.struct.s, dtype=int)
    na = np.array(a.struct.n, dtype=int)
    assert np.all(a.config.sym.fuse(tset, sa, 1) == na), 'charges of some block do not satisfy symmetry condition'
    for n in range(a.ndim_n):
        a.get_leg_structure(n, native=True)
    device = {a.config.backend.get_device(x) for x in a.A.values()}
    for s, hf in zip(a.struct.s, a.hard_fusion):
        assert s == hf.s[0]
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
