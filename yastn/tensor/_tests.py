""" Testing and controls. """
import numpy as np
from ._auxliary import _flatten, _unpack_axes

__all__ = ['are_independent', 'is_consistent']


class YastnError(Exception):
    """Errors cought by checks in yastn."""


def _test_can_be_combined(a, b):
    """Check if config's of two tensors allow for performing operations mixing them. """
    if a.device != b.device:
        raise YastnError('Devices of the two tensors do not match.')
    _test_configs_match(a.config, b.config)


def _test_configs_match(a_config, b_config):
    if a_config.sym.SYM_ID != b_config.sym.SYM_ID:
        raise YastnError('Two tensors have different symmetry rules.')
    if a_config.fermionic != b_config.fermionic:
        raise YastnError('Two tensors have different assigment of fermionic statistics.')
    if a_config.backend.BACKEND_ID != b_config.backend.BACKEND_ID:
        raise YastnError('Two tensors have different backends.')


def _test_tD_consistency(struct):
    tset = np.array(struct.t, dtype=int).reshape((len(struct.t), len(struct.s), len(struct.n)))
    Dset = np.array(struct.D, dtype=int).reshape((len(struct.D), len(struct.s)))
    for i in range(len(struct.s)):
        ti = [tuple(x.flat) for x in tset[:, i, :].reshape(len(tset), len(struct.n))]
        Di = Dset[:, i].reshape(-1)
        tDi = list(zip(ti, Di))
        if len(set(ti)) != len(set(tDi)):
            raise YastnError('Inconsist assigment of bond dimension to some charge.')


def _test_axes_match(a, b, sgn=1, axes=None):
    """
    Test if legs of a in axes[0] and legs ob b in axes[1] have matching signature and fusion structures.
    sgn in (-1, 1) is the sign of required match between signatures.

    Return meta-unfused axes and information if hard-fusion mask will be required.
    """

    if axes is None:
        if a.ndim != b.ndim:
            raise YastnError('Tensors have different number of legs.')
        axes = (tuple(range(a.ndim)), tuple(range(b.ndim)))
        uaxes = (tuple(range(a.ndim_n)), tuple(range(b.ndim_n)))
    else:
        if axes is not None and len(axes[0]) != len(axes[1]):
            raise YastnError('axes[0] and axes[1] indicated different number of legs.')
        if len(set(axes[0])) != len(axes[0]) or (len(set(axes[1])) != len(axes[1])):
            raise YastnError('Repeated axis in axes[0] or axes[1].')
        if len(set(axes[0]) - set(range(a.ndim))) > 0 or len(set(axes[1]) - set(range(b.ndim))) > 0:
            raise YastnError('Axis outside of tensor ndim.')
        ua, = _unpack_axes(a.mfs, axes[0])
        ub, = _unpack_axes(b.mfs, axes[1])
        uaxes = (ua, ub)

    if not all(a.struct.s[i1] == sgn * b.struct.s[i2] for i1, i2 in zip(*uaxes)):
        raise YastnError('Signatures do not match.')

    if any(a.mfs[i1] != b.mfs[i2] for i1, i2 in zip(*axes)):
        raise YastnError('Indicated axes of two tensors have different number of meta-fused legs or sub-fusions order.')

    needs_mask = False  # for hard-fused legs
    for i1, i2 in zip(*uaxes):
        if a.hfs[i1].tree != b.hfs[i2].tree or a.hfs[i1].op != b.hfs[i2].op:
            raise YastnError('Indicated axes of two tensors have different number of hard-fused legs or sub-fusions order.')
        if any(s1 != sgn * s2 for s1, s2 in zip(a.hfs[i1].s, b.hfs[i2].s)):
            raise YastnError('Signatures of hard-fused legs do not match.')
        if a.hfs[i1].t != b.hfs[i2].t or a.hfs[i1].D != b.hfs[i2].D:
            needs_mask = True
    return needs_mask, uaxes


def _test_axes_all(a, axes, native=False):
    axes = tuple(_flatten(axes))
    ndim = a.ndim_n if native else a.ndim
    if ndim != len(axes) or sorted(set(axes)) != list(range(ndim)):
        raise YastnError('Provided axes do not match tensor ndim.')


def are_independent(a, b):
    """
    Test if all elements of two yastn tensors are independent objects in memory.
    """
    test = []
    test.append(a is not b)
    test.append(a._data is not b._data)
    test.append(a.config.backend.is_independent(a._data, b._data))
    return all(test)


def is_consistent(a):
    """
    Test is yastn tensor is does not contain inconsistent structures

    Check that:
    1) tset and Dset correspond to A
    2) tset follow symmetry rule f(s@t)==n
    3) block dimensions are consistent (this requires config.test=True)
    """

    Dtot = 0
    for slc in a.slices:
        Dtot += slc.Dp
        assert slc.D[0] == slc.Dp if a.isdiag else np.prod(slc.D, dtype=int) == slc.Dp

    assert a.config.backend.get_shape(a._data) == (Dtot,)
    assert a.struct.size == Dtot

    assert len(a.struct.t) == len(a.struct.D)
    assert len(a.struct.t) == len(a.slices)

    for i in range(len(a.struct.t) - 1):
        assert a.struct.t[i] < a.struct.t[i + 1]

    tset = np.array(a.struct.t, dtype=int).reshape((len(a.struct.t), len(a.struct.s), len(a.struct.n)))
    sa = np.array(a.struct.s, dtype=int)
    na = np.array(a.struct.n, dtype=int)
    assert np.all(a.config.sym.fuse(tset, sa, 1) == na), 'charges of some block do not satisfy symmetry condition'
    _test_tD_consistency(a.struct)
    for s, hf in zip(a.struct.s, a.hfs):
        assert s == hf.s[0]
        assert len(hf.tree) == len(hf.op)
        assert len(hf.tree) == len(hf.s)
        assert len(hf.tree) == len(hf.t) + 1
        assert len(hf.tree) == len(hf.D) + 1
        assert all(y in ('p', 's') if x > 1 else 'n' for x, y in zip(hf.tree, hf.op))
    return True


def _get_tD_legs(struct):
    """ different views on struct.t and struct.D """
    tset = np.array(struct.t, dtype=int).reshape((len(struct.t), len(struct.s), len(struct.n)))
    Dset = np.array(struct.D, dtype=int).reshape((len(struct.t), len(struct.s)))
    tD_legs = [sorted(set((tuple(t.flat), D) for t, D in zip(tset[:, n, :], Dset[:, n]))) for n in range(len(struct.s))]
    tD_dict = [dict(tD) for tD in tD_legs]
    if any(len(x) != len(y) for x, y in zip(tD_legs, tD_dict)):
        raise YastnError('Bond dimensions related to some charge are not consistent.')
    tlegs = [tuple(tD.keys()) for tD in tD_dict]
    Dlegs = [tuple(tD.values()) for tD in tD_dict]
    return tlegs, Dlegs, tD_dict, tset, Dset
