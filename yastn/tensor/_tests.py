# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Testing and controls. """
from functools import reduce
from operator import mul

import numpy as np

from ._auxliary import _flatten, _unpack_axes, _struct

__all__ = ['are_independent', 'is_consistent', 'YastnError']


class YastnError(Exception):
    """Errors raised by yastn."""


def _test_can_be_combined(a, b):
    """Check if config's of two tensors allow for performing operations mixing them. """
    if type(a) is not type(b):
        raise YastnError('Operation requires two yastn.Tensor-s')
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
    tset = np.array(struct.t, dtype=np.int64).reshape((len(struct.t), len(struct.s), len(struct.n)))
    Dset = np.array(struct.D, dtype=np.int64).reshape((len(struct.D), len(struct.s)))
    for i in range(len(struct.s)):
        ti = list(map(tuple, tset[:, i, :].reshape(len(tset), len(struct.n)).tolist()))
        Di = Dset[:, i].tolist()
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
        if len(axes[0]) != len(axes[1]):
            raise YastnError('axes[0] and axes[1] indicate different number of legs.')
        sa0, sa1 = set(axes[0]), set(axes[1])
        if len(sa0) != len(axes[0]) or len(sa1) != len(axes[1]):
            raise YastnError('Repeated axis in axes[0] or axes[1].')
        if sa0 - set(range(a.ndim)) or sa1 - set(range(b.ndim)):
            raise YastnError('Axis outside of tensor ndim.')
        ua, = _unpack_axes(a.mfs, axes[0])
        ub, = _unpack_axes(b.mfs, axes[1])
        uaxes = (ua, ub)

    if not all(a.struct.s[i1] == sgn * b.struct.s[i2] for i1, i2 in zip(*uaxes)):
        raise YastnError('Signatures do not match.')

    if any(a.mfs[i1] != b.mfs[i2] for i1, i2 in zip(*axes)):
        raise YastnError('Indicated axes of two tensors have different number of meta-fused legs or sub-fusions order.')

    mask_needed = False  # for hard-fused legs
    for i1, i2 in zip(*uaxes):
        if a.hfs[i1].tree != b.hfs[i2].tree or a.hfs[i1].op != b.hfs[i2].op:
            raise YastnError('Indicated axes of two tensors have different number of hard-fused legs or sub-fusions order.')
        if any(s1 != sgn * s2 for s1, s2 in zip(a.hfs[i1].s, b.hfs[i2].s)):
            raise YastnError('Signatures of hard-fused legs do not match.')
        if a.hfs[i1].t != b.hfs[i2].t or a.hfs[i1].D != b.hfs[i2].D:
            mask_needed = True
    return mask_needed, uaxes


def _test_axes_all(a, axes, native=False):
    axes = tuple(_flatten(axes))
    ndim = a.ndim_n if native else a.ndim
    if ndim != len(axes) or sorted(set(axes)) != list(range(ndim)):
        raise YastnError('Provided axes do not match tensor ndim.')


def are_independent(a, b, independent=True):
    """
    Test if all elements of two yastn tensors are independent objects in memory.
    """
    test = []
    test.append(a is not b)
    test.append(a._data is not b._data)
    test.append(a.config.backend.is_independent(a._data, b._data))
    return all(test) == independent


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
        assert slc.D[0] == slc.Dp if a.isdiag else reduce(mul, slc.D, 1) == slc.Dp

    assert a.config.backend.get_shape(a._data) == (Dtot,)
    assert a.struct.size == Dtot

    assert len(a.struct.t) == len(a.struct.D)
    assert len(a.struct.t) == len(a.slices)

    for i in range(len(a.struct.t) - 1):
        assert a.struct.t[i] < a.struct.t[i + 1]

    tset = np.array(a.struct.t, dtype=np.int64).reshape((len(a.struct.t), len(a.struct.s), len(a.struct.n)))
    sa = np.array(a.struct.s, dtype=np.int64)
    na = np.array(a.struct.n, dtype=np.int64)
    assert np.all(a.config.sym.fuse(tset, sa, 1) == na), 'charges of some block do not satisfy symmetry condition'
    _test_tD_consistency(a.struct)
    for s, hf in zip(a.struct.s, a.hfs):
        assert s == hf.s[0]
        assert len(hf.tree) == len(hf.op)
        assert len(hf.tree) == len(hf.s)
        assert len(hf.tree) == len(hf.t) + 1
        assert len(hf.tree) == len(hf.D) + 1
        assert all(y in ('p', 's') if x > 1 else 'n' for x, y in zip(hf.tree, hf.op))
    # test that all elements of tensor are python int types
    _test_struct_types(a.struct)
    return True


def _test_struct_types(struct):
    assert isinstance(struct, _struct)
    assert isinstance(struct.s, tuple)
    assert all(isinstance(x, int) for x in struct.s)
    assert isinstance(struct.n, tuple)
    assert all(isinstance(x, int) for x in struct.n)
    assert isinstance(struct.diag, bool)
    assert isinstance(struct.t, tuple)
    assert all(isinstance(x, tuple) for x in struct.t)
    assert all(isinstance(y, int) for x in struct.t for y in x)
    assert isinstance(struct.D, tuple)
    assert all(isinstance(x, tuple) for x in struct.D)
    assert all(isinstance(y, int) for x in struct.D for y in x)
    assert isinstance(struct.D, tuple)
    assert isinstance(struct.size, int)


def _get_tD_legs(struct):
    """ different views on struct.t and struct.D """
    lt, ndim_n, nsym = len(struct.t), len(struct.s), len(struct.n)
    tset = np.array(struct.t, dtype=np.int64).reshape(lt, ndim_n, nsym)
    Dset = np.array(struct.D, dtype=np.int64).reshape(lt, ndim_n)
    tD_legs = [sorted(set((tuple(t), D) for t, D in zip(tset[:, n, :].tolist(), Dset[:, n].tolist()))) for n in range(ndim_n)]
    tD_dict = [dict(tD) for tD in tD_legs]
    if any(len(x) != len(y) for x, y in zip(tD_legs, tD_dict)):
        raise YastnError('Bond dimensions related to some charge are not consistent.')
    tlegs = [tuple(tD.keys()) for tD in tD_dict]
    Dlegs = [tuple(tD.values()) for tD in tD_dict]
    return tlegs, Dlegs, tD_dict
