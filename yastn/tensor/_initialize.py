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
""" Methods creating a new yastn.Tensor """
import os
from functools import reduce
from itertools import product
import numbers
from operator import mul, itemgetter

import numpy as np

from ._auxiliary import _flatten, _config, get_blocks, find_index, find_matching_indices, update_old_struct
from ._legbasic import legs_from_dict_v2
from ._tests import YastnError, _test_tD_consistency, _test_struct_types
from ..backend import backend_np
from ..sym import sym_none, sym_U1, sym_Z2, sym_Z3, sym_U1xU1, sym_U1xU1xZ2

__all__ = ['make_config']


_syms = {"dense": sym_none,
         "none": sym_none,
         "U1": sym_U1,
         "Z2": sym_Z2,
         "Z3": sym_Z3,
         "U1xU1": sym_U1xU1,
         "U1xU1xZ2": sym_U1xU1xZ2}


# def make_config(backend=backend_np, sym=sym_none, default_device='cpu',
#                 default_dtype='float64', fermionic=False,
#                 default_fusion='hard', force_fusion=None, tensordot_policy='fuse_contracted', **kwargs):
def make_config(**kwargs) -> _config:
    r"""
    Create structure with YASTN configuration

    Parameters
    ----------
    backend : backend module or str
        Specify ``backend`` providing linear algebra and base dense tensors.
        Currently supported backends are

            * NumPy as ``yastn.backend.backend_np``
            * PyTorch as ``yastn.backend.backend_torch``

        The above backends can be specified as strings: "np", "torch".
        Defaults to NumPy backend.

    sym : symmetry module or compatible object or str
        Specify abelian symmetry. To see how YASTN defines symmetries,
        see :class:`yastn.sym.sym_abelian`.
        Defaults to ``yastn.sym.sym_none``, effectively a dense tensor.
        For predefined symmetries, takes string input from
        'none' (or 'dense'), 'Z2', 'Z3', 'U1', 'U1xU1', 'U1xU1xZ2'.

    default_device : str
        Tensors can be stored on various devices as supported by ``backend``

            * NumPy supports only ``'cpu'`` device
            * PyTorch supports multiple devices, see
              https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

        If not specified, the default device is ``'cpu'``.

    default_dtype: str
        Default data type (dtype) of YASTN tensors. Supported options are: ``'float64'``,
        ``'complex128'``. If not specified, the default dtype is ``'float64'``.
    fermionic : bool or tuple[bool,...]
        Specify behavior of :meth:`yastn.swap_gate` function, allowing to introduce fermionic statistics.
        Allowed values: ``False``, ``True``, or a tuple ``(True, False, ...)`` with one bool for each component
        charge vector, i.e., of length sym.NSYM. The default is ``False``.
    default_fusion: str
        Specify default strategy to handle leg fusion: ``'hard'`` or ``'meta'``. See :meth:`yastn.Tensor.fuse_legs`
        for details. The default is ``'hard'``.
    force_fusion : str
        Overrides fusion strategy provided in :meth:`yastn.Tensor.fuse_legs`. The default is ``None``.
    tensordot_policy: str
        Contraction approach used by :meth:`yastn.tensordot`

            * ``'fuse_to_matrix'`` Tensordot involves suitable permutation of each tensor while performing a fusion of each tensor into a sequence of matrices and calling matrix-matrix multiplication. Postprocessing includes unfusioning the remaining legs in the result, which often copy data adding extra overhead.
            * ``'fuse_contracted'`` Tensordot involves suitable permutation of each tensor while performing a fusion of to-be-contracted legs of each tensor and calling multiplication. It involves a larger number of multiplication calls for smaller objects, but unfusing the legs of the result is not needed.
            * ``'no_fusion'`` Tensordot involves suitable permutation of tensor blocks and calling matrix-matrix multiplication for a potentially large number of small objects. Resulting contributions to new blocks get added. However, overheads of initial fusion (copying data) can sometimes be avoided in this approach.
    profile : bool
        If ``True``, enables profiling of tensor operations in backends supporting it.
        Currently, only PyTorch backend with NVTX support is available.
        Default is ``False``. If YASTN_PROFILE=1 is set in the environment, overrides this argument to ``True``.

    Example
    -------

    ::

        config = yastn.make_config(backend='np', sym='U1')
    """
    if "backend" not in kwargs or kwargs["backend"] == 'np':
        kwargs["backend"] = backend_np
    elif kwargs["backend"] == 'torch':
        from ..backend import backend_torch
        kwargs["backend"] = backend_torch
    elif kwargs["backend"] == 'torch_cpp':
        from ..backend import backend_torch_cpp  # pragma: no cover
        kwargs["backend"] = backend_torch_cpp  # pragma: no cover
    elif isinstance(kwargs["backend"], str):
        raise YastnError("backend encoded as string only supports: 'np', 'torch'")

    if "sym" not in kwargs:
        kwargs["sym"] = sym_none
    elif isinstance(kwargs["sym"], str):
        try:
            kwargs["sym"] = _syms[kwargs["sym"]]
        except KeyError:
            raise YastnError("sym encoded as string only supports: 'dense', 'Z2', 'Z3', 'U1', 'U1xU1', 'U1xU1xZ2'.")

    if "profile" not in kwargs:
        kwargs["profile"] = bool(int(os.getenv("YASTN_PROFILE","0")))

    return _config(**{a: kwargs[a] for a in _config._fields if a in kwargs})


def __setitem__(a, key, newvalue):
    """
    Update data of the selected block.

    The data (its shape) should be consistent with
    the dimensions of the charge sectors where the block belongs.

    Parameters
    ----------
    key : Sequence[int] | Sequence[Sequence[int]]
        charges of the block
    """
    try:
        key = np.array(key, dtype=np.int64).reshape(a.ndim_n, a.config.sym.NSYM)
        reverse_trans = np.argsort(a.trans)
        ukey = key[reverse_trans, :].ravel()
        bl = get_blocks(a.config.sym, a.struct.legs, a.struct.n, a.isdiag)
        ind = find_index(bl.t, ukey)
    except ValueError as exc:
        raise YastnError('Tensor does not have the block specified by key.') from exc
    slc = slice(*bl.slc[ind])
    Dt = bl.D[ind]
    Dr = tuple(Dt[ax] for ax in a.trans)
    if not a.isdiag:
        newvalue = a.config.backend.permute_dims(newvalue.reshape(Dr), reverse_trans)
    a._data[slc] = newvalue.reshape(-1)


def _fill_tensor(a, t=(), D=(), val='rand'):  # dtype = None
    r"""
    Create all allowed blocks based on signature ``s``, total charge ``n``,
    and a set of charge sectors ``t`` for each leg of the tensor.

    First, all allowed blocks are identified by checking the
    :ref:`selection rule<symmetry selection rule>`.
    Then each allowed block is created as a tensor with
    sizes specified in ``D`` and filled with value ``val``.

    .. note::
        This operation overwrites the data of the tensor.

    Parameters
    ----------
    a : yastn.Tensor

    t : Sequence[Sequence[int]] or Sequence[Sequence[Sequence[int]]]
        list of charge sectors for each leg of the tensor, see examples.
        In case of tensor without symmetry this argument is ignored.

    D : Sequence[int] or Sequence[Sequence[int]]
        list of sector sizes for each leg of the tensor, see examples.

    val : str
        ``'rand'``, ``'ones'``, or  ``'zeros'``
    """
    try:
        D = tuple(D)
    except TypeError:
        D = (D,)
    try:
        t = tuple(t)
    except TypeError:
        t = (t,)

    if a.config.sym.NSYM == 0:
        if a.isdiag and len(D) == 1:
            D = D + D
        D = tuple(x if x else (0,) for x in D)  # replace () with (0,)
        if len(D) != a.ndim_n:
            raise YastnError("Number of elements in D does not match tensor rank.")
        tset = np.zeros((1, a.ndim_n, a.config.sym.NSYM))
        Dset = np.array(D, dtype=np.int64).reshape(1, a.ndim_n)
    else:  # a.config.sym.NSYM >= 1
        D = (D,) if (a.ndim_n == 1 or a.isdiag) and isinstance(D[0], numbers.Number) else D
        t = (t,) if (a.ndim_n == 1 or a.isdiag) and isinstance(t[0], numbers.Number) else t
        D = D + D if a.isdiag and len(D) == 1 else D
        t = t + t if a.isdiag and len(t) == 1 else t

        D = list((x,) if isinstance(x, numbers.Number) else x for x in D)
        t = list((x,) if isinstance(x, numbers.Number) else x for x in t)

        if len(D) != a.ndim_n:
            raise YastnError("Number of elements in D does not match tensor rank.")
        if len(t) != a.ndim_n:
            raise YastnError("Number of elements in t does not match tensor rank.")
        for x, y in zip(D, t):
            if len(x) != len(y):
                raise YastnError("Elements of t and D do not match")

        comb_D = list(product(*D))
        comb_t = list(product(*t))
        lcomb_t = len(comb_t)
        comb_t = list(_flatten(comb_t))
        comb_t = np.array(comb_t, dtype=np.int64).reshape((lcomb_t, a.ndim_n, a.config.sym.NSYM))
        comb_D = np.array(comb_D, dtype=np.int64).reshape((lcomb_t, a.ndim_n))
        ind = np.all(a.config.sym.fuse(comb_t, a.s_n, 1) == a.struct.n, axis=1)
        tset = comb_t[ind]
        Dset = comb_D[ind]

    # eliminate zero blocks
    ind_nonzero = np.all(Dset, axis=1)
    tset = tset[ind_nonzero]
    Dset = Dset[ind_nonzero]

    if a.isdiag and np.any(Dset[:, 0] != Dset[:, 1]):
        raise YastnError("Diagonal tensor requires the same bond dimensions on both legs.")
    Dp = Dset[:, 0] if a.isdiag else np.prod(Dset, axis=1, dtype=np.int64)
    Dp = Dp.tolist()
    Dsize = sum(Dp)

    if len(tset) > 0:
        tset = tset.reshape(len(tset), a.ndim_n * a.config.sym.NSYM).tolist()
        Dset = Dset.tolist()
        meta = [(tuple(ts), tuple(Ds)) for ts, Ds in zip(tset, Dset)]
        meta = sorted(meta, key=itemgetter(0))
        a_t, a_D = zip(*meta)
    else:
        a_t, a_D = (), ()

    legs = legs_from_dict_v2({"s": a.s_n, "n": a.n, 't': a_t, "D": a_D})
    a.struct = a.struct._replace(t=a_t, D=a_D, size=Dsize, legs=legs)

    a._data = _init_block(a.config, Dsize, val, dtype=a.yastn_dtype, device=a.device)
    _test_tD_consistency(a.struct)
    _test_struct_types(a.struct)


def set_block(a, ts=(), Ds=None, val='zeros'):
    """
    Add new block to tensor or change the existing one.

    This is the intended way to add new blocks by hand.
    Checks if bond dimensions of the new block are consistent with the existing ones
    and updates the legs of the tensors accordingly.

    Parameters
    ----------
    ts : Sequence[int] | Sequence[Sequence[int]]
        Charges identifying the block. Ignored if tensor has no symmetry.

    Ds : Sequence[int]
        Dimensions of the block. If ``None``, tries to infer
        dimensions from legs of the tensor.

    val : tensor-like | str | tuple[str, tuple]
        recognized string values are ``'ones'``, ``'zeros'``, ``'normal'``, ``'rand'``,
        or a tuple ``('rand', distribution)``, for uniform distribution in range given by tuple lim.
        Otherwise any tensor-like format such as nested list, numpy.ndarray, etc.,
        can be used provided it is supported by :doc:`tensor's backend </tensor/configuration>`.
    """
    if a.trans != tuple(range(a.ndim_n)):
        raise YastnError("Setting block of transpoded tensor is not supported.")
    ts = np.array(ts, dtype=np.int64).ravel()
    nsym = a.config.sym.NSYM
    if a.isdiag and len(ts) == nsym:
        ts = np.hstack([ts, ts])
    if len(ts) != a.ndim_n * nsym:
        raise YastnError('Size of ts is not consistent with tensor rank and the number of symmetry sectors.')

    ats = ts.reshape((1, a.ndim_n, nsym))
    if not np.all(a.config.sym.fuse(ats, a.s_n, 1) == a.n):
        raise YastnError('Charges ts are not consistent with the symmetry rules: f(t @ s) == n')
    ats = ats[0]

    ts = tuple(ts.tolist())
    tss = tuple(ts[i * nsym: (i+1) * nsym] for i in range(a.ndim_n))

    if Ds is None:  # attempt to read Ds from existing blocks.
        try:
            Ds = tuple(leg[tt] for leg, tt in zip(a.legs, tss))
        except ValueError as err:
            raise YastnError('Provided Ds. Cannot infer all bond dimensions from existing blocks.') from err
    else:  # Ds was provided
        Ds = np.array(Ds, dtype=np.int64).ravel()
        if a.isdiag and len(Ds) == 1:
            Ds = np.hstack([Ds, Ds])
        Ds = tuple(Ds.tolist())
    if len(Ds) != a.ndim_n:
        raise YastnError('Size of Ds is not consistent with tensor rank.')

    if a.isdiag and Ds[0] != Ds[1]:
        raise YastnError("Diagonal tensor requires the same bond dimensions on both legs.")

    if any(tt in leg and leg[tt] != DD for leg, tt, DD in zip(a.legs, tss, Ds)):
        raise YastnError("Provided Ds is not consistent with dimensions of existing legs.")

    if any(tt not in leg for leg, tt in zip(a.legs, tss)):
        new_legs = tuple(leg.add_charge(tt, DD) for leg, tt, DD in zip(a.legs, tss, Ds) )
        embed_(a, new_legs)

    Dsize = Ds[0] if a.isdiag else reduce(mul, Ds, 1)
    new_block = _init_block(a.config, Dsize, val, dtype=a.yastn_dtype, device=a.device)

    bl = get_blocks(a.config.sym, a.legs, a.struct.n, a.isdiag)
    ind = find_index(bl.t, ats)
    slc = bl.slc[ind]
    a.data[slice(*slc)] = new_block


def embed_(a, legs_new):
    bl_old = get_blocks(a.config.sym, a.legs, a.struct.n, a.isdiag)
    bl_new = get_blocks(a.config.sym, legs_new, a.struct.n, a.isdiag)
    ind1, ind2 = find_matching_indices(bl_new.t, bl_old.t)
    sln, slo = bl_new.slc[ind1], bl_old.slc[ind2]
    meta = list(zip(sln, sln[:, 1] - sln[:, 0], slo, slo[:, 1] - slo[:, 0]))
    newdata = a.config.backend.embed_transpose(a.data, None, meta, bl_new.size)
    struct, slices = update_old_struct(bl_new)
    a.struct = struct
    a._data = newdata


def _init_block(config, Dsize, val, dtype, device):
    if isinstance(val, tuple) and val[0] == 'rand':
        return config.backend.rand((Dsize,), distribution=val[1], dtype=dtype, device=device)
    if isinstance(val, str):
        if val == 'zeros':
            return config.backend.zeros((Dsize,), dtype=dtype, device=device)
        if val == 'rand':
            return config.backend.rand((Dsize,), dtype=dtype, device=device)
        if val == 'normal':
            return config.backend.rand((Dsize,), distribution='normal', dtype=dtype, device=device)
        if val == 'ones':
            return config.backend.ones((Dsize,), dtype=dtype, device=device)
        raise YastnError('val should be in ("zeros", "ones", "rand") or an array of the correct size')
    x = config.backend.to_tensor(val, Ds=Dsize, dtype=dtype, device=device)
    if config.backend.get_size(x) == Dsize ** 2:
        x = config.backend.diag_get(x.reshape(Dsize, Dsize))
    return x
