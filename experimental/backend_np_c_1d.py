# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================

""" 
This module patches some functions from backend_np_1d.py
using a bit faster version written in C, 
where usage of OpenMP is simpler.

os.environ["OMP_NUM_THREADS"]  is set in backend_np_1d
"""

import os
_NUM_THREADS="2"
os.environ["OMP_NUM_THREADS"] = _NUM_THREADS        # this has to be set before importing numpy,scipy
os.environ["OPENBLAS_NUM_THREADS"] = _NUM_THREADS
os.environ["MKL_NUM_THREADS"] = _NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = _NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = _NUM_THREADS

import platform
from itertools import groupby
import ctypes, ctypes.util
import numpy as np

def randR(D, device='cpu', dtype=np.float64):
    return 2 * np.random.random_sample(D).astype(dtype) - 1


name = "tm_worker"
path = os.path.dirname(os.path.abspath(__file__)) + os.sep  # same path as this wrapper
if platform.system() == "Windows":
    name = path + name + ".dll"
elif platform.system() == "Linux":
    name = path + name + ".so"
elif platform.system() == "Darwin":
    name = path + name + ".dylib"
else:
    raise Exception(f"tm_worker.c library for platform {platform.system()} not implemented.")

# Load the C shared library.
# In case of 'No such file or directory' error 
# put the compiled dynamic library in the same dir as this file, or change the path above.
_lib = ctypes.CDLL(name)

_test_empty = _lib.test_empty

_tm_worker_parallel_float64 = _lib.tm_worker_parallel_float64
_tm_worker_parallel_float64.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ]
_tm_worker_parallel_float64.restype = ctypes.c_int

_tm_worker_parallel_complex128 = _lib.tm_worker_parallel_complex128
_tm_worker_parallel_complex128.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ]
_tm_worker_parallel_complex128.restype = ctypes.c_int

# -------------------------------------

def transpose_and_merge(data, order, meta_new, meta_mrg, Dsize):
    newdata = np.zeros((Dsize,), dtype=data.dtype)
    tasks = 0
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
        for (_, slo, Do, Dslc, Drsh) in gr:
            assert tn == t1    
            tasks += 1
    if tasks==0:
        return newdata

    len_Dn = len(meta_new[0][1])
    len_order = len(order)
    ct_tasks = (ctypes.c_int)(tasks)
    ct_len_order = (ctypes.c_int)(len_order)
    ct_nDn = (ctypes.c_int)(len_Dn)
    
    ct_order = (ctypes.c_int * len_order)()
    for i in range(len_order):
        ct_order[i] = order[i]
    ct_Do = (ctypes.c_int * (len_order * tasks))()
    ct_sln0 = (ctypes.c_int * tasks)()
    ct_Dn = (ctypes.c_int * (len_Dn * tasks))()
    ct_Dslc0 = (ctypes.c_int * (len_Dn * tasks))()
    ct_slo0 = (ctypes.c_int * tasks)()
    ct_Drsh = (ctypes.c_int * (len_Dn * tasks))()
    
    task = 0
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
        for (_, slo, Do, Dslc, Drsh) in gr:
            ct_sln0[task] = sln[0]
            for i in range(len_Dn):
                ct_Dn[i + task * len_Dn] = Dn[i]
                ct_Dslc0[i + task * len_Dn] = Dslc[i][0]
                ct_Drsh[i + task * len_Dn] = Drsh[i]
            ct_slo0[task] = slo[0]
            for i in range(len_order):
                ct_Do[i + task * len_order] = Do[i]
            task += 1
            
    if data.itemsize == 8:
        _tm_worker_parallel_float64(ct_tasks, newdata, ct_sln0, ct_nDn, ct_Dn, ct_Dslc0,  
                                              data, ct_slo0, ct_Do, ct_order, ct_Drsh, ct_len_order)
    elif data.itemsize == 16:
        _tm_worker_parallel_complex128(ct_tasks, newdata, ct_sln0, ct_nDn, ct_Dn, ct_Dslc0, 
                                                 data, ct_slo0, ct_Do, ct_order, ct_Drsh, ct_len_order)
    else:
        raise Exception(f"No _tm_worker_parallel_... implemented for {data.itemsize=}")
    return newdata
