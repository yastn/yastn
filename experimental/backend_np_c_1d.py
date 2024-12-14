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

""" 
This module patches some functions from backend_np_1d.py
using a bit faster version written in C, 
where usage of OpenMP is simpler.

os.environ["OMP_NUM_THREADS"]  is set in backend_np_1d
"""

import os
_NUM_THREADS="1"
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
