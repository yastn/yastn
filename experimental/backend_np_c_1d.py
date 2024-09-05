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
this module patches some functions from backend_np_1d.py
using faster versions written in C, 
where usage of OpenMP is simpler and faster.

"OMP_NUM_THREADS" is set in backend_np_1d
"""

from backend_np_1d import *

import os
import platform

from itertools import groupby
import ctypes, ctypes.util
import numpy as np

name = "tm_worker"
path = os.path.dirname(os.path.abspath(__file__)) + os.sep  # same path as this wrapper
if platform.system() == "Windows":
    name = path + name + ".dll"
elif platform.system() == "Linux":
    name = path + name + ".so"
elif platform.system() == "Darwin":
    name = path + name + ".dylib"   # 2024-08-31  - not tested with MacOS yet
else:
    raise Exception(f"tm_worker.c library for platform {platform.system()} not implemented.")

# Load the C shared library.
# In case of 'No such file or directory' error either:
#  1) put the compiled dynamic library in the same dir as tm_wrapper.py
#  or 2) put the compiled dynamic library in the working directory and change path above to "./"
#  or 3) os.environ["PATH"] += os.pathsep + os.path.dirname(os.path.abspath(__file__))
#  or 4) add the path to the library to the   os.environ["LD_LIBRARY_PATH"] 

_lib = ctypes.CDLL(name)

_test_empty = _lib.test_empty

_tm_worker_parallel_float64 = _lib.tm_worker_parallel_float64
_tm_worker_parallel_float64.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
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
    ctypes.POINTER(ctypes.c_int),
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
    
    ct_tasks = (ctypes.c_int)(tasks)
    ct_len_order = (ctypes.c_int)(len(order))
    ct_order = (ctypes.c_int * len(order))()
    for i in range(len(order)):
        ct_order[i] = order[i]
    ct_Do = (ctypes.c_int * (len(order) * tasks))()
    ct_sln0 = (ctypes.c_int * tasks)()
    ct_Dn1 = (ctypes.c_int * tasks)()
    ct_Dslc0 = (ctypes.c_int * tasks)()
    ct_Dslc1 = (ctypes.c_int * tasks)()
    ct_slo0 = (ctypes.c_int * tasks)()
    ct_Drsh1 = (ctypes.c_int * tasks)()
    
    task = 0
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
        for (_, slo, Do, Dslc, Drsh) in gr:
            ct_sln0[task] = sln[0]
            ct_Dn1[task] = Dn[1]
            ct_Dslc0[task] = Dslc[0][0]
            ct_Dslc1[task] = Dslc[1][0]
            ct_slo0[task] = slo[0]
            ct_Drsh1[task] = Drsh[1]
            for i in range(len(order)):
                ct_Do[i + task * len(order)] = Do[i]
            task += 1

    # even with a single thread available, 
    # calling tm_worker only once with gathered data is 20% faster
    if data.itemsize == 8:
        _tm_worker_parallel_float64(ct_tasks, newdata, ct_sln0, ct_Dn1, ct_Dslc0, ct_Dslc1, 
                                              data, ct_slo0, ct_Do, ct_order, ct_Drsh1, ct_len_order)
    elif data.itemsize == 16:
        _tm_worker_parallel_complex128(ct_tasks, newdata, ct_sln0, ct_Dn1, ct_Dslc0, ct_Dslc1, 
                                                 data, ct_slo0, ct_Do, ct_order, ct_Drsh1, ct_len_order)
    else:
        raise Exception(f"No _tm_worker_parallel_... implemented for {data.itemsize=}")
    return newdata
