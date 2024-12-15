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
import os
_NUM_THREADS="8"
os.environ["OMP_NUM_THREADS"] = _NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _NUM_THREADS
os.environ["MKL_NUM_THREADS"] = _NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = _NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = _NUM_THREADS

from itertools import groupby
from operator import itemgetter
import numpy as np


def randR(D, device='cpu', dtype=np.float64):
    return 2 * np.random.random_sample(D).astype(dtype) - 1


def transpose_and_merge(data, order, meta_new, meta_mrg, Dsize):
    newdata = np.zeros((Dsize,), dtype=data.dtype)
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=itemgetter(0))):
        assert tn == t1
        temp = newdata[slice(*sln)].reshape(Dn)
        for (_, slo, Do, Dslc, Drsh) in gr:
            slcs = tuple(slice(*x) for x in Dslc)
            temp[slcs] = data[slice(*slo)].reshape(Do).transpose(order).reshape(Drsh)
    return newdata
