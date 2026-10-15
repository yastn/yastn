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
