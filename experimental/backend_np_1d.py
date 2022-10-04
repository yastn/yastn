import os
_NUM_THREADS="1"
os.environ["OMP_NUM_THREADS"] = _NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _NUM_THREADS
os.environ["MKL_NUM_THREADS"] = _NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = _NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = _NUM_THREADS

from itertools import groupby
import numpy as np


def randR(D, device='cpu', dtype=np.float64):
    return 2 * np.random.random_sample(D).astype(dtype) - 1


def transpose_and_merge(data, order, meta_new, meta_mrg, Dsize):
    newdata = np.zeros((Dsize,), dtype=data.dtype)
    for (tn, Dn, sln), (t1, gr) in zip(meta_new, groupby(meta_mrg, key=lambda x: x[0])):
        assert tn == t1
        temp = newdata[slice(*sln)].reshape(Dn)
        for (_, slo, Do, Dslc, Drsh) in gr:
            slcs = tuple(slice(*x) for x in Dslc)
            temp[slcs] = data[slice(*slo)].reshape(Do).transpose(order).reshape(Drsh)
    return newdata
