import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

def randR(D, device='cpu'):
    return 2 * np.random.random_sample(D) - 1


def dot(A, B, meta_dot):
    C = {}
    for (out, ina, inb) in meta_dot:
        C[out] = A[ina] @ B[inb]
    return C

def merge_to_matrix(A, order, meta_new, meta_mrg, device='cpu'):
    """ New dictionary of blocks after merging into matrix. """
    Anew = {u: np.zeros(Du, dtype=np.float64) for (u, Du) in meta_new}
    for (tn, to, Dsl, Dl, Dsr, Dr) in meta_mrg:
        Anew[tn][slice(*Dsl), slice(*Dsr)] = A[to].transpose(order).reshape(Dl, Dr)
    return Anew
