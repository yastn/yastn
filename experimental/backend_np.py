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
