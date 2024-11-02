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
""" Building Krylov space. """

def expand_krylov_space(self, f, tol, ncv, hermitian, V, H=None, **kwargs):
    """
    Expand the Krylov base up to ``ncv`` states or until reaching desired tolerance ``tol``. 
    Implementation for :class:`yastn.Tensor`.
    """
    if H is None:
        H = {}
    happy = False
    for j in range(len(V)-1, ncv):
        w = f(V[-1])
        if not hermitian:  # Arnoldi
            for i in range(j + 1):
                H[(i, j)] = V[i].vdot(w)
                w = w.apxb(V[i], x=-H[(i, j)])
        else:  # Lanczos
            if j > 0:
                H[(j - 1, j)] = H[(j, j - 1)]
                w = w.apxb(V[j - 1], x=-H[(j - 1, j)])
            H[(j, j)] = V[j].vdot(w)
            w = w.apxb(V[j], x=-H[(j, j)])
        H[(j + 1, j)] = w.norm()
        if H[(j + 1, j)] < tol:
            happy = True
            H.pop((j + 1, j))
            break
        V.append(w / H[(j + 1, j)])
    return V, H, happy


def linear_combination(self, *vectors, amplitudes, **kwargs):
    """ Linear combination of :meth:`yastn.Tensors` with given amplitudes. """
    v = amplitudes[0] * vectors[0]
    for x, b in zip(amplitudes[1:], vectors[1:]):
        v = v.apxb(b, x=x)
    return v
