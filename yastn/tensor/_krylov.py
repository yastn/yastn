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
""" Building Krylov space. """

def expand_krylov_space(self, f, tol, ncv, hermitian, V, H=None, **kwargs):
    """
    Expand the Krylov base up to ``ncv`` states or until reaching desired tolerance ``tol``.
    Implementation for :class:`yastn.Tensor`.
    """
    if H is None:
        H = {}
    happy = False
    for j in range(len(V) - 1, ncv):
        w = f(V[-1])
        if not hermitian:  # Arnoldi
            amplitudes = [1]
            for i in range(j + 1):
                H[(i, j)] = V[i].vdot(w)
                amplitudes.append(-H[(i, j)])
            w = w.add(*V, amplitudes=amplitudes, **kwargs)
        else:  # Lanczos
            H[(j, j)] = V[j].vdot(w)
            if j == 0:
                w = w.add(V[j], amplitudes=[1, -H[(j, j)]], **kwargs)
            else:
                H[(j - 1, j)] = H[(j, j - 1)]
                w = w.add(V[j - 1], V[j], amplitudes=[1, -H[(j - 1, j)], -H[(j, j)]], **kwargs)
        H[(j + 1, j)] = w.norm()
        if H[(j + 1, j)] < tol:
            happy = True
            H.pop((j + 1, j))
            break
        V.append(w / H[(j + 1, j)])
    return V, H, happy
