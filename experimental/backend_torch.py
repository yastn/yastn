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
import torch

def randR(D, device='cpu'):
    return 2 * torch.rand(D, device=device) - 1


def dot(A, B, meta_dot):
    C = {}
    for (out, ina, inb) in meta_dot:
        C[out] = A[ina] @ B[inb]
    return C

def merge_to_matrix(A, order, meta_new, meta_mrg, device='cpu'):
    """ New dictionary of blocks after merging into matrix. """
    Anew = {u: torch.zeros(Du, dtype=torch.float64, device=device) for (u, Du) in meta_new}
    for (tn, to, Dsl, Dl, Dsr, Dr) in meta_mrg:
        Anew[tn][slice(*Dsl), slice(*Dsr)] = A[to].permute(order).reshape(Dl, Dr)
    return Anew
