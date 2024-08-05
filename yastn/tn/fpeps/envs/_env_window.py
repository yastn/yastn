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
from __future__ import annotations
from ... import mps
from .... import YastnError


class EnvWindow:
    """ EnvWindow class for expectation values within PEPS with CTM boundary. """

    def __init__(self, env_ctm, xrange, yrange):
        self.psi = env_ctm.psi
        self.env_ctm = env_ctm
        self.xrange = xrange
        self.yrange = yrange

    def __getitem__(self, ind) -> yastn.tn.mps.MpsMpoOBC:
        """
        Boundary MPS build of CTM tensors.

        Parameters
        ----------
        (n, dirn): tuple[int, str]
            n is a row or column index for the MPS boundary of the transfer matrix specified by n and dirn.
            Consistent with PEPS indexing.
            dirn is 't', 'b', 'l', and 'r' for top, bottom, left, and right boundaries, respectively.
            't' and 'b' refer to a horizontal direction (n specifies a row).
            'l' and 'r' refer to a vertical direction (n specifies a column).
        """
        n, dirn = ind
        Nx = self.xrange[1] - self.xrange[0]
        Ny = self.yrange[1] - self.yrange[0]
        li, ri = self.yrange[0], self.yrange[1] - 1
        ti, bi = self.xrange[0], self.xrange[1] - 1

        if dirn in 'lr' and not li <= n <= ri:
            raise YastnError(f"{n=} not within {self.yrange=}")

        if dirn == 'l':
            psi = mps.Mps(Nx + 2)
            for ind, nx in enumerate(range(*self.xrange), start=1):
                psi[ind] = self.env_ctm[nx, n].l.transpose(axes=(2, 1, 0))
            psi[0] = self.env_ctm[ti, n].tl.add_leg(axis=2).transpose(axes=(2, 1, 0))
            psi[Nx + 1] = self.env_ctm[bi, n].bl.add_leg(axis=0).transpose(axes=(2, 1, 0))
            return psi

        if dirn == 'r':
            psi = mps.Mps(Nx + 2)
            for ind, nx in enumerate(range(*self.xrange), start=1):
                psi[ind] =self. env_ctm[nx, n].r.conj()
            psi[0] =self. env_ctm[ti, n].tr.add_leg(axis=0).conj()
            psi[Nx + 1] =self. env_ctm[bi, n].br.add_leg(axis=2).conj()
            return psi

        if dirn in 'tb' and not ti <= n <= bi:
            raise YastnError(f"{n=} not within {self.xrange=}")

        if dirn == 't':
            psi = mps.Mps(Ny + 2)
            for ind, ny in enumerate(range(*self.yrange), start=1):
                psi[ind] =self. env_ctm[n, ny].t
            psi[0] =self. env_ctm[n, li].tl.add_leg(axis=0)
            psi[Ny + 1] =self. env_ctm[n, ri].tr.add_leg(axis=2)
            return psi

        if dirn == 'b':
            psi = mps.Mps(Ny + 2)
            for ind, ny in enumerate(range(*self.yrange), start=1):
                psi[ind] =self. env_ctm[n, ny].b.transpose(axes=(2, 1, 0)).conj()
            psi[0] =self. env_ctm[n, li].bl.add_leg(axis=2).transpose(axes=(2, 1, 0)).conj()
            psi[Ny + 1] =self. env_ctm[n, ri].br.add_leg(axis=0).transpose(axes=(2, 1, 0)).conj()
            return psi

        raise YastnError(f"{dirn=} not recognized. Should be 't', 'b', 'l', or 'r'.")


    def transfer_mpo(self, n, dirn='v'):
        if dirn == 'h':
            Ny = self.yrange[1] - self.yrange[0]
            op = mps.Mpo(N = Ny + 2)
            for ind, ny in enumerate(range(*self.yrange), start=1):
                op.A[ind] = self.psi[n, ny].transpose(axes=(1, 2, 3, 0))
            op.A[0] = self.env_ctm[n, self.yrange[0]].l.add_leg(axis=0)
            op.A[Ny + 1] = self.env_ctm[n, self.yrange[1] - 1].r.add_leg(axis=3).transpose(axes=(1, 2, 3, 0))
        elif dirn == 'v':
            Nx = self.xrange[1] - self.xrange[0]
            op = mps.Mpo(N = Nx + 2)
            for ind, nx in enumerate(range(*self.xrange), start=1):
                op.A[ind] = self.psi[nx, n].transpose(axes=(0, 3, 2, 1))
            op.A[0] = self.env_ctm[self.xrange[0], n].t.add_leg(axis=0).transpose(axes=(0, 3, 2, 1))
            op.A[Nx + 1] = self.env_ctm[self.xrange[1] - 1, n].b.add_leg(axis=3).transpose(axes=(1, 0, 3, 2))
        return op

