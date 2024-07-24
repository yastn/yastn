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


class EnvWindow:
    """ EnvWindow class for expectation values within PEPS. """

    def __init__(self, env_ctm, xrange, yrange):
        self.psi = env_ctm.psi
        self.env_ctm = env_ctm
        self._env = {}
        self.xrange = xrange
        self.yrange = yrange
        Nx = self.xrange[1] - self.xrange[0]
        Ny = self.yrange[1] - self.yrange[0]

        li, ri = self.yrange[0], self.yrange[1] - 1
        ti, bi = self.xrange[0], self.xrange[1] - 1
        self['l', li] = mps.Mps(Nx + 2)
        self['r', ri] = mps.Mps(Nx + 2)
        self['t', ti] = mps.Mps(Ny + 2)
        self['b', bi] = mps.Mps(Ny + 2)

        for ind, ny in enumerate(range(*self.yrange), start=1):
            self['t', ti][ind] = env_ctm[ti, ny].t
            self['b', bi][ind] = env_ctm[bi, ny].b

        self['t', ti][0] = env_ctm[ti, li].tl.add_leg(axis=0)
        self['t', ti][Ny + 1] = env_ctm[ti, ri].tr.add_leg(axis=2)
        self['b', bi][0] = env_ctm[bi, li].bl.add_leg(axis=2)
        self['b', bi][Ny + 1] = env_ctm[bi, ri].br.add_leg(axis=0)

        for ind in range(Ny + 2):
            self['b', bi][ind] = self['b', bi][ind].transpose(axes=(2, 1, 0)).conj()

        for ind, nx in enumerate(range(*self.xrange), start=1):
            self['l', li][ind] = env_ctm[nx, li].l
            self['r', ri][ind] = env_ctm[nx, ri].r

        self['l', li][0] = env_ctm[ti, li].tl.add_leg(axis=2)
        self['l', li][Nx + 1] = env_ctm[bi, li].bl.add_leg(axis=0)
        self['r', ri][0] = env_ctm[ti, ri].tr.add_leg(axis=0)
        self['r', ri][Nx + 1] = env_ctm[bi, ri].br.add_leg(axis=2)

        for ind in range(Nx + 2):
            self['l', li][ind] = self['l', li][ind].transpose(axes=(2, 1, 0))
            self['r', ri][ind] = self['r', ri][ind].conj()

    def __getitem__(self, ind):
        return self._env[ind]

    def __setitem__(self, ind, value):
        self._env[ind] = value

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

