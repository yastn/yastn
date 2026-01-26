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
from ._env_contractions import *
from .._evolution import BondMetric
from .._doublePepsTensor import DoublePepsTensor
from ... import mps
from ....initialize import eye
from ....tensor import YastnError, tensordot


_hair_dirn = {'t': hair_t, 'l': hair_l, 'b': hair_b, 'r': hair_r}
_axis_dirn ={'t': 2, 'l': 3, 'b': 0, 'r': 1}


class EnvApproximate:
    def __init__(self, psi, which='65', opts_svd=None, opts_var=None, update_sweeps=None):
        r"""
        Supports calculation of bond metric for larger local clusters,
        which are contracted approximately using the boundary MPS approach.

        Parameters
        ----------
        psi: yastn.tn.Peps
            Peps state being evolved.

        which: str
            Type of environment from '43', '43+', '65', '65+', '87', '87+'

        opts_svd: dict
            Passed to :meth:`yastn.tn.mps.zipper` for truncation in boundary MPS calculation.

        opts_var:
            Passed to :meth:`yastn.tn.mps.compression_` for boundary MPS variational fine-tunning.

        update_sweeps:
            Passed as max_sweeps to :meth:`yastn.tn.mps.compression_` for boundary MPS variational fine-tunning.
        """
        self.psi = psi
        self._set_which(which)
        self.opts_var = {'max_sweeps': 2,} if opts_var is None else opts_var
        self.opts_svd = opts_svd
        self.update_sweeps = update_sweeps
        self._envs = {}

    def _get_which(self):
        return self._which

    def _set_which(self, which):
        if which not in ('43', '43+', '65', '65+', '87', '87+'):
            raise YastnError(f" Type of EnvApprox {which=} not recognized.")
        self._which = which
        if which in ('43', '43+'):
            self.Nl, self.Nw = 2, 1
        if which in ('65', '65+'):
            self.Nl, self.Nw = 3, 2
        if which in ('87', '87+'):
            self.Nl, self.Nw = 4, 3
        self._hairs = '+' in which

    which = property(fget=_get_which, fset=_set_which)

    def __getitem__(self, key):
        return self._envs[key]

    def __setitem__(self, key, value):
        self._envs[key] = value

    def apply_patch(self):
        pass

    def move_to_patch(self, sites):
        pass

    def pre_truncation_(env, bond):
        pass

    def post_truncation_(env, bond, **kwargs):
        pass

    def bond_metric(self, Q0, Q1, s0, s1, dirn):
        """
        Calculates bond metric. The environment size is controlled by ``which``.

        Approximately contract the diagram using boundary MPS approach, parallel to the metric (open) bond direction.

        ::

            If which == '65':

                (-2 -2)═(-2 -1)═(-2 +0)══(-2 +1)═(-2 +2)═(-2 +3)
                   ║       ║       ║        ║        ║        ║
                (-1 -2)═(-1 -1)═(-1 +0)══(-1 +1)═(-1 +2)═(-1 +3)
                   ║       ║       ║        ║        ║        ║
                (+0 -2)═(+0 -1)════Q0══   ══Q1═══(+0 +2)═(+0 +3)
                   ║       ║       ║        ║        ║        ║
                (+1 -2)═(+1 -1)═(+1 +0)══(+1 +1)═(+1 +2)═(+1 +3)
                   ║       ║       ║        ║        ║        ║
                (+2 -2)═(+2 -1)═(+2 +0)══(+2 +1)═(+2 +2)═(+2 +3)


            If which == '65+':

                        (-3 -2) (-3 -1) (-3 +0)  (-3 +1) (-3 +2)  (-3 +3)
                           ║       ║       ║        ║       ║        ║
                (-2 -3)═(-2 -2)═(-2 -1)═(-2 +0)══(-2 +1)═(-2 +2)══(-2 +3)═(-2 +4)
                           ║       ║       ║        ║       ║        ║
                (-1 -3)═(-1 -2)═(-1 -1)═(-1 +0)══(-1 +1)═(-1 +2)══(-1 +3)═(-1 +4)
                           ║       ║       ║        ║       ║        ║
                (+0 -3)═(+0 -2)═(+0 -1)════Q0══   ══Q1═══(+0 +2)══(+0 +3)═(+0 +4)
                           ║       ║       ║        ║       ║        ║
                (+1 -3)═(+1 -2)═(+1 -1)═(+1 +0)══(+1 +1)═(+1 +2)══(+1 +3)═(+1 +4)
                           ║       ║       ║        ║       ║        ║
                (+2 -3)═(+2 -2)═(+2 -1)═(+2 +0)══(+2 +1)═(+2 +2)══(+2 +3)═(+2 +4)
                           ║       ║       ║        ║       ║        ║
                        (+3 -2) (+3 -1) (+3 +0)  (+3 +1) (+3 +2)  (+3 +3)

        """
        assert self.psi.nn_site(s0, (0, 1) if dirn in ('h', 'lr') else (1, 0)) == s1
        bd = (s0, s1, dirn)

        try:
            if self.update_sweeps:
                self.update_env(bd)
            else:
                self.initialize_env(bd)
        except (KeyError, YastnError):
            self.initialize_env(bd)

        tmpo = self.transfer_mpo(bd, n=0, Q0=Q0, Q1=Q1)
        env = mps.Env(self[bd, -1], [tmpo, self[bd, 1]])
        Nl = self.Nl
        for n in range(Nl):
            env.update_env_(n, to='last')
            env.update_env_(2 * Nl - n - 1, to='first')
        g = tensordot(env.F[Nl-1, Nl], env.F[Nl, Nl-1], axes=((0, 2), (0, 2)))
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))

    def initialize_env(self, bd):
        if bd[2] in ("h", "lr"):
            self[bd, self.Nw + 1] = self.boundary_mps(bd, 't')
            for nx in range(-self.Nw, 0):  # [bd, 1]
                tmpo = self.transfer_mpo(bd, n=nx)
                self[bd, -nx] = mps.zipper(tmpo, self[bd, -nx+1], opts_svd=self.opts_svd)
                mps.compression_(self[bd, -nx], (tmpo, self[bd, -nx+1]), **self.opts_var)

            self[bd, -self.Nw - 1] = self.boundary_mps(bd, 'b').conj()
            for nx in range(self.Nw, 0, -1):  # [bd, -1]
                tmpo = self.transfer_mpo(bd, n=nx).H
                self[bd, -nx] = mps.zipper(tmpo, self[bd, -nx-1], opts_svd=self.opts_svd)
                mps.compression_(self[bd, -nx], (tmpo, self[bd, -nx-1]), **self.opts_var)

        else:  # dirn == "v":
            self[bd, self.Nw + 1] = self.boundary_mps(bd, 'r')
            for ny in range(self.Nw, 0, -1):  # [bd, 1]
                tmpo = self.transfer_mpo(bd, n=ny)
                self[bd, ny] = mps.zipper(tmpo, self[bd, ny+1], opts_svd=self.opts_svd)
                mps.compression_(self[bd, ny], (tmpo, self[bd, ny+1]), **self.opts_var)

            self[bd, -self.Nw - 1] = self.boundary_mps(bd, 'l').conj()
            for ny in range(-self.Nw, 0):  # [bd, -1]
                tmpo = self.transfer_mpo(bd, n=ny).H
                self[bd, ny] = mps.zipper(tmpo, self[bd, ny-1], opts_svd=self.opts_svd)
                mps.compression_(self[bd, ny], (tmpo, self[bd, ny-1]), **self.opts_var)

    def update_env(self, bd):
        if bd[2] in ("h", "lr"):
            self[bd, self.Nw + 1] = self.boundary_mps(bd, 't')
            for nx in range(-self.Nw, 0):  # [bd, 1]
                tmpo = self.transfer_mpo(bd, n=nx)
                mps.compression_(self[bd, -nx], (tmpo, self[bd, -nx+1]), max_sweeps=self.update_sweeps)

            self[bd, -self.Nw - 1] = self.boundary_mps(bd, 'b').conj()
            for nx in range(self.Nw, 0, -1):  # [bd, -1]
                tmpo = self.transfer_mpo(bd, n=nx).H
                mps.compression_(self[bd, -nx], (tmpo, self[bd, -nx-1]), max_sweeps=self.update_sweeps)

        else:  # dirn == "v":
            self[bd, self.Nw + 1] = self.boundary_mps(bd, 'r')
            for ny in range(self.Nw, 0, -1):  # [bd, 1]
                tmpo = self.transfer_mpo(bd, n=ny)
                mps.compression_(self[bd, ny], (tmpo, self[bd, ny+1]), max_sweeps=self.update_sweeps)

            self[bd, -self.Nw - 1] = self.boundary_mps(bd, 'l').conj()
            for ny in range(-self.Nw, 0):  # [bd, -1]
                tmpo = self.transfer_mpo(bd, n=ny).H
                mps.compression_(self[bd, ny], (tmpo, self[bd, ny-1]), max_sweeps=self.update_sweeps)

    def transfer_mpo(self, bd, n, Q0=None, Q1=None):
        H = mps.Mpo(N = 2 * self.Nl)
        s0, s1, dirn = bd

        if dirn in ("h", "lr"):
            nx = n
            d = {(nx, ny): self.psi.nn_site(s0, d=(nx, ny))
                for ny in range(-self.Nl + 1 - self._hairs, self.Nl + 1 + self._hairs)}
            tensors_from_psi(d, self.psi)
            if nx == 0:
                d[0, 0], d[0, 1] = Q0, Q1

            ny = -self.Nl + 1
            hl = hair_l(d[nx, ny - 1]) if self._hairs else None
            H.A[H.first] = edge_l(d[nx, ny], hl=hl).add_leg(s=-1, axis=0)
            for site in H.sweep(to='last', dl=1, df=1):
                ny += 1
                H.A[site] = DoublePepsTensor(bra=d[nx, ny], ket=d[nx, ny], trans=(1, 2, 3, 0))
            ny += 1
            hr = hair_r(d[nx, ny + 1]) if self._hairs else None
            H.A[H.last] = edge_r(d[nx, ny], hr=hr).add_leg(s=1).transpose(axes=(1, 2, 3, 0))

        else:  # dirn == "v":
            ny = n
            d = {(nx, ny): self.psi.nn_site(s0, d=(nx, ny))
                for nx in range(-self.Nl + 1 - self._hairs, self.Nl + 1 + self._hairs)}
            tensors_from_psi(d, self.psi)
            if ny == 0:
                d[0, 0], d[1, 0] = Q0, Q1

            nx = -self.Nl + 1
            ht = hair_t(d[nx - 1, ny]) if self._hairs else None
            H.A[H.first] = edge_t(d[nx, ny], ht=ht).add_leg(s=-1, axis=0)
            for site in H.sweep(to='last', dl=1, df=1):
                nx += 1
                H.A[site] = DoublePepsTensor(bra=d[nx, ny], ket=d[nx, ny])
            nx += 1
            hb = hair_b(d[nx + 1, ny]) if self._hairs else None
            H.A[H.last] = edge_b(d[nx, ny], hb=hb).add_leg(s=1).transpose(axes=(1, 2, 3, 0))
        return H

    def boundary_mps(self, bd, dirn='r'):
        if dirn in 'lr':
            ny = self.Nw + 1 if dirn == 'r' else -self.Nw - 1
            nns = [(nx, ny) for nx in range(-self.Nl + 1, self.Nl + 1)]
        else: # dirn in 'tb':
            nx = self.Nw + 1 if dirn == 'b' else -self.Nw - 1
            nns = [(nx, ny) for ny in range(-self.Nl + 1, self.Nl + 1)]
        d = {nxy: self.psi.nn_site(bd[0], d=nxy) for nxy in nns}
        tensors_from_psi(d, self.psi)
        if self._hairs:
            hair_dirn = _hair_dirn[dirn]
            return mps.product_mps([hair_dirn(d[nxy]).fuse_legs(axes=[(0, 1)]).conj() for nxy in nns])

        vectors, config = [], self.psi.config
        axis = _axis_dirn[dirn]
        for nxy in nns:
            leg = d[nxy].get_legs(axes=axis)  # peps tensor are now [t l] [b r] s  # TODO check directions
            vectors.append(eye(config, legs=[leg, leg.conj()], isdiag=False).fuse_legs(axes=[(0, 1)]))
        return mps.product_mps(vectors)
