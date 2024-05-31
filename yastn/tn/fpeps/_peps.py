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
from ...tn.mps import Mpo
from ._doublePepsTensor import DoublePepsTensor
from ._geometry import SquareLattice, CheckerboardLattice


class Peps():

    def __init__(self, geometry):
        """
        Empty PEPS instance on a lattice specified by provided geometry.

        Empty PEPS has no tensors assigned.
        Supports [] notation to get/set individual tensors.
        Inherits methods from geometry.

        Example
        -------

        ::

            geometry = fpeps.CheckerboardLattice()
            psi = fpeps.Peps(geometry)

            config = yastn.make_config(sym='U1')
            leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
            A00 = yastn.rand(config, legs=[leg.conj(), leg, leg.conj(), leg])
            psi[0, 0] = A00
            ten = psi[0, 0]
            assert ten.ndim == 3
            # Currently, Peps tensor with 5 legs gets fused as (oo)(oo)o
            # in the process of setting to limit the number of tensor blocks.

            A00 = yastn.rand(config, legs=[leg.conj(), leg, leg.conj()])
            psi[0, 0] = A00
            ten = psi[0, 0]
            assert ten.ndim == 4
             # Peps with no physical legs are also possible and handled by algorithms like ctmrg.
        """
        self.geometry = geometry
        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "nn_bond_type", "f_ordered"]:
            setattr(self, name, getattr(geometry, name))
        self._data = {self.site2index(site): None for site in self.sites()}

    @property
    def config(self):
        return self[0, 0].config

    def has_physical(self):
        """ Whether PEPS has phyical leg"""
        return self[0, 0].ndim in (3, 5)

    def __getitem__(self, site):
        """ Get tensor for site. """
        return self._data[self.site2index(site)]

    def __setitem__(self, site, obj):
        """
        Set tensor at site.
        5-leg tensors are fused to 3-leg: [t l] [b r] sa
        2-leg tensors are unfused to 4-leg: t l b r
        """
        if hasattr(obj, 'ndim') and obj.ndim == 5 :
            obj = obj.fuse_legs(axes=((0, 1), (2, 3), 4))
        if hasattr(obj, 'ndim') and obj.ndim == 2 :
            obj = obj.unfuse_legs(axes=(0, 1))
        self._data[self.site2index(site)] = obj

    def save_to_dict(self):
        """
        Serialize PEPS into a dictionary.
        """
        if isinstance(self.geometry, CheckerboardLattice):
            lattice = "checkerboard"
        elif isinstance(self.geometry, SquareLattice):
            lattice = "square"

        d = {'lattice': lattice,
             'dims': self.dims,
             'boundary': self.boundary,
             'data': {}}

        for site in self.sites():
            d['data'][site] = self[site].save_to_dict()

        return d

    def copy(self):
        r"""
        Makes a copy of PEPS by :meth:`copying<yastn.Tensor.copy>` all :class:`yastn.Tensor<yastn.Tensor>`'s
        into a new and independent :class:`yastn.tn.fpeps.Peps`.
        """
        psi = Peps(geometry=self.geometry)
        for ind in self._data:
            psi._data[ind] = self._data[ind].copy()
        return psi

    def clone(self):
        r"""
        Makes a clone of PEPS by :meth:`copying<yastn.Tensor.clone>` all :class:`yastn.Tensor<yastn.Tensor>`'s
        into a new and independent :class:`yastn.tn.fpeps.Peps`.
        """
        psi = Peps(geometry=self.geometry)
        for ind in self._data:
            psi._data[ind] = self._data[ind].copy()
        return psi

    def transfer_mpo(self, n=0, dirn='v') -> yastn.tn.mps.MpsMpo:
        """
        Converts a specific row or column of PEPS into MPO.

        For tensors with physical leg, Mpo consists of DoublePepsTensor (2-layers).
        For tensors without a physical leg directly use Peps tensors (1-layer).

        Parameters
        ----------
        n: int
            index of row or column.
        dirn: str
            'v' for column, 'h' for row.
        """

        if dirn == 'h':
            op = Mpo(N=self.Ny)
            for ny in range(self.Ny):
                site = (n, ny)
                top = self[site]
                op.A[ny] = top.transpose(axes=(1, 2, 3, 0)) if top.ndim == 4 else \
                           DoublePepsTensor(top=top, btm=top, transpose=(1, 2, 3, 0))
        elif dirn == 'v':
            periodic = (self.boundary == "cylinder")
            op = Mpo(N=self.Nx, periodic=periodic)
            op.tol = self._data['tol'] if 'tol' in self._data else None
            for nx in range(self.Nx):
                site = (nx, n)
                top = self[site]
                op.A[nx] = top if top.ndim == 4 else \
                           DoublePepsTensor(top=top, btm=top)
        return op


class Peps2Layers():

    def __init__(self, bra, ket=None):
        """
        Initialize empty PEPS on a lattice specified by provided geometry.

        Empty PEPS has no tensors assigned.
        """
        self.bra = bra
        self.ket = bra if ket is None else ket
        self.geometry = bra.geometry

        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "nn_bond_type", "f_ordered"]:
            setattr(self, name, getattr(bra.geometry, name))

    @property
    def config(self):
        return self.ket.config

    def has_physical(self) -> bool:
        return False

    def __getitem__(self, site) -> yastn.tn.fpeps.DoublePepsTensor:
        """ Get tensor for site. """
        return DoublePepsTensor(top=self.ket[site], btm=self.bra[site])
