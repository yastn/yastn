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
from typing import Sequence, Union
import yastn
from yastn import Tensor
from ...tn.mps import Mpo
from ._doublePepsTensor import DoublePepsTensor
from ._geometry import SquareLattice, CheckerboardLattice, RectangularUnitcell


class Peps():

    def __init__(self, geometry=None, tensors : Union[None, Sequence[Sequence[Tensor]], dict[tuple[int,int],Tensor] ]= None):
        """
        A PEPS instance on a specified lattice can be initialized as empty or with optional tensors already assigned to each lattice site.

            i), ii)
            iii) geometry and tensors. Here, the tensors and geometry must be compatible in terms of non-equivalent sites ?

        PEPS inherits key methods (e.g., sites, bonds, dims) from the associated lattice geometry.

        Empty PEPS has no tensors assigned.
        Supports :code:`[]` notation to get/set individual tensors.
        PEPS tensors can be either rank-5 (including physical legs) or 
        rank-4 (without physical legs). Leg enumeration follows the 
        order: top, left, bottom, right, and physical leg 

        Example 1
        ---------

        ::

            import yastn
            import yastn.tn.fpeps as fpeps

            geometry = fpeps.CheckerboardLattice()
            psi = fpeps.Peps(geometry)

            config = yastn.make_config(sym='U1')
            leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
            #
            # for rank-5 tensors
            #
            A00 = yastn.rand(config, legs=[leg.conj(), leg, leg, leg.conj(), leg])
            psi[0, 0] = A00
            # Currently, 5-leg PEPS tensors are fused by __setitem__ as ((top-left)(bottom-right) physical).
            # This is done to work with object having smaller number of blocks.
            assert psi[0, 0].ndim == 3
            assert (psi[0, 0].unfuse_legs(axes=(0, 1)) - A00).norm() < 1e-13

            # PEPS with no physical legs is also possible.
            #
            # for rank-4 tensors
            #
            B00 = yastn.rand(config, legs=[leg.conj(), leg, leg, leg.conj()])
            psi[0, 0] = B00
            assert psi[0, 0].ndim == 4

        Example 2
        ---------

        ::
        
            # directly pass the pattern of tensors in the unit cell as a dictionary. The geometry is created implicitly.
            psi = fpeps.PepsExtended(tensors={ (0,0):A00, (0,1):A01, (1,0):A01, (1,1):A00 })
            #
            # or equivalently
            # psi = fpeps.PepsExtended(tensors=[[A00, A01], [A01, A00]])
        """
        if geometry is not None and isinstance(tensors,dict):
            self.geometry = geometry
        elif geometry is None and isinstance(tensors,dict):
            id_map= { uuid: i for i,uuid in enumerate( set([id(t) for t in tensors.values() ])) } # convert to small integers
            self.geometry= RectangularUnitcell(pattern={ site: id_map[id(t)] for site,t in tensors.items() })

        elif geometry is None and isinstance(tensors,Sequence) and set(map(type(row) for row in tensors))==set(Sequence,):
            # TODO
            # for geometry passed as list[list[Tensor]]
            raise NotImplementedError()
        elif geometry is not None and tensors is None:
            self.geometry= geometry

        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "nn_bond_type", "f_ordered"]:
            setattr(self, name, getattr(geometry, name))
        self._data = {self.site2index(site): None for site in self.sites()}

        if isinstance(tensors,dict):
            assert set(self.sites()) <= set(tensors.keys()),"geometry and tensors are not compatible"
            # self._data = {self.site2index(site): tensors[site] for site in self.sites()}
            for site in self.sites():
                self[site] = tensors[site]

    @property
    def config(self):
        return self[0, 0].config

    def has_physical(self) -> bool:
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

    def __dict__(self):
        """
        Serialize PEPS into a dictionary.
        """
        d = {'lattice': type(self.geometry).__name__,
             'dims': self.dims,
             'boundary': self.boundary,
             'pattern': self.geometry.__dict__(),
             'data': {}}

        for site in self.sites():
            d['data'][site] = self[site].save_to_dict()

        return d

    def save_to_dict(self) -> dict:
        """
        Serialize PEPS into a dictionary.
        """
        d= self.__dict__()
        if isinstance(self.geometry, CheckerboardLattice):
            d['lattice'] = "checkerboard"
        elif isinstance(self.geometry, SquareLattice):
            d['lattice'] = "square"
        return d

    def __repr__(self):
        return f"Peps(geometry={self.geometry.__repr__()}, tensors={ self._data })"

    def copy(self):
        r"""
        Returns an independent copy of the PEPS instance with each :class:`yastn.Tensor<yastn.Tensor>` object in
        the network sharing the same data blocks as the original instance. This method does not create a deep copy of tensors;
        each tensor in the copied PEPS will reference the same blocks as in the original.
        """
        psi = Peps(geometry=self.geometry)
        for ind in self._data:
            psi._data[ind] = self._data[ind].copy()
        return psi

    def clone(self):
        r"""
        Returns a deep copy of the PEPS instance by :meth:`cloning<yastn.Tensor.clone>` each tensor in
        the network. Each tensor in the cloned PEPS will contain its own independent data blocks.
        """
        psi = Peps(geometry=self.geometry)
        for ind in self._data:
            psi._data[ind] = self._data[ind].copy()
        return psi

    def transfer_mpo(self, n=0, dirn='v') -> yastn.tn.mps.MpsMpo:
        """
        Converts a specified row or column of the PEPS into a Matrix Product Operator (MPO) representation,
        facilitating boundary or contraction calculations along that direction.

        For tensors with physical legs, the MPO is composed of `DoublePepsTensor` instances (2-layered tensors, see
        :class:`yastn.tn.fpeps.DoublePepsTensor`)
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
                psi = self[site]
                op.A[ny] = psi.transpose(axes=(1, 2, 3, 0)) if psi.ndim == 4 else \
                           DoublePepsTensor(bra=psi, ket=psi, transpose=(1, 2, 3, 0))
        elif dirn == 'v':
            periodic = (self.boundary == "cylinder")
            op = Mpo(N=self.Nx, periodic=periodic)
            op.tol = self._data['tol'] if 'tol' in self._data else None
            for nx in range(self.Nx):
                site = (nx, n)
                psi = self[site]
                op.A[nx] = psi if psi.ndim == 4 else \
                           DoublePepsTensor(bra=psi, ket=psi)
        return op


class Peps2Layers():

    def __init__(self, bra, ket=None):
        """
        PEPS class supporting <bra|ket> contraction.

        If ket is not provided, ket = bra.
        """
        self.bra = bra
        self.ket = bra if ket is None else ket
        assert self.ket.geometry == self.bra.geometry
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
        return DoublePepsTensor(bra=self.bra[site], ket=self.ket[site])
