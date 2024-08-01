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
from yastn import Tensor
from ...tn.mps import Mpo
from ._doublePepsTensor import DoublePepsTensor
from ._geometry import SquareLattice, CheckerboardLattice, RectangularUnitcell


class Peps():

    def __init__(self, geometry=None, tensors : Union[None, Sequence[Sequence[Tensor]], dict[tuple[int,int],Tensor] ]= None):
        """
        Empty PEPS instance on a lattice specified by provided geometry and optionally tensors.

            i), ii)
            iii) geometry and tensors. Here, the tensors and geometry must be compatible in terms of non-equivalent sites ?

        Inherits methods from geometry.

        Empty PEPS has no tensors assigned.
        Supports :code:`[]` notation to get/set individual tensors.
        Intended both for PEPS with physical legs (rank-5 PEPS tensors)
        and without physical legs (rank-4 PEPS tensors).

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
            # Currently, PEPS tensor with 5 legs gets fused as (oo)(oo)o by __setitem__.
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
        return DoublePepsTensor(top=self.ket[site], btm=self.bra[site])
