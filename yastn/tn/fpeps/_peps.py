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
from ... import Tensor, YastnError, block, eye
from ...tn.mps import Mpo, MpsMpoOBC, MpoPBC
from ._doublePepsTensor import DoublePepsTensor
from ._gates_auxiliary import apply_gate_onsite, gate_from_mpo, gate_fix_swap_gate, fill_eye_in_gate

class Peps():

    def __init__(self, geometry, tensors: Union[None, Sequence[Sequence[Tensor]], dict[tuple[int,int],Tensor]]= None):
        r"""
        A PEPS instance on a specified lattice can be initialized as empty (with no tensors assiged) or with tensors assigned to each unique lattice site.

        PEPS inherits key methods (e.g., sites, bonds, dims) from the associated lattice geometry.
        Supports :code:`[]` notation to get/set individual tensors.
        PEPS tensors can be either rank-5 (including physical legs) or rank-4 (without physical legs).
        Leg enumeration follows the order: top, left, bottom, right, and physical leg.

        Parameters
        ----------
        geometry: SquareLattice | CheckerboardLattice | RectangularUnitcell
            Specify lattice geometry.

        tensors: Optional[Sequence[Sequence[Tensor]] | dict[tuple[int,int],Tensor]]]
            Fill in the Peps lattice with tensors.
            Each unique sites should get assigned in a way consistent with the geometry.

        Example
        -------

        ::

            import yastn
            import yastn.tn.fpeps as fpeps

            # PEPS with CheckerboarLattice geometry and no tensors assigned.
            #
            geometry = fpeps.CheckerboardLattice()
            psi = fpeps.Peps(geometry)

            config = yastn.make_config(sym='U1')
            leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
            #
            # for rank-5 tensors
            #
            A00 = yastn.rand(config, legs=[leg.conj(), leg, leg, leg.conj(), leg])
            psi[0, 0] = A00
            #
            # Currently, 5-leg PEPS tensors are fused by __setitem__ as ((top-left)(bottom-right) physical).
            # This is done to work with objects having a smaller number of blocks.
            assert psi[0, 0].ndim == 5
            assert (psi[0, 0] - A00).norm() < 1e-13

            # PEPS with no physical legs is also possible.
            #
            # for rank-4 tensors
            #
            B00 = yastn.rand(config, legs=[leg.conj(), leg, leg, leg.conj()])
            psi[0, 0] = B00
            assert psi[0, 0].ndim == 4

            # PEPS with tensors assigned during initialization
            #
            psi = fpeps.Peps(geometry, tensors={(0, 0): A00, (0, 1): A01})
            #
            # or equivalently
            #
            psi = fpeps.Peps(geometry, tensors=[[A00, A01], [A01, A00]])
            # above, some provided tensors are redundant, although this redundancy is consistent with the geometry.

        """
        self.geometry = geometry
        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "nn_bond_type", "f_ordered", "nn_bond_dirn"]:
            setattr(self, name, getattr(geometry, name))
        self._data = {self.site2index(site): None for site in self.sites()}

        if tensors is not None:
            try:
                if isinstance(tensors, Sequence):
                    dict_tensors = {}
                    for nx, row in enumerate(tensors):
                        for ny, tensor in enumerate(row):
                            dict_tensors[nx, ny] = tensor
                    tensors = dict_tensors
                tmp_tensors = {}  # TODO: remove it when extra fusion of Peps tensors is removed
                for site, tensor in tensors.items():
                    if self[site] is None:
                        self[site] = tensor
                        tmp_tensors[self.site2index(site)] = tensor
                    if tmp_tensors[self.site2index(site)] is not tensor:
                        raise YastnError("Peps: Non-unique assignment of tensor to unique lattice sites.")
            except (KeyError, TypeError):
                raise YastnError("Peps: tensors assigned outside of the lattice geometry.")
            if any(tensor is None for tensor in self._data.values()):
                raise YastnError("Peps: Not all unique lattice sites got assigned with a tensor.")

    @property
    def config(self):
        assert all(self[site].config == self[self.sites()[0]].config for site in self.sites())
        return self[self.sites()[0]].config

    def has_physical(self) -> bool:
        """ Whether PEPS has phyical leg"""
        site0 = self.sites()[0]
        return self[site0].ndim in (3, 5)

    def __getitem__(self, site) -> Tensor:
        """ Get tensor for site. """
        return self._data[self.site2index(site)]

    def __setitem__(self, site, obj):
        """
        Set tensor at site.
        5-leg tensors are fused to 3-leg: [t l] [b r] sa
        2-leg tensors are unfused to 4-leg: t l b r
        """
        if hasattr(obj, 'ndim') and obj.ndim == 2 :
            obj = obj.unfuse_legs(axes=(0, 1))
        self._data[self.site2index(site)] = obj

    def __dict__(self) -> dict:
        """
        Serialize PEPS into a dictionary.
        """
        d = {**self.geometry.__dict__(),
             'data': {}}
        for site in self.sites():
            d['data'][site] = self[site].save_to_dict()
        return d

    def save_to_dict(self) -> dict:
        """
        Serialize PEPS into a dictionary.
        """
        return self.__dict__()

    def __repr__(self) -> str:
        return f"Peps(geometry={self.geometry.__repr__()}, tensors={ self._data })"

    def clone(self) -> Peps:
        r"""
        Returns a deep clone of the PEPS instance by :meth:`cloning<yastn.Tensor.clone>` each tensor in
        the network. Each tensor in the cloned PEPS will contain its own independent data blocks.
        """
        psi = Peps(geometry=self.geometry)
        for ind in self._data:
            psi._data[ind] = self._data[ind].clone()
        return psi

    def copy(self) -> Peps:
        r"""
        Returns a deep copy of the PEPS instance by :meth:`copy<yastn.Tensor.copy>` each tensor in
        the network. Each tensor in the copied PEPS will contain its own independent data blocks.
        """
        psi = Peps(geometry=self.geometry)
        for ind in self._data:
            psi._data[ind] = self._data[ind].copy()
        return psi

    def shallow_copy(self) -> Peps:
        r"""
        New instance of :class:`yastn.tn.mps.Peps` pointing to the same tensors as the old one.

        Shallow copy is usually sufficient to retain the old PEPS.
        """
        psi = Peps(geometry=self.geometry)
        for ind in self._data:
            psi._data[ind] = self._data[ind]
        return psi

    def transfer_mpo(self, n=0, dirn='v') -> MpsMpoOBC | MpoPBC:
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
                op.A[nx] = psi.transpose(axes=(0, 3, 2, 1)) if psi.ndim == 4 else \
                           DoublePepsTensor(bra=psi, ket=psi).transpose(axes=(0, 3, 2, 1))
        return op

    def get_bond_dimensions(self):
        out = {}
        for bond in self.bonds():
            dirn, l_ordered = self.nn_bond_type(bond)
            s0 = bond[0] if l_ordered else bond[1]
            axes = 3 if dirn == 'h' else 2  # if dirn == 'v'
            out[bond] = self[s0].get_shape(axes=axes)
        return out

    def to_tensor(self) -> Tensor:
        r"""
        Contract PEPS into a single tensor.

        It should only be used for a system with a very few sites.
        It works only for a finite system.
        The order of legs is consistent with the PEPS fermionic order.
        System and ancilla legs are unfused.

        Note that ancilla legs might carry charges offsetting the particle number of resulting tensor to zero.
        This might give unexpected behavior when adding two states with different structure of offsets,
        which is set in :meth:`yastn.tn.fpeps.product_peps`.
        """
        if 'ii' == self.geometry._periodic:
            raise YastnError("to_tensor() works only for a finite PEPS.")
        Nx, Ny = self.Nx, self.Ny
        psi = self[0, 0].remove_leg(axis=1)
        for nx in range(1, Nx):
            ten = self[nx, 0].remove_leg(axis=1)
            psi = psi.tensordot(ten, axes=(2 * nx - 1, 0))
        axs = list(range(1, 2 * Nx - 2, 2)) + [2 * Nx]
        psi = psi.swap_gate(axes=(0, axs))
        psi = psi.trace(axes=(0, 2 * Nx - 1))

        for ny in range(1, Ny):
            ten = self[0, ny]
            dny = (ny - 1) * Nx
            psi = psi.tensordot(ten, axes=(dny, 1))
            for nx in range(1, Nx):
                axs = list(range(dny + nx, dny + 2 * Nx - nx, 2))
                psi = psi.swap_gate(axes=(dny + 2 * Nx + nx + 1, axs))
                ten = self[nx, ny]
                psi = psi.tensordot(ten, axes=((dny + 2 * Nx + nx - 1, dny + nx), (0, 1)))

            dny = ny * Nx
            axs = list(range(dny + 1, dny + 2 * Nx - 2, 2)) + [dny + 2 * Nx]
            psi = psi.swap_gate(axes=(dny, axs))
            psi = psi.trace(axes=(dny, dny + 2 * Nx - 1))

        dny = (Ny - 1) * Nx
        for nx in range(Nx):
            psi = psi.remove_leg(axis=dny + nx)

        if psi.get_legs(axes=0).is_fused():
            psi = psi.unfuse_legs(axes=list(range(psi.ndim)))
            N = psi.ndim
            for n in range(1, N, 2):
                axs = list(range(n + 1, N))
                psi = psi.swap_gate(axes=(n, axs))
        return psi

    def apply_gate_(self, gate):
        """
        Apply gate to PEPS state psi, changing respective tensors in place.
        """
        G = gate_from_mpo(gate.G) if isinstance(gate.G, MpsMpoOBC) else gate.G

        if len(G) == 2 and len(gate.sites) > 2:  # fill-in identities
            G = fill_eye_in_gate(self, G, gate.sites)

        dirns, g0, s0 = '', G[0], gate.sites[0]
        for g1, s1 in zip(G[1:], gate.sites[1:]):
            dirn = self.nn_bond_dirn(s0, s1)
            g0, g1 = gate_fix_swap_gate(g0, g1, dirn, self.f_ordered(s0, s1))
            dirns += dirn
            self[s0] = apply_gate_onsite(self[s0], g0, dirn=dirns[:-1])
            dirns = dirns[-1]
            g0, s0 = g1, s1
        self[s0] = apply_gate_onsite(self[s0], g0, dirn=dirns)

    def __add__(self, other) -> Peps:
        return add(self, other)


def add(*states, amplitudes=None, **kwargs) -> MpsMpoOBC:
    r"""
    Linear superposition of several PEPSs with specific amplitudes, i.e., :math:`\sum_j \textrm{amplitudes[j]}{\times}\textrm{states[j]}`.

    Compression (truncation of bond dimensions) is not performed.

    Parameters
    ----------
    states: Sequence[yastn.tn.mps.Peps]

    amplitudes: Sequence[scalar]
        If ``None``, all amplitudes are set to :math:`1`.
    """
    if amplitudes is not None and len(states) != len(amplitudes):
        raise YastnError('Number of Peps-s to add must be equal to the number of coefficients in amplitudes.')

    if any(not isinstance(state, Peps) for state in states):
        raise YastnError('All added states should be Peps-s.')

    geometry = states[0].geometry
    if any(geometry != state.geometry for state in states):
        raise YastnError('All added states should have the same geometry.')

    phi = Peps(geometry)
    for site in geometry.sites():
        tens = {}
        t = geometry.nn_site(site, 't') is not None
        l = geometry.nn_site(site, 'l') is not None
        b = geometry.nn_site(site, 'b') is not None
        r = geometry.nn_site(site, 'r') is not None
        for n, state in enumerate(states):
            ten = state[site]
            if site == (0, 0) and amplitudes is not None:
                ten = amplitudes[n] * ten
            tens[n * t, n * l, n * b, n * r] = ten
        phi[site] = block(tens, common_legs=4)
    return phi


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

        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "nn_bond_type", "f_ordered", "nn_bond_dirn"]:
            setattr(self, name, getattr(bra.geometry, name))

    @property
    def config(self):
        return self.ket.config

    def has_physical(self) -> bool:
        return False

    def __getitem__(self, site) -> DoublePepsTensor:
        """ Get tensor for site. """
        return DoublePepsTensor(bra=self.bra[site], ket=self.ket[site])
