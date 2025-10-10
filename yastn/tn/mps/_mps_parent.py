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
""" MpsMpoParent structure and basic methods common for OBC and PBC. """
from __future__ import annotations
from typing import Iterator, Sequence
from numbers import Number, Integral
from ... import YastnError, Tensor, Leg


class _MpsMpoParent:
    # The basic structure of MPS/MPO with `N` sites.

    def __init__(self, N=1, nr_phys=1):
        r"""
        Parent class extracting common elements of periodic and open boundary MpsMpo
        """
        if not isinstance(N, Integral) or N <= 0:
            raise YastnError('Number of Mps sites N should be a positive integer.')
        if nr_phys not in (1, 2):
            raise YastnError('Number of physical legs, nr_phys, should be 1 or 2.')
        self._N = N
        self.A = {i: None for i in range(N)}  # dict of mps tensors; indexed by integers
        self.pC = None  # index of the central block, None if it does not exist
        self._first = 0  # index of the first lattice site
        self._last = N - 1  # index of the last lattice site
        self._nr_phys = nr_phys
        self.factor = 1  # multiplicative factor is real and positive (e.g. norm)

    @property
    def first(self):
        return self._first

    @property
    def last(self):
        return self._last

    @property
    def nr_phys(self):
        return self._nr_phys

    @property
    def N(self):
        return self._N

    def __len__(self):
        return self._N

    @property
    def config(self):
        return self.A[0].config

    def sweep(self, to='last', df=0, dl=0) -> Iterator[int]:
        r"""
        Generator of indices of all sites going from the first site to the last site, or vice-versa.

        Parameters
        ----------
        to: str
            'first' or 'last'.
        df: int
            shift iterator by :math:`{\rm df}\ge 0` from the first site.
        dl: int
            shift iterator by :math:`{\rm dl}\ge 0` from the last site.
        """
        if to == 'last':
            return range(df, self.N - dl)
        if to == 'first':
            return range(self.N - 1 - dl, df - 1, -1)
        raise YastnError('"to" in sweep should be in "first" or "last"')

    def __getitem__(self, n) -> Tensor:
        """ Return tensor corresponding to n-th site."""
        try:
            return self.A[n]
        except KeyError as e:
            raise YastnError(f"MpsMpoOBC does not have site with index {n}") from e

    def __setitem__(self, n, tensor):
        """
        Assign tensor to n-th site of Mps or Mpo.

        Assigning central block is not supported.
        """
        if not isinstance(n, Integral) or n < self.first or n > self.last:
            raise YastnError("MpsMpoOBC: n should be an integer in 0, 1, ..., N-1")
        if tensor.ndim != self.nr_phys + 2:
            raise YastnError(f"MpsMpoOBC: Tensor rank should be {self.nr_phys + 2}")
        self.A[n] = tensor

    def shallow_copy(self) -> _MpsMpoParent:
        r"""
        New instance of :class:`yastn.tn.mps.MpsMpoOBC` pointing to the same tensors as the old one.

        Shallow copy is usually sufficient to retain the old MPS/MPO.
        """
        phi = type(self)(N=self.N, nr_phys=self.nr_phys)
        phi.A = dict(self.A)
        phi.pC = self.pC
        phi.factor = self.factor
        if hasattr(self, 'flag'):
            phi.flag = self.flag
        return phi

    def on_bra(self) -> _MpsMpoParent:
        r"""
        A shallow copy of the tensor with an added ``on_bra`` flag.

        The flag is only relevant in functions using :meth:`Env()<yastn.tn.mps.Env>`.
        It makes the Mpo operator acting on an Mpo state to be applied on the bra legs (or auxiliary legs),
        instead of a default application on ket legs.
        For instance, :code:`Heff = [-H, H.on_bra()]` can be used to evolve an operator in the Heisenberg picture.

        This flag gets propagated by __mul__, conj, transpose, and other functions employing shallow_copy.
        It is not saved/loaded, or propagated by more complicated functions.
        """
        phi = self.shallow_copy()
        phi.flag = 'on_bra'
        return phi

    def clone(self) -> _MpsMpoParent:
        r"""
        Makes a clone of MPS or MPO by :meth:`cloning<yastn.Tensor.clone>`
        all :class:`yastn.Tensor<yastn.Tensor>`'s into a new and independent :class:`yastn.tn.mps.MpsMpoOBC`.
        """
        phi = self.shallow_copy()
        for ind, ten in phi.A.items():
            phi.A[ind] = ten.clone()  # TODO clone factor ?
        return phi

    def copy(self) -> _MpsMpoParent:
        r"""
        Makes a copy of MPS or MPO by :meth:`copying<yastn.Tensor.copy>` all :class:`yastn.Tensor<yastn.Tensor>`'s
        into a new and independent :class:`yastn.tn.mps.MpsMpoOBC`.
        """
        phi = self.shallow_copy()
        for ind, ten in phi.A.items():
            phi.A[ind] = ten.copy()
        return phi

    def conj(self) -> _MpsMpoParent:
        """ Makes a conjugation of the object. """
        phi = self.shallow_copy()
        for ind, ten in phi.A.items():
            phi.A[ind] = ten.conj()
        return phi

    def transpose(self) -> _MpsMpoParent:
        """ Transpose of MPO. For MPS, return self."""
        if self.nr_phys == 1:
            return self
        phi = self.shallow_copy()
        for n in phi.sweep(to='last'):
            phi.A[n] = phi.A[n].transpose(axes=(0, 3, 2, 1))
        return phi

    def conjugate_transpose(self) -> _MpsMpoParent:
        """ Transpose and conjugate MPO. For MPS, return self.conj()."""
        if self.nr_phys == 1:
            return self.conj()
        phi = self.shallow_copy()
        for n in phi.sweep(to='last'):
            phi.A[n] = phi.A[n].transpose(axes=(0, 3, 2, 1)).conj()
        return phi

    @property
    def T(self) -> _MpsMpoParent:
        r""" Transpose of MPO. For MPS, return self.

        Same as :meth:`self.transpose()<yastn.tn.mps.MpsMpoOBC.transpose>` """
        return self.transpose()

    @property
    def H(self) -> _MpsMpoParent:
        r""" Transpose and conjugate of MPO. For MPS, return self.conj().

        Same as :meth:`self.conjugate_transpose()<yastn.tn.mps.MpsMpoOBC.conjugate_transpose>` """
        return self.conjugate_transpose()

    def reverse_sites(self) -> _MpsMpoParent:
        r"""
        New MPS/MPO with reversed order of sites and respectively transposed virtual tensor legs.
        """
        phi = type(self)(N=self.N, nr_phys=self.nr_phys)
        phi.factor = self.factor
        axes = (2, 1, 0) if self.nr_phys == 1 else (2, 1, 0, 3)
        for n in phi.sweep(to='last'):
            phi.A[n] = self.A[self.N - n - 1].transpose(axes=axes)
        if self.pC is None:
            phi.pC = None
        else:
            phi.pC = (self.N - self.pC[1] - 1, self.N - self.pC[0] - 1)
            phi.A[phi.pC] = self.A[self.pC].transpose(axes=(1, 0))
        return phi

    def __mul__(self, number) -> _MpsMpoParent:
        """ New MPS/MPO with the first tensor multiplied by a scalar. """
        phi = self.shallow_copy()
        am = abs(number)
        if am > 0:
            phi.factor = am * self.factor
            phi.A[0] = phi.A[0] * (number / am)
        else:
            phi.factor = am * self.factor
        return phi

    def __rmul__(self, number) -> _MpsMpoParent:
        """ New MPS/MPO with the first tensor multiplied by a scalar. """
        return self.__mul__(number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """ This is to circumvent problems with np.float64 * Mps. """
        if ufunc.__name__ == 'multiply':
            lhs, rhs = inputs
            return rhs.__mul__(lhs)
        raise YastnError(f"Only np.float * Mps is supported; {ufunc.__name__} was called.")

    def __neg__(self):
        return self.__mul__(-1)

    def __truediv__(self, number) -> _MpsMpoParent:
        """ Divide MPS/MPO by a scalar. """
        return self.__mul__(1 / number)

    def virtual_leg(self, ind):
        if ind == 'first':
            return self.A[self.first].get_legs(axes=0)
        if ind == 'last':
            return self.A[self.last].get_legs(axes=2)

    def get_bond_dimensions(self) -> Sequence[int]:
        r"""
        Returns total bond dimensions of all virtual spaces along MPS/MPO from
        the first to the last site, including trivial leftmost and rightmost virtual spaces.
        This gives a tuple with N+1 elements.
        """
        Ds = [self.A[n].get_shape(axes=0) for n in self.sweep(to='last')]
        Ds.append(self.A[self.last].get_shape(axes=2))
        return tuple(Ds)

    def get_bond_charges_dimensions(self) -> Sequence[dict[Sequence[int], int]]:
        r"""
        Returns list of charge sectors and their dimensions for all virtual spaces along MPS/MPO
        from the first to the last site, including trivial leftmost and rightmost virtual spaces.
        Each element of the list is a dictionary {charge: sectorial bond dimension}.
        This gives a list with N+1 elements.
        """
        tDs = []
        for n in self.sweep(to='last'):
            leg = self.A[n].get_legs(axes=0)
            tDs.append(leg.tD)
        leg = self.A[self.last].get_legs(axes=2)
        tDs.append(leg.tD)
        return tDs

    def get_virtual_legs(self) -> Sequence[Leg]:
        r"""
        Returns :class:`yastn.Leg` of all virtual spaces along MPS/MPO from
        the first to the last site, in the form of the `0th` leg of each MPS/MPO tensor.
        Finally, append the rightmost virtual spaces, i.e., `2nd` leg of the last tensor,
        conjugating it so that all legs have signature `-1`.
        This gives a list with N+1 elements.
        """
        legs = [self.A[n].get_legs(axes=0) for n in self.sweep(to='last')]
        legs.append(self.A[self.last].get_legs(axes=2).conj())
        return legs

    def get_physical_legs(self) -> Sequence[Leg] | Sequence[tuple[Leg, Leg]]:
        r"""
        Returns :class:`yastn.Leg` of all physical spaces along MPS/MPO from
        the first to the last site. For MPO return a tuple of ket and bra spaces for each site.
        """
        if self.nr_phys == 2:
            return [self.A[n].get_legs(axes=(1, 3)) for n in self.sweep(to='last')]
        return [self.A[n].get_legs(axes=1) for n in self.sweep(to='last')]


    def save_to_dict(self) -> dict[str, dict | Number]:
        r"""
        Serialize MPS/MPO into a dictionary.

        Each element represents serialized :class:`yastn.Tensor`
        (see, :meth:`yastn.Tensor.save_to_dict`) of the MPS/MPO.
        Absorbs central block if it exists.
        """
        psi = self.shallow_copy()
        psi.absorb_central_()  # make sure central block is eliminated
        out_dict = {
            'N': psi.N,
            'nr_phys': psi.nr_phys,
            'factor': psi.factor, #.item(),
            'A': {}
        }
        for n in psi.sweep(to='last'):
            out_dict['A'][n] = psi[n].save_to_dict()
        return out_dict

    def save_to_hdf5(self, file, my_address):
        r"""
        Save MPS/MPO into a HDF5 file.

        Parameters
        ----------
        file: File
            A `pointer` to a file opened by the user

        my_address: str
            Name of a group in the file, where the Mps will be saved, e.g., 'state/'
        """
        psi = self.shallow_copy()
        try:
            factor = psi.config.backend.to_numpy(psi.factor)
        except:
            factor = psi.factor
        psi.absorb_central_()  # make sure central block is eliminated
        file.create_dataset(my_address+'/N', data=psi.N)
        file.create_dataset(my_address+'/nr_phys', data=psi.nr_phys)
        file.create_dataset(my_address+'/factor', data=factor)
        for n in self.sweep(to='last'):
            psi[n].save_to_hdf5(file, my_address+'/A/'+str(n))

    def to_matrix(self) -> Tensor:
        r"""
        Contract MPS/MPO to a single tensor and then reshape to vector/matrix by fusing all physical legs into a single leg.
        For MPOs also all dual(bra) legs are fused into a single leg.
        """
        axes = (tuple(range(self.N)),) if self.nr_phys == 1 else (tuple(range(0, 2 * self.N, 2)), tuple(range(1, 2 * self.N, 2)))
        return self.to_tensor().fuse_legs(axes=axes)
