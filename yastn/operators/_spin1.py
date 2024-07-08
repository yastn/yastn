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
""" Generator of basic local spin-1 operators. """
from __future__ import annotations
import numpy as np
from .. import block
from ..tensor import YastnError, Tensor, Leg
from ._meta_operators import meta_operators

class Spin1(meta_operators):
    # Predefine operators for spin-1 system.
    def __init__(self, sym='dense', **kwargs):
        r"""
        A set of standard operators for 3-dimensional Hilbert space as Spin-1 representation
        of su(2) algebra. Defines identity, :math:`S^z,\ S^x,\ S^y` operators
        and :math:`S^+,\ S^-` raising and lowering operators (if allowed by symmetry).
        Define eigenvectors of :math:`S^z`, :math:`S^x`, :math:`S^y`, and local Hilbert space as a :class:`yastn.Leg`.

        Parameters
        ----------
        sym : str
            Explicit symmetry to used. Allowed options are :code:`'dense'`, ``'Z3'``, or ``'U1'``.

        kwargs
            Other YASTN configuration parameters can be provided, see :meth:`yastn.make_config`.

        Notes
        -----
        The following basis ordering and charge conventions are assumed

            * For :code:`sym='dense'`, the basis order is (sz=+1, sz=0, sz=-1).
            * For :code:`sym='Z3'`, charge t=0 -> sz=+1, t=1 -> sz=0; t=2 -> sz=-1.
            * For :code:`sym='U1'`, charge t=-1 -> sz=-1, t=0 -> sz=0, t=1 -> sz=1; i.e., sz = t.

        When using :meth:`yastn.to_numpy` to recover usual dense representation of the algebra
        for :code:`sym='U1'` symmetry, :code:`reverse=True` is required
        since by default the charges are ordered in the increasing order.

        Default configuration sets :code:`fermionic` to :code:`False`.
        """
        if sym not in ('dense', 'Z3', 'U1'):
            raise YastnError("For Spin1 sym should be in ('dense', 'Z3', 'U1').")
        kwargs['fermionic'] = False
        kwargs['sym'] = sym
        super().__init__(**kwargs)
        self._sym = sym
        self.operators = ('I', 'sx', 'sy', 'isy', 'sz', 'sp', 'sm')

    def space(self) -> yastn.Leg:
        r""" :class:`yastn.Leg` describing local Hilbert space. """
        if self._sym == 'dense':
            leg = Leg(self.config, s=1, D=(3,))
        if self._sym == 'Z3':
            leg = Leg(self.config, s=1, t=(0, 1, 2), D=(1, 1, 1))
        if self._sym == 'U1':
            leg = Leg(self.config, s=1, t=(-1, 0, 1), D=(1, 1, 1))
        return leg

    def I(self) -> yastn.Tensor:
        r""" Identity operator. """
        if self._sym == 'dense':
            I = Tensor(config=self.config, s=self.s)
            I.set_block(val=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], Ds=(3, 3))
        if self._sym in 'Z3':
            I = Tensor(config=self.config, s=self.s, n=0)
            I.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            I.set_block(ts=(1, 1), Ds=(1, 1), val=1)
            I.set_block(ts=(2, 2), Ds=(1, 1), val=1)
        if self._sym in 'U1':
            I = Tensor(config=self.config, s=self.s, n=0)
            I.set_block(ts=(1, 1), Ds=(1, 1), val=1)
            I.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            I.set_block(ts=(-1, -1), Ds=(1, 1), val=1)
        return I

    def sx(self) -> yastn.Tensor:
        r""" Spin-1 :math:`S^x` operator. """
        isq2 = 1 / np.sqrt(2)
        if self._sym == 'dense':
            sx = Tensor(config=self.config, s=self.s)
            sx.set_block(val=[[0, isq2, 0], [isq2, 0, isq2], [0, isq2, 0]], Ds=(3, 3))
        if self._sym in ('Z3', 'U1'):
            raise YastnError('Cannot define sx operator for U1 or Z3 symmetry.')
        return sx

    def sy(self) -> yastn.Tensor:
        r""" Spin-1 :math:`S^y` operator. """
        iisq2 = 1j / np.sqrt(2)
        if self._sym == 'dense':
            sy = Tensor(config=self.config, s=self.s, dtype='complex128')
            sy.set_block(val=[[0, -iisq2, 0], [iisq2, 0, -iisq2], [0, iisq2, 0]], Ds=(3, 3))
        if self._sym in ('Z3', 'U1'):
            raise YastnError('Cannot define sy operator for U1 or Z3 symmetry.')
        return sy

    def isy(self) -> yastn.Tensor:
        r""" Spin-1 :math:`i \cdot S^y` operator with real representation. """
        isq2 = 1 / np.sqrt(2)
        if self._sym == 'dense':
            sy = Tensor(config=self.config, s=self.s)
            sy.set_block(val=[[0, isq2, 0], [-isq2, 0, isq2], [0, -isq2, 0]], Ds=(3, 3))
        if self._sym in ('Z3', 'U1'):
            raise YastnError('Cannot define sy operator for U1 or Z3 symmetry.')
        return sy

    def sz(self) -> yastn.Tensor:
        r""" Spin-1 :math:`S^z` operator. """
        if self._sym == 'dense':
            sz = Tensor(config=self.config, s=self.s)
            sz.set_block(val=[[1, 0, 0], [0, 0, 0], [0, 0, -1]], Ds=(3, 3))
        if self._sym == 'Z3':
            sz = Tensor(config=self.config, s=self.s, n=0)
            sz.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            sz.set_block(ts=(1, 1), Ds=(1, 1), val=0)
            sz.set_block(ts=(2, 2), Ds=(1, 1), val=-1)
        if self._sym == 'U1':
            sz = Tensor(config=self.config, s=self.s, n=0)
            sz.set_block(ts=(1, 1), Ds=(1, 1), val=1)
            sz.set_block(ts=(0, 0), Ds=(1, 1), val=0)
            sz.set_block(ts=(-1, -1), Ds=(1, 1), val=-1)
        return sz

    def vec_z(self, val=1) -> yastn.Tensor:
        r""" Normalized eigenvectors of :math:`S^z`. """
        if self._sym == 'dense' and val == 1:
            vec = Tensor(config=self.config, s=(1,))
            vec.set_block(val=[1, 0, 0], Ds=(3,))
        elif self._sym == 'dense' and val == 0:
            vec = Tensor(config=self.config, s=(1,))
            vec.set_block(val=[0, 1, 0], Ds=(3,))
        elif self._sym == 'dense' and val == -1:
            vec = Tensor(config=self.config, s=(1,))
            vec.set_block(val=[0, 0, 1], Ds=(3,))
        elif self._sym == 'Z3' and val == 1:
            vec = Tensor(config=self.config, s=(1,), n=0)
            vec.set_block(val=[1], ts=(0,), Ds=(1,))
        elif self._sym == 'Z3' and val == 0:
            vec = Tensor(config=self.config, s=(1,), n=1)
            vec.set_block(val=[1], ts=(1,), Ds=(1,))
        elif self._sym == 'Z3' and val == -1:
            vec = Tensor(config=self.config, s=(1,), n=2)
            vec.set_block(val=[1], ts=(2,), Ds=(1,))
        elif self._sym == 'U1' and val == 1:
            vec = Tensor(config=self.config, s=(1,), n=1)
            vec.set_block(val=[1], ts=(1,), Ds=(1,))
        elif self._sym == 'U1' and val == 0:
            vec = Tensor(config=self.config, s=(1,), n=0)
            vec.set_block(val=[1], ts=(0,), Ds=(1,))
        elif self._sym == 'U1' and val == -1:
            vec = Tensor(config=self.config, s=(1,), n=-1)
            vec.set_block(val=[1], ts=(-1,), Ds=(1,))
        else:
            raise YastnError('Eigenvalues val should be in (-1, 0, 1).')
        return vec

    def vec_x(self, val=1) -> yastn.Tensor:
        r""" Normalized eigenvectors of :math:`S^x`. """
        isq2 = 1 / np.sqrt(2)
        if self._sym == 'dense' and val == 1:
            vec = Tensor(config=self.config, s=(1,))
            vec.set_block(val=[1/2, isq2, 1/2], Ds=(3,))
        elif self._sym == 'dense' and val == 0:
            vec = Tensor(config=self.config, s=(1,))
            vec.set_block(val=[-isq2, 0, isq2], Ds=(3,))
        elif self._sym == 'dense' and val == -1:
            vec = Tensor(config=self.config, s=(1,))
            vec.set_block(val=[1/2, -isq2, 1/2], Ds=(3,))
        else:
            raise YastnError('Eigenvalues val should be in (-1, 0, 1) and eigenvectors of Sx are well defined only for dense tensors.')
        return vec

    def vec_y(self, val=1) -> yastn.Tensor:
        r""" Normalized eigenvectors of :math:`S^y`. """
        isq2 = 1 / np.sqrt(2)
        if self._sym == 'dense' and val == 1:
            vec = Tensor(config=self.config, s=(1,), dtype='complex128')
            vec.set_block(val=[-1 / 2, -1j * isq2, 1 / 2], Ds=(3,))
        elif self._sym == 'dense' and val == 0:
            vec = Tensor(config=self.config, s=(1,))
            vec.set_block(val=[isq2, 0, isq2], Ds=(3,))
        elif self._sym == 'dense' and val == -1:
            vec = Tensor(config=self.config, s=(1,), dtype='complex128')
            vec.set_block(val=[-1 / 2, 1j * isq2, 1 / 2], Ds=(3,))
        else:
            raise YastnError('Eigenvalues val should be in (-1, 0, 1) and eigenvectors of Sy are well defined only for dense tensors.')
        return vec

    def sp(self) -> yastn.Tensor:
        r""" Spin-1 raising operator :math:`S^+=S^x + iS^y`. """
        sq2 = np.sqrt(2)
        if self._sym == 'dense':
            sp = Tensor(config=self.config, s=self.s)
            sp.set_block(val=[[0, sq2, 0], [0, 0, sq2], [0, 0, 0]], Ds=(3, 3))
        if self._sym == 'Z3':
            sp = Tensor(config=self.config, s=self.s, n=2)
            sp.set_block(ts=(0, 1), Ds=(1, 1), val=sq2)
            sp.set_block(ts=(1, 2), Ds=(1, 1), val=sq2)
        if self._sym == 'U1':
            sp = Tensor(config=self.config, s=self.s, n=1)
            sp.set_block(ts=(0, -1), Ds=(1, 1), val=sq2)
            sp.set_block(ts=(1, 0), Ds=(1, 1), val=sq2)
        return sp

    def sm(self) -> yastn.Tensor:
        r""" Spin-1 lowering operator :math:`S^-=S^x - iS^y`. """
        sq2 = np.sqrt(2)
        if self._sym == 'dense':
            sm = Tensor(config=self.config, s=self.s)
            sm.set_block(val=[[0, 0, 0], [sq2, 0, 0], [0, sq2, 0]], Ds=(3, 3))
        if self._sym == 'Z3':
            sm = Tensor(config=self.config, s=self.s, n=1)
            sm.set_block(ts=(1, 0), Ds=(1, 1), val=sq2)
            sm.set_block(ts=(2, 1), Ds=(1, 1), val=sq2)
        if self._sym == 'U1':
            sm = Tensor(config=self.config, s=self.s, n=-1)
            sm.set_block(ts=(0, 1), Ds=(1, 1), val=sq2)
            sm.set_block(ts=(-1, 0), Ds=(1, 1), val=sq2)
        return sm

    def vec_s(self) -> yastn.Tensor:
        r"""
        Returns vector of Spin-1 generators, in order: :math:`S^z, S^+, S^-`.
        The generators are indexed by first index of the resulting rank-3 tensors.
        Signature convention is

        ::

                   2(-1)
                   |
            0(-1)--S
                   |
                   1(+1)
        """
        vec_s= block({i: t.add_leg(axis=0, s=-1) for i, t in enumerate(
                        [self.sz(), self.sp(), self.sm()])}, common_legs=(1, 2))
        vec_s= vec_s.drop_leg_history(axes=0)
        return vec_s

    def g(self) -> yastn.Tensor:
        r"""
        Returns rank-2 metric tensor g, such that the quadratic Casimir in terms of basis
        :math:`\vec{S} = [S^z; S^+; S^-]` can be computed as :math:`\vec{S}^T g \vec{S}`. The signature of g is

        ::

            0(+1)--g--1(+1)
        """
        if self._sym == 'dense':
            g = Tensor(config=self.config, s=(1, 1))
            g.set_block(val=[[1, 0, 0], [0, 0, 0.5], [0, 0.5, 0]], Ds=(3, 3))
        if self._sym in 'Z3':
            g = Tensor(config=self.config, s=(1, 1))
            g.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            g.set_block(ts=(1, 2), Ds=(1, 1), val=0.5)
            g.set_block(ts=(2, 1), Ds=(1, 1), val=0.5)
        if self._sym in 'U1':
            g = Tensor(config=self.config, s=(1, 1))
            g.set_block(ts=(1, -1), Ds=(1, 1), val=0.5)
            g.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            g.set_block(ts=(-1, 1), Ds=(1, 1), val=0.5)
        return g

    def to_dict(self):
        """
        Returns
        -------
        dict(str,yastn.Tensor)
            a map from strings to operators
        """
        return {'I': lambda j: self.I(),
                'sx': lambda j: self.sx(),
                'sy': lambda j: self.sy(),
                'sz': lambda j: self.sz(),
                'sp': lambda j: self.sp(),
                'sm': lambda j: self.sm()}
