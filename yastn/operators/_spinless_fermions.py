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
""" Generator of basic local spinless-fermion operators. """
from __future__ import annotations

from typing import Callable

from ._meta_operators import meta_operators
from ..tensor import YastnError, Tensor, Leg


class SpinlessFermions(meta_operators):
    """ Predefine operators for spinless fermions. """

    def __init__(self, **kwargs):
        r"""
        Standard operators for single fermionic species and 2-dimensional Hilbert space.
        Defines identity, creation, annihilation, and density operators.
        Defines vectors for empty and occupied states, and local Hilbert space as a :class:`yastn.Leg`.

        Parameters
        ----------
            sym : str
                Explicit symmetry to be used. Allowed options are :code:`'Z2'`, or :code:`'U1'`.

            kwargs : any
                Passed to :meth:`yastn.make_config` to change backend,
                default_device or other config parameters.

        Fixes :code:`fermionic` fields in config to :code:`True`.
        """
        if 'fermionic' not in kwargs:
            kwargs['fermionic'] = True
        super().__init__(**kwargs)
        if self._sym not in ('Z2', 'U1'):
            raise YastnError("For SpinlessFermions sym should be in ('Z2', 'U1').")
        if self.config.fermionic != True:
            raise YastnError("For SpinlessFermions config.fermionic should be True.")
        self.operators = ('I', 'n', 'c', 'cp')

    def space(self) -> Leg:
        r""" :class:`yastn.Leg` object describing local Hilbert space. """
        return Leg(self.config, s=1, t=(0, 1), D=(1, 1))  # the same for U1 and Z2

    def key(self) -> Callable | None:
        r"""
        Provide key parameter for :meth:`yastn.to_nonsymmetric`,
        used while sorting block charges,
        so that all symmetries result in the same dense representation of physical space,
        using the order (|0>, |1>).
        """
        return None

    def I(self) -> Tensor:
        r""" Identity operator. """
        I = Tensor(config=self.config, s=self.s, n=0)
        I.set_block(ts=(0, 0), Ds=(1, 1), val=1)
        I.set_block(ts=(1, 1), Ds=(1, 1), val=1)
        return I

    def n(self) -> Tensor:
        r""" Particle number operator. """
        n = Tensor(config=self.config, s=self.s, n=0)
        # n.set_block(ts=(0, 0), Ds=(1, 1), val=0)
        n.set_block(ts=(1, 1), Ds=(1, 1), val=1)
        return n

    def vec_n(self, val=0) -> Tensor:
        r""" State with occupation 0 or 1. """
        if val not in (0, 1):
            raise YastnError("Occupation val should be in (0, 1).")
        vec = Tensor(config=self.config, s=(1,), n=val)
        vec.set_block(ts=(val,), Ds=(1,), val=1)
        return vec

    def cp(self) -> Tensor:
        r""" Raising operator. """
        cp = Tensor(config=self.config, s=self.s, n=1)
        cp.set_block(ts=(1, 0), Ds=(1, 1), val=1)
        return cp

    def c(self) -> Tensor:
        r""" Lowering operator. """
        n = 1 if self._sym == 'Z2' else -1
        c = Tensor(config=self.config, s=self.s, n=n)
        c.set_block(ts=(0, 1), Ds=(1, 1), val=1)
        return c

    def to_dict(self):
        return {'I': lambda j: self.I(),
                'n': lambda j: self.n(),
                'c': lambda j: self.c(),
                'cp': lambda j: self.cp()}
