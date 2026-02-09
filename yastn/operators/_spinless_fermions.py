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
""" Generator of basic local spinless-fermion operators. """
from __future__ import annotations

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
