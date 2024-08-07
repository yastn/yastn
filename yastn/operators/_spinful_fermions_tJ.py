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
""" Generator of basic local spinful-fermion operators for tJ model. """
from __future__ import annotations
from ..tensor import YastnError, Tensor, Leg
from ._meta_operators import meta_operators

class SpinfulFermions_tJ(meta_operators):
    """ Predefine operators for spinful fermions. """

    def __init__(self, **kwargs):
        r"""
        Generator of standard operators for local Hilbert space with two fermionic species
        and 3-dimensional Hilbert space where double occupancy is excluded.

        Predefine identity, creation, annihilation, and density operators.
        Defines vectors with possible occupations, and local Hilbert space as a :class:`yastn.Leg`.

        Parameters
        ----------

        **kwargs : any
            Passed to :meth:`yastn.make_config` to change backend, default_device or other config parameters.

        Notes
        -----
        For 'U1xU1xZ2', the two species (spin-up and spin-down) are treated as indistinguishable.
        In that case, creation and annihilation operators of the two species anti-commute
        (fermionic statistics is encoded in the Z2 channel).
        """
        if 'fermionic' not in kwargs and isinstance(kwargs['sym'], str):
            kwargs['fermionic'] = (False, False, True) if kwargs['sym'] == 'U1xU1xZ2' else True
        super().__init__(**kwargs)

        sym = self._sym
        fer = self.config.fermionic

        if sym not in ('Z2', 'U1', 'U1xU1', 'U1xU1xZ2'):
            raise YastnError("For SpinfulFermions_tJ sym should be in ('Z2', 'U1', 'U1xU1', 'U1xU1xZ2').")
        if (sym == 'U1xU1xZ2' and fer != (False, False, True)) or \
           (sym in ('Z2', 'U1') and fer != True) or \
           (sym == 'U1xU1' and fer not in (True, (True, True))):
            raise YastnError("For SpinfulFermions_tJ config.sym does not match config.fermionic.")
        self.operators = ('I', 'n', 'c', 'cp')

    def space(self) -> yastn.Leg:
        r""" :class:`yastn.Leg` describing local Hilbert space. """
        if self._sym == 'Z2':  # charges: 0 = |00>; 1 = (|10>, |01>)  |occ_u, occ_d>
            return Leg(self.config, s=1, t=(0, 1), D=(1, 2))
        if self._sym == 'U1':  # charges: 0 = |00>; 1 = (|10>, |01>)
            return Leg(self.config, s=1, t=(0, 1), D=(1, 2))
        if self._sym == 'U1xU1xZ2':  # charges == (occ_u, occ_d, parity)
            return Leg(self.config, s=1, t=((0, 0, 0), (0, 1, 1), (1, 0, 1)), D=(1, 1, 1))
        if self._sym == 'U1xU1':  # charges == (occ_u, occ_d)
            return Leg(self.config, s=1, t=((0, 0), (0, 1), (1, 0)), D=(1, 1, 1))

    def vec_n(self, val=(0, 0)) -> yastn.Tensor:
        r""" Vector with occupation (u, d). """
        if val == (0, 0):
            if self._sym == 'Z2':  # charges: 0 = |00>; 1 = (|10>, |01>)  |occ_u, occ_d>
                vec = Tensor(config=self.config, s=(1,), n=0)
                vec.set_block(ts=(0,), Ds=(1,), val=1)
            elif self._sym == 'U1':  # charges: 0 = |00>; 1 = (|10>, |01>)
                vec = Tensor(config=self.config, s=(1,), n=0)
                vec.set_block(ts=(0,), Ds=(1,), val=1)
            if self._sym == 'U1xU1xZ2':  # charges == (occ_u, occ_d, parity)
                vec = Tensor(config=self.config, s=(1,), n=(0, 0, 0))
                vec.set_block(ts=((0, 0, 0),), Ds=(1,), val=1)
            elif self._sym == 'U1xU1':  # charges == (occ_u, occ_d)
                vec = Tensor(config=self.config, s=(1,), n=(0, 0))
                vec.set_block(ts=((0, 0),), Ds=(1,), val=1)
        elif val == (1, 0):
            if self._sym == 'Z2':  # charges: 0 = |00>; 1 = (|10>, |01>)  |occ_u, occ_d>
                vec = Tensor(config=self.config, s=(1,), n=1)
                vec.set_block(ts=(1,), Ds=(2,), val=[1, 0])
            elif self._sym == 'U1':  # charges: 0 = |00>; 1 = (|10>, |01>)
                vec = Tensor(config=self.config, s=(1,), n=1)
                vec.set_block(ts=(1,), Ds=(2,), val=[1, 0])
            elif self._sym == 'U1xU1xZ2':  # charges == (occ_u, occ_d, parity)
                vec = Tensor(config=self.config, s=(1,), n=(1, 0, 1))
                vec.set_block(ts=((1, 0, 1),), Ds=(1,), val=1)
            elif self._sym == 'U1xU1':  # charges == (occ_u, occ_d)
                vec = Tensor(config=self.config, s=(1,), n=(1, 0))
                vec.set_block(ts=((1, 0),), Ds=(1,), val=1)
        elif val == (0, 1):
            if self._sym == 'Z2':  # charges: 0 = |00>; 1 = (|10>, |01>)  |occ_u, occ_d>
                vec = Tensor(config=self.config, s=(1,), n=1)
                vec.set_block(ts=(1,), Ds=(2,), val=[0, 1])
            elif self._sym == 'U1':  # charges: 0 = |00>; 1 = (|10>, |01>)
                vec = Tensor(config=self.config, s=(1,), n=1)
                vec.set_block(ts=(1,), Ds=(2,), val=[0, 1])
            elif self._sym == 'U1xU1xZ2':  # charges == (occ_u, occ_d, parity)
                vec = Tensor(config=self.config, s=(1,), n=(0, 1, 1))
                vec.set_block(ts=((0, 1, 1),), Ds=(1,), val=1)
            elif self._sym == 'U1xU1':  # charges == (occ_u, occ_d)
                vec = Tensor(config=self.config, s=(1,), n=(0, 1))
                vec.set_block(ts=((0, 1),), Ds=(1,), val=1)
        else:
            raise YastnError('For SpinfulFermions_tJ val in vec_n should be (0, 0), (1, 0), or (0, 1).')
        return vec

    def I(self) -> yastn.Tensor:
        r""" Identity operator in 4-dimensional Hilbert space. """
        I = Tensor(config=self.config, s=self.s)
        if self._sym == 'Z2':  # charges: 0 = |00>; 1 = (|10>, |01>)  |occ_u, occ_d>
            I.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            I.set_block(ts=(1, 1), Ds=(2, 2), val=[[1, 0], [0, 1]])
        elif self._sym == 'U1':  # charges: 0 = |00>; 1 = (|10>, |01>)
            I.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            I.set_block(ts=(1, 1), Ds=(2, 2), val=[[1, 0], [0, 1]])
        elif self._sym == 'U1xU1xZ2':  # charges == (occ_u, occ_d, parity)
            for t in [(0, 0, 0), (0, 1, 1), (1, 0, 1)]:
                I.set_block(ts=(t, t), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1':  # charges == (occ_u, occ_d)
            for t in [(0, 0), (0, 1), (1, 0)]:
                I.set_block(ts=(t, t), Ds=(1, 1), val=1)
        return I

    def n(self, spin='u'):
        """ Particle number operator, with spin='u' for spin-up, and 'd' for spin-down. """
        n = Tensor(config=self.config, s=self.s)
        if spin == 'u':
            if self._sym == 'Z2':  # charges: 0 = |00>; 1 = (|10>, |01>)  |occ_u, occ_d>
                n.set_block(ts=(1, 1), Ds=(2, 2), val=[[1, 0], [0, 0]])
            elif self._sym == 'U1':  # charges: 0 = |00>; 1 = (|10>, |01>)
                n.set_block(ts=(1, 1), Ds=(2, 2), val=[[1, 0], [0, 0]])
            elif self._sym == 'U1xU1xZ2':  # charges == (occ_u, occ_d, parity)
                n.set_block(ts=((1, 0, 1), (1, 0, 1)), Ds=(1, 1), val=1)
            elif self._sym == 'U1xU1':  # charges == (occ_u, occ_d)
                n.set_block(ts=((1, 0), (1, 0)), Ds=(1, 1), val=1)
        elif spin == 'd':
            if self._sym == 'Z2':  # charges: 0 = |00>; 1 = (|10>, |01>)  |occ_u, occ_d>
                n.set_block(ts=(1, 1), Ds=(2, 2), val=[[0, 0], [0, 1]])
            elif self._sym == 'U1':  # charges: 0 = |00>; 1 = (|10>, |01>)
                n.set_block(ts=(1, 1), Ds=(2, 2), val=[[0, 0], [0, 1]])
            elif self._sym == 'U1xU1xZ2':  # charges == (occ_u, occ_d, parity)
                n.set_block(ts=((0, 1, 1), (0, 1, 1)), Ds=(1, 1), val=1)
            elif self._sym == 'U1xU1':  # charges == (occ_u, occ_d)
                n.set_block(ts=((0, 1), (0, 1)), Ds=(1, 1), val=1)
        else:
            raise YastnError("spin should be equal 'u' or 'd'.")
        return n

    def h(self):
        """ hole number operator"""
        if self._sym == 'Z2':  # charges: 0 = |00>; 1 = (|10>, |01>)  |occ_u, occ_d>
            h = Tensor(config=self.config, s=self.s, n=0)
            h.set_block(ts=(0, 0), Ds=(1, 1), val=1)
        elif self._sym == 'U1':  # charges: 0 = |00>; 1 = (|10>, |01>)
            h = Tensor(config=self.config, s=self.s, n=0)
            h.set_block(ts=(0, 0), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1xZ2':  # charges == (occ_u, occ_d, parity)
            h = Tensor(config=self.config, s=self.s, n=(0, 0, 0))
            h.set_block(ts=((0, 0, 0), (0, 0, 0)), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1':  # charges == (occ_u, occ_d)
            h = Tensor(config=self.config, s=self.s, n=(0, 0))
            h.set_block(ts=((0, 0), (0, 0)), Ds=(1, 1), val=1)
        return h

    def cp(self, spin='u') -> yastn.Tensor:
        r""" Creation operator, with spin='u' for spin-up, and 'd' for spin-down. """
        if spin == 'u':
            if self._sym == 'Z2':  # charges: 0 = |00>; 1 = (|10>, |01>)  |occ_u, occ_d>
                cp = Tensor(config=self.config, s=self.s, n=1)
                cp.set_block(ts=(1, 0), Ds=(2, 1), val=[1, 0,])
            elif self._sym == 'U1':  # charges: 0 = |00>; 1 = (|10>, |01>)
                cp = Tensor(config=self.config, s=self.s, n=1)
                cp.set_block(ts=(1, 0), Ds=(2, 1), val=[1, 0])
            elif self._sym == 'U1xU1xZ2':  # charges == (occ_u, occ_d, parity)
                cp = Tensor(config=self.config, s=self.s, n=(1, 0, 1))
                cp.set_block(ts=((1, 0, 1), (0, 0, 0)), Ds=(1, 1), val=1)
            elif self._sym == 'U1xU1':  # charges == (occ_u, occ_d)
                cp = Tensor(config=self.config, s=self.s, n=(1, 0))
                cp.set_block(ts=((1, 0), (0, 0)), Ds=(1, 1), val=1)
        elif spin == 'd':
            if self._sym == 'Z2':  # charges: 0 = |00>; 1 = (|10>, |01>)  |occ_u, occ_d>
                cp = Tensor(config=self.config, s=self.s, n=1)
                cp.set_block(ts=(1, 0), Ds=(2, 1), val=[0, 1])
            elif self._sym == 'U1':  # charges: 0 = |00>; 1 = (|10>, |01>)
                cp = Tensor(config=self.config, s=self.s, n=1)
                cp.set_block(ts=(1, 0), Ds=(2, 1), val=[0, 1])
            elif self._sym == 'U1xU1xZ2':  # charges == (occ_u, occ_d, parity)
                cp = Tensor(config=self.config, s=self.s, n=(0, 1, 1))
                cp.set_block(ts=((0, 1, 1), (0, 0, 0)), Ds=(1, 1), val=1)
            elif self._sym == 'U1xU1':  # charges == (occ_u, occ_d)
                cp = Tensor(config=self.config, s=self.s, n=(0, 1))
                cp.set_block(ts=((0, 1), (0, 0)), Ds=(1, 1), val=1)
        else:
            raise YastnError("spin should be equal 'u' or 'd'.")
        return cp

    def c(self, spin='u') -> yastn.Tensor:
        r""" Annihilation operator, with spin='u' for spin-up, and 'd' for spin-down. """
        if spin == 'u':
            if self._sym == 'Z2': # charges: 0 <-> (|00>,); 1 <-> (|10>, |01>)
                c = Tensor(config=self.config, s=self.s, n=1)
                c.set_block(ts=(0, 1), Ds=(1, 2), val=[1, 0])
            elif self._sym == 'U1':  # charges: 0 = |00>; 1 = (|10>, |01>)
                c = Tensor(config=self.config, s=self.s, n=-1)
                c.set_block(ts=(0, 1), Ds=(1, 2), val=[1, 0])
            elif self._sym == 'U1xU1xZ2': # charges <-> (ocupation up, occupation down, total_parity)
                c = Tensor(config=self.config, s=self.s, n=(-1, 0, 1))
                c.set_block(ts=((0, 0, 0), (1, 0, 1)), Ds=(1, 1), val=1)
            elif self._sym == 'U1xU1':  # charges <-> (ocupation up, occupation down)
                c = Tensor(config=self.config, s=self.s, n=(-1, 0))
                c.set_block(ts=((0, 0), (1, 0)), Ds=(1, 1), val=1)
        elif spin == 'd':
            if self._sym == 'Z2':
                c = Tensor(config=self.config, s=self.s, n=1)
                c.set_block(ts=(0, 1), Ds=(1, 2), val=[0, 1])
            elif self._sym == 'U1':  # charges: 0 = |00>; 1 = (|10>, |01>)
                c = Tensor(config=self.config, s=self.s, n=-1)
                c.set_block(ts=(0, 1), Ds=(1, 2), val=[0, 1])
            elif self._sym == 'U1xU1xZ2':
                c = Tensor(config=self.config, s=self.s, n=(0, -1, 1))
                c.set_block(ts=((0, 0, 0), (0, 1, 1)), Ds=(1, 1), val=1)
            elif self._sym == 'U1xU1':
                c = Tensor(config=self.config, s=self.s, n=(0, -1))
                c.set_block(ts=((0, 0), (0, 1)), Ds=(1, 1), val=1)
        else:
            raise YastnError("spin should be equal 'u' or 'd'.")
        return c

    def Sz(self) -> yastn.Tensor:
        """ Return Sz operator for spin-1/2 fermions. """
        return 0.5 * (self.n('u') - self.n('d'))

    def Sp(self) -> yastn.Tensor:
        """ Return Sp operator for spin-1/2 fermions. """
        return self.cp('u') @ self.c('d')

    def Sm(self) -> yastn.Tensor:
        """ Return Sm operator for spin-1/2 fermions. """
        return self.cp('d') @ self.c('u')

    def to_dict(self):
        return {'I': lambda j: self.I(),
                'nu': lambda j: self.n(spin='u'),
                'cu': lambda j: self.c(spin='u'),
                'cpu': lambda j: self.cp(spin='u'),
                'nd': lambda j: self.n(spin='d'),
                'cd': lambda j: self.c(spin='d'),
                'cpd': lambda j: self.cp(spin='d'),
                'h': lambda j: self.h(),
                'Sz': lambda j: self.Sz(),
                'Sp': lambda j: self.Sp(),
                'Sm': lambda j: self.Sm()}
