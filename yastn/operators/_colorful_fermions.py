# Copyright 2026 The YASTN Authors. All Rights Reserved.
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
""" Generator of basic local operators for fermions with three color species. """
from __future__ import annotations

from typing import Callable

import numpy as np

from ._meta_operators import meta_operators
from ..tensor import YastnError, Tensor, Leg

_COLORS = ('r', 'g', 'b')

# Occupations |n_r n_g n_b> in the common dense order, grouped by parity and then by total occupation.
_STATES = ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))

# Dense 8x8 operators in the basis of _STATES; rows index output states, columns index input states.
#              |000> |110> |101> |011> |100> |010> |001> |111>
_N = {'r': np.diag([0, 1, 1, 0, 1, 0, 0, 1]),
      'g': np.diag([0, 1, 0, 1, 0, 1, 0, 1]),
      'b': np.diag([0, 0, 1, 1, 0, 0, 1, 1])}

# Creation operators for indistinguishable species (Z2, U1, U1xU1xU1xZ2),
# with |n_r n_g n_b> = (cp_r)^{n_r} (cp_g)^{n_g} (cp_b)^{n_b} |000>, i.e.,
# cp_g gets the sign (-1)^{n_r} and cp_b gets the sign (-1)^{n_r + n_g}.
_CP = {'r': np.array([[0, 0, 0, 0, 0, 0, 0, 0],    # <000|
                      [0, 0, 0, 0, 0, 1, 0, 0],    # <110|  cp_r |010> =  |110>
                      [0, 0, 0, 0, 0, 0, 1, 0],    # <101|  cp_r |001> =  |101>
                      [0, 0, 0, 0, 0, 0, 0, 0],    # <011|
                      [1, 0, 0, 0, 0, 0, 0, 0],    # <100|  cp_r |000> =  |100>
                      [0, 0, 0, 0, 0, 0, 0, 0],    # <010|
                      [0, 0, 0, 0, 0, 0, 0, 0],    # <001|
                      [0, 0, 0, 1, 0, 0, 0, 0]]),  # <111|  cp_r |011> =  |111>
       'g': np.array([[0, 0, 0, 0, 0, 0, 0, 0],    # <000|
                      [0, 0, 0, 0, -1, 0, 0, 0],   # <110|  cp_g |100> = -|110>
                      [0, 0, 0, 0, 0, 0, 0, 0],    # <101|
                      [0, 0, 0, 0, 0, 0, 1, 0],    # <011|  cp_g |001> =  |011>
                      [0, 0, 0, 0, 0, 0, 0, 0],    # <100|
                      [1, 0, 0, 0, 0, 0, 0, 0],    # <010|  cp_g |000> =  |010>
                      [0, 0, 0, 0, 0, 0, 0, 0],    # <001|
                      [0, 0, -1, 0, 0, 0, 0, 0]]), # <111|  cp_g |101> = -|111>
       'b': np.array([[0, 0, 0, 0, 0, 0, 0, 0],    # <000|
                      [0, 0, 0, 0, 0, 0, 0, 0],    # <110|
                      [0, 0, 0, 0, -1, 0, 0, 0],   # <101|  cp_b |100> = -|101>
                      [0, 0, 0, 0, 0, -1, 0, 0],   # <011|  cp_b |010> = -|011>
                      [0, 0, 0, 0, 0, 0, 0, 0],    # <100|
                      [0, 0, 0, 0, 0, 0, 0, 0],    # <010|
                      [1, 0, 0, 0, 0, 0, 0, 0],    # <001|  cp_b |000> =  |001>
                      [0, 1, 0, 0, 0, 0, 0, 0]])}  # <111|  cp_b |110> =  |111>

# Creation operators for distinguishable species (U1xU1xU1); no Jordan-Wigner signs,
# as fermionic statistics between species is handled by swap gates.
_CP_U1xU1xU1 = {'r': _CP['r'],
                'g': np.array([[0, 0, 0, 0, 0, 0, 0, 0],    # <000|
                               [0, 0, 0, 0, 1, 0, 0, 0],    # <110|  cp_g |100> =  |110>
                               [0, 0, 0, 0, 0, 0, 0, 0],    # <101|
                               [0, 0, 0, 0, 0, 0, 1, 0],    # <011|  cp_g |001> =  |011>
                               [0, 0, 0, 0, 0, 0, 0, 0],    # <100|
                               [1, 0, 0, 0, 0, 0, 0, 0],    # <010|  cp_g |000> =  |010>
                               [0, 0, 0, 0, 0, 0, 0, 0],    # <001|
                               [0, 0, 1, 0, 0, 0, 0, 0]]),  # <111|  cp_g |101> =  |111>
                'b': np.array([[0, 0, 0, 0, 0, 0, 0, 0],    # <000|
                               [0, 0, 0, 0, 0, 0, 0, 0],    # <110|
                               [0, 0, 0, 0, 1, 0, 0, 0],    # <101|  cp_b |100> =  |101>
                               [0, 0, 0, 0, 0, 1, 0, 0],    # <011|  cp_b |010> =  |011>
                               [0, 0, 0, 0, 0, 0, 0, 0],    # <100|
                               [0, 0, 0, 0, 0, 0, 0, 0],    # <010|
                               [1, 0, 0, 0, 0, 0, 0, 0],    # <001|  cp_b |000> =  |001>
                               [0, 1, 0, 0, 0, 0, 0, 0]])}  # <111|  cp_b |110> =  |111>

_SQ3 = 1 / np.sqrt(3)
_GELL_MANN = {1: [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
              2: [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
              3: [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
              4: [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
              5: [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
              6: [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
              7: [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
              8: [[_SQ3, 0, 0], [0, _SQ3, 0], [0, 0, -2 * _SQ3]]}


class ColorfulFermions(meta_operators):
    """ Predefine operators for fermions with three color species. """

    def __init__(self, **kwargs):
        r"""
        Generator of standard operators for local Hilbert space with three fermionic species
        (colors :code:`'r'`, :code:`'g'`, :code:`'b'`) and 8-dimensional Hilbert space.

        Predefine identity, creation, annihilation, density operators, and SU(3) generators.
        Defines vectors with possible occupations, and local Hilbert space as a :class:`yastn.Leg`.

        Parameters
        ----------
        sym : str
            Explicit symmetry to be used. Allowed options are :code:`'Z2'`, :code:`'U1'`,
            :code:`'U1xU1xU1'`, or :code:`'U1xU1xU1xZ2'`.

        kwargs : any
            Passed to :meth:`yastn.make_config` to change backend, default_device or other config parameters.

        Notes
        -----
        Basis states are :math:`|n_r n_g n_b\rangle = (c^\dagger_r)^{n_r} (c^\dagger_g)^{n_g} (c^\dagger_b)^{n_b} |000\rangle`.

        Fermionic field in config is fixed such that:

            * For :code:`'Z2'`, :code:`'U1'` and :code:`'U1xU1xU1xZ2'`, the three species are treated as
              indistinguishable. In that case, creation and annihilation operators of different species
              anti-commute (fermionic statistics is encoded in the Z2 channel of U1xU1xU1xZ2).
            * For :code:`'U1xU1xU1'` the three species are treated as distinguishable.
              In that case, creation and annihilation operators of different species commute.
        """
        if 'fermionic' not in kwargs and isinstance(kwargs.get('sym', None), str):
            kwargs['fermionic'] = (False, False, False, True) if kwargs['sym'] == 'U1xU1xU1xZ2' else True
        super().__init__(**kwargs)

        sym = self._sym
        fer = self.config.fermionic

        if sym not in ('Z2', 'U1', 'U1xU1xU1', 'U1xU1xU1xZ2'):
            raise YastnError("For ColorfulFermions sym should be in ('Z2', 'U1', 'U1xU1xU1', 'U1xU1xU1xZ2').")
        if (sym == 'U1xU1xU1xZ2' and fer != (False, False, False, True)) or \
           (sym in ('Z2', 'U1') and fer != True) or \
           (sym == 'U1xU1xU1' and fer not in (True, (True, True, True))):
            raise YastnError("For ColorfulFermions config.sym does not match config.fermionic.")
        self.operators = ('I', 'n', 'c', 'cp')

        # charge sectors; states within each sector follow the order in _STATES
        charges = [self._charge(st) for st in _STATES]
        self._sectors = {}
        for ind, t in enumerate(charges):
            self._sectors.setdefault(t, []).append(ind)
        self._state_charge = dict(zip(_STATES, charges))

    def _charge(self, st) -> tuple[int, ...]:
        """ Charge of occupation state st = (n_r, n_g, n_b). """
        if self._sym == 'Z2':
            return (sum(st) % 2,)
        if self._sym == 'U1':
            return (sum(st),)
        if self._sym == 'U1xU1xU1':
            return tuple(st)
        return tuple(st) + (sum(st) % 2,)  # U1xU1xU1xZ2

    def _dense_cp(self, color) -> np.ndarray:
        """ Creation operator as 8x8 matrix in the basis of _STATES. """
        self._color_index(color)
        if self._sym == 'U1xU1xU1':
            return _CP_U1xU1xU1[color]
        return _CP[color]

    def _from_dense(self, mat, name='operator') -> Tensor:
        """ Build symmetric operator from 8x8 matrix in the basis of _STATES. """
        iscomplex = np.iscomplexobj(mat) and np.any(np.abs(np.imag(mat)) > 0)
        if not iscomplex:
            mat = np.real(mat)
        blocks, ns = [], set()
        for t_out, i_out in self._sectors.items():
            for t_in, i_in in self._sectors.items():
                sub = mat[np.ix_(i_out, i_in)]
                if np.any(sub != 0):
                    ts = np.array([[t_out, t_in]], dtype=np.int64)
                    ns.add(tuple(self.config.sym.fuse(ts, (1, -1), 1)[0].tolist()))
                    blocks.append((t_out + t_in, sub))
        if len(ns) > 1:
            raise YastnError(f"Cannot define {name} for {self._sym} symmetry.")
        n = ns.pop() if ns else None
        dtype = 'complex128' if iscomplex else self.config.default_dtype
        op = Tensor(config=self.config, s=self.s, n=n, dtype=dtype)
        for ts, sub in blocks:
            op.set_block(ts=ts, Ds=sub.shape, val=sub.tolist())
        return op

    @staticmethod
    def _color_index(color) -> int:
        if color not in _COLORS:
            raise YastnError("color should be equal 'r', 'g', or 'b'.")
        return _COLORS.index(color)

    def space(self) -> Leg:
        r""" :class:`yastn.Leg` object describing local Hilbert space. """
        t = tuple(self._sectors)
        D = tuple(len(v) for v in self._sectors.values())
        return Leg(self.config, s=1, t=t, D=D)

    def key(self) -> Callable | None:
        r"""
        Provide key parameter for :meth:`yastn.to_nonsymmetric`,
        used while sorting block charges,
        so that all symmetries result in the same dense representation of physical space,
        using the order (|000>, |110>, |101>, |011>, |100>, |010>, |001>, |111>) for |n_r n_g n_b>.
        """
        if self._sym == 'Z2':
            return None
        if self._sym == 'U1':
            return lambda t: (t[0] % 2, t[0])
        # U1xU1xU1, U1xU1xU1xZ2
        return lambda t: ((t[0] + t[1] + t[2]) % 2, t[0] + t[1] + t[2], -t[0], -t[1], -t[2])

    def vec_n(self, val=(0, 0, 0)) -> Tensor:
        r""" State with occupation given by tuple (n_r, n_g, n_b). """
        val = tuple(val) if isinstance(val, (tuple, list)) else val
        try:
            t = self._state_charge[val]
        except (KeyError, TypeError):  # TypeError for unhashable val
            raise YastnError('Occupations given by val should be a tuple (n_r, n_g, n_b) with elements 0 or 1.') from None
        inds = self._sectors[t]
        vals = [1 if _STATES[ind] == val else 0 for ind in inds]
        vec = Tensor(config=self.config, s=(1,), n=t)
        vec.set_block(ts=t, Ds=(len(inds),), val=vals)
        return vec

    def I(self) -> Tensor:
        r""" Identity operator in 8-dimensional Hilbert space. """
        return self._from_dense(np.eye(8))

    def n(self, color='r') -> Tensor:
        r""" Particle number operator for :code:`color` in :code:`'r'`, :code:`'g'`, or :code:`'b'`. """
        self._color_index(color)
        return self._from_dense(_N[color])

    def cp(self, color='r') -> Tensor:
        r""" Creation operator for :code:`color` in :code:`'r'`, :code:`'g'`, or :code:`'b'`. """
        return self._from_dense(self._dense_cp(color))

    def c(self, color='r') -> Tensor:
        r""" Annihilation operator for :code:`color` in :code:`'r'`, :code:`'g'`, or :code:`'b'`. """
        return self._from_dense(self._dense_cp(color).T)

    def gell_mann(self, k) -> Tensor:
        r"""
        Gell-Mann operator :math:`G_k = \sum_{a,b} c^\dagger_a (\lambda_k)_{ab} c_b`, for :math:`k = 1, \ldots, 8`.

        :math:`\lambda_k` are standard Gell-Mann matrices in the color basis (r, g, b), without the factor 1/2;
        see :meth:`su3` for SU(3) generators. For :math:`k = 2, 5, 7` the operator is complex.
        Off-diagonal operators (:math:`k \neq 3, 8`) change individual color occupations
        and cannot be defined for :code:`'U1xU1xU1'` and :code:`'U1xU1xU1xZ2'` symmetries.
        """
        if k not in _GELL_MANN:
            raise YastnError("Gell-Mann index k should be in 1, 2, ..., 8.")
        lam = np.array(_GELL_MANN[k])
        cps = [self._dense_cp(color) for color in _COLORS]
        mat = sum(lam[a, b] * (cps[a] @ cps[b].T) for a in range(3) for b in range(3) if lam[a, b] != 0)
        return self._from_dense(mat, name=f"Gell-Mann operator k={k}")

    def su3(self, k) -> Tensor:
        r""" SU(3) generator :math:`T_k = G_k / 2`, with :math:`G_k` given by :meth:`gell_mann`. """
        return self.gell_mann(k) / 2
