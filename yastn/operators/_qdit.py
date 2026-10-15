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
from __future__ import annotations

from ._meta_operators import meta_operators
from ..initialize import eye
from ..tensor import YastnError, Leg, Tensor, diag


class Qdit(meta_operators):
    # Predefine dense operators with set dimension of the local space.
    def __init__(self, d=2, **kwargs):
        r"""
        Algebra of d-dimensional Hilbert space with only identity operator, and local Hilbert space as a :class:`yastn.Leg`.

        Parameters
        ----------
        d : int
            Default Hilbert space dimension.

        kwargs
            Other YASTN configuration parameters can be provided, see :meth:`yastn.make_config`.

        Notes
        -----
        Default configuration sets :code:`fermionic` to :code:`False`.
        """
        super().__init__(**kwargs)
        if self._sym!= 'dense':
            raise YastnError("For Qdit sym should be 'dense'.")
        if self.config.fermionic != False:
            raise YastnError("For Qdit config.fermionic should be False.")
        self._d = d
        self.operators = ('I',)

    def space(self, d=None) -> Leg:
        r""" :class:`yastn.Leg` object describing local Hilbert space. Can override default dimension by providing d. """
        return Leg(self.config, s=1, D=(self._d if d is None else d,))

    def key(self) -> None:
        return None

    def I(self, d=None) -> Tensor:
        """ Identity operator. Can override default dimension by providing d."""
        return diag(eye(config=self.config, s=self.s, D=self._d if d is None else d))

    def to_dict(self):
        """
        Returns
        -------
        dict(str,yastn.Tensor)
            a map from strings to operators
        """
        return {'I': lambda j: self.I()}
