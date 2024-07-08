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
from ..initialize import eye
from .. import diag
from ..tensor import YastnError, Leg
from ._meta_operators import meta_operators

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

    def space(self, d=None) -> yastn.Leg:
        r""" :class:`yastn.Leg` describing local Hilbert space. Can override default dimension by providing d. """
        return Leg(self.config, s=1, D=(self._d if d is None else d,))

    def I(self, d=None) -> yastn.Tensor:
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
