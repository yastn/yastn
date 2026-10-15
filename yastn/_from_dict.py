# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
""" yastn.from_dict() handling imports of all major classes """
from .tensor import Tensor
from .tn.fpeps import Lattice, Peps, Peps2Layers, DoublePepsTensor
from .tn.fpeps import EnvBoundaryMPS, EnvBP, EnvCTM, EnvCTM_c4v
from .tn.mps import MpsMpoOBC, MpoPBC


types = {"Tensor": Tensor,
         "MpsMpoOBC": MpsMpoOBC,
         "MpoPBC": MpoPBC,
         "Lattice": Lattice,
         "Peps": Peps,
         "Peps2Layers": Peps2Layers,
         "DoublePepsTensor": DoublePepsTensor,
         "EnvBoundaryMPS": EnvBoundaryMPS,
         "EnvBP": EnvBP,
         "EnvCTM": EnvCTM,
         "EnvCTM_c4v": EnvCTM_c4v
         }

def from_dict(d, config=None):
    r"""
    De-serializes an object from the dictionary ``d``.
    Object type is read from ``d``.

    Parameters
    ----------
    d : dict
        Object stored in form of a dictionary created with ``to_dict`` method.

    config : Optional[module | _config(NamedTuple)]
        :ref:`YASTN configuration <tensor/configuration:yastn  configuration>`
        If provided, overrides configuration stored in `d`.
    """
    return types[d['type']].from_dict(d, config)
