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
from . import operators
from ._profile import trace_flops, active_flop_tracer
from ._from_dict import from_dict
from ._split_combine_dict import *
from .initialize import *
from .krylov import *
from .tensor.oe_blocksparse import get_contraction_path, contract_with_unroll, SlicedLeg, make_sliced_legs
from .tensor import *
try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:  # pragma: no cover
    __version__ = "unknown version"  # pragma: no cover
    version_tuple = (0, 0, "unknown version")  # pragma: no cover
