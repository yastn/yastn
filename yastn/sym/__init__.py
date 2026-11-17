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
from .sym_none import sym_none
from .sym_U1 import sym_U1
from .sym_U1xU1 import sym_U1xU1
from .sym_U1xU1xZ2 import sym_U1xU1xZ2
from .sym_U1xU1xU1 import sym_U1xU1xU1
from .sym_U1xU1xU1xZ2 import sym_U1xU1xU1xZ2
from .sym_Z2 import sym_Z2
from .sym_Z3 import sym_Z3
from .sym_abelian import sym_abelian
from .sym_Z2xU1 import sym_Z2xU1

# Symmetries that can be selected by name, e.g., yastn.make_config(sym='U1').
# This is the single place to register a new predefined symmetry.
SYMMETRIES = {"dense": sym_none,
              "none": sym_none,
              "Z2": sym_Z2,
              "Z3": sym_Z3,
              "U1": sym_U1,
              "U1xU1": sym_U1xU1,
              "U1xU1xZ2": sym_U1xU1xZ2,
              "U1xU1xU1": sym_U1xU1xU1,
              "U1xU1xU1xZ2": sym_U1xU1xU1xZ2}
