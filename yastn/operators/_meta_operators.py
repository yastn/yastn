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
""" Parent class for operator classes. """
from ..tensor import make_config


class meta_operators():
    # Predefine common elements of all operator classes.
    def __init__(self, **kwargs):
        r""" Common elements for all operator class. """
        self.config = make_config(**kwargs)
        self.s = (1, -1)
        self._sym = self.config.sym.SYM_ID

    def random_seed(self, seed):
        """ Set the seed of random number generator in the backend. """
        self.config.backend.random_seed(seed)
