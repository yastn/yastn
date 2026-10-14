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
""" Define rules for Z2xU1 symmetry. """
import numpy as np

from .sym_abelian import sym_abelian


class sym_Z2xU1(sym_abelian):
    """ Z2xU1 symmetry"""

    SYM_ID = "Z2xU1"
    NSYM = 2

    def fuse(charges, signatures, new_signature):
        """ Fusion rule for Z2xU1 symmetry. """
        teff = new_signature * (charges.swapaxes(1,2) @ signatures)
        teff[:, 0] = np.mod(teff[:, 0], 2)
        return teff
