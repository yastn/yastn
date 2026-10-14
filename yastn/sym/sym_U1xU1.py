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
""" Define rules for U1xU1 symmetry. """
from .sym_abelian import sym_abelian


class sym_U1xU1(sym_abelian):
    """U1xU1 symmetry"""

    SYM_ID = 'U1xU1'
    NSYM = 2  # two ints are used to distinguish symmetry sectors

    @classmethod
    def fuse(cls, charges, signatures, new_signature):
        """ Fusion rule for U1xU1 symmetry. """
        return new_signature * (charges.swapaxes(1, 2) @ signatures)
