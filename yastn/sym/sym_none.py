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
""" Define trivial rules for dense tensor. """
from .sym_abelian import sym_abelian


class sym_none(sym_abelian):
    """No symmetry"""

    SYM_ID = 'dense'
    NSYM = 0  # nothing to distinguish symmetry sector

    @classmethod
    def fuse(cls, charges, signatures, new_signature):
        """ Trivial fusion rule for no symmetry needed for interface consistency. """
        # charges is an empty matrix
        # swap to properly match non-zero dimensions of returned tset
        return charges.swapaxes(1, 2) @ signatures
