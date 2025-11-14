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
