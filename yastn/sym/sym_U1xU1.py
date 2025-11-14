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
