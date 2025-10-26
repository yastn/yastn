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
