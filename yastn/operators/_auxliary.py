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

import numpy as np

def swap_charges(charges_0, charges_1, fss):
    """ Calculates a sign accumulated while swaping lists of charges."""
    if not fss:
        return 1
    t0 = np.array(charges_0, dtype=np.int64)
    t1 = np.array(charges_1, dtype=np.int64)
    if fss is True:
        return 1 - 2 * (np.sum(t0 * t1, dtype=np.int64).item() % 2)
    return 1 - 2 * (np.sum((t0 * t1)[:, fss], dtype=np.int64).item() % 2)
