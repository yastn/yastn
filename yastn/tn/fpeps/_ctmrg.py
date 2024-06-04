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
""" Functions performing many CTMRG steps until convergence and return of CTM environment tensors for mxn lattice. """
from typing import NamedTuple


class CTMRGout(NamedTuple):
    sweeps : int = 0
    env : dict = None


def ctmrg_(env, max_sweeps=1, iterator_step=1, fix_signs=None, opts_svd=None):
    r"""
    Generator for ctmrg().
    """

    for sweep in range(1, max_sweeps + 1):
        env.update_()

        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield CTMRGout(sweep, env)
    yield CTMRGout(sweep, env)
