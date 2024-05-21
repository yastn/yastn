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
""" Special contractions. Introduced to dispatch equivalent contractions to more complex meta-tensors."""
from ._contractions import ncon

def _attach_01(M, T):
    return ncon([T, M], ((-0, 1, 2, -2), (2, 1, -1, -3)))

def _attach_23(M, T):
    return ncon([T, M], ((-0, 1, 2, -2), (-1, -3, 2, 1)))

def _attach_12(M, T):
    return ncon([T, M], ((-0, 1, 2, -2), (-3, 2, 1, -1)))

def _attach_30(M, T):
    return ncon([T, M], ((-0, 1, 2, -2), (1, -1, -3, 2)))


