# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
""" yastn.from_dict() handling imports off all major classes """
from .tensor import Tensor
from .tn.mps import MpsMpoOBC, MpoPBC
from .tn.fpeps import Lattice, Peps, Peps2Layers
from .tn.fpeps import EnvBoundaryMPS, EnvBP, EnvCTM, EnvCTM_c4v


types = {"Tensor": Tensor,
         "MpsMpoOBC": MpsMpoOBC,
         "MpoPBC": MpoPBC,
         "Lattice": Lattice,
         "Peps": Peps,
         "Peps2Layers": Peps2Layers,
         "EnvBoundaryMPS": EnvBoundaryMPS,
         "EnvBP": EnvBP,
         "EnvCTM": EnvCTM,
         "EnvCTM_c4v": EnvCTM_c4v
         }

def from_dict(d, config=None):
    return types[d['type']].from_dict(d, config)
