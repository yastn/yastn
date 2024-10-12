
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
""" Contractions for benchmarks: yastn with ctm tensors with no legs fused. """
from __future__ import annotations
from model_parent import CtmBenchParent
import yastn
import yastn.tn.fpeps as fpeps


class CtmBenchYastnfPeps(CtmBenchParent):

    def __init__(self, args):
        """ Initialize tensors for contraction. """
        super().__init__(args.fname)

        config = yastn.make_config(sym=self.d["symmetry"], backend=args.backend, device=args.device)

        legs = {k: yastn.Leg(config, s=v['signature'], t=v['charges'], D=v['dimensions'])
                for k, v in self.d.items() if "leg" in k}

        legs_a = ["a_leg_t", "a_leg_l", "a_leg_b", "a_leg_r", "a_leg_s", "a_leg_a"]
        legs_a = [legs[k] for k in legs_a if k in legs]
        self.a = yastn.rand(config, legs=legs_a)

        legs_Tt = [legs["Tt_leg_l"], legs["a_leg_t"].conj(), legs["a_leg_t"], legs["Tt_leg_r"]]
        self.Tt = yastn.rand(config, legs=legs_Tt)

        legs_Tr = [legs["Tr_leg_t"], legs["a_leg_r"].conj(), legs["a_leg_r"], legs["Tr_leg_b"]]
        self.Tr = yastn.rand(config, legs=legs_Tr)

        legs_Ctr = [legs["Tt_leg_r"].conj(), legs["Tr_leg_t"].conj()]
        self.Ctr = yastn.rand(config, legs=legs_Ctr)

        self.a = self.a.fuse_legs(axes=((0, 1), (2, 3), 4))
        self.Tt = self.Tt.fuse_legs(axes=(0, (1, 2), 3))
        self.Tr = self.Tr.fuse_legs(axes=(0, (1, 2), 3))
        self.A = fpeps.DoublePepsTensor(self.a, self.a)

    def print_info(self):
        print("==================== ")
        print(" As used in yastn.tn.fpeps; Do not form double-layer tensor; Extensive use of fusions. ")
        print("Config: ")
        print(self.a.config)
        print("a tensor shape, size:", self.a.get_shape(), self.a.size)
        print("Tt tensor shape, size:", self.Tt.get_shape(), self.Tt.size)
        print("Tr tensor shape, size:", self.Tr.get_shape(), self.Tr.size)
        print("Ctr tensor shape, size:", self.Ctr.get_shape(), self.Ctr.size)

        print("  ")
        print(self.a.config)

    def enlarged_corner(self):
        vec_tr = self.Tt @ (self.Ctr @ self.Tr)
        self.C2x2_tr = self.A._attach_30(vec_tr)

    def fuse_enlarged_corner(self):
        self.C2x2_mat = self.C2x2_tr.fuse_legs(axes=((0, 1), (2, 3)))

    def svd_enlarged_corner(self):
        self.U, self.S, self.V = self.C2x2_mat.svd()
