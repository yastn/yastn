
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


class CtmBenchYastnBasicDL(CtmBenchParent):

    def __init__(self, fname, config):
        """ Initialize tensors for contraction. """
        super().__init__(fname)

        config = yastn.make_config(sym=self.d["symmetry"], **config)

        legs = {k: yastn.Leg(config, s=v['signature'], t=v['charges'], D=v['dimensions'])
                for k, v in self.d.items() if "leg" in k}

        legs_a = ["a_leg_s", "a_leg_a", "a_leg_t", "a_leg_l", "a_leg_b", "a_leg_r"]
        legs_a = [legs[k] for k in legs_a if k in legs]
        self.a = yastn.rand(config, legs=legs_a)
        if self.a.ndim == 6:  # ancilla leg is present
            self.a = self.a.fuse_legs(axes=((0, 1), 2, 3, 4, 5))  # system and ancilla legs are fused

        legs_Tt = [legs["Tt_leg_l"], legs["a_leg_t"].conj(), legs["a_leg_t"], legs["Tt_leg_r"]]
        self.Tt = yastn.rand(config, legs=legs_Tt)

        legs_Tr = [legs["Tr_leg_t"], legs["a_leg_r"].conj(), legs["a_leg_r"], legs["Tr_leg_b"]]
        self.Tr = yastn.rand(config, legs=legs_Tr)

        legs_Ctr = [legs["Tt_leg_r"].conj(), legs["Tr_leg_t"].conj()]
        self.Ctr = yastn.rand(config, legs=legs_Ctr)

    def print_header(self, file=None):
        print("Attach a and a* sequentially; no fusion in building tensors.", file=file)

    def print_properties(self, file=None):
        print("Config:", file=file)
        for k, v in self.a.config._asdict().items():
            print(k, v, file=file)
        print("a tensor properties:", file=file)
        self.a.show_properties(file=file)
        print("Tt tensor properties:", file=file)
        self.Tt.show_properties(file=file)
        print("Tr tensor properties:", file=file)
        self.Tr.show_properties(file=file)
        print("Ctr tensor properties:", file=file)
        self.Ctr.show_properties(file=file)
        print("C2x2tr tensor properties:", file=file)
        self.C2x2tr.show_properties(file=file)
        print("C2x2mat tensor properties:", file=file)
        self.C2x2mat.show_properties(file=file)

    def enlarged_corner(self):
        """
        Contract the network

        a(0-)--Tt--(3+)A(0-)--Ctr
              / |              |
             (1+)(2-)         (1+)
             C   E             B
             |   |            (0-)
        b----a---|-----D(1+)---Tr
             | G |            / |
        c----|---a*----F(2-)-/  |
             |   |            (3+)
             e   f              d
        """
        self.C2x2tr = yastn.einsum('aCEA,AB,BDFd,GCbeD,GEcfF->abcdef',
                            self.Tt, self.Ctr, self.Tr, self.a, self.a.conj(),
                            order='ABCDEFG')

    def fuse_enlarged_corner(self):
        self.C2x2mat = self.C2x2tr.fuse_legs(axes=((0, 1, 2), (3, 4, 5)))

    def svd_enlarged_corner(self):
        self.U, self.S, self.V = self.C2x2mat.svd()

    def final_cleanup(self):
        yastn.clear_cache()  # yastn is using lru_cache to store contraction logic
