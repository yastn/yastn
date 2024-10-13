
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
""" Parent class specifying contractions for benchmarks. """
import abc
import json


class CtmBenchParent(metaclass=abc.ABCMeta):

    def __init__(self, fname, *args):
        """
        Read tensors legs and other information into dictionary self.d
        """

        self.bench_pipeline = ["enlarged_corner", "fuse_enlarged_corner", "svd_enlarged_corner"]

        with open(fname, "r") as f:
            self.d = json.load(f)

    def print_header(self, file=None):
        print(" No benchmark ", file=file)

    def print_properties(self, file=None):
        print(" Fill-in contractions. ", file=file)

    def enlarged_corner(self):
        """
        Contract the network

        -----Tt---Ctr
            / |    |
           |  |    |
        ---a--|----Tr
           |\ |   /|
        ---|--a*-/ |
           |  |    |
        """
        self.C2x2_tr = None

    def fuse_enlarged_corner(self):
        """
        From block-sparse tensor to block-sparse matrix

        (0----C2x2  -> 0--C2x2
         1----|   |       |
         2)---|___|       1
              | | |
             (4 5 3)

        """
        self.C2x2_mat = None

    def svd_enlarged_corner(self):
        """
        Perform svd of block-sparse matrix
        """
        self.U, self.S, self.V = None, None, None

    def final_cleanup(self):
        """ For operations done after executing benchmarks """
        pass
