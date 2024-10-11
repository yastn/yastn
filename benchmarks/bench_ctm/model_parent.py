
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
from __future__ import annotations
import abc
import json
import os


class CtmBenchParent(metaclass=abc.ABCMeta):

    def __init__(self, fname):
        """
        Read tensor legs for testing.
        """
        dirname = os.path.dirname(__file__)
        fpath = os.path.join(dirname, "input_shapes/" + fname)

        with open(fpath, "r") as f:
            self.d = json.load(f)


    def print_info(self):
        print(" Fill-in contractions. ")

    def enlarged_corner(self):
        """
        Contract the network

        a(0-)--T_t---(3+)A(0-)---C
              / |                |
             (1+)(2-)           (1+)
             C   E               B
             |   |              (0-)
        b----a---|--------D(1+)--T_r
             | \ |              /|
        c----|---a*--------F(2-) |
             |   |              (3+)
             e   f               d
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

