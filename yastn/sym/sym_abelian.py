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
""" Parent class for defining symmetry rules. """
class sym_meta(type):
    def __str__(cls):
        return cls.SYM_ID

    # def __repr__(cls):
    #     """ standard path to predefined symmetries """
    #     return "yastn.sym.sym_" + cls.SYM_ID


class sym_abelian(metaclass=sym_meta):
    """
    Interface to be subclassed for concrete symmetry implementations.
    """
    SYM_ID = 'symmetry-name'
    NSYM = len('length-of-charge-vector')

    @classmethod
    def fuse(cls, charges, signatures, new_signature):
        """
        Fusion rule for abelian symmetry. An `i`-th row ``charges[i,:,:]`` contains `m` length-`NSYM`
        charge vectors, where `m` is the number of legs being fused.
        For each row, the charge vectors are added up (fused) with selected ``signatures``
        according to the group addition rules.

        Parameters
        ----------
            charges: numpy.ndarray(int)
                `k x m x nsym` matrix, where `k` is the number of
                independent blocks, and `m` is the number of fused legs.

            signatures: numpy.ndarray(int)
                integer vector with `m` elements in `{-1, +1}`

            new_signature: int

        Returns
        -------
            numpy.ndarray(int)
                integer matrix with shape (k, NSYM) of fused charges;
                includes multiplication by ``new_signature``
        """
        raise NotImplementedError("Subclasses need to override the fuse function")  # pragma: no cover
