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

import numpy as np

class sym_meta(type):
    def __str__(cls):
        return cls.SYM_ID

class sym_abelian(metaclass=sym_meta):
    """
    Interface to be subclassed for concrete symmetry implementations.
    """
    SYM_ID = 'symmetry-name'
    NSYM = len('length-of-charge-vector')

    @classmethod
    def zero(cls):
        """ Zero charge. """
        return (0,) * cls.NSYM

    @classmethod
    def fuse(cls, charges, signatures, new_signature):
        """
        Fusion rule for abelian symmetry.

        An `i`-th row ``charges[i,:,:]`` contains `m` length-`NSYM`
        charge vectors, where `m` is the number of legs being fused.
        For each row, the charge vectors are added up (fused) with selected ``signatures``
        according to the group addition rules.

        Parameters
        ----------
        charges: numpy.ndarray(int)
            :math:`k \times m \times nsym` matrix, where :math:`k` is the number of
            independent blocks, and :math:`m` is the number of fused legs.

        signatures: numpy.ndarray(int)
            integer vector with :math:`m` elements in :math:`{-1, +1}`.

        new_signature: int
            new signature for a new leg from fused legs.

        Returns
        -------
        numpy.ndarray(int)
            integer matrix with shape (k, NSYM) of fused charges;
            includes multiplication by ``new_signature``.
        """
        raise NotImplementedError("Subclasses need to override the fuse function")

    @classmethod
    def add_charges(cls, *charges, s=None, new_s=1):
        """
        Auxiliary function for adding tensor charges.
        It employs fuse function and returns resulting charge as a tuple.

        Parameters
        ----------
        charges: tuple[int, ...]
            Sequence of charges to be added.

        s: None | Sequence[int]
            The default None uses signature 1 for all charges.
            For subtraction, it is possible to provide signatures by hand.

        new_s: int
            The default is 1.

        Returns
        -------
        tuple
            resulting tuple of charges
        """
        if len(charges) == 0:
            return cls.zero()
        if s is None:
            s = (1,) * len(charges)
        charges = np.array(charges, dtype=np.int64).reshape(1, len(charges), cls.NSYM)
        new_charge = cls.fuse(charges, s, new_s)
        return tuple(new_charge.reshape(cls.NSYM).tolist())
