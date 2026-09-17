"""Non-abelian SU(2)-symmetric tensors.

The legacy :class:`yastn.Tensor` stores one dense block per charge tuple and
therefore implements Abelian symmetries only.  This module has a separate,
compatible-on-import representation: a block contains *degeneracy* indices
only, while all magnetic indices are supplied by Clebsch--Gordan tensors.
Existing tensors and their serialised dictionaries are untouched.
"""
from ._tensor import Leg, SU2U1Leg, SU2Tensor, SU2U1Tensor, tensordot
from .linalg import MultipletMask, truncation_mask

__all__ = ['Leg', 'SU2U1Leg', 'SU2Tensor', 'SU2U1Tensor', 'tensordot', 'MultipletMask', 'truncation_mask']
