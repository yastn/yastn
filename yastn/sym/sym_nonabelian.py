# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Common protocol for non-Abelian symmetry categories."""
from __future__ import annotations

import numpy as np

from .sym_abelian import sym_meta


class sym_nonabelian(metaclass=sym_meta):
    """Base class for symmetries with branching irrep fusion.

    Subclasses provide canonical charges, conjugation, irrep dimensions, and
    the unordered tensor-product outcomes.  This class supplies the common
    selection-rule and compatibility API consumed by :mod:`yastn.tensor`.
    """

    SYM_ID = 'nonabelian-symmetry-name'
    NSYM = 0
    IS_ABELIAN = False

    @classmethod
    def zero(cls):
        return (0,) * cls.NSYM

    @classmethod
    def canonical_charge(cls, charge):
        """Validate and return a canonical charge tuple."""
        raise NotImplementedError

    @classmethod
    def conj_charge(cls, charge):
        """Return the dual irrep."""
        raise NotImplementedError

    @classmethod
    def fusion_outcomes(cls, *charges):
        """Return fusion outcomes, retaining outer multiplicities."""
        raise NotImplementedError

    @classmethod
    def irrep_dimension(cls, charge):
        """Return the dense dimension of an irrep."""
        raise NotImplementedError

    @classmethod
    def signed_fusion_outcomes(cls, charges, signatures=None, new_signature=1):
        """Fusion outcomes with leg orientations applied through duality."""
        if signatures is None:
            signatures = (1,) * len(charges)
        oriented = tuple(cls.canonical_charge(c) if s == new_signature else cls.conj_charge(c)
                         for c, s in zip(charges, signatures))
        return cls.fusion_outcomes(*oriented)

    @classmethod
    def signed_fusion_paths(cls, charges, target, signatures=None, new_signature=1):
        """Fusion paths after applying leg orientations through duality."""
        if signatures is None:
            signatures = (1,) * len(charges)
        oriented = tuple(cls.canonical_charge(c) if s == new_signature else cls.conj_charge(c)
                         for c, s in zip(charges, signatures))
        target = cls.canonical_charge(target)
        return cls.fusion_paths(oriented, target)

    @classmethod
    def fusion_multiplicity(cls, charges, target, signatures=None, new_signature=1):
        target = cls.canonical_charge(target)
        return cls.signed_fusion_outcomes(charges, signatures, new_signature).count(target)

    @classmethod
    def can_fuse(cls, charges, target, signatures=None, new_signature=1):
        return cls.fusion_multiplicity(charges, target, signatures, new_signature) > 0

    @classmethod
    def fuse(cls, charges, signatures, new_signature):
        """Serve legacy callers only when each fusion has one unique output."""
        charges = np.asarray(charges, dtype=np.int64)
        if charges.ndim != 3 or charges.shape[2] != cls.NSYM:
            raise ValueError(f"{cls.SYM_ID} charges must have shape (blocks, legs, {cls.NSYM}).")
        out = []
        for row in charges:
            outcomes = set(cls.signed_fusion_outcomes(tuple(map(tuple, row)), signatures, new_signature))
            if len(outcomes) != 1:
                raise TypeError(f"{cls.SYM_ID} fusion branches; use fusion_outcomes/can_fuse instead of fuse.")
            out.append(outcomes.pop())
        return np.asarray(out, dtype=np.int64).reshape(len(out), cls.NSYM)
