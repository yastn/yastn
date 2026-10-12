# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Common protocol for non-Abelian symmetry categories."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
import numpy as np

from .sym_abelian import sym_meta


class _nonabelian_meta(sym_meta, ABCMeta):
    """Keep symmetry-class names while exposing abstract interface methods."""


class sym_nonabelian(metaclass=_nonabelian_meta):
    """Base class for symmetries with branching irrep fusion.

    Subclasses provide canonical charges, conjugation, irrep dimensions,
    tensor-product outcomes, CG isometries, and F/R coefficients. This class
    supplies common path enumeration, selection rules, and compatibility
    helpers. The identity charge defaults to all zeros; override ``zero``
    when using another labelling convention.
    """

    SYM_ID = 'nonabelian-symmetry-name'
    NSYM = 0
    IS_ABELIAN = False

    @classmethod
    def zero(cls):
        return (0,) * cls.NSYM

    @classmethod
    @abstractmethod
    def canonical_charge(cls, charge):
        """Validate and return a canonical charge tuple."""
        raise NotImplementedError(f"{cls.SYM_ID} must define canonical_charge.")

    @classmethod
    @abstractmethod
    def conj_charge(cls, charge):
        """Return the dual irrep."""
        raise NotImplementedError(f"{cls.SYM_ID} must define conj_charge.")

    @classmethod
    @abstractmethod
    def fusion_outcomes(cls, *charges):
        """Return canonical outcome tuples, retaining outer multiplicities.

        No inputs give ``(zero(),)``; one input gives its canonical irrep.
        With several inputs use left-associated tensor products.
        """
        raise NotImplementedError(f"{cls.SYM_ID} must define fusion_outcomes.")

    @classmethod
    @abstractmethod
    def irrep_dimension(cls, charge):
        """Return the dense dimension of an irrep."""
        raise NotImplementedError(f"{cls.SYM_ID} must define irrep_dimension.")

    @classmethod
    @abstractmethod
    def fusion_isometry(cls, left, right, target=None):
        """Return a CG map with target states as rows and product states as columns.

        If ``target`` is omitted, return a mapping from output irreps to maps.
        """
        raise NotImplementedError(f"{cls.SYM_ID} must define fusion_isometry.")

    @classmethod
    @abstractmethod
    def f_symbol(cls, left1, left2, left3, total, channel12, channel23):
        """Return the recoupling coefficient between the two three-irrep trees."""
        raise NotImplementedError(f"{cls.SYM_ID} must define f_symbol.")

    @classmethod
    @abstractmethod
    def braiding_phase(cls, left, right, total):
        """Return the exchange coefficient in the specified fusion channel."""
        raise NotImplementedError(f"{cls.SYM_ID} must define braiding_phase.")

    @classmethod
    def fusion_paths(cls, charges, target):
        """Return left-associated intermediate charges leading to ``target``.

        A path contains ``max(len(charges) - 2, 0)`` canonical charge tuples.
        Repeated paths retain outer multiplicities, including multiplicity
        at the final binary fusion. An empty product has only the unit irrep.
        Concrete symmetries may override this enumeration for efficiency.
        """
        charges = tuple(cls.canonical_charge(c) for c in charges)
        target = cls.canonical_charge(target)
        if not charges:
            return ((),) if target == cls.zero() else ()
        if len(charges) == 1:
            return ((),) if charges[0] == target else ()
        states = ((charges[0], ()),)
        for right in charges[1:-1]:
            states = tuple((out, path + (out,)) for left, path in states
                           for out in cls.fusion_outcomes(left, right))
        return tuple(path for left, path in states
                     for out in cls.fusion_outcomes(left, charges[-1]) if out == target)

    @classmethod
    def add_charges(cls, *charges, signatures=None, new_signature=1):
        """Return a unique total irrep, rejecting branching charge addition."""
        outcomes = set(cls.signed_fusion_outcomes(charges, signatures, new_signature))
        if len(outcomes) == 1:
            return outcomes.pop()
        raise TypeError(f"Adding {cls.SYM_ID} charges requires an explicit fusion channel.")

    @staticmethod
    def _validate_signatures(count, signatures, new_signature):
        if isinstance(new_signature, (bool, np.bool_)) or new_signature not in (-1, 1):
            raise ValueError("new_signature must be +1 or -1.")
        if signatures is None:
            return (1,) * count
        signatures = tuple(signatures)
        if len(signatures) != count:
            raise ValueError("There must be one signature per charge.")
        if any(isinstance(s, (bool, np.bool_)) or s not in (-1, 1) for s in signatures):
            raise ValueError("Signatures must be +1 or -1.")
        return signatures

    @classmethod
    def _oriented_charges(cls, charges, signatures, new_signature):
        charges = tuple(charges)
        signatures = cls._validate_signatures(len(charges), signatures, new_signature)
        return tuple(cls.canonical_charge(c) if s == new_signature else cls.conj_charge(c)
                     for c, s in zip(charges, signatures))

    @classmethod
    def signed_fusion_outcomes(cls, charges, signatures=None, new_signature=1):
        """Fusion outcomes with leg orientations applied through duality."""
        oriented = cls._oriented_charges(charges, signatures, new_signature)
        return cls.fusion_outcomes(*oriented)

    @classmethod
    def signed_fusion_paths(cls, charges, target, signatures=None, new_signature=1):
        """Fusion paths after applying leg orientations through duality."""
        oriented = cls._oriented_charges(charges, signatures, new_signature)
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
        charges = np.asarray(charges)
        if charges.ndim != 3 or charges.shape[2] != cls.NSYM:
            raise ValueError(f"{cls.SYM_ID} charges must have shape (blocks, legs, {cls.NSYM}).")
        signatures = cls._validate_signatures(charges.shape[1], signatures, new_signature)
        out = []
        for row in charges:
            outcomes = set(cls.signed_fusion_outcomes(tuple(map(tuple, row)), signatures, new_signature))
            if len(outcomes) != 1:
                raise TypeError(f"{cls.SYM_ID} fusion branches; use fusion_outcomes/can_fuse instead of fuse.")
            out.append(outcomes.pop())
        return np.asarray(out, dtype=np.int64).reshape(len(out), cls.NSYM)
