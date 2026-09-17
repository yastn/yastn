# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Fusion rules for the direct product :math:`SU(2) \times U(1)`."""
from __future__ import annotations

from numbers import Integral

from .sym_nonabelian import sym_nonabelian


class sym_SU2xU1(sym_nonabelian):
    """Irreps labelled by ``(2*j, q)`` with additive integer ``q``."""

    SYM_ID = 'SU2xU1'
    NSYM = 2

    @classmethod
    def zero(cls):
        return (0, 0)

    @classmethod
    def canonical_charge(cls, charge):
        try:
            two_j, q = charge
        except (TypeError, ValueError) as exc:
            raise ValueError("An SU2xU1 charge must be a (2*j, q) pair.") from exc
        if isinstance(two_j, bool) or not isinstance(two_j, Integral) or two_j < 0:
            raise ValueError("The SU2 label must be a non-negative integer 2*j.")
        if isinstance(q, bool) or not isinstance(q, Integral):
            raise ValueError("The U1 charge must be an integer.")
        return int(two_j), int(q)

    @classmethod
    def conj_charge(cls, charge):
        two_j, q = cls.canonical_charge(charge)
        return two_j, -q

    @classmethod
    def fusion_outcomes(cls, *charges):
        if not charges:
            return (cls.zero(),)
        two_j, q = cls.canonical_charge(charges[0])
        outcomes = [(two_j, q)]
        for charge in charges[1:]:
            right_j, right_q = cls.canonical_charge(charge)
            outcomes = [(total_j, left_q + right_q)
                        for left_j, left_q in outcomes
                        for total_j in range(abs(left_j - right_j), left_j + right_j + 1, 2)]
        return tuple(outcomes)

    @classmethod
    def fusion_paths(cls, charges, target):
        """SU(2) intermediate labels with the additive U(1) charge attached."""
        charges = tuple(cls.canonical_charge(c) for c in charges)
        target = cls.canonical_charge(target)
        if sum(q for _, q in charges) != target[1]:
            return ()
        if len(charges) < 2:
            return ((),) if (not charges and target == cls.zero()) or (charges and charges[0] == target) else ()
        states = ((charges[0], ()),)
        for right in charges[1:-1]:
            states = tuple((out, path + (out,)) for left, path in states
                           for out in cls.fusion_outcomes(left, right))
        return tuple(path for left, path in states if target in cls.fusion_outcomes(left, charges[-1]))

    @classmethod
    def irrep_dimension(cls, charge):
        two_j, _ = cls.canonical_charge(charge)
        return two_j + 1

    @classmethod
    def add_charges(cls, *charges, signatures=None, new_signature=1):
        if not charges:
            return cls.zero()
        if signatures is None:
            signatures = (1,) * len(charges)
        canonical = tuple(cls.canonical_charge(c) for c in charges)
        if any(two_j != 0 for two_j, _ in canonical):
            raise TypeError("Adding non-singlet SU2xU1 tensor charges requires an explicit fusion channel.")
        q = new_signature * sum(s * charge[1] for s, charge in zip(signatures, canonical))
        return 0, q
