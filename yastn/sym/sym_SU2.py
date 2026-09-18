# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Representation data for the compact group :math:`SU(2)`.

Charges are stored as ``2*j``. This keeps half-integer representations exact.
"""
from __future__ import annotations

from numbers import Integral
from functools import lru_cache
from math import factorial, sqrt

import numpy as np

from .sym_nonabelian import sym_nonabelian


class sym_SU2(sym_nonabelian):
    """Fusion-category interface for SU(2) irreducible representations."""

    SYM_ID = "SU2"
    NSYM = 1

    @classmethod
    def zero(cls) -> tuple[int]:
        return (0,)

    @staticmethod
    def _label(charge) -> int:
        if isinstance(charge, (tuple, list)):
            if len(charge) != 1:
                raise ValueError("An SU2 charge must contain one 2*j label.")
            charge = charge[0]
        if isinstance(charge, bool) or not isinstance(charge, Integral) or charge < 0:
            raise ValueError("SU2 charges are non-negative integer 2*j labels.")
        return int(charge)

    @classmethod
    def canonical_charge(cls, charge) -> tuple[int]:
        return (cls._label(charge),)

    @classmethod
    def conj_charge(cls, charge) -> tuple[int]:
        return cls.canonical_charge(charge)

    @classmethod
    def fusion_outcomes(cls, *charges) -> tuple[tuple[int], ...]:
        """Return all left-associated fusion outcomes, retaining multiplicity."""
        if not charges:
            return (cls.zero(),)
        outcomes = [cls._label(charges[0])]
        for charge in charges[1:]:
            right = cls._label(charge)
            outcomes = [total for left in outcomes
                        for total in range(abs(left - right), left + right + 1, 2)]
        return tuple((x,) for x in outcomes)

    @classmethod
    def fusion_paths(cls, charges, target):
        """Return left-associated intermediate-irrep paths to ``target``.

        For ``n`` external irreps a path contains ``n - 2`` labels.  A
        two-leg fusion therefore has the empty path ``()``.
        """
        charges = tuple(cls._label(c) for c in charges)
        target = cls._label(target)
        if not charges:
            return ((),) if target == 0 else ()
        if len(charges) == 1:
            return ((),) if charges[0] == target else ()
        states = ((charges[0], ()),)
        for right in charges[1:-1]:
            states = tuple((out, path + (out,)) for left, path in states
                           for (out,) in cls.fusion_outcomes(left, right))
        return tuple(path for left, path in states
                     if (target,) in cls.fusion_outcomes(left, charges[-1]))

    @classmethod
    def irrep_dimension(cls, charge) -> int:
        return cls._label(charge) + 1

    @classmethod
    def add_charges(cls, *charges, signatures=None, new_signature=1):
        """Combine tensor *total* charges.

        The first implementation supports invariant tensors, whose total
        charge is the singlet.  Leg fusion uses :meth:`fusion_outcomes`.
        """
        if not charges or all(cls.canonical_charge(x) == cls.zero() for x in charges):
            return cls.zero()
        raise TypeError("Adding non-singlet SU2 tensor charges requires an explicit fusion channel.")

    @staticmethod
    def _triangle(a, b, c):
        return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0

    @staticmethod
    def _half_factorial(x):
        return factorial(x // 2) if x >= 0 and x % 2 == 0 else 0

    @classmethod
    @lru_cache(maxsize=None)
    def clebsch_gordan(cls, j1, m1, j2, m2, J, M):
        """Condon--Shortley ``<j1 m1,j2 m2|J M>`` using doubled labels."""
        j1, j2, J = cls._label(j1), cls._label(j2), cls._label(J)
        m1, m2, M = int(m1), int(m2), int(M)
        if M != m1 + m2 or not cls._triangle(j1, j2, J) or abs(M) > J:
            return 0.0
        hf = cls._half_factorial
        numerator = ((J + 1) * hf(J + j1 - j2) * hf(J - j1 + j2) * hf(j1 + j2 - J)
                     * hf(J + M) * hf(J - M) * hf(j1 - m1) * hf(j1 + m1)
                     * hf(j2 - m2) * hf(j2 + m2))
        denominator = hf(j1 + j2 + J + 2)
        if not denominator:
            return 0.0
        total = 0.0
        for k in range((j1 + j2 - J) // 2 + 1):
            args = (2 * k, j1 + j2 - J - 2 * k, j1 - m1 - 2 * k,
                    j2 + m2 - 2 * k, J - j2 + m1 + 2 * k,
                    J - j1 - m2 + 2 * k)
            fs = tuple(hf(x) for x in args)
            if all(fs):
                total += (-1.) ** k / _product(fs)
        return sqrt(numerator / denominator) * total

    @classmethod
    @lru_cache(maxsize=None)
    def wigner_6j(cls, j1, j2, j12, j3, J, j23):
        """Wigner 6-j symbol ``{j1 j2 j12; j3 J j23}`` (doubled labels)."""
        vals = tuple(cls._label(x) for x in (j1, j2, j12, j3, J, j23))
        a, b, c, d, e, f = vals
        triangles = ((a, b, c), (a, e, f), (d, b, f), (d, e, c))
        if not all(cls._triangle(*tri) for tri in triangles):
            return 0.0

        def delta(x, y, z):
            return sqrt(factorial((x + y - z) // 2) * factorial((x - y + z) // 2)
                        * factorial((-x + y + z) // 2) / factorial((x + y + z) // 2 + 1))

        A = ((a + b + c) // 2, (a + e + f) // 2,
             (d + b + f) // 2, (d + e + c) // 2)
        B = ((a + b + d + e) // 2, (a + c + d + f) // 2,
             (b + c + e + f) // 2)
        total = 0.0
        for z in range(max(A), min(B) + 1):
            den = _product(factorial(z - x) for x in A) * _product(factorial(x - z) for x in B)
            total += (-1.) ** z * factorial(z + 1) / den
        return _product(delta(*tri) for tri in triangles) * total

    @classmethod
    def f_symbol(cls, j1, j2, j3, J, j12, j23):
        """Unitary recoupling coefficient between the two three-irrep trees."""
        labels = tuple(cls._label(x) for x in (j1, j2, j3, J, j12, j23))
        j1, j2, j3, J, j12, j23 = labels
        phase = (-1.) ** ((j1 + j2 + j3 + J) // 2)
        return phase * sqrt((j12 + 1) * (j23 + 1)) * cls.wigner_6j(j1, j2, j12, j3, J, j23)

    @classmethod
    def braiding_phase(cls, left, right, total):
        """CG phase for exchanging the two inputs of a fusion vertex."""
        left, right, total = (cls._label(x) for x in (left, right, total))
        if not cls._triangle(left, right, total):
            return 0.0
        return (-1.) ** ((left + right - total) // 2)

    @classmethod
    def fusion_isometry(cls, j1, j2, J=None):
        """CG isometry from product magnetic basis to coupled basis.

        With ``J`` specified, returns an array of shape
        ``(J + 1, (j1 + 1) * (j2 + 1))``. Without ``J``, returns all allowed
        output sectors as ``{(J,): matrix}``.
        """
        j1, j2 = cls._label(j1), cls._label(j2)
        if J is None:
            return {out: cls.fusion_isometry(j1, j2, out) for out in cls.fusion_outcomes(j1, j2)}
        J = cls._label(J)
        if (J,) not in cls.fusion_outcomes(j1, j2):
            raise ValueError(f"Irrep {J} is not present in {j1} x {j2}.")
        m1s, m2s, Ms = range(-j1, j1 + 1, 2), range(-j2, j2 + 1, 2), range(-J, J + 1, 2)
        return np.asarray([[cls.clebsch_gordan(j1, m1, j2, m2, J, M)
                            for m1 in m1s for m2 in m2s] for M in Ms], dtype=float)


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result
