"""Fusion rules for the non-abelian product group ``SU(2) x U(1)``."""
from __future__ import annotations

from .sym_SU2 import sym_SU2


class sym_SU2xU1:
    """Irreps labelled as ``(two_j, q)``.

    ``two_j`` is twice the spin and ``q`` is the additive U(1) charge.  The
    product has branching only in its SU(2) component:
    ``(j1, q1) x (j2, q2) = sum_j (j, q1 + q2)``.
    """
    SYM_ID = 'SU2xU1'
    NSYM = 2
    is_abelian = False

    @classmethod
    def zero(cls):
        return (0, 0)

    @classmethod
    def validate_irrep(cls, charge):
        try:
            two_j, q = charge
        except (TypeError, ValueError) as exc:
            raise ValueError('An SU2xU1 irrep must be a (two_j, q) pair') from exc
        return sym_SU2.validate_irrep(two_j), int(q)

    @classmethod
    def fusion_outcomes(cls, left, right):
        j0, q0 = cls.validate_irrep(left)
        j1, q1 = cls.validate_irrep(right)
        return tuple((j, q0 + q1) for j in sym_SU2.fusion_outcomes(j0, j1))

    @classmethod
    def conj_charge(cls, charge):
        two_j, q = cls.validate_irrep(charge)
        return two_j, -q

    @classmethod
    def fuse(cls, *args, **kwargs):
        raise TypeError(
            'SU2xU1 has branching SU2 fusion rules; use the reduced non-abelian '
            'tensor API rather than legacy yastn.Tensor.')
