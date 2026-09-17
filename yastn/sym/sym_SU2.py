"""Rules and elementary data for the non-abelian :math:`SU(2)` symmetry.

Irreducible representations are labelled by ``two_j = 2j``.  Keeping this
label integral makes half-integer spins exact and is also the convention used
by :mod:`yastn.su2`.
"""
from __future__ import annotations

from numbers import Integral


class sym_SU2:
    """The SU(2) irrep fusion rules (labels are twice the physical spin).

    This deliberately does *not* inherit :class:`sym_abelian`: an SU(2)
    fusion has more than one possible result, whereas the latter's ``fuse``
    API promises exactly one.  Use :class:`yastn.SU2Tensor` for SU(2)
    tensors; passing this class to the legacy ``Tensor`` is rejected.
    """
    SYM_ID = 'SU2'
    NSYM = 1
    is_abelian = False

    @classmethod
    def zero(cls):
        return (0,)

    @classmethod
    def validate_irrep(cls, two_j):
        if not isinstance(two_j, Integral) or isinstance(two_j, bool) or two_j < 0:
            raise ValueError("SU2 irreps must be non-negative integer two_j labels")
        return int(two_j)

    @classmethod
    def fusion_outcomes(cls, left, right):
        """Return irreps in ``left ⊗ right``, in ascending ``two_j`` order."""
        left, right = cls.validate_irrep(left), cls.validate_irrep(right)
        return tuple(range(abs(left - right), left + right + 1, 2))

    @classmethod
    def conj_charge(cls, charge):
        """All SU(2) irreps are self-dual."""
        if isinstance(charge, tuple):
            if len(charge) != 1:
                raise ValueError("An SU2 charge has one two_j component")
            charge = charge[0]
        return (cls.validate_irrep(charge),)

    @classmethod
    def fuse(cls, *args, **kwargs):
        raise TypeError(
            "SU2 is non-abelian and has branching fusion rules; use "
            "yastn.SU2Tensor instead of the legacy yastn.Tensor API.")
