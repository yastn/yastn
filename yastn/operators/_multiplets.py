"""Rank-three irreducible operators and their singlet products.

The third leg carries the operator irrep; its multiplicity is one.
These helpers keep representation pairing separate from fermionic swaps.
"""
from ..tensor import Tensor, YastnError


def _is_multiplet(op):
    return (isinstance(op, Tensor) and not getattr(op.config.sym, 'IS_ABELIAN', True)
            and op.ndim == 3 and op.get_legs(2).D == (1,))


def _multiplet_pair(O, P):
    """Close two irreducible operator legs; return bond endpoints (+,-).

    Physical legs remain (out,in). Spherical vectors use S dot S (no 1/3),
    while a spinor and its dual use sum_sigma cp_sigma c_sigma (no 1/2).
    Bending a spin-1 leg introduces the SU2 metric and its minus sign.
    This is distinct from fermionic parity crossings, added by the caller.
    """
    if not (_is_multiplet(O) and _is_multiplet(P)) or O.config != P.config:
        raise YastnError('Multiplet operators must occur in compatible pairs.')
    for op in (O, P):
        if op.s[:2] != (1, -1) or op.n != op.config.sym.zero() or op.get_legs(2).D != (1,):
            raise YastnError('A multiplet must be invariant with physical signatures (+,-) and one string irrep.')
    if O.s[2] == P.s[2]:
        # Both S() legs are covariant. The dual metric maps
        # (Sm/sqrt2,Sz,-Sp/sqrt2) to (Sp/sqrt2,-Sz,-Sm/sqrt2).
        j2 = O.get_legs(2).t[0][0]
        if j2 != 2:
            raise YastnError('Spinor pairs require one spinor and its dual (cp_spinor(), c_spinor()).')
        O = -O.switch_signature(2)
    if O.s[2] == -1:
        O, P = O.switch_signature(2), P.switch_signature(2)
        O = O * (-1) ** O.get_legs(2).t[0][0]
    if O.get_legs(2).conj() != P.get_legs(2):
        raise YastnError('Multiplet string irreps or particle charges do not form a singlet.')
    return O, P


def _operator_factors(operators, sites):
    """Local factors and singlet pairs, in their original multiplication order."""
    factors, k = [], 0
    while k < len(operators):
        if _is_multiplet(operators[k]):
            if k + 1 == len(operators):
                raise YastnError('Multiplet operators must occur in compatible pairs.')
            if not _is_multiplet(operators[k + 1]):
                raise YastnError('Multiplet operators must occur in compatible pairs.')
            factors.append((operators[k], operators[k + 1], sites[k], sites[k + 1]))
            k += 2
        else:
            factors.append((operators[k], sites[k]))
            k += 1
    return factors
