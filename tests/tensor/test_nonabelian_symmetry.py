"""Common non-Abelian protocol, including fusion rules beyond SU(2)."""
import inspect
from itertools import product

import numpy as np
import pytest

from yastn.sym import sym_nonabelian, sym_SU2, sym_SU2xU1


class GenericSU2(sym_SU2):
    fusion_paths = classmethod(sym_nonabelian.fusion_paths.__func__)


class GenericSU2xU1(sym_SU2xU1):
    fusion_paths = classmethod(sym_nonabelian.fusion_paths.__func__)


class A4FusionRules(sym_nonabelian):
    """Selection-rule fixture; tensor CG and F/R data are intentionally absent.

    The three singlets are labelled 0, 1, 2 and the triplet is labelled 3.
    Unlike SU(2), triplet x triplet contains the triplet twice.
    """
    SYM_ID = 'A4'
    NSYM = 1

    @classmethod
    def canonical_charge(cls, charge):
        label, = charge
        if label not in (0, 1, 2, 3):
            raise ValueError('Unknown A4 irrep.')
        return (int(label),)

    @classmethod
    def conj_charge(cls, charge):
        label, = cls.canonical_charge(charge)
        return ((0, 2, 1, 3)[label],)

    @classmethod
    def irrep_dimension(cls, charge):
        return 3 if cls.canonical_charge(charge) == (3,) else 1

    @classmethod
    def fusion_outcomes(cls, *charges):
        outcomes = (0,)
        for charge in charges:
            right, = cls.canonical_charge(charge)
            updated = []
            for left in outcomes:
                if left == right == 3:
                    updated.extend((0, 1, 2, 3, 3))
                else:
                    updated.append(3 if 3 in (left, right) else (left + right) % 3)
            outcomes = updated
        return tuple((label,) for label in outcomes)


def test_nonabelian_abstract_protocol():
    assert inspect.isabstract(sym_nonabelian)
    assert sym_nonabelian.__abstractmethods__ == {
        'canonical_charge', 'conj_charge', 'fusion_outcomes', 'irrep_dimension',
        'fusion_isometry', 'f_symbol', 'braiding_phase'}
    with pytest.raises(TypeError, match='abstract'):
        sym_nonabelian()
    for sym in (sym_SU2, sym_SU2xU1, GenericSU2, GenericSU2xU1):
        assert not inspect.isabstract(sym)
    assert str(sym_SU2) == 'SU2'
    assert str(sym_SU2xU1) == 'SU2xU1'


@pytest.mark.parametrize('reference, generic', [(sym_SU2, GenericSU2), (sym_SU2xU1, GenericSU2xU1)])
@pytest.mark.parametrize('rank', [0, 1, 2, 4, 6])
def test_nonabelian_generic_paths(reference, generic, rank):
    charges = tuple((i % 3,) if reference.NSYM == 1 else (i % 3, i - 2) for i in range(rank))
    signs = tuple((-1) ** i for i in range(rank))
    for spin in range(7):
        for q in (sum(-s * t[-1] for s, t in zip(signs, charges)), 100):
            target = (spin,) if reference.NSYM == 1 else (spin, q)
            expected = reference.signed_fusion_paths(charges, target, signs, -1)
            if reference.NSYM == 1:
                expected = tuple(tuple((j,) for j in path) for path in expected)
            actual = generic.signed_fusion_paths(charges, target, signs, -1)
            assert actual == expected
            assert len(actual) == generic.fusion_multiplicity(charges, target, signs, -1)


def test_nonabelian_outer_multiplicities_and_duals():
    sym = A4FusionRules
    assert sym.fusion_paths((), (0,)) == ((),)
    assert sym.fusion_paths((), (1,)) == ()
    assert sym.fusion_paths(((1,),), (1,)) == ((),)
    assert sym.fusion_paths(((1,),), (2,)) == ()
    assert sym.fusion_paths(((3,), (3,)), (3,)) == ((), ())
    assert sym.fusion_paths(((3,),) * 3, (0,)) == (((3,),), ((3,),))
    assert len(sym.fusion_paths(((3,),) * 3, (3,))) == 7
    for charges in product(((0,), (1,), (2,), (3,)), repeat=3):
        multiplicities = [len(sym.fusion_paths(charges, (t,))) for t in range(4)]
        assert multiplicities == [sym.fusion_multiplicity(charges, (t,)) for t in range(4)]
        assert sum(n * sym.irrep_dimension((t,)) for t, n in enumerate(multiplicities)) == np.prod(
            [sym.irrep_dimension(t) for t in charges])
    assert sym.add_charges((1,), (2,), signatures=(1, -1)) == (2,)
    assert sym.add_charges((1,), new_signature=-1) == (2,)
    with pytest.raises(TypeError, match='explicit fusion channel'):
        sym.add_charges((3,), (3,))


@pytest.mark.parametrize('sym', [sym_SU2, sym_SU2xU1, A4FusionRules])
def test_nonabelian_unique_charge_and_empty_products(sym):
    zero = sym.zero()
    assert sym.add_charges() == zero
    assert sym.add_charges(zero, zero) == zero
    empty_blocks = sym.fuse(np.empty((0, 2, sym.NSYM), dtype=int), (1, -1), 1)
    assert empty_blocks.shape == (0, sym.NSYM)
    empty_products = sym.fuse(np.empty((3, 0, sym.NSYM), dtype=int), (), -1)
    np.testing.assert_array_equal(empty_products, np.tile(zero, (3, 1)))
    assert np.issubdtype(empty_products.dtype, np.integer)
    assert sym.signed_fusion_outcomes(iter((zero, zero)), iter((1, -1))) == (zero,)


@pytest.mark.parametrize('sym', [sym_SU2, sym_SU2xU1])
@pytest.mark.parametrize('signs, new_sign', [((1,), 1), ((1, 1, 1), 1),
                                           ((1, 0), 1), ((1, True), 1),
                                           ((1, 1), 0), ((1, 1), False)])
def test_nonabelian_invalid_signatures(sym, signs, new_sign):
    charges = (sym.zero(),) * 2
    with pytest.raises(ValueError, match='signature|Signature'):
        sym.signed_fusion_outcomes(charges, signs, new_sign)
    with pytest.raises(ValueError, match='signature|Signature'):
        sym.signed_fusion_paths(charges, sym.zero(), signs, new_sign)
    with pytest.raises(ValueError, match='signature|Signature'):
        sym.add_charges(*charges, signatures=signs, new_signature=new_sign)
    with pytest.raises(ValueError, match='signature|Signature'):
        sym.fuse(np.array([charges]), signs, new_sign)


@pytest.mark.parametrize('sym', [sym_SU2, sym_SU2xU1])
def test_nonabelian_legacy_fuse_validates_charges(sym):
    charges = np.zeros((1, 1, sym.NSYM))
    charges[0, 0, 0] = 0.5
    with pytest.raises(ValueError):
        sym.fuse(charges, (1,), 1)
    with pytest.raises(ValueError, match='shape'):
        sym.fuse(np.zeros((1, sym.NSYM), dtype=int), (1,), 1)
