"""U1 sectors share numerical spin coefficients without bypassing selection rules."""
from itertools import product

import pytest
import yastn
from yastn.sym import sym_SU2, sym_SU2xU1
from yastn.tensor import _merging, _nonabelian


@pytest.fixture
def small_coefficient_caches():
    names = ('nonabelian_fusion_coefficient', 'adjacent_swap_coefficient',
             'nonabelian_fusion_u1', 'adjacent_swap_u1')
    sizes = {name: yastn.get_cache_info()[name].maxsize for name in names}
    yastn.set_cache_maxsize(dict.fromkeys(names, 8))
    try:
        yield
    finally:
        yastn.set_cache_maxsize(sizes)


def fusion_args(sym, q, intermediate=0):
    charge = lambda j, n: (j, n) if sym is sym_SU2xU1 else (j,)
    channel = (intermediate, 2 * q, 1, q) if sym is sym_SU2xU1 else (intermediate, 1)
    return (sym, (charge(1, q),) * 4, (1, 1, -1, -1), charge(0, 0),
            channel, ((0, 1), (2, 3)), (charge(intermediate, 2 * q),) * 2,
            (1, -1), (), ((), ()))


def test_fusion_shares_spin_cache(small_coefficient_caches):
    references = {j: _nonabelian.fusion_coefficient(*fusion_args(sym_SU2, 0, j)) for j in (0, 2)}
    assert all(abs(value) > 0 for value in references.values())
    for _ in range(2):
        for q in range(-16, 16):
            for j, reference in references.items():
                assert _nonabelian.fusion_coefficient(*fusion_args(sym_SU2xU1, q, j)) == pytest.approx(reference)
    info = yastn.get_cache_info()['nonabelian_fusion_coefficient']
    assert info.currsize == 2
    assert info.misses == 2
    assert info.hits == 128
    outer = yastn.get_cache_info()['nonabelian_fusion_u1']
    assert outer.currsize == 8 and outer.misses == 128


def test_adjacent_swap_shares_spin_cache(small_coefficient_caches):
    reference = _merging._adjacent_swap_coefficient(sym_SU2, (1,), (1,), (1,), (1,), (0,), (2,))
    assert abs(reference) > 0
    for _ in range(2):
        for q in range(-16, 16):
            result = _merging._adjacent_swap_coefficient(
                sym_SU2xU1, (1, q), (1, 2 * q), (1, -q), (1, 2 * q), (0, 3 * q), (2, 0))
            assert result == pytest.approx(reference)
    info = yastn.get_cache_info()['adjacent_swap_coefficient']
    assert info.currsize == 1
    assert info.misses == 1
    assert info.hits == 64
    outer = yastn.get_cache_info()['adjacent_swap_u1']
    assert outer.currsize == 8 and outer.misses == 64


def test_u1_lookup_caches_are_independent(small_coefficient_caches):
    fusion = fusion_args(sym_SU2xU1, 3)
    swap = (sym_SU2xU1, (1, 3), (1, 6), (1, -3), (1, 6), (0, 9), (2, 0))
    for _ in range(3):
        assert abs(_nonabelian.fusion_coefficient(*fusion)) > 0
        assert abs(_merging._adjacent_swap_coefficient(*swap)) > 0
    for inner, outer in (('nonabelian_fusion_coefficient', 'nonabelian_fusion_u1'),
                         ('adjacent_swap_coefficient', 'adjacent_swap_u1')):
        infos = yastn.get_cache_info()
        assert infos[outer].hits == 2 and infos[outer].misses == 1
        assert infos[inner].hits == 0 and infos[inner].misses == 1
        # Resize only the outer cache. Its miss must reuse the retained spin entry.
        yastn.set_cache_maxsize({outer: 1})
        assert yastn.get_cache_info()[inner] == infos[inner]
    _nonabelian.fusion_coefficient(*fusion)
    _merging._adjacent_swap_coefficient(*swap)
    infos = yastn.get_cache_info()
    for name in ('nonabelian_fusion_coefficient', 'adjacent_swap_coefficient'):
        assert infos[name].hits == 1 and infos[name].misses == 1
    yastn.clear_cache()
    for name in ('nonabelian_fusion_coefficient', 'adjacent_swap_coefficient',
                 'nonabelian_fusion_u1', 'adjacent_swap_u1'):
        assert yastn.get_cache_info()[name].currsize == 0


def test_u1_churn_does_not_recompute_spin_networks(small_coefficient_caches, monkeypatch):
    # Two spin networks fit exactly. U1 aliases must not displace either one.
    references = {j: _nonabelian.fusion_coefficient(*fusion_args(sym_SU2, 0, j)) for j in (0, 2)}
    yastn.set_cache_maxsize({'nonabelian_fusion_coefficient': 2})
    yastn.clear_cache()
    evaluate = _nonabelian._contract_scalar
    evaluations = []

    def counted(factors):
        evaluations.append(1)
        return evaluate(factors)

    monkeypatch.setattr(_nonabelian, '_contract_scalar', counted)
    for q in range(32):
        for j, expected in references.items():
            actual = _nonabelian.fusion_coefficient(*fusion_args(sym_SU2xU1, q, j))
            assert actual == pytest.approx(expected)
    assert len(evaluations) == 2
    assert yastn.get_cache_info()['nonabelian_fusion_u1'].misses == 64


def test_adjacent_swap_preserves_u1_selection_rules():
    sym = sym_SU2xU1
    nonzero = 0
    for qx, ql, qr in product((-2, 0, 1), repeat=3):
        for jy, jyp in product((0, 2), repeat=2):
            valid = ((1, qx), (1, ql), (1, qr), (1, qx + ql + qr),
                     (jy, qx + ql), (jyp, qx + qr))
            # Break each conservation relation, including external charges.
            cases = [valid]
            for i in range(6):
                case = list(valid)
                j, q = case[i]
                case[i] = (j, q + 1)
                cases.append(tuple(case))
            for x, left, right, z, y, yp in cases:
                expected = sum(sym.f_symbol(x, left, right, z, y, middle)
                               * sym.braiding_phase(left, right, middle)
                               * sym.f_symbol(x, right, left, z, yp, middle)
                               for middle in sym.fusion_outcomes(left, right))
                actual = _merging._adjacent_swap_coefficient(sym, x, left, right, z, y, yp)
                assert actual == pytest.approx(expected, abs=1e-14)
                nonzero += abs(expected) > 1e-14
    assert nonzero == 108
