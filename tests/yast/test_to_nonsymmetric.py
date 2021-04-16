try:
    import yast
except ModuleNotFoundError:
    import fix_path
    import yast
import config_dense_R
import config_U1_R
import pytest
from math import isclose

tol = 1e-12


def test_to_nonsymmetric_0():
    a = yast.rand(config=config_dense_R, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    b = yast.rand(config=config_dense_R, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))

    an = a.to_nonsymmetric()
    bn = b.to_nonsymmetric()
    assert isclose(an.vdot(bn), a.vdot(b), rel_tol=tol)
    assert isclose(a.vdot(bn), a.vdot(b), rel_tol=tol)   # for dense to_nonsymetric should result in the same config
    assert an.are_independent(a)
    assert bn.are_independent(b)
    assert an.is_consistent()
    assert bn.is_consistent()


def test_to_nonsymmetric_1():
    a = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                  t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                  t=((-2, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    lsa = {0: a.get_leg_structure(axis=0)}
    lsb = {0: b.get_leg_structure(axis=0)}
    
    an = a.to_nonsymmetric(leg_structures=lsb)
    bn = b.to_nonsymmetric(leg_structures=lsa)
    assert isclose(an.vdot(bn), a.vdot(b), rel_tol=tol)
    with pytest.raises(yast.YastError):
        a.vdot(bn)
    assert an.are_independent(a)
    assert bn.are_independent(b)
    assert an.is_consistent()
    assert bn.is_consistent()


if __name__ == '__main__':
    test_to_nonsymmetric_0()
    test_to_nonsymmetric_1()
