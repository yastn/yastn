import yamps.tensor as tensor
import settings_full
import settings_U1
import settings_Z2_U1
import settings_U1_U1
import pytest


def test_dot0():
    a = tensor.rand(settings=settings_full, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    b = tensor.rand(settings=settings_full, s=(1, -1, 1), D=(2, 3, 5))

    settings_full.dot_merge = True
    c1 = a.dot(b, axes=((0, 1), (0, 1)))
    settings_full.dot_merge = False
    c2 = b.dot(a, axes=((1, 0), (1, 0)))
    c2 = c2.transpose(axes=(1, 2, 0))
    assert 0 == pytest.approx(c1.norm_diff(c2), abs=1e-8)


def test_dot1():
    a = tensor.rand(settings=settings_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b = tensor.rand(settings=settings_U1, s=(1, -1, 1),
                    t=((-1, 2), (1, 2), (-1, 1)),
                    D=((1, 3), (5, 6), (10, 11)))

    settings_U1.dot_merge = True
    c1 = a.dot(b, axes=((0, 1), (0, 1)))
    settings_U1.dot_merge = False
    c2 = b.dot(a, axes=((1, 0), (1, 0)))
    c2 = c2.transpose(axes=(1, 2, 0))
    assert 0 == pytest.approx(c1.norm_diff(c2), abs=1e-8)


def test_dot2():
    a = tensor.rand(settings=settings_Z2_U1, s=(-1, 1, 1, -1),
                    t=((-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)),
                    D=((1, 2), (1, 2), (3, 4), (3, 4), (5, 6), (5, 6), (7, 8), (7, 8)))
    b = tensor.rand(settings=settings_Z2_U1, s=(1, -1, 1),
                    t=((-1, 1), (-1, 1), (-1, 1), (-1, 1), (0, 2), (0, 2)),
                    D=((1, 2), (1, 2), (3, 4), (3, 4), (7, 8), (7, 8)))

    settings_Z2_U1.dot_merge = False
    c1 = a.dot(b, axes=((0, 1), (0, 1)))
    settings_Z2_U1.dot_merge = True
    c2 = b.dot(a, axes=((0, 1), (0, 1)))
    c2 = c2.transpose(axes=(1, 2, 0))
    assert 0 == pytest.approx(c1.norm_diff(c2), abs=1e-8)


if __name__ == '__main__':
    test_dot0()
    test_dot1()
    test_dot2()
