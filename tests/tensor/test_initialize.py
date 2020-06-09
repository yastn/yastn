from yamps.tensor import Tensor
from yamps.tensor import match_legs
import settings_full
import settings_U1
import settings_Z2_U1
import settings_U1_U1
import pytest


def test_reset0():
    a = Tensor(settings=settings_full, s=(-1, 1, 1))
    a.reset_tensor(D=(1, 2, 3), val='rand')
    a.show_properties()
    npa = a.to_numpy()
    assert npa.shape == (1, 2, 3)
    assert a.tset.shape == (1, 3, 0)

    a0 = Tensor(settings=settings_full, s=())
    a0.reset_tensor(t=(), D=(), val='randC')
    a0.show_properties()
    npa0 = a0.to_numpy()
    assert npa0.shape == ()
    assert a0.tset.shape == (1, 0, 0)


def test_reset1():
    a = Tensor(settings=settings_U1, s=(-1, 1, 1))
    a.reset_tensor(t=((-2, 0, 2), (0, 2), (-2, 0, 2)),
                   D=((1, 2, 3), (1, 2), (1, 2, 3)),
                   val='ones')
    a.show_properties()
    npa = a.to_numpy()
    assert npa.shape == (6, 3, 6)
    assert a.tset.shape == (5, 3, 1)

    a0 = Tensor(settings=settings_U1, s=())
    a0.reset_tensor(t=(), D=(), val='zeros')
    a0.show_properties()
    npa0 = a0.to_numpy()
    assert npa0.shape == ()
    assert a0.tset.shape == (1, 0, 1)

    a0 = Tensor(settings=settings_U1, s=())
    a0.reset_tensor(t=(), D=(), val='ones')
    a0.show_properties()
    npa0 = a0.to_numpy()
    assert npa0.shape == ()
    assert a0.tset.shape == (1, 0, 1)


def test_reset2():
    a = Tensor(settings=settings_Z2_U1, s=(-1, 1, 1))
    a.reset_tensor(t=((0, 1), (0, 2), 0, (-2, 2), (-1, 0, 1), (-2, 0, 2)),
                   D=((1, 2), (1, 2), 1, (1, 2), (1, 2, 3), (1, 2, 3)),
                   val='rand')
    a.show_properties()
    npa = a.to_numpy()
    assert npa.shape == (9, 3, 36)
    assert a.tset.shape == (9, 3, 2)

    b = Tensor(settings=settings_Z2_U1, s=(-1, 1, 1))
    b.reset_tensor(t=[[(0, 1), (1, 0)], [(0, 0)], [(0, 1), (1, 0), (2, 1)]],
                   D=[[1, 2], 3, [1, 2, 3]],
                   val='rand')
    b.show_properties()
    npb = b.to_numpy()
    assert npb.shape == (3, 3, 6)
    assert b.tset.shape == (3, 3, 2)


def test_reset_examples():
    a = Tensor(settings=settings_U1, s=(-1, 1, 1))
    a.reset_tensor(t=[0, (-2, 0), (2, 0)],
                   D=[1, (1, 2), (1, 3)],
                   val='ones')
    a.show_properties()

    b = Tensor(settings=settings_U1_U1, s=(-1, 1, 1))
    b.reset_tensor(t=[0, 0, (-2, 0), (-2, 0), (2, 0), (2, 0)],
                   D=[1, 1, (1, 2), (1, 2), (1, 3), (1, 3)],
                   val='ones')
    b.show_properties()

    c = Tensor(settings=settings_U1_U1, s=(-1, 1, 1))
    c.reset_tensor(t=[[(0, 0)], [(-2, -2), (0, 0), (-2, 0), (0, -2)], [(2, 2), (0, 0), (2, 0), (0, 2)]],
                   D=[1, (1, 4, 2, 2), (1, 9, 3, 3)],
                   val='ones')
    c.show_properties()
    assert pytest.approx(b.norm_diff(c)) == 0
    assert pytest.approx(c.norm_diff(b)) == 0

    ta, da = a.get_tD()
    tb, db = b.get_tD()
    tc, dc = c.get_tD()
    a1 = Tensor(settings=settings_U1, s=(-1, 1, 1))
    a1.reset_tensor(t=ta, D=da, val='ones')
    b1 = Tensor(settings=settings_U1_U1, s=(-1, 1, 1))
    b1.reset_tensor(t=tb, D=db, val='ones')
    c1 = Tensor(settings=settings_U1_U1, s=(-1, 1, 1))
    c1.reset_tensor(t=tc, D=dc, val='ones')
    assert pytest.approx(a.norm_diff(a1)) == 0
    assert pytest.approx(b.norm_diff(b1)) == 0
    assert pytest.approx(c.norm_diff(c1)) == 0


if __name__ == '__main__':
    test_reset_examples()
