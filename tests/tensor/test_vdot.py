""" yast.vdot """
import pytest
import yast
from .configs import config_dense, config_U1

tol = 1e-12


def scalar_vs_numpy(a, b):
    tDsa = {ii: b.get_leg_structure(ii) for ii in range(b.get_ndim())}
    tDsb = {ii: a.get_leg_structure(ii) for ii in range(a.get_ndim())}
    na = a.to_dense(tDsa)
    nb = b.to_dense(tDsb)
    ns = na.conj().reshape(-1) @ nb.reshape(-1)
    sab = a.vdot(b)
    sba = b.vdot(a)
    assert abs(ns - sab) < tol  # == 0.0
    assert abs(ns.conj() - sba) < tol  # == 0.0


def test_scalar_0():
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='complex128')
    b = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='complex128')
    scalar_vs_numpy(a, b)


def test_scalar_1R():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 0, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 2, 12)))
    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)))
    c = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=(1, -1, 2, 0),
                  D=(2, 4, 9, 2))
    scalar_vs_numpy(a, b)
    scalar_vs_numpy(a, c)
    scalar_vs_numpy(c, b)


def test_scalar_1C():
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 0, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 2, 12)), dtype='complex128')
    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)), dtype='complex128')
    c = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=(1, -1, 2, 0),
                  D=(2, 4, 9, 2), dtype='complex128')
    scalar_vs_numpy(a, b)
    scalar_vs_numpy(a, c)
    scalar_vs_numpy(c, b)


def test_scalar_exceptions():
    a = yast.Tensor(config=config_U1, s=(), dtype='complex128')
    b = yast.Tensor(config=config_U1, s=(), dtype='complex128')
    scalar_vs_numpy(a, b)

    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1), n=1,
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 0, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 2, 12)), dtype='complex128')

    b = yast.rand(config=config_U1, s=(-1, 1, 1, -1), n=-1,
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)), dtype='complex128')

    c = yast.rand(config=config_U1, s=(-1, -1, 1, -1), n=1,
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)), dtype='complex128')

    d = yast.rand(config=config_U1, s=(-1, 1, 1, -1), n=1,
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)), dtype='complex128')
    d = d.fuse_legs(axes=(0, (1, 2), 3))

    e = yast.rand(config=config_U1, s=(1, -1, -1, 1), n=1,
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)), dtype='complex128')

    assert abs(a.vdot(b)) < tol
    assert abs(a.vdot(e, conj=(0, 0))) < tol
    assert abs(a.vdot(e, conj=(1, 1))) < tol

    with pytest.raises(yast.YastError):
        a.vdot(c)
    with pytest.raises(yast.YastError):
        a.vdot(d)


if __name__ == '__main__':
    test_scalar_0()
    test_scalar_1R()
    test_scalar_1C()
    test_scalar_exceptions()
