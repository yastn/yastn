from numpy.lib.function_base import _cov_dispatcher
import yamps.yast as yast
import config_dense_C
import config_U1_R
import config_U1_C
import pytest
import numpy as np


def scalar_vs_numpy(a, b):
    outa = tuple(ii for ii in range(a.ndim))
    outb = tuple(ii for ii in range(b.ndim))
    tDsa = {ii: b.get_leg_tD(ii) for ii in range(b.ndim)}
    tDsb = {ii: a.get_leg_tD(ii) for ii in range(a.ndim)}
    na = a.to_dense(tDsa)
    nb = b.to_dense(tDsb)
    ns = na.conj().reshape(-1) @ nb.reshape(-1)
    sab = a.scalar(b)
    sba = b.scalar(a)
    assert pytest.approx(ns) == sab
    assert pytest.approx(ns) == np.conj(sba)

def scalar_catch_error(a, b):
    try:
        a.scalar(b)        
        assert False
    except yast.FatalError:
        assert True

def test_scalar_0():
    a = yast.rand(config=config_dense_C, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    b = yast.rand(config=config_dense_C, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    scalar_vs_numpy(a, b)


def test_scalar_1R():
    a = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 0, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 2, 12)))
    b = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)))
    c = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                  t=(1, -1, 2, 0),
                  D=(2, 4, 9, 2))
    scalar_vs_numpy(a, b)
    scalar_vs_numpy(a, c)
    scalar_vs_numpy(c, b)


def test_scalar_1C():
    a = yast.rand(config=config_U1_C, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 0, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 2, 12)))
    b = yast.rand(config=config_U1_C, s=(-1, 1, 1, -1),
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)))
    c = yast.rand(config=config_U1_C, s=(-1, 1, 1, -1),
                  t=(1, -1, 2, 0),
                  D=(2, 4, 9, 2))
    scalar_vs_numpy(a, b)
    scalar_vs_numpy(a, c)
    scalar_vs_numpy(c, b)


def test_scalar_exceptions():
    a = yast.Tensor(config=config_U1_C, s=())
    b = yast.Tensor(config=config_U1_C, s=())
    scalar_vs_numpy(a, b)

    a = yast.rand(config=config_U1_C, s=(-1, 1, 1, -1), n=1,
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 0, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 2, 12)))
    b = yast.rand(config=config_U1_C, s=(-1, 1, 1, -1), n=-1,
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)))
    c = yast.rand(config=config_U1_C, s=(-1, -1, 1, -1), n=1,
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)))
    d = yast.rand(config=config_U1_C, s=(-1, 1, 1, -1), n=1,
                  t=((-1, 2), (1, 2), (-1, 1), (-1, 0, 1, 2)),
                  D=((1, 3), (5, 6), (7, 8), (10, 2, 11, 12)))

    d = d.fuse_legs(axes=(0, (1, 2), 3))

    scalar_catch_error(a, b)
    scalar_catch_error(a, c)
    scalar_catch_error(a, d)


if __name__ == '__main__':
    test_scalar_0()
    test_scalar_1R()
    test_scalar_1C()
    test_scalar_exceptions()
    