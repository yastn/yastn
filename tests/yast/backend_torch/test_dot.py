import yamps.yast as yast
import config_dense_R
import config_U1_R
import config_Z2_U1_R
from math import isclose
import numpy as np

tol = 1e-12

def dot_vs_numpy(a, b, axes, conj):
    outa = tuple(ii for ii in range(a.ndim) if ii not in axes[0])
    outb = tuple(ii for ii in range(b.ndim) if ii not in axes[1])
    tDs = {nn: a.get_leg_tD(ii) for nn, ii in enumerate(outa)}
    tDs.update({nn + len(outa): b.get_leg_tD(ii) for nn, ii in enumerate(outb)})
    tDsa = {ia: b.get_leg_tD(ib) for  ia, ib in zip(*axes)}
    tDsb = {ib: a.get_leg_tD(ia) for  ia, ib in zip(*axes)}
    na = a.to_numpy(tDsa)
    nb = b.to_numpy(tDsb)
    if conj[0]:
        na = na.conj()
    if conj[1]:
        nb = nb.conj()
    nab = np.tensordot(na, nb, axes)
    c = a.dot(b, axes, conj)
    nc = c.to_numpy(tDs)
    assert c.is_consistent()
    assert a.is_independent(c)
    assert c.is_independent(b)
    assert isclose(np.linalg.norm(nc - nab), 0, rel_tol=tol, abs_tol=tol)


def test_dot_0():
    a = yast.rand(config=config_dense_R, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    b = yast.rand(config=config_dense_R, s=(1, -1, 1), D=(2, 3, 5))

    dot_vs_numpy(a, b, axes=((0, 3), (0, 2)), conj=(0, 0))
    dot_vs_numpy(b, a, axes=((2, 0), (3, 0)), conj=(1, 1))


def test_dot_1():
    a = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b = yast.rand(config=config_U1_R, s=(1, -1, 1),
                    t=((-1, 2), (1, 2), (-1, 1)),
                    D=((1, 3), (5, 6), (10, 11)))

    dot_vs_numpy(a, b, ((0, 1), (0, 1)), (0, 0))
    dot_vs_numpy(a, b, ((1, 3), (1, 2)), (0, 0))

def test_dot_1_sparse():
    a = yast.Tensor(config=config_U1_R, s=(-1, 1, 1, -1), n=-2)
    a.set_block(ts=(2, 1, 0, 1), Ds=(2, 1, 10, 1), val='rand')
    b = yast.Tensor(config=config_U1_R, s=(-1, 1, 1, -1), n=1)
    b.set_block(ts=(1, 2, 0, 0), Ds=(1, 2, 10, 10), val='rand')

    dot_vs_numpy(a, b, ((2, 1), (1, 2)), (1, 0))

    a.set_block(ts=(1, 1, -1, 1), Ds=(1, 1, 11, 1), val='rand')
    a.set_block(ts=(2, 2, -1, 1), Ds=(2, 2, 11, 1), val='rand')
    a.set_block(ts=(3, 3, -1, 1), Ds=(3, 3, 11, 1), val='rand')
    b.set_block(ts=(1, 1, 1, 0), Ds=(1, 1, 1, 10), val='rand')
    b.set_block(ts=(3, 3, 1, 0), Ds=(3, 3, 1, 10), val='rand')
    b.set_block(ts=(3, 3, 2, 1), Ds=(3, 3, 2, 1), val='rand')

    dot_vs_numpy(a, b, ((0, 1), (0, 1)), (0, 1))
    dot_vs_numpy(a, b, ((0, 3, 1), (1, 2, 0)), (0, 0))

def test_dot_2():
    t1 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    t2 = [(0, 0), (0, 2), (2, 0), (2, 2)]
    a = yast.rand(config=config_Z2_U1_R, s=(-1, 1, 1, -1),
                    t=(t1, t1, t1, t1),
                    D=((1, 2, 2, 4), (9, 4, 3, 2), (5, 6, 7, 8), (7, 8, 9, 10)))
    b = yast.rand(config=config_Z2_U1_R, s=(1, -1, 1),
                    t=(t1, t1, t2),
                    D=((1, 2, 2, 4), (9, 4, 3, 2), (5, 6, 7, 8,)))

    dot_vs_numpy(a, b, ((0, 1), (0, 1)), (0, 0))
    dot_vs_numpy(b, a, ((1, 0), (1, 0)), (0, 0))


if __name__ == '__main__':
    test_dot_0()
    test_dot_1()
    test_dot_1_sparse()
    test_dot_2()
