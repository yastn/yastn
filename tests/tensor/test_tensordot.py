import numpy as np
import pytest
import yast
try:
    from .configs import config_dense, config_U1, config_Z2_U1
except ImportError:
    from configs import config_dense, config_U1, config_Z2_U1

tol = 1e-12  #pylint: disable=invalid-name


def dot_vs_numpy(a, b, axes, conj, policy=None):
    outa = tuple(ii for ii in range(a.ndim) if ii not in axes[0])
    outb = tuple(ii for ii in range(b.ndim) if ii not in axes[1])
    tDs = {nn: a.get_leg_structure(ii) for nn, ii in enumerate(outa)}
    tDs.update({nn + len(outa): b.get_leg_structure(ii) for nn, ii in enumerate(outb)})
    tDsa = {ia: b.get_leg_structure(ib) for ia, ib in zip(*axes)}
    tDsb = {ib: a.get_leg_structure(ia) for ia, ib in zip(*axes)}
    na = a.to_numpy(tDsa)
    nb = b.to_numpy(tDsb)
    if conj[0]:
        na = na.conj()
    if conj[1]:
        nb = nb.conj()
    nab = np.tensordot(na, nb, axes)
    c = yast.tensordot(a, b, axes, conj, policy=policy)
    nc = c.to_numpy(tDs)
    assert c.is_consistent()
    assert a.are_independent(c)
    assert c.are_independent(b)
    assert np.linalg.norm(nc - nab) < tol  # == 0.0


def test_dot_0():
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='complex128')
    b = yast.rand(config=config_dense, s=(1, -1, 1), D=(2, 3, 5), dtype='complex128')

    dot_vs_numpy(a, b, axes=((0, 3), (0, 2)), conj=(0, 0))
    dot_vs_numpy(b, a, axes=((2, 0), (3, 0)), conj=(1, 1))


@pytest.mark.parametrize("policy", ["direct", "hybrid", "merge"])
def test_dot_1(policy):
    a = yast.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))

    b = yast.rand(config=config_U1, s=(1, -1, 1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 0, 1)),
                  D=((1, 2, 3), (4, 5, 6), (10, 7, 11)))

    dot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 0), policy=policy)
    dot_vs_numpy(a, b, axes=((1, 3), (1, 2)), conj=(0, 0), policy=policy)

    fa = a.fuse_legs(axes=(0, 2, (1, 3)))
    fb = b.fuse_legs(axes=((1, 2), 0))

    dot_vs_numpy(fa, fb, axes=((2,), (0,)), conj=(0, 0), policy=policy)


@pytest.mark.parametrize("policy", ["direct", "hybrid", "merge"])
def test_dot_1_sparse(policy):
    a = yast.Tensor(config=config_U1, s=(-1, 1, 1, -1), n=-2)
    a.set_block(ts=(2, 1, 0, 1), Ds=(2, 1, 10, 1), val='rand')
    b = yast.Tensor(config=config_U1, s=(-1, 1, 1, -1), n=1)
    b.set_block(ts=(1, 2, 0, 0), Ds=(1, 2, 10, 10), val='rand')

    dot_vs_numpy(a, b, axes=((2, 1), (1, 2)), conj=(1, 0), policy=policy)

    a.set_block(ts=(1, 1, -1, 1), Ds=(1, 1, 11, 1), val='rand')
    a.set_block(ts=(2, 2, -1, 1), Ds=(2, 2, 11, 1), val='rand')
    a.set_block(ts=(3, 3, -1, 1), Ds=(3, 3, 11, 1), val='rand')
    b.set_block(ts=(1, 1, 1, 0), Ds=(1, 1, 1, 10), val='rand')
    b.set_block(ts=(3, 3, 1, 0), Ds=(3, 3, 1, 10), val='rand')
    b.set_block(ts=(3, 3, 2, 1), Ds=(3, 3, 2, 1), val='rand')

    dot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 1), policy=policy)
    dot_vs_numpy(a, b, axes=((0, 3, 1), (1, 2, 0)), conj=(0, 0), policy=policy)

    fa = a.fuse_legs(axes=((1, 0), (3, 2)))
    fb = b.fuse_legs(axes=((1, 0), (3, 2)))

    dot_vs_numpy(fa, fb, axes=((0,), (0,)), conj=(0, 1), policy=policy)


@pytest.mark.parametrize("policy", ["direct", "hybrid", "merge"])
def test_dot_2(policy):
    t1 = [(0, -1), (0, 1), (1, -1), (1, 1)]
    t2 = [(0, 0), (0, 2), (2, 0), (2, 2)]
    a = yast.rand(config=config_Z2_U1, s=(-1, 1, 1, -1),
                  t=(t1, t1, t1, t1),
                  D=((1, 2, 2, 4), (9, 4, 3, 2), (5, 6, 7, 8), (7, 8, 9, 10)))
    b = yast.rand(config=config_Z2_U1, s=(1, -1, 1),
                  t=(t1, t1, t2),
                  D=((1, 2, 2, 4), (9, 4, 3, 2), (5, 6, 7, 8,)))

    dot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 0), policy=policy)
    dot_vs_numpy(b, a, axes=((1, 0), (1, 0)), conj=(0, 0), policy=policy)


def test_broadcast():
    t1 = [(0, -1), (0, 1), (1, -1), (1, 1)]
    D1 = (1, 2, 2, 4)

    t2 = [(0, -1), (0, 1), (1, -1), (0, 0)]
    D2 = (1, 2, 2, 5)

    a = yast.rand(config=config_Z2_U1, s=(-1, 1, 1, -1),
                  t=(t1, t1, t1, t1),
                  D=(D1, (9, 4, 3, 2), (5, 6, 7, 8), (7, 8, 9, 10)))
    b = yast.rand(config=config_Z2_U1, s=(1, -1), t = [t2, t2], D=[D2, D2], isdiag=True)
    b2 = b.diag()

    c1 = a.broadcast(b, axis=0)
    c2 = a.broadcast(b, axis=0, conj=(0, 1))
    c3 = b2.tensordot(a, axes=(0, 0))

    assert(yast.norm_diff(c1, c2)) < tol
    assert(yast.norm_diff(c1, c3)) < tol
    assert c3.get_shape() == (5, 18, 26, 34)


if __name__ == '__main__':
    test_dot_0()
    test_dot_1(policy=None)
    test_dot_1_sparse(policy=None)
    test_dot_2(policy=None)
    test_broadcast()
