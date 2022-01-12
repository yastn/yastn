""" yast.add_leg() yast.remove_leg() """
import pytest
import yast
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def _test_add_remove_leg(a):
    """ run a sequence of adding and removing axis operations. """
    b = yast.add_leg(a)  # new axis with s=1 added as the last one; makes copy
    assert b.is_consistent()
    assert yast.are_independent(a, b)
    assert all(x == 0 for x in b.struct.n)  # tensor charge is set here to 0
    b.add_leg(axis=1, s=-1, inplace=True)
    b.is_consistent()

    c = b.remove_leg()  # removes last axis by default
    assert c.is_consistent()
    assert c.struct.n == a.struct.n
    c.remove_leg(axis=1, inplace=True)
    assert c.is_consistent()
    assert yast.norm(a - c) < tol


def test_add_leg_basic():
    """ add_leg for various symmetries """
    # dense
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    _test_add_remove_leg(a)

    # U1
    a = yast.Tensor(config=config_U1, s=(-1, 1), n=-1)
    a.set_block(ts=(1, 0), Ds=(1, 1), val=1)  # creation operator
    b = yast.Tensor(config=config_U1, s=(-1, 1), n=1)
    b.set_block(ts=(0, 1), Ds=(1, 1), val=1)  # anihilation operator
    _test_add_remove_leg(a)
    _test_add_remove_leg(b)

    ab1 = yast.tensordot(a, b, axes=((), ()))
    a.add_leg(s=1, inplace='True')
    b.add_leg(s=-1, inplace='True')
    ab2 = yast.tensordot(a, b, axes=(2, 2))
    assert yast.norm(ab1 - ab2) < tol

    # Z2xU1
    a = yast.rand(config=config_Z2xU1, s=(-1, 1, 1), n=(1, 2),
                  t=[[(0, 0), (1, 0), (0, 2), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)]],
                  D=((1, 2, 2, 4), (2, 3), (2, 4, 6, 3, 6, 9)))
    assert a.get_shape() == (9, 3, 25)
    _test_add_remove_leg(a)

    # new axis with tensor charge set by hand.
    a.add_leg(s=-1, axis=0, t=(0, 2), inplace=True)
    assert a.struct.n == (1, 0)
    a.is_consistent()
    a.add_leg(s=1, t=(1, 0), inplace=True)
    assert a.struct.n == (0, 0)
    a.is_consistent()

    # mix adding/removing axes with fusions
    assert a.get_shape() == (1, 9, 3, 25, 1)
    a.fuse_legs(axes=((1, 0), 2, (3, 4)), inplace=True, mode='hard')
    assert a.get_shape() == (9, 3, 25)
    a.fuse_legs(axes=((0, 1), 2), inplace=True, mode='meta')
    assert a.get_shape() == (27, 25)

    a.add_leg(axis=1, inplace=True)
    assert a.get_shape() == (27, 1, 25)
    a.add_leg(axis=3, inplace=True)
    assert a.get_shape() == (27, 1, 25, 1)
    a.unfuse_legs(axes=0, inplace=True)
    assert a.get_shape() == (9, 3, 1, 25, 1)
    a.unfuse_legs(axes=(0, 3), inplace=True)
    assert a.get_shape(native=True) == (9, 1, 3, 1, 25, 1, 1)
    a.is_consistent()


def test_operators_chain():
    """
    Consider a sequence of operators "cdag cdag c c"
    add virtual legs connecting them, starting from the end
    """

    cdag = yast.Tensor(config=config_U1, s=(1, -1), n=1)
    cdag.set_block(ts=(1, 0), Ds=(1, 1), val=1)
    c = yast.Tensor(config=config_U1, s=(1, -1), n=-1)
    c.set_block(ts=(0, 1), Ds=(1, 1), val=1)

    nn = (0,) * len(c.n)
    o4 = yast.add_leg(c, axis=-1, t=nn, s=-1)
    o4 = yast.add_leg(o4, axis=0, s=1)
    tD = o4.get_leg_structure(axis=0)
    nn = next(iter(tD))

    o3 = yast.add_leg(c, axis=-1, t=nn, s=-1)
    o3 = yast.add_leg(o3, axis=0, s=1)
    tD = o3.get_leg_structure(axis=0)
    nn = next(iter(tD))

    o2 = yast.add_leg(cdag, axis=-1, t=nn, s=-1)
    o2 = yast.add_leg(o2, axis=0, s=1)
    tD = o2.get_leg_structure(axis=0)
    nn = next(iter(tD))

    o1 = yast.add_leg(cdag, axis=-1, t=nn, s=-1)
    o1 = yast.add_leg(o1, axis=0, s=1)
    tD = o1.get_leg_structure(axis=0)
    nn = next(iter(tD))

    assert nn == (0,) * len(c.n)

    T1 = yast.ncon([cdag, cdag, c, c], [(-1, -5), (-2, -6), (-3 ,-7), (-4, -8)])
    T2 = yast.ncon([o1, o2, o3, o4], [(4, -1, -5, 1), (1, -2, -6, 2), (2, -3 ,-7, 3), (3, -4, -8, 4)])
    assert yast.norm(T1 -  T2) < tol

    # special case when there are no blocks in the tensor
    a = yast.Tensor(config=config_U1, s=(1, -1, 1, -1), n=1)
    a.remove_leg(axis=1, inplace=True)
    assert a.struct.s == (1, 1, -1)
    a.remove_leg(axis=1, inplace=True)
    assert a.struct.s == (1, -1)
    assert a.struct.n == (1,)


def test_add_leg_exceptions():
    """ handling exceptions in yast.add_leg()"""
    with pytest.raises(yast.YastError):
        a = yast.eye(config=config_U1, t=(-1, 0, 1), D=(2, 3, 4))
        a.add_leg(s=1)  # Cannot add axis to a diagonal tensor.
    with pytest.raises(yast.YastError):
        a = yast.ones(config=config_U1, s=(1, -1), t=((-1, 0, 1), (-1, 0, 1)), D=((2, 3, 4), (2, 3, 4)), n=1)
        a.add_leg(s=1, t=(1, 0))  # len(t) does not match the number of symmetry charges.
    with pytest.raises(yast.YastError):
        a = yast.ones(config=config_U1, s=(1, -1), t=((-1, 0, 1), (-1, 0, 1)), D=((2, 3, 4), (2, 3, 4)), n=1)
        a.add_leg(s=2)  # Signature of the new axis should be 1 or -1.


def test_remove_leg_exceptions():
    """ handling exceptions in yast.remove_leg()"""
    t, D = (-1, 0, 1), (2, 3, 4)
    with pytest.raises(yast.YastError):
        a = yast.eye(config=config_U1, t=t, D=D)
        a.remove_leg(axis=1)  # Cannot remove axis to a diagonal tensor.
    with pytest.raises(yast.YastError):
        a = yast.ones(config=config_U1, s=(1, -1), t=(t, t), D=(D, D), n=1)
        scalar = yast.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))
        _ = scalar.remove_leg(axis=0)  # Cannot remove axis of a scalar tensor.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(1, -1, 1), t=(t, t, t), D=(D, D, D))
        a.fuse_legs(axes=((0, 1), 2), mode='meta', inplace=True)
        _ = a.remove_leg(axis=0)  # Axis to be removed cannot be fused.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(1, -1, 1, 1), t=(t, t, t, t), D=(D, D, D, D))
        a.fuse_legs(axes=((0, 1), 2, 3), mode='meta', inplace=True)
        a.fuse_legs(axes=(0, (1, 2)), mode='hard', inplace=True)
        _ = a.remove_leg(axis=0)  # Axis to be removed cannot be fused.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(1, -1, 1, 1), t=(t, t, t, t), D=(D, D, D, D))
        _ = a.remove_leg(axis=1)  # Axis to be removed must have single charge of dimension one.


if __name__ == '__main__':
    test_add_leg_basic()
    test_operators_chain()
    test_add_leg_exceptions()
    test_remove_leg_exceptions()
