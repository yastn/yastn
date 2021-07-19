""" yast.add_leg """
import yast
try:
    from .configs import config_dense, config_U1, config_Z2_U1
except ImportError:
    from configs import config_dense, config_U1, config_Z2_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_aux_0():
    """ add_leg for dense; nsym=0 """
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    c = yast.add_leg(a)
    c.is_consistent()
    assert a.are_independent(c)
    a.add_leg(inplace=True)
    assert yast.norm_diff(a, c) < tol
    assert a.are_independent(c)


def test_aux_1():
    """ add_leg with nsym=1 """
    a = yast.Tensor(config=config_U1, s=(-1, 1), n=-1)
    a.set_block(ts=(1, 0), Ds=(1, 1), val=1)

    b = yast.Tensor(config=config_U1, s=(-1, 1), n=1)
    b.set_block(ts=(0, 1), Ds=(1, 1), val=1)

    ab1 = yast.tensordot(a, b, axes=((), ()))

    yast.add_leg(a, s=1, inplace='True')
    yast.add_leg(b, s=-1, inplace='True')
    a.is_consistent()
    b.is_consistent()

    ab2 = yast.tensordot(a, b, axes=(2, 2))
    assert yast.norm_diff(ab1, ab2) < tol


def test_aux_2():
    """ add_leg with nsym=2 """
    a = yast.rand(config=config_Z2_U1, s=(-1, 1, 1), n=(1, 2),
                  t=[[(0, 0), (1, 0), (0, 2), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 2), (1, -2), (1, 0), (1, 2)]],
                  D=((1, 2, 2, 4), (2, 3), (2, 6, 3, 6, 9)), dtype='float64')
    b = yast.rand(config=config_Z2_U1, s=(-1, 1, 1), n=(1, 0),
                  t=[[(0, 0), (1, 0), (0, 2), (1, 2)], [(0, -2), (0, 2)], [(0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)]],
                  D=((1, 2, 2, 4), (2, 3), (2, 4, 6, 3, 6, 9)), dtype='float64')

    a.add_leg(s=1, t=(1, 0), inplace=True)
    a.add_leg(s=-1, axis=0, t=(0, 2), inplace=True)
    assert a.struct.n == (0, 0)
    a.is_consistent()

    assert b.get_shape() == (9, 5, 30)
    b.fuse_legs(axes=(0, (1, 2)), inplace=True)
    assert b.get_shape() == (9, 75)

    b.add_leg(axis=1, inplace=True)
    b.add_leg(axis=3, inplace=True)
    assert b.get_shape() == (9, 1, 75, 1)
    assert b.get_shape(native=True) == (9, 1, 5, 30, 1)

    b.is_consistent()


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
    assert yast.norm_diff(T1, T2) < tol


if __name__ == '__main__':
    test_aux_0()
    test_aux_1()
    test_aux_2()
    test_operators_chain()
