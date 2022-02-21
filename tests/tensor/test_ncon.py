""" Test yast.ncon """
import pytest
import yast
try:
    from .configs import config_dense, config_U1
except ImportError:
    from configs import config_dense, config_U1

tol = 1e-12  # pylint: disable=invalid-name


def test_ncon_syntax():
    # create a set of U(1)-symmetric tensors
    a = yast.rand(config=config_U1, s=[-1, 1, -1], n=0,
                  D=((20, 10), (3, 3), (1, 1)), t=((1, 0), (1, 0), (1, 0)))
    b = yast.rand(config=config_U1, s=[1, 1, 1], n=1,
                  D=((4, 4), (2, 2), (20, 10)), t=((1, 0), (1, 0), (1, 0)))
    c = yast.rand(config=config_U1, s=[1, 1, 1, -1], n=1,
                  D=((20, 10), (30, 20), (10, 5), (10, 5)), t=((1, 0), (1, 0), (1, 0), (1, 0)))
    d = yast.rand(config=config_U1, s=[1, 1, -1, -1], n=0,
                  D=((30, 20), (10, 5), (20, 10), (10, 5)), t=((1, 0), (1, 0), (1, 0), (1, 0)))

    # Perform basic contraction of two tensors - equivalent to a single tensordot call
    #           _                 _                        _    _
    #  (-1) 1--|a|--0 (1) (1) 2--|b|--0 (-0) = (-1)    1--|a|--|b|--0    (-0)
    #  (-3) 2--|_|               |_|--1 (-2)   (-3) 3<-2--|_|  |_|--1->2 (-2)
    #
    # The uncontracted indices, labeled by negative integers, are ordered according in
    # descending fashion on resulting tensor 
    e = yast.ncon([a, b], [[1, -1, -3], [-0, -2, 1]])
    assert e.get_shape() == (8, 6, 4, 2)

    # Network composed of several tensors can be contracted by a single ncon call,
    # including traces and conjugations
    #           _                 _                       _    _    __    __
    #  (-2) 1--|a|--0 (4) (4) 0--|c|--2 (1) = (-2) 2<-1--|a|--|c|--|d*|--|b*|--0->3 (-3)
    #  (-0) 2--|_|               |_|--3 (1)   (-0) 0<-2--|_|  |_|  |__|  |__|--1->1 (-1)
    #                             |
    #                             1 (3)
    #                             0 (3) 
    #           __                |_
    #  (-3) 0--|b*|--2(5) (5) 2--|d*|--1 (2)
    #  (-1) 1--|__|              |__|--3 (2)
    #
    f = yast.ncon([a, b, c, d], [[4, -2, -0], [-3, -1, 5], [4, 3, 1, 1], [3, 2, 5, 2]],
                  conjs=(0, 1, 0, 1))
    assert f.get_shape() == (2, 4, 6, 8)
    

def test_ncon_basic():
    """ tests of ncon executing a series of tensor contractions. """
    a = yast.rand(s=(1, 1, 1), D=(20, 3, 1), config=config_dense, dtype='complex128')
    b = yast.rand(s=(1, 1, -1), D=(4, 2, 20), config=config_dense, dtype='complex128')
    c = yast.rand(s=(-1, 1, 1, -1), D=(20, 30, 10, 10), config=config_dense, dtype='complex128')
    d = yast.rand(s=(1, 1, 1, -1), D=(30, 10, 20, 10), config=config_dense, dtype='complex128')
    e = yast.rand(s=(1, 1), D=(3, 1), config=config_dense, dtype='complex128')
    f = yast.rand(s=(-1, -1), D=(2, 4), config=config_dense, dtype='complex128')
    g = yast.rand(s=(1,), D=(5,), config=config_dense, dtype='complex128')
    h = yast.rand(s=(-1,), D=(6,), config=config_dense, dtype='complex128')

    x = yast.ncon([a, b], ((1, -1, -3), (-0, -2, 1)))
    assert x.get_shape() == (4, 3, 2, 1)

    y = yast.ncon([a, b, c, d], ((4, -2, -0), (-3, -1, 5), (4, 3, 1, 1), (3, 2, 5, 2)), (0, 1, 0, 1))
    assert y.get_shape() == (1, 2, 3, 4)

    z1 = yast.ncon([e, f], ((-2, -0), (-1, -3)), conjs=[0, 1])
    z2 = yast.ncon([f, e], ((-1, -3), (-2, -0)), conjs=[1, 0])
    z3 = yast.ncon([e, f], ((-2, -0), (-1, -3)), conjs=[0, 0])
    z4 = yast.ncon([f, e], ((-1, -3), (-2, -0)), conjs=[1, 1])
    assert all(x.get_shape() == (1, 2, 3, 4) for x in (z1, z2, z3, z4))
    assert yast.norm(z1 - z2) < tol  # == 0.0
    assert yast.norm(z3 - z4.conj()) < tol  # == 0.0

    y1 = yast.ncon([a, b, c, d, g, h], ((4, -2, -0), (-3, -1, 5), (4, 3, 1, 1), (3, 2, 5, 2), (-4,), (-5,)),
                   conjs=(1, 0, 1, 0, 1, 0))
    assert y1.get_shape() == (1, 2, 3, 4, 5, 6)

    y2 = yast.ncon([a, g, a], ((1, 2, 3), (-0,), (1, 2, 3)), conjs=(0, 0, 1))
    assert y2.get_shape() == (5,)

    y3 = yast.ncon([a, g, a, b, a], [[1, 2, 3], [-4], [1, 2, 3], [-3, -1, 4], [4, -2, -0]],
                   conjs=(0, 0, 1, 0, 0))
    assert y3.get_shape() == (1, 2, 3, 4, 5)

    y4 = yast.ncon([a, a, b, b], [[1, 2, 3], [1, 2, 3], [6, 5, 4], [6, 5, 4]],
                   conjs=(0, 1, 0, 1))
    y5 = yast.ncon([a, a, b, b], [[6, 5, 4], [6, 5, 4], [1, 3, 2], [1, 3, 2]],
                   conjs=(0, 1, 0, 1))
    assert isinstance(y4.item(), complex)
    assert isinstance(y5.item(), complex)
    assert yast.norm(y4 - y5) / yast.norm(y4) < tol
    assert pytest.approx((a.norm().item() ** 2) * (b.norm().item() ** 2), rel=tol) == y4.item()

    a = yast.rand(config=config_U1, s=[-1, 1, -1], n=0,
                  D=((20, 10), (3, 3), (1, 1)), t=((1, 0), (1, 0), (1, 0)))
    b = yast.rand(config=config_U1, s=[1, 1, 1], n=1,
                  D=((4, 4), (2, 2), (20, 10)), t=((1, 0), (1, 0), (1, 0)))
    c = yast.rand(config=config_U1, s=[1, 1, 1, -1], n=1,
                  D=((20, 10), (30, 20), (10, 5), (10, 5)), t=((1, 0), (1, 0), (1, 0), (1, 0)))
    d = yast.rand(config=config_U1, s=[1, 1, -1, -1], n=0,
                  D=((30, 20), (10, 5), (20, 10), (10, 5)), t=((1, 0), (1, 0), (1, 0), (1, 0)))

    e = yast.ncon([a, b], [[1, -1, -3], [-0, -2, 1]])
    assert e.get_shape() == (8, 6, 4, 2)
    f = yast.ncon([a, b, c, d], [[4, -2, -0], [-3, -1, 5], [4, 3, 1, 1], [3, 2, 5, 2]],
                  conjs=(0, 1, 0, 1))
    assert f.get_shape() == (2, 4, 6, 8)
    g = yast.ncon([a, a, a, b], [[1, 2, 3], [1, 2, 3], [4, -2, -0], [-3, -1, 4]],
                  conjs=(0, 1, 1, 1))
    assert g.get_shape() == (2, 4, 6, 8)

    a = 1j * yast.ones(config=config_dense, s=(1, 1, -1, -1), D=(1, 1, 1, 1))
    assert abs(yast.ncon([a], [(1, 2, 2, 1)], conjs=[1]).item() + 1j) < tol
    assert abs(yast.ncon([a], [(1, -1, -0, 1)], conjs=[1]).item() + 1j) < tol
    assert abs(yast.ncon([a], [(-0, -1, -2, -3)], conjs=[1]).item() + 1j) < tol
    assert abs(yast.ncon([a], [(1, 2, 2, 1)], conjs=[0]).item() - 1j) < tol


def test_ncon_exceptions():
    """ capturing some exception by ncon. """
    t, D = (0, 1), (2, 3)
    a = yast.rand(config=config_U1, s=[-1, 1, -1], n=0,
                  t=(t, t, t), D=(D, D, D))
    with pytest.raises(yast.YastError):
        _ = yast.ncon([a, a], [(1, 2, 3)])
        # Number of tensors and groups of indices do not match.
    with pytest.raises(yast.YastError):
        _ = yast.ncon([a, a], [(1, 2, -1), (1, 2)])
        # Number of legs of one of the tensors do not match provided indices.
    with pytest.raises(yast.YastError):
        _ = yast.ncon([a, a], [(1, 2, -1), (1, 3, -2)])
        # Indices of legs to contract do not match
    with pytest.raises(yast.YastError):
        _ = yast.ncon([a, a], [(1, 2, -1), (1, 1, -2)])
        # Indices of legs to contract do not match
    with pytest.raises(yast.YastError):
        _ = yast.ncon([a, a], [(3, 3, 2), (1, 1, 2)], conjs=[0, 1])
        # Likely inefficient order of contractions. Do all traces before tensordot.
        # Call all axes connecting two tensors one after another.
    with pytest.raises(yast.YastError):
        _ = yast.ncon([a, a, a], [(1, 2, 3), (1, 3, 4), (2, 4, -0)], conjs=[0, 1, 0])
        # Likely inefficient order of contractions. Do all traces before tensordot.
        # Call all axes connecting two tensors one after another.
    with pytest.raises(yast.YastError):
        yast.ncon([a], [(-1, -1, -0)])
        # Repeated non-positive (outgoing) index is ambiguous.


if __name__ == '__main__':
    test_ncon_basic()
    test_ncon_exceptions()
