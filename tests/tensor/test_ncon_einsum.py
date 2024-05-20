""" Test yastn.ncon """
import pytest
import yastn
try:
    from .configs import config_dense, config_U1
except ImportError:
    from configs import config_dense, config_U1

tol = 1e-12  # pylint: disable=invalid-name


def test_ncon_einsum_syntax():
    # create a set of U(1)-symmetric tensors
    a = yastn.rand(config=config_U1, s=[-1, 1, -1], n=0,
                  D=((20, 10), (3, 3), (1, 1)), t=((1, 0), (1, 0), (1, 0)))
    b = yastn.rand(config=config_U1, s=[1, 1, 1], n=1,
                  D=((4, 4), (2, 2), (20, 10)), t=((1, 0), (1, 0), (1, 0)))
    c = yastn.rand(config=config_U1, s=[1, 1, 1, -1], n=1,
                  D=((20, 10), (30, 20), (10, 5), (10, 5)), t=((1, 0), (1, 0), (1, 0), (1, 0)))
    d = yastn.rand(config=config_U1, s=[1, 1, -1, -1], n=0,
                  D=((30, 20), (10, 5), (20, 10), (10, 5)), t=((1, 0), (1, 0), (1, 0), (1, 0)))

    # Perform basic contraction of two tensors - equivalent to a single tensordot call
    #           _                 _                        _    _
    #  (-1) 1--|a|--0 (1) (1) 2--|b|--0 (-0) = (-1)    1--|a|--|b|--0    (-0)
    #  (-3) 2--|_|               |_|--1 (-2)   (-3) 3<-2--|_|  |_|--1->2 (-2)
    #
    # The uncontracted indices, labeled by negative integers, are ordered according in
    # descending fashion on resulting tensor
    e = yastn.ncon([a, b], [[1, -1, -3], [-0, -2, 1]])
    assert e.get_shape() == (8, 6, 4, 2)

    # The same can be obtained using einsum function, which tries to mimic the syntax of np.einsum
    e1 = yastn.einsum('xbd,acx->abcd', a, b)
    assert yastn.norm(e1 - e) < tol

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
    f = yastn.ncon([a, b, c, d], [[4, -2, -0], [-3, -1, 5], [4, 3, 1, 1], [3, 2, 5, 2]],
                  conjs=(0, 1, 0, 1))
    assert f.get_shape() == (2, 4, 6, 8)

    # In yastn.einsum(subscripts, *operands, order='Alphabetic')
    #
    # order specifies the order of contraction, otherwise alphabetic order of contracted indices is used
    # character '*' can be used in einsum subscripts to conjugate respective tensor
    # spaces in subscripts are ignored
    f1 = yastn.einsum('nCA, *DBo, nmkk, *mlol -> ABCD', a, b, c, d, order='klmno')
    assert yastn.norm(f1 - f) < tol


def test_ncon_einsum_basic():
    """ tests of ncon executing a series of tensor contractions. """
    a = yastn.rand(s=(1, 1, 1), D=(20, 3, 1), config=config_dense, dtype='complex128')
    b = yastn.rand(s=(1, 1, -1), D=(4, 2, 20), config=config_dense, dtype='complex128')
    c = yastn.rand(s=(-1, 1, 1, -1), D=(20, 30, 10, 10), config=config_dense, dtype='complex128')
    d = yastn.rand(s=(1, 1, 1, -1), D=(30, 10, 20, 10), config=config_dense, dtype='complex128')
    e = yastn.rand(s=(1, 1), D=(3, 1), config=config_dense, dtype='complex128')
    f = yastn.rand(s=(-1, -1), D=(2, 4), config=config_dense, dtype='complex128')
    g = yastn.rand(s=(1,), D=(5,), config=config_dense, dtype='complex128')
    h = yastn.rand(s=(-1,), D=(6,), config=config_dense, dtype='complex128')

    x = yastn.ncon([a, b], ((1, -1, -3), (-0, -2, 1)))
    assert x.get_shape() == (4, 3, 2, 1)
    x1 = yastn.einsum('abc,dea->dbec', a, b)
    assert yastn.norm(x - x1) < tol

    y = yastn.ncon([a, b, c, d], ((4, -2, -0), (-3, -1, 5), (4, 3, 1, 1), (3, 2, 5, 2)), (0, 1, 0, 1))
    assert y.get_shape() == (1, 2, 3, 4)
    y1 = yastn.einsum('abc, def*, aghh, gifi* -> cebd', a, b, c, d, order='higaf')
    assert yastn.norm(y - y1) < tol

    z1 = yastn.ncon([e, f], ((-2, -0), (-1, -3)), conjs=[0, 1])
    z2 = yastn.ncon([f, e], ((-1, -3), (-2, -0)), conjs=[1, 0])
    z3 = yastn.ncon([e, f], ((-2, -0), (-1, -3)), conjs=[0, 0])
    z4 = yastn.ncon([f, e], ((-1, -3), (-2, -0)), conjs=[1, 1])
    assert all(x.get_shape() == (1, 2, 3, 4) for x in (z1, z2, z3, z4))
    assert yastn.norm(z1 - z2) < tol  # == 0.0
    assert yastn.norm(z3 - z4.conj()) < tol  # == 0.0

    y1 = yastn.ncon([a, b, c, d, g, h], ((4, -2, -0), (-3, -1, 5), (4, 3, 1, 1), (3, 2, 5, 2), (-4,), (-5,)),
                   conjs=(1, 0, 1, 0, 1, 0))
    assert y1.get_shape() == (1, 2, 3, 4, 5, 6)

    y2 = yastn.ncon([a, g, a], ((1, 2, 3), (-0,), (1, 2, 3)), conjs=(0, 0, 1))
    assert y2.get_shape() == (5,)

    y3 = yastn.ncon([a, g, a, b, a], [[1, 2, 3], [-4], [1, 2, 3], [-3, -1, 4], [4, -2, -0]],
                   conjs=(0, 0, 1, 0, 0))
    assert y3.get_shape() == (1, 2, 3, 4, 5)

    y4 = yastn.ncon([a, a, b, b], [[1, 2, 3], [1, 2, 3], [6, 5, 4], [6, 5, 4]],
                   conjs=(0, 1, 0, 1))
    y5 = yastn.ncon([a, a, b, b], [[6, 5, 4], [6, 5, 4], [1, 3, 2], [1, 3, 2]],
                   conjs=(0, 1, 0, 1))
    assert isinstance(y4.item(), complex)
    assert isinstance(y5.item(), complex)
    assert yastn.norm(y4 - y5) / yastn.norm(y4) < tol
    assert pytest.approx((a.norm().item() ** 2) * (b.norm().item() ** 2), rel=tol) == y4.item()

    a = yastn.rand(config=config_U1, s=[-1, 1, -1], n=0,
                  D=((20, 10), (3, 3), (1, 1)), t=((1, 0), (1, 0), (1, 0)))
    b = yastn.rand(config=config_U1, s=[1, 1, 1], n=1,
                  D=((4, 4), (2, 2), (20, 10)), t=((1, 0), (1, 0), (1, 0)))
    c = yastn.rand(config=config_U1, s=[1, 1, 1, -1], n=1,
                  D=((20, 10), (30, 20), (10, 5), (10, 5)), t=((1, 0), (1, 0), (1, 0), (1, 0)))
    d = yastn.rand(config=config_U1, s=[1, 1, -1, -1], n=0,
                  D=((30, 20), (10, 5), (20, 10), (10, 5)), t=((1, 0), (1, 0), (1, 0), (1, 0)))

    e = yastn.ncon([a, b], [[1, -1, -3], [-0, -2, 1]])
    assert e.get_shape() == (8, 6, 4, 2)
    e = yastn.ncon([b, a], [[-0, -1, 1], [1, -2, -3]])
    e1 = yastn.einsum('abc,cde', b, a)
    assert yastn.norm(e - e1) < tol
    assert e.get_shape() == (8, 4, 6, 2)

    f = yastn.ncon([a, b, c, d], [[4, -2, -0], [-3, -1, 5], [4, 3, 1, 1], [3, 2, 5, 2]],
                  conjs=(0, 1, 0, 1))
    assert f.get_shape() == (2, 4, 6, 8)
    g = yastn.ncon([a, a, a, b], [[1, 2, 3], [1, 2, 3], [4, -2, -0], [-3, -1, 4]],
                  conjs=(0, 1, 1, 1))
    assert g.get_shape() == (2, 4, 6, 8)

    a = 1j * yastn.ones(config=config_dense, s=(1, 1, -1, -1), D=(1, 1, 1, 1))
    assert abs(yastn.ncon([a], [(1, 2, 2, 1)], conjs=[1]).item() + 1j) < tol
    assert abs(yastn.ncon([a], [(1, -1, -0, 1)], conjs=[1]).item() + 1j) < tol
    assert abs(yastn.ncon([a], [(-0, -1, -2, -3)], conjs=[1]).item() + 1j) < tol
    assert abs(yastn.ncon([a], [(1, 2, 2, 1)], conjs=[0]).item() - 1j) < tol


def test_ncon_einsum_exceptions():
    """ capturing some exception by ncon. """
    t, D = (0, 1), (2, 3)
    a = yastn.rand(config=config_U1, s=[-1, 1, -1], n=0,
                  t=(t, t, t), D=(D, D, D))
    with pytest.raises(yastn.YastnError):
        _ = yastn.ncon([a, a], [(1, 2, 3)])
        # Number of tensors and groups of indices do not match.
    with pytest.raises(yastn.YastnError):
        _ = yastn.ncon([a, a], [(1, 2, -1), (1, 2)])
        # Number of legs of one of the tensors do not match provided indices.
    with pytest.raises(yastn.YastnError):
        _ = yastn.ncon([a, a], [(1, 2, -1), (1, 3, -2)])
        # Indices of legs to contract do not match
    with pytest.raises(yastn.YastnError):
        _ = yastn.ncon([a, a], [(1, 2, -1), (1, 1, -2)])
        # Indices of legs to contract do not match
    with pytest.raises(yastn.YastnError):
        _ = yastn.ncon([a, a], [(3, 3, 2), (1, 1, 2)], conjs=[0, 1])
        # Likely inefficient order of contractions. Do all traces before tensordot.
        # Call all axes connecting two tensors one after another.
    with pytest.raises(yastn.YastnError):
        _ = yastn.ncon([a, a, a], [(1, 2, 3), (1, 3, 4), (2, 4, -0)], conjs=[0, 1, 0])
        # Likely inefficient order of contractions. Do all traces before tensordot.
        # Call all axes connecting two tensors one after another.
    with pytest.raises(yastn.YastnError):
        yastn.ncon([a], [(-1, -1, -0)])
        # Repeated non-positive (outgoing) index is ambiguous.
    with pytest.raises(yastn.YastnError):
        yastn.ncon([a], [(-555, -542, -0)])
        # ncon requires indices to be between -256 and 256.
    with pytest.raises(yastn.YastnError):
        yastn.einsum(a, a, order='alphabetic')
        # The first argument should be a string
    with pytest.raises(yastn.YastnError):
        yastn.einsum('xkl, xmn-> klmm -> kl', a, a)
        # Subscript should have at most one separator ->
    with pytest.raises(yastn.YastnError):
        yastn.einsum('xy;, xyn -> ;n', a, a)
        # Only alphabetic characters can be used to index legs.
    with pytest.raises(yastn.YastnError):
        yastn.einsum('-;k, -;l -> kl', a, a)
        # Only alphabetic characters can be used to index legs.
    with pytest.raises(yastn.YastnError):
        yastn.einsum('klm, klm-> mm', a, a)
        # Repeated index after ->
    with pytest.raises(yastn.YastnError):
        yastn.einsum('klm, *klm->', a, a, order='kl')
        # order does not cover all contracted indices


if __name__ == '__main__':
    test_ncon_einsum_syntax()
    test_ncon_einsum_basic()
    test_ncon_einsum_exceptions()
