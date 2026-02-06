# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Test yastn.ncon() """
import pytest
import yastn

tol = 1e-12  # pylint: disable=invalid-name


def test_ncon_einsum_syntax(config_kwargs):
    # Create a set of U1-symmetric tensors
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg1 = yastn.Leg(config_U1, s=1, t=(1, 0), D=(1, 1))
    leg2 = yastn.Leg(config_U1, s=1, t=(1, 0), D=(2, 2))
    leg3 = yastn.Leg(config_U1, s=1, t=(1, 0), D=(3, 3))
    leg4 = yastn.Leg(config_U1, s=1, t=(1, 0), D=(4, 4))
    leg5 = yastn.Leg(config_U1, s=1, t=(1, 0), D=(20, 10))
    leg6 = yastn.Leg(config_U1, s=1, t=(1, 0), D=(30, 20))
    leg7 = yastn.Leg(config_U1, s=1, t=(1, 0), D=(10, 5))
    a = yastn.rand(config=config_U1, legs=[leg5.conj(), leg3, leg1.conj()], n=0)
    b = yastn.rand(config=config_U1, legs=[leg4, leg2, leg5], n=1)
    c = yastn.rand(config=config_U1, legs=[leg5, leg6, leg7, leg7.conj()], n=1)
    d = yastn.rand(config=config_U1, legs=[leg6, leg7, leg5.conj(), leg7.conj()], n=1)

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
    assert yastn.norm(e1 - e) < 1e-12

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
    f = yastn.ncon([a, b, c, d],
                   [[4, -2, -0], [-3, -1, 5], [4, 3, 1, 1], [3, 2, 5, 2]],
                  conjs=(0, 1, 0, 1))
    assert f.get_shape() == (2, 4, 6, 8)
    #
    # with two equivalent syntaxes to specify tensor conjugation
    f1 = yastn.ncon([a, b.conj(), c, d.conj()],
                    [[4, -2, -0], [-3, -1, 5], [4, 3, 1, 1], [3, 2, 5, 2]])
    assert yastn.norm(f1 - f) < 1e-12
    #
    # In yastn.einsum(subscripts, *operands, order=None)
    #
    # Order specifies the order of contraction.
    # Otherwise, the alphabetic order of contracted indices is used.
    # Character '*' can be used in einsum subscripts to conjugate respective tensor.
    # Spaces in subscripts are ignored
    f2 = yastn.einsum('nCA, *DBo, nmkk, *mlol -> ABCD', a, b, c, d, order='klmno')
    assert yastn.norm(f2 - f) < 1e-12
    #
    # yastn.ncon() also takes order argument to specify contraction order.
    # Otherwise, an ascending order of positive contracted indices is used.
    f3 = yastn.ncon([a, b.conj(), c, d.conj()],
                    [[1, -2, -0], [-3, -1, 2], [1, 3, 4, 4], [3, 5, 2, 5]],
                    order = [4, 5, 3, 1, 2])
    assert yastn.norm(f3 - f) < 1e-12


def test_ncon_einsum_basic(config_kwargs):
    """ tests of ncon executing a series of tensor contractions. """
    config_dense = yastn.make_config(sym='none', **config_kwargs)
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
    #
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
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

    # test order
    inds1 = ((4, -2, -0), (-3, -1, 5), (4, 3, 1, 1), (3, 2, 5, 2))
    inds2 = ((11, -4, -1), (-6, -3, 2), (11, 3, 8, 8), (3, 1, 2, 1))
    order2 = (8, 1, 3, 11, 2)
    ins1 = yastn.tensor._einsum._meta_ncon(inds1, None, ())
    ins2 = yastn.tensor._einsum._meta_ncon(inds2, order2, ())
    assert ins1 == ins2


def test_ncon_einsum_exceptions(config_kwargs):
    """ capturing some exception by ncon. """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    t, D = (0, 1), (2, 3)
    a = yastn.rand(config=config_U1, s=[-1, 1, -1], n=0,
                  t=(t, t, t), D=(D, D, D))
    with pytest.raises(yastn.YastnError,
                      match="Number of tensors and indices do not match."):
        _ = yastn.ncon([a, a], [(1, 2, 3)])
    with pytest.raises(yastn.YastnError,
                       match="Number of legs of one of the tensors do not match provided indices."):
        _ = yastn.ncon([a, a], [(1, 2, -1), (1, 2)])
    with pytest.raises(yastn.YastnError,
                       match="Indices of legs to contract do not match."):
        _ = yastn.ncon([a, a], [(1, 2, -1), (1, 3, -2)])
    with pytest.raises(yastn.YastnError,
                       match="Indices of legs to contract do not match."):
        _ = yastn.ncon([a, a], [(1, 2, -1), (1, 1, -2)])
    with pytest.raises(yastn.YastnError,
                       match="Likely inefficient order of contractions. Do all traces before tensordot."):
        _ = yastn.ncon([a, a], [(3, 3, 2), (1, 1, 2)], conjs=[0, 1])
    with pytest.raises(yastn.YastnError,
                       match="Likely inefficient order of contractions. Do all traces before tensordot."):
        _ = yastn.ncon([a, a, a], [(1, 2, 3), (1, 3, 4), (2, 4, -0)], conjs=[0, 1, 0])
    with pytest.raises(yastn.YastnError,
                       match="Repeated non-positive"): # (outgoing) index is ambiguous.
        yastn.ncon([a], [(-1, -1, -0)])
    with pytest.raises(yastn.YastnError,
                      match="Ncon requires indices to be between -256 and 256."):
        yastn.ncon([a], [(-555, -542, -0)])
    with pytest.raises(yastn.YastnError,
                       match="Order should be a list of positive ints with no repetitions."):
        yastn.ncon([a, a, a], [(1, 2, -1), (1, 3, -2), (2, 3, -3)], order=[1, 2, 2])
        # order should be a list of positive ints with no repetitions.
    with pytest.raises(yastn.YastnError,
                       match="Order should be a list of positive ints with no repetitions."):
        yastn.ncon([a, a, a], [(1, 2, -1), (1, 3, -2), (2, 3, -3)], order=[1, 2, 3, -2])
    with pytest.raises(yastn.YastnError,
                       match="Positive ints in ins and order should match."):
        yastn.ncon([a, a, a], [(1, 2, -1), (1, 3, -2), (2, 3, -3)], order=[1, 2, 4])
    with pytest.raises(yastn.YastnError,
                       match="The first argument should be a string."):
        yastn.einsum(a, a, order='alphabetic')
    with pytest.raises(yastn.YastnError,
                       match="Subscript should have at most one separator ->"):
        yastn.einsum('xkl, xmn-> klmm -> kl', a, a)
    with pytest.raises(yastn.YastnError,
                       match="Only alphabetic characters can be used to index legs."):
        yastn.einsum('xy;, xyn -> ;n', a, a)
    with pytest.raises(yastn.YastnError,
                       match="Only alphabetic characters can be used to index legs."):
        yastn.einsum('-;k, -;l -> kl', a, a)
    with pytest.raises(yastn.YastnError,
                       match="Repeated index after ->"):
        yastn.einsum('klm, klm-> mm', a, a)
    with pytest.raises(yastn.YastnError,
                       match="Order does not cover all contracted indices."):
        yastn.einsum('klm, *klm->', a, a, order='kl')


def test_ncon_einsum_swaps(config_kwargs):
    """ tests of ncon executing a series of tensor contractions. """
    config_Z2 = yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
    l = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(1, 1))
    lc = l.conj()
    #
    # first diagram
    a = yastn.rand(config=config_Z2, legs=[l, lc])
    b = yastn.rand(config=config_Z2, legs=[l, lc])
    #
    x = yastn.ncon([a, b], ((1, 2), (2, 1)), swap=[(1, 2)])
    y = yastn.einsum('ab,ba', a, b, swap='ab')
    #
    r = a.swap_gate(axes=(0, 1))
    r = yastn.tensordot(r, b, axes=((0, 1), (1, 0)))
    #
    assert (x - r).norm() < tol * r.norm()
    assert (y - r).norm() < tol * r.norm()
    #
    # second diagram
    a = yastn.rand(config=config_Z2, legs=[l, l, lc, l, lc])
    b = yastn.rand(config=config_Z2, legs=[l, lc, l])
    c = yastn.rand(config=config_Z2, legs=[l, lc, l])
    #
    x = yastn.ncon([a, b, c, c], ((1, 4, 2, -0, 1), (2, 3, -1), (3, 4, -2), (-3, -4, -5)), swap=((-0, 3), (-0, 1), (-1, -2), (-3, -5), (-4, -2)))
    y = yastn.einsum('adbAa,bcB,cdC,DEF->ABCDEF', a, b, c, c, swap='Ac,Aa,BC,CE,DF')
    #
    # reference
    d = a.swap_gate(axes=(3, 4))
    r = yastn.trace(d, axes=(0, 4))
    r = yastn.tensordot(r, b, axes=(1, 0))
    r = r.swap_gate(axes=(1, 2))
    r = yastn.tensordot(r, c, axes=((2, 0), (0, 1)))
    r = r.swap_gate(axes=(1, 2))
    e = c.swap_gate(axes=(0, 2))
    r = yastn.tensordot(r, e, axes=((), ()))
    r = r.swap_gate(axes=(2, 4))
    #
    assert (x - r).norm() < tol * r.norm()
    assert (y - r).norm() < tol * r.norm()
    #
    # third diagram to test different contraction orders
    a = yastn.rand(config=config_Z2, n=1, legs=[l, l, l, l])
    b = yastn.rand(config=config_Z2, n=1, legs=[l, l, l, l, lc])
    c = yastn.rand(config=config_Z2, n=1, legs=[l, l, lc, lc])
    d = yastn.rand(config=config_Z2, n=1, legs=[l, lc, lc, lc, lc])
    e = yastn.rand(config=config_Z2, n=1, legs=[l, lc, lc, lc])
    f = yastn.rand(config=config_Z2, n=1, legs=[lc, lc])
    #
    # reference
    r = yastn.tensordot(a, b, axes=(0, 4))
    r = r.swap_gate(axes=(0, (4, 5, 6), 1, 6))
    r = yastn.tensordot(r, c, axes=((0, 3), (2, 3)))
    r = r.swap_gate(axes=((0, 2, 3), 6))
    r = yastn.tensordot(r, d, axes=((0, 2, 3, 5), (1, 3, 2, 4)))
    r = yastn.tensordot(r, e, axes=((1, 2, 3), (1, 2, 3)))
    r = yastn.tensordot(r, f, axes=((0, 1), (0, 1)))
    #
    for order in [None]:  # (2, 4, 7, 1, 3, 5, 6, 11, 8, 9, 10, 12)]:
        x = yastn.ncon([a, b, c, d, e, f], ((1, 2, 4, 11), (3, 5, 6, 8, 1), (7, 9, 2, 3), (10, 4, 6, 5, 7), (12, 8, 9, 10), (11, 12)),
                       swap=((2, 8), (2, 5), (2, 6), (4, 8), (9, 6), (9, 5), (4, 9)), order=order)
        assert (x - r).norm() < tol * r.norm()

    for order in [None]: #, 'bdgacefkhijl']:
        y = yastn.einsum('abdk,cefha,gibc,jdfeg,lhij,kl', a, b, c, d, e, f,
                         swap='bh,be,bf,dh,ei,fi,di', order=order)  #
        assert (y - r).norm() < tol * r.norm()



if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
