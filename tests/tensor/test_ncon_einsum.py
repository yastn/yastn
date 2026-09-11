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
import random
import signal
import sys
import pytest
import yastn
from yastn.tensor._einsum import _meta_ncon

tol = 1e-12  # pylint: disable=invalid-name


def _plan(inds, swap, order):
    """Plan the network; order None means the default (ascending) order."""
    return _meta_ncon(tuple(tuple(i) for i in inds),
                      tuple(order) if order is not None else None,
                      tuple(tuple(s) for s in swap))


def _mini_tensors(config, mk, inds, charges):
    """Random tensors with legs matched per edge label; first occurrence gets a fresh Leg."""
    legobj = {}
    ts = []
    for ind in inds:
        ll = []
        for e in ind:
            if e not in legobj:
                legobj[e] = mk()
                ll.append(legobj[e])
            else:
                ll.append(legobj[e].conj())
        ts.append(yastn.rand(config=config, n=charges[len(ts)], legs=ll))
    return ts


def _rand_net(rng, cfg, sym):
    """Random fermionic network: 3-6 tensors, self-loops, open legs, 1-5 random swaps."""
    nt = rng.randint(3, 6)
    # random multigraph with self loops; each tensor 2-5 legs
    legs = [[] for _ in range(nt)]
    edges = []
    eid = 1
    for t in range(nt):
        while len(legs[t]) < rng.randint(2, 4):
            u = rng.randrange(nt)
            if u == t and rng.random() < 0.7:
                continue
            legs[t].append(eid); legs[u].append(eid); edges.append((t, u)); eid += 1
    # a few open legs
    nopen = rng.randint(0, 2)
    for k in range(nopen):
        t = rng.randrange(nt); legs[t].append(-k)
    for t in range(nt):
        rng.shuffle(legs[t])
    # leg objects: each contracted edge shares a Leg (conj on the other side)
    if sym == 'Z2':
        mk = lambda: yastn.Leg(cfg, s=1, t=(0, 1), D=(rng.randint(1, 2), rng.randint(1, 2)))
    else:
        mk = lambda: yastn.Leg(cfg, s=1, t=(-1, 0, 1), D=(1, rng.randint(1, 2), 1))
    legobj = {}
    ts = []
    for t in range(nt):
        ll = []
        for e in legs[t]:
            if e not in legobj:
                legobj[e] = mk(); ll.append(legobj[e])
            else:
                ll.append(legobj[e].conj())
        n = rng.choice([0, 1]) if sym == 'Z2' else rng.choice([-1, 0, 1])
        ts.append(yastn.rand(config=cfg, n=n, legs=ll))
    # swaps: random pairs of distinct edges (contracted or open)
    all_e = list(range(1, eid)) + [-k for k in range(nopen)]
    swaps = []
    for _ in range(rng.randint(1, 5)):
        a, b = rng.sample(all_e, 2)
        swaps.append((a, b))
    order = list(range(1, eid)); rng.shuffle(order)
    return ts, [tuple(l) for l in legs], swaps, order


def _max_legs(cmds, inds):
    nl = {j: len(x) for j, x in enumerate(inds)}
    mx = max(nl.values())
    for c in cmds:
        if c[0].startswith('tensordot'):
            nl[c[1]] = nl.pop(c[2][0]) + nl.pop(c[2][1]) - 2 * len(c[3][0]) + \
                (2 * len(c[4]) if c[0] == 'tensordot_psplit' else 0)
        elif c[0].startswith('trace'):
            nl[c[1]] = nl.pop(c[2]) - 2 * len(c[3][0]) + \
                (2 * len(c[4]) if c[0] == 'trace_psplit' else 0)
        mx = max(mx, max(nl.values()))
    return mx


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


def test_ncon_trace_swap_value(config_kwargs):
    r"""Swapped trace should give the same value as an explicit swapped trace, with no gadget."""
    config_Z2 = yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
    l = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(1, 1))
    lc = l.conj()

    A = yastn.rand(config=config_Z2, n=0, legs=[l, lc])
    B = yastn.rand(config=config_Z2, n=1, legs=[l])
    C = yastn.rand(config=config_Z2, n=1, legs=[lc])

    inds = ((1, 1), (2,), (2,))
    swap = ((1, 2),)
    plan = _plan(inds, swap, None)
    assert not any(c[0].endswith('_psplit') for c in plan)  # trace row is the coboundary of {B}

    x = yastn.ncon([A, B, C], inds, swap=swap)

    ref = yastn.trace(A.swap_gate(axes=(0, 1)), axes=(0, 1)).item()
    ref *= yastn.tensordot(B, C, axes=(0, 0)).item()
    assert abs(x.item() - ref) < tol


def test_ncon_trace_swap_multiple_pairs_value(config_kwargs):
    r"""Multiple swapped trace pairs should give the right value with no gadget."""
    config_Z2 = yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
    l = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(1, 1))
    lc = l.conj()

    A = yastn.rand(config=config_Z2, n=0, legs=[l, lc, l, lc])
    B = yastn.rand(config=config_Z2, n=1, legs=[l])
    C = yastn.rand(config=config_Z2, n=1, legs=[lc])
    D = yastn.rand(config=config_Z2, n=1, legs=[l])
    E = yastn.rand(config=config_Z2, n=1, legs=[lc])

    inds = ((1, 1, 2, 2), (3,), (3,), (4,), (4,))
    swap = ((1, 3), (2, 4))
    plan = _plan(inds, swap, None)
    assert not any(c[0].endswith('_psplit') for c in plan)  # both trace rows are coboundaries

    x = yastn.ncon([A, B, C, D, E], inds, swap=swap)

    ref = A.swap_gate(axes=(0, 1)).swap_gate(axes=(2, 3))
    ref = yastn.trace(ref, axes=(0, 1))
    ref = yastn.trace(ref, axes=(0, 1)).item()
    ref *= yastn.tensordot(B, C, axes=(0, 0)).item()
    ref *= yastn.tensordot(D, E, axes=(0, 0)).item()
    assert abs(x.item() - ref) < tol


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
    for order in [None,
                  (2, 4, 7, 1, 3, 5, 6, 11, 8, 9, 10, 12),
                  (6, 5, 8, 10, 9, 7, 3, 2, 1, 4, 11, 12),
                  (12, 10, 1, 7, 9, 3, 6, 8, 11, 2, 5, 4),
                  (12, 11, 6, 5, 1, 4, 10, 8, 3, 7, 9, 2)]:
        x = yastn.ncon([a, b, c, d, e, f], ((1, 2, 4, 11), (3, 5, 6, 8, 1), (7, 9, 2, 3), (10, 4, 6, 5, 7), (12, 8, 9, 10), (11, 12)),
                       swap=((2, 8), (2, 5), (2, 6), (4, 8), (9, 6), (9, 5), (4, 9)), order=order)
        assert (x - r).norm() < tol * r.norm()

    for order in [None, 'bdgacefkhijl']:
        y = yastn.einsum('abdk,cefha,gibc,jdfeg,lhij,kl', a, b, c, d, e, f,
                         swap='bh,be,bf,dh,ei,fi,di', order=order)
        assert (y - r).norm() < tol * r.norm()


def test_einsum_scalar_swap_order(config_kwargs):
    r"""Scalar fermionic einsum should be invariant to contraction order."""
    config_Z2 = yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
    l = yastn.Leg(config_Z2, s=1, t=(0, 1), D=(2, 2))
    lc = l.conj()

    A = yastn.rand(config=config_Z2, n=0, legs=[l, l, l, l])
    B = yastn.rand(config=config_Z2, n=0, legs=[lc, lc, lc])
    C = yastn.rand(config=config_Z2, n=0, legs=[lc, l, l, l])
    D = yastn.rand(config=config_Z2, n=0, legs=[l, lc])
    E = yastn.rand(config=config_Z2, n=0, legs=[lc, lc])
    F = yastn.rand(config=config_Z2, n=0, legs=[lc, l, lc])

    inds = ((9,1,2,3), (9, 2,3), (1,4,5,8), (7,8), (4,6), (5,6,7))
    orders = [[9,2,3,1,4,5,6,7,8], [4,5,6,7,8,9,2,3,1], [8,1,6,4,5,7,9,2,3]]
    swap = [(9,4), (9,5), (2,8), (3,8)]

    def _assert_all_orders():
        vals = [yastn.ncon((A, B, C, D, E, F), inds, swap=swap, order=order) for order in orders]
        for v in vals:
            assert v.ndim == 0
        ref = vals[0]
        den = ref.norm()
        den = den.item() if hasattr(den, "item") else den
        den = max(float(den), 1.0)
        for v in vals[1:]:
            assert (v - ref).norm() < tol * den

    _assert_all_orders()


# entry-23 pattern in miniature: two tensors share two contracted edges while the bad swap's
# third-party edge lies on a cycle of the remaining network (which the A,B,C,D tensors of the
# original note cannot close); adding a shared edge 6 between C and D makes edge 5 lie on the
# cycle C-5-6-D that avoids A and B.
_IND_CYCLE = ((1, 2, 3), (1, 2, 4), (3, 5, 6), (4, 5, 6))
_SWAP_CYCLE = ((1, 5),)
_FORCED_CYCLE = (1, 2, 3, 4, 5, 6)  # contracts A,B over edges 1 and 2 first -> one gadget
_CLEAN_CYCLE = (3, 4, 5, 6, 1, 2)   # swap (1, 5) only becomes same-tensor at the end -> no gadget


def test_ncon_gadget_two_edge_step(config_kwargs):
    """A step contracting two edges with an obstinate bad swap must use exactly one gadget."""
    config_Z2 = yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
    mk = lambda: yastn.Leg(config_Z2, s=1, t=(0, 1), D=(1, 1))
    p_forced = _plan(_IND_CYCLE, _SWAP_CYCLE, _FORCED_CYCLE)
    gadgets = [c for c in p_forced if c[0].endswith('_psplit')]
    assert len(gadgets) == 1
    assert gadgets[0][0] == 'tensordot_psplit'  # one step needs a gadget
    assert len(gadgets[0][4]) == 1              # exactly one recorded axis
    p_clean = _plan(_IND_CYCLE, _SWAP_CYCLE, _CLEAN_CYCLE)
    assert not any(c[0].endswith('_psplit') for c in p_clean)
    patterns = ((1, 1, 1, 1), (1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 1, 0), (0, 1, 0, 1))
    for seed, charges in enumerate(patterns):
        ts = _mini_tensors(config_Z2, mk, _IND_CYCLE, charges)
        x = yastn.ncon(ts, _IND_CYCLE, swap=_SWAP_CYCLE, order=_FORCED_CYCLE)
        ref = yastn.ncon(ts, _IND_CYCLE, swap=_SWAP_CYCLE, order=_CLEAN_CYCLE)
        # agreement with the gadget-free order is the reference
        assert abs(x.item() - ref.item()) < tol * max(abs(ref.item()), 1e-30)


@pytest.mark.parametrize('fermionic', [(False, True), True])
def test_ncon_gadget_product_symmetry(config_kwargs, fermionic):
    """The same gadget step with a product symmetry; one or two fermionic charge components."""
    config = yastn.make_config(sym='U1xU1', fermionic=fermionic, **config_kwargs)
    tt = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1))
    mk = lambda: yastn.Leg(config, s=1, t=tt, D=(1,) * len(tt))
    for seed in range(3):  # same plans as in test_ncon_gadget_two_edge_step
        config.backend.random_seed(seed)
        ts = _mini_tensors(config, mk, _IND_CYCLE, [(0, 0)] * 4)
        x = yastn.ncon(ts, _IND_CYCLE, swap=_SWAP_CYCLE, order=_FORCED_CYCLE)
        ref = yastn.ncon(ts, _IND_CYCLE, swap=_SWAP_CYCLE, order=_CLEAN_CYCLE)
        no_swap = yastn.ncon(ts, _IND_CYCLE, order=_CLEAN_CYCLE)
        assert abs(ref.item() - no_swap.item()) > 1e-3 * abs(ref.item())  # the swap is not trivial
        assert abs(x.item() - ref.item()) < tol * max(abs(ref.item()), 1e-30)


def test_ncon_order_independence_random(config_kwargs):
    """Seeded mini-fuzz: random fermionic networks must give order-independent values."""
    executed = 0
    gadget_cases = 0
    for i in range(40):
        rng = random.Random(i)
        sym = 'Z2' if i % 2 == 0 else 'U1'
        cfg = yastn.make_config(sym=sym, fermionic=True, **config_kwargs)
        ts, inds, swaps, order = _rand_net(rng, cfg, sym)
        small = []
        attempts = 0
        cand = order
        while len(small) < 3 and attempts < 60:
            if _max_legs(_plan(inds, swaps, cand), inds) <= 9:
                small.append(cand)
            cand = order[:]
            rng.shuffle(cand)
            attempts += 1
        if len(small) < 3:  # no 3 orders with intermediates of <= 9 legs
            continue
        plans = [(o, _plan(inds, swaps, o)) for o in small]
        vals = [yastn.ncon(ts, inds, swap=swaps, order=o) for o, _ in plans]
        ref = vals[0]
        den = max(float(ref.norm().item()), 1.0)
        for v in vals[1:]:
            assert (v - ref).norm().item() < 1e-10 * den
        if any(any(c[0].endswith('_psplit') for c in p) for _, p in plans):
            gadget_cases += 1
        executed += 1
    assert executed > 0
    assert gadget_cases >= 5  # otherwise the test is not exercising the gadget path


def test_ncon_no_hang_regression(config_kwargs):
    """A trace pair with a bad swap against an open leg used to hang the old planner forever."""
    if sys.platform == 'win32':
        pytest.skip('signal.alarm is not supported on Windows')
    config_Z2 = yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
    mk = lambda: yastn.Leg(config_Z2, s=1, t=(0, 1), D=(1, 1))
    inds = [(-1, 2, 1), (1, 3), (4, 4), (5, 2, 0), (3, 6, 6, 5)]
    swap = [(5, 1), (6, 0), (3, 1), (6, 1), (0, 4)]
    ts = _mini_tensors(config_Z2, mk, inds, (0, 1, 0, 1, 0))
    inds = [tuple(x) for x in inds]
    order_hang = (4, 3, 1, 2, 5, 6)
    order_plain = (1, 2, 3, 4, 5, 6)

    class _Timeout(Exception):
        pass

    def _alarm(*a):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _alarm)
    try:
        signal.alarm(30)
        try:
            _plan(inds, swap, order_hang)
        finally:
            signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, old)

    va = yastn.ncon(ts, inds, swap=swap, order=order_hang)
    vb = yastn.ncon(ts, inds, swap=swap, order=order_plain)
    assert (va - vb).norm().item() < 1e-10 * max(float(va.norm().item()), 1e-30)


def test_ncon_gadget_autograd(config_kwargs):
    """Gradients through the gadget path must equal gradients through a gadget-free order."""
    if config_kwargs['backend'] != 'torch':
        pytest.skip('requires torch backend')
    import torch  # pylint: disable=import-outside-toplevel
    config_Z2 = yastn.make_config(sym='Z2', fermionic=True, **config_kwargs)
    mk = lambda: yastn.Leg(config_Z2, s=1, t=(0, 1), D=(1, 1))

    def grads(order, ts):
        cts = [t.clone() for t in ts]
        cts[0].requires_grad_()
        x = yastn.ncon(cts, _IND_CYCLE, swap=_SWAP_CYCLE, order=order)
        loss = abs(x) ** 2
        loss.to_number().backward()
        return cts[0].grad()

    for _ in range(2):
        ts = _mini_tensors(config_Z2, mk, _IND_CYCLE, (1, 1, 1, 1))
        g_forced = grads(_FORCED_CYCLE, ts)
        g_clean = grads(_CLEAN_CYCLE, ts)
        assert torch.isfinite(g_forced.data).all().item()
        assert torch.isfinite(g_clean.data).all().item()
        assert (g_forced - g_clean).norm().item() <= 1e-8 * max(g_clean.norm().item(), 1e-30)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--tensordot_policy", "fuse_to_matrix"])
    pytest.main([__file__, "-vs", "--durations=0", "--tensordot_policy", "fuse_contracted"])
    pytest.main([__file__, "-vs", "--durations=0", "--tensordot_policy", "no_fusion"])
    #pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--tensordot_policy", "fuse_to_matrix"])
    #pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--tensordot_policy", "fuse_contracted"])
    #pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--tensordot_policy", "no_fusion"])
