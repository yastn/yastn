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
""" test algebraic expressions using magic methods, e.g.: a + 2. * b,  a / 2. - c * 3.,  a + b ** 2"""
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def algebra_vs_numpy(f, a, b):
    """
    f is lambda expression using magic methods on a and b tensors
    e.g. f = lambda x, y: x + y
    """
    legs_for_a = dict(enumerate(b.get_legs()))
    legs_for_b = dict(enumerate(a.get_legs()))
    na = a.to_numpy(legs=legs_for_a) # makes sure nparrays have consistent shapes
    nb = b.to_numpy(legs=legs_for_b) # makes sure nparrays have consistent shapes
    nc = f(na, nb)

    c = f(a, b)
    assert c.is_consistent()
    assert all(c.are_independent(x) for x in (a, b))

    cn = c.to_numpy()
    assert np.linalg.norm(cn - nc) < tol
    return c


def combine_tests(a, b):
    """ some standard set of tests """
    r1 = algebra_vs_numpy(lambda x, y: x + 2 * y, a, b)
    r2 = yastn.add(a, b, amplitudes=(None, 2))
    r3 = yastn.add(a, b, b)
    r4 = algebra_vs_numpy(lambda x, y: 2 * x + y, b, a)
    assert all(yastn.norm(r1 - x, p=p) < tol for x in (r2, r3, r4) for p in ('fro', 'inf'))  # == 0.0
    # additionally tests norm


def test_algebra_basic(config_kwargs):
    """ test basic algebra for various symmetries """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    a = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='float64')
    b = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype='float64')
    combine_tests(a, b)

    c = yastn.rand(config=config_dense, isdiag=True, D=5, dtype='float64')
    d, e = c.copy(), c.clone()
    r4 = algebra_vs_numpy(lambda x, y: 2. * x - (y + y), c, d)
    r5 = algebra_vs_numpy(lambda x, y: -y + 2. * x - y, c, e)
    assert r4.norm() < tol  # == 0.0
    assert r5.norm() < tol  # == 0.0
    assert all(yastn.are_independent(c, x) for x in (d, e))

    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg0a = yastn.Leg(config_U1, s=-1, t=(-1, 1, 0), D=(1, 2, 3))
    leg0b = yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(1, 2, 3))
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6))
    leg2 = yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9))
    leg3 = yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))

    a = yastn.rand(config=config_U1, legs=[leg0a, leg1, leg2, leg3], dtype='float64')
    b = yastn.rand(config=config_U1, legs=[leg0b, leg1, leg2, leg3], dtype='float64')
    combine_tests(a, b)

    c = yastn.eye(config=config_U1, t=1, D=5)
    d = yastn.eye(config=config_U1, t=2, D=5)
    r4 = algebra_vs_numpy(lambda x, y: 2. * x + y, c, d)
    r5 = algebra_vs_numpy(lambda x, y: x - (2 * y) ** 2 + 2 * y, c, d)
    r6 = algebra_vs_numpy(lambda x, y: x - y / 0.5, d, c)
    assert all(pytest.approx(x.norm().item(), rel=tol) == 5 for x in (r4, r5, r6))
    assert all(pytest.approx(x.norm(p='inf').item(), rel=tol) == 2 for x in (r4, r5, r6))

    e = yastn.ones(config=config_U1, s=(1, -1), n=1, t=(1, 0), D=(5, 5))
    f = yastn.ones(config=config_U1, s=(1, -1), n=1, t=(2, 1), D=(5, 5))
    r7 = algebra_vs_numpy(lambda x, y: x - y / 0.5, e, f)
    assert pytest.approx(r7.norm().item(), rel=tol) == 5 * np.sqrt(5)
    assert pytest.approx(r7.norm(p='inf').item(), rel=tol) == 2

    # Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    leg0a = yastn.Leg(config_Z2xU1, s=-1, t=[(0, 0), (0, 2), (1, 2)], D=[1, 2, 4])
    leg0b = yastn.Leg(config_Z2xU1, s=-1, t=[(0, 0), (0, 1), (1, 2)], D=[1, 3, 4])
    leg1 = yastn.Leg(config_Z2xU1, s=1, t=[(0, -2), (0, 2)], D=[2, 3])
    leg2a = yastn.Leg(config_Z2xU1, s=1, t=[(0, -2), (0, 2), (1, -2), (1, 0), (1, 2)], D=[2, 6, 3, 6, 9])
    leg2b = yastn.Leg(config_Z2xU1, s=1, t=[(0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)], D=[2, 4, 6, 3, 6, 9])
    leg3a = yastn.Leg(config_Z2xU1, s=1, t=[(0, 0), (0, 2)], D=[4, 7])
    leg3b = yastn.Leg(config_Z2xU1, s=1, t=[(0, 0)], D=[4])

    a = yastn.randC(config=config_Z2xU1, legs=[leg0a, leg1, leg2a, leg3a])
    b = yastn.randC(config=config_Z2xU1, legs=[leg0b, leg1, leg2b, leg3b])
    combine_tests(a, b)


def test_add_diagonal(config_kwargs):
    """
    Addition of diagonal tensors matches sectorial bond dimensions,
    filling in missing zeros at the end of each sector.
    """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg0 = yastn.Leg(config_U1, s=-1, t=(-1, 0, 1, 3), D=(2, 2, 3, 2))
    leg1 = yastn.Leg(config_U1, s=-1, t=(-1, 0, 1, 2), D=(1, 5, 3, 2))
    a = yastn.eye(config_U1, legs=leg0)
    b = yastn.eye(config_U1, legs=leg1)
    c1 = a + b
    c2 = b + a
    c3 = a - b
    assert pytest.approx(yastn.trace(c1).item(), rel=tol) == 20
    assert pytest.approx(yastn.trace(c2).item(), rel=tol) == 20
    assert pytest.approx(yastn.trace(c3).item(), rel=tol) == -2
    assert all(x.get_legs(axes=0).D == (2, 5, 3, 2, 2) for x in [c1, c2, c3])


def test_algebra_functions(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.Tensor(config=config_U1, isdiag=True)
    a.set_block(ts=1, Ds=3, val=[1, 0.01, 0.0001])
    a.set_block(ts=-1, Ds=3, val=[1, 0.01, 0.0001])
    b = yastn.sqrt(a)
    assert pytest.approx(b.norm().item(), rel=tol) == np.sqrt(2.0202)
    b = yastn.rsqrt(a, cutoff=0.004)
    assert pytest.approx(b.norm().item(), rel=tol) == np.sqrt(202)
    b = yastn.reciprocal(a, cutoff=0.001)
    assert pytest.approx(b.norm().item(), rel=tol) == np.sqrt(20002)
    b = yastn.exp(1j * a)
    assert pytest.approx(b.norm().item(), rel=tol) == np.sqrt(6)

    c = a * (4j + 3)
    assert yastn.norm(c.imag() - 4 * a) < tol
    assert yastn.norm(c.real() - 3 * a) < tol
    assert yastn.norm(abs(c) - 5 * a) < tol


def test_algebra_fuse_meta(config_kwargs):
    """ test basic algebra on meta-fused tensor. """
    config_Z2 = yastn.make_config(sym='Z2', **config_kwargs)
    a = yastn.rand(config=config_Z2, s=(-1, 1, 1, -1),
                  t=((0, 1), (0,), (0, 1), (1,)), D=((1, 2), (3,), (4, 5), (7,)))
    b = yastn.rand(config=config_Z2, s=(-1, 1, 1, -1),
                  t=((0,), (0, 1), (0, 1), (0, 1)), D=((1,), (3, 4), (4, 5), (6, 7)))
    ma = a.fuse_legs(axes=((0, 3), (2, 1)), mode='meta')
    mb = b.fuse_legs(axes=((0, 3), (2, 1)), mode='meta')
    mc = algebra_vs_numpy(lambda x, y: x + y, ma, mb)
    uc = mc.unfuse_legs(axes=(0, 1)).transpose(axes=(0, 3, 2, 1))
    assert yastn.norm(uc - a - b) < tol


def algebra_hf(f, a, b, hf_axes1=(0, (1, 2), 3)):
    """
    Test operations on a and b combined with application of fuse_legs(..., mode='hard').

    f is lambda expresion using magic methods on a and b tensors.
    hf_axes1 are axes of first hard fusion resulting in 3 legs; without transpose.
    """
    fa = yastn.fuse_legs(a, axes=hf_axes1, mode='hard')
    fb = yastn.fuse_legs(b, axes=hf_axes1, mode='hard')
    ffa = yastn.fuse_legs(fa, axes=(1, (2, 0)), mode='hard')
    ffb = yastn.fuse_legs(fb, axes=(1, (2, 0)), mode='hard')
    fffa = yastn.fuse_legs(ffa, axes=[(0, 1)], mode='hard')
    fffb = yastn.fuse_legs(ffb, axes=[(0, 1)], mode='hard')

    c = algebra_vs_numpy(f, a, b)
    cf = yastn.fuse_legs(c, axes=hf_axes1, mode='hard')
    fc = f(fa, fb)
    fcf = yastn.fuse_legs(fc, axes=(1, (2, 0)), mode='hard')
    ffc = f(ffa, ffb)
    ffcf = yastn.fuse_legs(ffc, axes=[(0, 1)], mode='hard')
    fffc = f(fffa, fffb)
    assert all(yastn.norm(x - y) < tol for x, y in zip((fc, ffc, fffc), (cf, fcf, ffcf)))

    uffc = fffc.unfuse_legs(axes=0)
    uufc = uffc.unfuse_legs(axes=1).transpose(axes=(2, 0, 1))
    uf_axes = tuple(i for i, a in enumerate(hf_axes1) if not isinstance(a, int))
    uuuc = uufc.unfuse_legs(axes=uf_axes)
    assert all(yastn.norm(x - y) < tol for x, y in zip((ffc, fc, c), (uffc, uufc, uuuc)))
    assert all(x.is_consistent() for x in (fc, fcf, ffc, ffcf, fffc, uffc, uufc, uuuc))


def test_algebra_fuse_hard(config_kwargs):
    """ execute tests of additions after hard fusion for several tensors. """
    # U1 with 4 legs
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                t=((0,), (0,), (-1, 0, 1), (-1, 0, 1)),
                D=((2,), (5,), (7, 8, 9), (10, 11, 12)))
    a.set_block(ts=(1, 1, 0, 0), Ds=(3, 6, 8, 11))
    b = yastn.rand(config=config_U1, s=(1, -1, -1, 1),
                t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1), (-2, 0, 2)),
                D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    c = yastn.rand(config=config_U1, s=(1, -1, -1, 1),
                t=((1,), (1,), (0, 1), (0, 1)),
                D=((3,), (6,), (8, 9), (11, 12)))

    algebra_hf(lambda x, y: x / 0.5 + y * 3, b, c)
    algebra_hf(lambda x, y: x - y ** 2, a.conj(), c)
    algebra_hf(lambda x, y: 0.5 * x + y * 3, a, b.conj())

    # U1 with 6 legs
    t1, t2, t3 = (-1, 0, 1), (-2, 0, 2), (-3, 0, 3)
    Da, Db, Dc = (1, 3, 2), (3, 3, 4), (5, 3, 6)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t1, t1, t2, t2, t3, t3), D=(Da, Db, Db, Da, Da, Db))
    b = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t2, t2, t3, t3, t1, t1), D=(Db, Dc, Da, Dc, Da, Db))
    algebra_hf(lambda x, y: x / 0.5 + y * 3, a, b, hf_axes1=((0, 1), (2, 3), (4, 5)))
    b.set_block(ts=(2, 2, 1, -2, -3, 0), Ds=(4, 6, 1, 1, 1, 3), val='normal')
    algebra_hf(lambda x, y: x - 3 * y, a, b, hf_axes1=((0, 1), (2, 3), (4, 5)))

    # Z2xU1 with 4 legs
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    leg1 = yastn.Leg(config_Z2xU1, s=1, t=[(0, -1), (0, 1), (1, -1), (1, 1)], D=[1, 2, 3, 4])
    leg2 = yastn.Leg(config_Z2xU1, s=1, t=[(0, 0), (0, 1), (1, 1)], D=[5, 2, 4])
    a = yastn.rand(config=config_Z2xU1, legs=[leg2.conj(), leg2, leg1, leg1.conj()])
    b = yastn.rand(config=config_Z2xU1, legs=[leg1, leg1.conj(), leg2.conj(), leg2])

    algebra_hf(lambda x, y: x / 0.5 + y * 3, a, b.conj())

    a.set_block(ts=((1, 2), (1, 2), (1, 2), (1, 2)), Ds=(6, 6, 6, 6), val='normal')
    a.set_block(ts=((1, -1), (1, -1), (1, -1), (1, -1)), Ds=(3, 3, 3, 3), val='normal')
    algebra_hf(lambda x, y: x - y ** 3, a.conj(), b)


def test_algebra_allclose(config_kwargs):
    # U1 with 4 legs
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                t=((0,), (0,), (-1, 0, 1), (-1, 0, 1)),
                D=((2,), (5,), (7, 8, 9), (10, 11, 12)))

    assert yastn.allclose(a, a) is True
    assert yastn.allclose(a, a.flip_signature()) is False

    b = 0 * a
    b.set_block(ts=(1, 1, 1, 1), Ds=(1, 1, 9, 12), val='zeros')

    c = a + b  # differs from `a` by a single zero block
    assert yastn.allclose(a, c, rtol=1e-13, atol=1e-13) is False
    assert (a - c).norm() < 1e-13


def test_algebra_exceptions(config_kwargs):
    """ test handling exceptions """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    leg2 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 5))
    leg3 = yastn.Leg(config_U1, s=1, t=(-1, 0), D=(2, 4))

    with pytest.raises(yastn.YastnError,
                       match="Cannot add diagonal tensor to non-diagonal tensor."):
        a = yastn.eye(config=config_U1, legs=[leg1.conj(), leg1])
        b = yastn.ones(config=config_U1, legs=[leg1.conj(), leg1])
        _ = a + b
    with pytest.raises(yastn.YastnError,
                       match="Signatures do not match."):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1, leg2.conj()])
        b = yastn.rand(config=config_U1, legs=[leg1, leg2.conj(), leg1, leg2.conj()])
        _ = a + b
    with pytest.raises(yastn.YastnError,
                       match="Tensors have different number of legs."):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1, leg2.conj()])
        b = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1])
        _ = a + b
    with pytest.raises(yastn.YastnError,
                       match="Bond dimensions do not match."):
        a = yastn.rand(config=config_U1, legs=[leg1, leg1.conj(), leg1])
        b = yastn.rand(config=config_U1, legs=[leg1, leg2.conj(), leg1])
        _ = a + b
    with pytest.raises(yastn.YastnError,
                       match="Bond dimensions do not match."):
        a = yastn.rand(config=config_U1, legs=[leg1, leg1.conj(), leg1])
        b = yastn.rand(config=config_U1, legs=[leg1, leg3.conj(), leg1])
        _ = a + b
    with pytest.raises(yastn.YastnError,
                       match="Bond dimensions related to some charge are not consistent."):
        # Here, individual blocks between a na b are consistent, but cannot form consistent sum.
        a = yastn.Tensor(config=config_U1, s=(1, -1, 1, -1))
        a.set_block(ts=(1, 1, 0, 0), Ds=(2, 2, 1, 1), val='normal')
        b = yastn.Tensor(config=config_U1, s=(1, -1, 1, -1))
        b.set_block(ts=(1, 1, 1, 1), Ds=(1, 1, 1, 1), val='normal')
        _ = a + b
    with pytest.raises(yastn.YastnError,
                       match="Indicated axes of two tensors have different number of meta-fused legs or sub-fusions order."):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1, leg2.conj()])
        b = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1, leg2.conj()])
        a = a.fuse_legs(axes=(0, 1, (2, 3)), mode='meta')
        b = b.fuse_legs(axes=((0, 1), 2, 3), mode='meta')
        _ = a + b
    with pytest.raises(yastn.YastnError,
                       match="Tensor charges do not match."):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1], n=0)
        b = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1], n=1)
        _ = a + b
    with pytest.raises(yastn.YastnError,
                       match="Number of tensors and amplitudes do not match."):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1], n=0)
        yastn.add(a, a, a, amplitudes=(1, 1))
    with pytest.raises(yastn.YastnError,
                       match="p should be 'fro', or 'inf'."):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1], n=0)
        _ = yastn.norm(a, p='wrong_order')



def test_hf_union_exceptions(config_kwargs):
    """ exceptions happening in resolving hard-fusion mismatches. """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(2, 3, 2))
    leg2 = yastn.Leg(config_U1, s=1, t=(-2, 0, 2), D=(2, 3, 5))
    leg3 = yastn.Leg(config_U1, s=1, t=(-2, 0, 2), D=(2, 5, 2))
    with pytest.raises(yastn.YastnError):
        a = yastn.Tensor(config=config_U1, s=(1, -1, 1, -1))
        a.set_block(ts=(1, 1, 0, 0), Ds=(2, 2, 1, 1), val='normal')
        b = yastn.Tensor(config=config_U1, s=(1, -1, 1, -1))
        b.set_block(ts=(1, 1, 1, 1), Ds=(1, 1, 1, 1), val='normal')
        a = a.fuse_legs(axes=[(0, 1, 2, 3)], mode='hard')
        b = b.fuse_legs(axes=[(0, 1, 2, 3)], mode='hard')
        _ = a + b
        # Bond dimensions of fused legs do not match.
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config=config_U1, legs=[leg1, leg1.conj(), leg1])
        b = yastn.rand(config=config_U1, legs=[leg2, leg3.conj(), leg2])
        a = a.fuse_legs(axes=((0, 2), 1), mode='hard')
        b = b.fuse_legs(axes=((0, 2), 1), mode='hard')
        _ = a + b
        # Bond dimensions do not match.
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1.conj(), leg2.conj()])
        b = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2.conj(), leg1, leg2.conj()])
        a = a.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        b = b.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        _ = a + b
        # Signatures of hard-fused legs do not match.
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1.conj(), leg2.conj()])
        b = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1.conj(), leg2.conj()])
        a = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        b = b.fuse_legs(axes=((0, 1, 2), 3), mode='hard')
        _ = a + b
        # Indicated axes of two tensors have different number of hard-fused legs or sub-fusions order.

    test_array = a.config.backend.to_tensor([[1], [2]], device=a.device)
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1.conj(), leg2.conj()])
        _ = a * test_array
        # Multiplication cannot change data size; broadcasting not supported.
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1.conj(), leg2.conj()])
        _ = a / test_array
        # truediv cannot change data size; broadcasting not supported.
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1.conj(), leg2.conj()])
        _ = a ** test_array
        # Exponent cannot change data size; broadcasting not supported.
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1.conj(), leg2.conj()])
        _ = np.float64(10) + a
        # Only np.float * Mps is supported; add was called.
    with pytest.raises(yastn.YastnError):
        a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1.conj(), leg2.conj()])
        _ = a + 1
        # Operation requires two yastn.Tensor-s


def test_auxiliary():
    """ test some auxiliary functions that join slices. """
    # _join_contiguous_slices
    slc1 = ((0, 10), (10, 20), (30, 40), (40, 50))
    slc2 = ((0, 10), (10, 20), (20, 30), (40, 50))
    meta = yastn.tensor._auxliary._join_contiguous_slices(slc1, slc2)
    assert meta == (((0, 20), (0, 20)), ((30, 40), (20, 30)), ((40, 50), (40, 50)))

    slc1 = ((0, 10), (10, 20), (20, 30), (40, 50))
    slc2 = ((10, 20), (20, 30), (30, 40), (40, 50))
    meta = yastn.tensor._auxliary._join_contiguous_slices(slc1, slc2)
    assert meta == (((0, 30), (10, 40)), ((40, 50), (40, 50)))

    # _slices_to_negate
    slices = (yastn.tensor._auxliary._slc(((0, 10),)),
              yastn.tensor._auxliary._slc(((10, 20),)),
              yastn.tensor._auxliary._slc(((20, 30),)),
              yastn.tensor._auxliary._slc(((30, 40),)))

    negate_slices = yastn.tensor._contractions._slices_to_negate([0, 0, 0, 0], slices)
    assert negate_slices == ()
    negate_slices = yastn.tensor._contractions._slices_to_negate([0, 1, 1, 0], slices)
    assert negate_slices == ((10, 30),)
    negate_slices = yastn.tensor._contractions._slices_to_negate([1, 0, 1, 1], slices)
    assert negate_slices == ((0, 10), (20, 40))


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
