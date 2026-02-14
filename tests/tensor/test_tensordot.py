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
""" yastn.tensordot() """
import numpy as np
import pytest
import yastn
import re

tol = {"float64": 1e-12, "complex128": 1e-12,  #pylint: disable=invalid-name
       "float32": 1e-5, "complex64": 1e-5}  #pylint: disable=invalid-name


torch_test = pytest.mark.skipif("'torch' not in config.getoption('--backend')",
                                reason="Uses torch.autograd.gradcheck().")


def tensordot_vs_numpy(a, b, axes, conj, dtype):
    outa = tuple(ii for ii in range(a.ndim) if ii not in axes[0])
    outb = tuple(ii for ii in range(b.ndim) if ii not in axes[1])

    legs_a = a.get_legs()
    legs_b = b.get_legs()

    legs_a_out = {nn: legs_a[ii] for nn, ii in enumerate(outa)}
    legs_b_out = {nn + len(outa): legs_b[ii] for nn, ii in enumerate(outb)}

    conj_in = False if conj[0] + conj[1] == 1 else True
    na = a.to_numpy(legs={ia: legs_b[ib].conj() if conj_in else legs_b[ib] for ia, ib in zip(*axes)})
    nb = b.to_numpy(legs={ib: legs_a[ia].conj() if conj_in else legs_a[ia] for ia, ib in zip(*axes)})
    if conj[0]:
        na = na.conj()
        legs_a_out = {nn: leg.conj() for nn, leg in legs_a_out.items()}
    if conj[1]:
        nb = nb.conj()
        legs_b_out = {nn: leg.conj() for nn, leg in legs_b_out.items()}
    nab = np.tensordot(na, nb, axes)

    c = yastn.tensordot(a, b, axes, conj)

    print(a.yastn_dtype, b.yastn_dtype, c.yastn_dtype, dtype)
    assert c.yastn_dtype == dtype

    nc = c.to_numpy(legs={**legs_a_out, **legs_b_out})
    assert c.is_consistent()
    assert a.are_independent(c)
    assert b.are_independent(c)
    assert np.linalg.norm(nc - nab) < tol[dtype]
    return c


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_dot_basic_dense(config_kwargs, dtype):
    """ test tensordot for different symmetries. """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    config_dense.backend.random_seed(1)
    a = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5), dtype=dtype)
    b = yastn.rand(config=config_dense, s=(1, -1, 1), D=(2, 3, 5), dtype=dtype)
    c1 = tensordot_vs_numpy(a, b, axes=((0, 3), (0, 2)), conj=(0, 0), dtype=dtype)
    c2 = tensordot_vs_numpy(b, a, axes=((2, 0), (3, 0)), conj=(1, 1), dtype=dtype)
    assert yastn.norm(c1.conj() - c2.transpose(axes=(1, 2, 0))) < tol[dtype]
    #
    # outer product
    tensordot_vs_numpy(a, b, axes=((), ()), conj=(0, 0), dtype=dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_dot_basic_dense_1(config_kwargs,dtype):
    """ test tensordot for different symmetries. """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    config_dense.backend.random_seed(1)
    a = yastn.rand(config=config_dense, s=(-1, 1), D=(2, 2), dtype=dtype)
    b = yastn.rand(config=config_dense, s=(1, 1), D=(2, 2), dtype=dtype)
    tensordot_vs_numpy(a, b, axes=((1,), (1,)), conj=(0, 1), dtype=dtype)


def test_dot_basic_dense2(config_kwargs):
    """ test tensordot for different symmetries. """
    # dense
    dtype = 'float64'
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    config_dense.backend.random_seed(1)
    a = yastn.rand(config=config_dense, s=(-1, 1), D=(2, 3), dtype=dtype)
    b = yastn.rand(config=config_dense, s=(1, -1), D=(2, 3), dtype=dtype)
    c1 = tensordot_vs_numpy(a, b, axes=((0,), (0,)), conj=(0, 0), dtype=dtype)
    c2 = tensordot_vs_numpy(b, a, axes=((0,), (0,)), conj=(1, 1), dtype=dtype)
    assert yastn.norm(c1.conj() - c2.transpose(axes=(1,0))) < tol[dtype]
    # outer product
    tensordot_vs_numpy(a, b, axes=((), ()), conj=(0, 0), dtype=dtype)


def test_dot_basic_dense3(config_kwargs):
    """ test tensordot for different symmetries. """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    config_dense.backend.random_seed(1)
    dtype = 'float64'
    a = yastn.rand(config=config_dense, s=(-1, 1, -1), D=(2, 4, 5), dtype=dtype)
    b = yastn.rand(config=config_dense, s=(1, -1, 1, 1), D=(2, 3, 1, 5), dtype=dtype)
    c1 = tensordot_vs_numpy(a, b, axes=((0, 2), (0, 3)), conj=(0, 0), dtype=dtype) # a1 b1 b2
    c2 = tensordot_vs_numpy(b, a, axes=((3, 0), (2, 0)), conj=(1, 1), dtype=dtype) # b1* b2* a1*
    assert yastn.norm(c1.conj() - c2.transpose(axes=(2, 0, 1))) < tol[dtype]
    # outer product
    tensordot_vs_numpy(a, b, axes=((), ()), conj=(0, 0), dtype=dtype)


def test_dot_basic_U1(config_kwargs):
    """ test tensordot for different symmetries. """
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    dtype = 'float64'
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(1, -1, 1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 0, 1)),
                  D=((1, 2, 3), (4, 5, 6), (10, 7, 11)), dtype=dtype)
    tensordot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 0), dtype=dtype)
    tensordot_vs_numpy(a, b, axes=((1, 3), (1, 2)), conj=(0, 0), dtype=dtype)

    a = yastn.Tensor(config=config_U1, s=(-1, 1, 1, -1), n=-2)
    a.set_block(ts=(2, 1, 0, 1), Ds=(2, 1, 10, 1), val='rand')
    b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, -1), n=1)
    b.set_block(ts=(1, 2, 0, 0), Ds=(1, 2, 10, 10), val='rand')
    c = tensordot_vs_numpy(a, b, axes=((2, 1), (1, 2)), conj=(1, 0), dtype=dtype)
    assert c.struct.n == (3,)
    a.set_block(ts=(1, 1, -1, 1), Ds=(1, 1, 11, 1), val='rand')
    a.set_block(ts=(2, 2, -1, 1), Ds=(2, 2, 11, 1), val='rand')
    a.set_block(ts=(3, 3, -1, 1), Ds=(3, 3, 11, 1), val='rand')
    b.set_block(ts=(1, 1, 1, 0), Ds=(1, 1, 1, 10), val='rand')
    b.set_block(ts=(3, 3, 1, 0), Ds=(3, 3, 1, 10), val='rand')
    b.set_block(ts=(3, 3, 2, 1), Ds=(3, 3, 2, 1), val='rand')
    tensordot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 1), dtype=dtype)
    tensordot_vs_numpy(a, b, axes=((0, 3, 1), (1, 2, 0)), conj=(0, 0), dtype=dtype)
    #
    # corner cases
    a = yastn.rand(config=config_U1, s=(-1, 1, 1),
                  t=((-1, 1, 0), (-1, 1, 0), (-1, 1, 0)),
                  D=((1, 2, 3), (3, 2, 1), (1, 2, 2)), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(-1, 1, 1),
                  t=((-2, 2), (-1, 1, -3), (-1, 1, -3)),
                  D=((1, 2), (3, 2, 1), (1, 2, 2)), dtype=dtype)
    # some charges are missing
    assert a.size > 0 and b.size > 0
    tensordot_vs_numpy(b, a, axes=((2,), (0,)), conj=(0, 0), dtype=dtype)
    #
    # no matching charges
    c = tensordot_vs_numpy(a, b, axes=((2,), (0,)), conj=(0, 0), dtype=dtype)
    assert c.size == 0
    assert c.norm() < tol[dtype]
    #
    # outer product
    c1 = tensordot_vs_numpy(a, b, axes=((), ()), conj=(0, 0), dtype=dtype)
    c2 = tensordot_vs_numpy(b, a, axes=((), ()), conj=(1, 1), dtype=dtype)
    assert yastn.norm(c1.conj() - c2.transpose(axes=(3, 4, 5, 0, 1, 2))) < tol[dtype]


def test_dot_basic_Z2(config_kwargs):
    """ test tensordot for different symmetries. """
    # Z2
    config_Z2 = yastn.make_config(sym='Z2', **config_kwargs)
    dtype = 'float64'
    a = yastn.rand(config=config_Z2, s=(-1, 1, 1, -1),
                  t=((0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5 ,6), (7, 8)), dtype=dtype)
    b = yastn.rand(config=config_Z2, s=(1, -1, 1),
                  t=((0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (7, 8)), dtype=dtype)
    tensordot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 0), dtype=dtype)
    tensordot_vs_numpy(a, b, axes=((1, 3), (1, 2)), conj=(0, 0), dtype=dtype)

    import pdb; pdb.set_trace()

    a = yastn.Tensor(config=config_Z2, s=(-1, 1, 1, -1), n=-2)
    a.set_block(ts=(2, 1, 0, 1), Ds=(2, 1, 10, 1), val='rand')
    b = yastn.Tensor(config=config_Z2, s=(-1, 1, 1, -1), n=1)
    b.set_block(ts=(1, 2, 0, 0), Ds=(1, 2, 10, 10), val='rand')
    c = tensordot_vs_numpy(a, b, axes=((2, 1), (1, 2)), conj=(1, 0), dtype=dtype)
    assert c.struct.n == (3,)
    a.set_block(ts=(1, 1, -1, 1), Ds=(1, 1, 11, 1), val='rand')
    a.set_block(ts=(2, 2, -1, 1), Ds=(2, 2, 11, 1), val='rand')
    a.set_block(ts=(3, 3, -1, 1), Ds=(3, 3, 11, 1), val='rand')
    b.set_block(ts=(1, 1, 1, 0), Ds=(1, 1, 1, 10), val='rand')
    b.set_block(ts=(3, 3, 1, 0), Ds=(3, 3, 1, 10), val='rand')
    b.set_block(ts=(3, 3, 2, 1), Ds=(3, 3, 2, 1), val='rand')
    tensordot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 1), dtype=dtype)
    tensordot_vs_numpy(a, b, axes=((0, 3, 1), (1, 2, 0)), conj=(0, 0), dtype=dtype)
    #
    # corner cases
    a = yastn.rand(config=config_Z2, s=(-1, 1, 1),
                  t=((-1, 1, 0), (-1, 1, 0), (-1, 1, 0)),
                  D=((1, 2, 3), (3, 2, 1), (1, 2, 2)), dtype=dtype)
    b = yastn.rand(config=config_Z2, s=(-1, 1, 1),
                  t=((-2, 2), (-1, 1, -3), (-1, 1, -3)),
                  D=((1, 2), (3, 2, 1), (1, 2, 2)), dtype=dtype)
    # some charges are missing
    assert a.size > 0 and b.size > 0
    tensordot_vs_numpy(b, a, axes=((2,), (0,)), conj=(0, 0), dtype=dtype)
    #
    # no matching charges
    c = tensordot_vs_numpy(a, b, axes=((2,), (0,)), conj=(0, 0), dtype=dtype)
    assert c.size == 0
    assert c.norm() < tol[dtype]
    #
    # outer product
    c1 = tensordot_vs_numpy(a, b, axes=((), ()), conj=(0, 0), dtype=dtype)
    c2 = tensordot_vs_numpy(b, a, axes=((), ()), conj=(1, 1), dtype=dtype)
    assert yastn.norm(c1.conj() - c2.transpose(axes=(3, 4, 5, 0, 1, 2))) < tol[dtype]


def test_dot_basic_U1_2(config_kwargs):
    """ test tensordot for different symmetries. """
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    dtype = 'float64'
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(1, -1, 1),
                  t=((-1, 1, 2), (-1, 1, 2), (-1, 0, 1)),
                  D=((1, 2, 2), (2, 2, 2), (2, 2, 2)), dtype=dtype)
    tensordot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 0), dtype=dtype)
    tensordot_vs_numpy(a, b, axes=((1, 3), (1, 2)), conj=(0, 0), dtype=dtype)


def test_dot_basic_U1_3(config_kwargs):
    """ BUG cutensor 2.3.1 """
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    dtype = 'float64'
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                     t=((1,2,3,), (1,2,3,), (-1,0,), (1,)),
                     D=((1, 2, 3), (2, 2, 3), (11, 10), (1,)),
                     n=-2, dtype=dtype)
    b = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                     t=((1,3,), (1,2,3,), (0,1,2,), (0,1)),
                     D=((1, 3), (2, 2, 3), (10, 2, 2), (11, 1)),
                     n=1, dtype=dtype)
    tensordot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 1), dtype=dtype)

    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                     t=((1,2,3,), (1,2,3,), (-1,0,), (1,)),
                     D=((1, 2, 3), (2, 2, 3), (11, 10), (1,)),
                     n=-2, dtype=dtype)
    b = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                     t=((1,3,), (1,2,3,), (0,1,2,), (0,1)),
                     D=((1, 3), (2, 2, 3), (10, 1, 2), (10, 1)),
                     n=1, dtype=dtype)
    tensordot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 1), dtype=dtype)


def test_dot_basic_Z2xU1(config_kwargs):
    """ test tensordot for different symmetries. """
    # Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    t1 = [(0, -1), (0, 1), (1, -1), (1, 1)]
    t2 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    dtype = 'float64'
    a = yastn.rand(config=config_Z2xU1, s=(-1, 1, 1, -1),
                  t=(t1, t1, t1, t1),
                  D=((1, 2, 2, 4), (9, 4, 3, 2), (5, 6, 7, 8), (7, 8, 9, 10)), dtype=dtype)
    b = yastn.rand(config=config_Z2xU1, s=(1, -1, 1),
                  t=(t1, t1, t2),
                  D=((1, 2, 2, 4), (9, 4, 3, 2), (5, 6, 7, 8,)), dtype=dtype)

    tensordot_vs_numpy(a, b, axes=((0, 1), (0, 1)), conj=(0, 0), dtype=dtype)
    tensordot_vs_numpy(b, a, axes=((1, 0), (1, 0)), conj=(0, 0), dtype=dtype)

    # corner cases;
    a = yastn.rand(config=config_Z2xU1, s=(-1, 1),
                  t=(t1, t1), D=((1, 2, 3, 4), (2, 3, 4, 5)), dtype=dtype)
    b = yastn.rand(config=config_Z2xU1, s=(-1, 1),
                  t=(t2, t2), D=((1, 2, 3, 4), (2, 3, 4, 5)), dtype=dtype)
    #
    # no matching charges
    c = tensordot_vs_numpy(a, b, axes=((1,), (0,)), conj=(0, 0), dtype=dtype)
    assert c.size == 0
    assert c.norm() < tol[dtype]


def test_tensordot_diag(config_kwargs):
    # Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    t1 = [(0, -1), (0, 1), (1, -1), (1, 1)]
    D1 = (1, 2, 2, 4)

    t2 = [(0, -1), (0, 1), (1, -1), (0, 0)]
    D2 = (1, 2, 2, 5)

    dtype = 'float64'
    a = yastn.rand(config=config_Z2xU1, s=(-1, 1, 1, -1),
                  t=(t1, t1, t1, t1),
                  D=(D1, (9, 4, 3, 2), (5, 6, 7, 8), (7, 8, 9, 10)), dtype=dtype)
    b = yastn.rand(config=config_Z2xU1, s=(1, -1), t = [t2, t2], D=[D2, D2], isdiag=True, dtype=dtype)
    b2 = b.diag()

    c1 = b.broadcast(a, axes=0)
    c2 = b.conj().broadcast(a, axes=0)
    c3 = b2.tensordot(a, axes=(0, 0))

    assert(yastn.norm(c1 - c2)) < tol[dtype]
    assert(yastn.norm(c1 - c3)) < tol[dtype]
    assert c3.get_shape() == (5, 18, 26, 34)


def tensordot_hf(a, b, hf_axes1, dtype):
    """ Test vdot of a and b combined with application of fuse_legs(..., mode='hard'). """
    fa = yastn.fuse_legs(a, axes=hf_axes1, mode='hard')
    fb = yastn.fuse_legs(b, axes=hf_axes1, mode='hard')
    ffa = yastn.fuse_legs(fa, axes=(0, (2, 1)), mode='hard')
    ffb = yastn.fuse_legs(fb, axes=(0, (2, 1)), mode='hard')
    c = tensordot_vs_numpy(a, b, axes=((1, 2, 3, 4, 5), (1, 2, 3, 4, 5)), conj=(1, 0), dtype=dtype)
    fc = yastn.tensordot(fa, fb, axes=((1, 2), (1, 2)), conj=(1, 0))
    ffc = yastn.tensordot(ffa, ffb, axes=(1, 1), conj=(1, 0))
    assert ffc.yastn_dtype == dtype
    assert all(yastn.norm(c - x) < tol[dtype] for x in (fc, ffc))


def test_tensordot_fuse_hard(config_kwargs):
    """ test tensordot combined with hard-fusion."""
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    dtype = 'float64'
    # will have some leg-structure mismatches after fusion to resolve.
    t1, t2, t3 = (-1, 0, 1), (-2, 0, 2), (-3, 0, 3)
    D1, D2, D3 = (1, 3, 2), (3, 3, 4), (5, 3, 6)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t1, t1, t2, t2, t3, t3), D=(D1, D2, D2, D1, D1, D2), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t2, t2, t3, t3, t1, t1), D=(D2, D3, D1, D3, D1, D2), dtype=dtype)
    tensordot_hf(a, b, hf_axes1=(0, (4, 3, 1), (5, 2)), dtype=dtype)
    tensordot_hf(a, b, hf_axes1=(0, (4, 3, 1, 5), 2), dtype=dtype)
    #
    # other corner case; no matching charges
    a = yastn.rand(config=config_U1, s=(1, 1, 1),
                  t=((-1, 1, 0), (-1, 1, 0), (-1, 1, 0)),
                  D=((1, 2, 3), (1, 2, 3), (1, 2, 3)), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(-1, -1, -1),
                  t=((-2, 2), (-1, 1), (-1, 1, -3)),
                  D=((4, 4), (1, 2), (1, 2, 5)), dtype=dtype)
    c = tensordot_vs_numpy(a, b, axes=((1, 2), (0, 1)), conj=(0, 0), dtype=dtype)
    af = a.fuse_legs(axes=(0, (1, 2)), mode='hard')
    bf = b.fuse_legs(axes=((0, 1), 2), mode='hard')
    cf = tensordot_vs_numpy(af, bf, axes=((1,), (0,)), conj=(0, 0), dtype=dtype)
    assert (cf - c).norm() < tol[dtype]
    assert cf.size == 0
    assert cf.norm() < tol[dtype]


def test_tensordot_fuse_meta(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    dtype = 'float64'
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1),
                  t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8), (3, 4)), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1),
                  t=((-1, 0, 1), (1,), (-1, 1), (0, 1), (0, 1, 2)),
                  D=((2, 1, 2), (4,), (4, 6), (7, 8), (3, 4, 5)), dtype=dtype)

    c = tensordot_vs_numpy(a, b, axes=((0, 3, 4), (0, 3, 4)), conj=(0, 1), dtype=dtype)
    fa = a.fuse_legs(axes=(0, (2, 1), (4, 3)), mode='meta')
    fb = b.fuse_legs(axes=(0, (2, 1), (4, 3)), mode='meta')
    assert fa.trans == (0, 2, 1, 4, 3)
    assert fb.trans == (0, 2, 1, 4, 3)
    #
    fc = tensordot_vs_numpy(fa, fb, axes=((2, 0), (2, 0)), conj=(0, 1), dtype=dtype)
    assert fc.trans == (0, 1, 2, 3) and fc.ndim == 2
    #
    fa = fa.fuse_legs(axes=((0, 2), 1), mode='meta')
    fb = fb.fuse_legs(axes=((0, 2), 1), mode='meta')
    assert fa.trans == (0, 4, 3, 2, 1)
    assert fb.trans == (0, 4, 3, 2, 1)
    #
    ffc = tensordot_vs_numpy(fa, fb, axes=((0,), (0,)), conj=(0, 1), dtype=dtype)
    assert ffc.trans == (0, 1, 2, 3) and ffc.ndim == 2
    #
    cf = c.fuse_legs(axes=((1, 0), (3, 2)), mode='meta')
    assert cf.trans == (1, 0, 3, 2) and cf.ndim == 2
    #
    assert all(yastn.norm(cf - x) < tol[dtype] for x in (fc, ffc))


def test_tensordot_exceptions(config_kwargs):
    """ special cases and exceptions"""
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    t1, t2 = (-1, 0, 1), (-1, 0, 2)
    D1, D2 = (2, 3, 4), (2, 4, 5)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1),
                    t=(t1, t1, t1, t1), D=(D1, D1, D1, D1))
    b = yastn.rand(config=config_U1, s=(-1, -1, 1, -1),
                    t=(t1, t1, t2, t1), D=(D1, D1, D1, D2))
    with pytest.raises(yastn.YastnError,
                       match="Signatures do not match."):
        _ = yastn.tensordot(a, b, axes=((0, 1, 2), (0, 1, 2)), conj=(0, 1))
    with pytest.raises(yastn.YastnError,
                       match="indicate different number of legs."):
        _ = yastn.tensordot(a, b, axes=((0, 1, 2), (0, 1)), conj=(1, 0))
        # axes[0] and axes[1] indicated different number of legs.
    with pytest.raises(yastn.YastnError,
                       match="Repeated axis in"):
        _ = yastn.tensordot(a, b, axes=((0, 1), (0, 0)), conj=(1, 0))
        # Repeated axis in axes[0] or axes[1].
    with pytest.raises(yastn.YastnError,
                       match="Bond dimensions do not match."):
        _ = yastn.tensordot(a, b, axes=((2, 3), (2, 3)), conj=(1, 0))
    with pytest.raises(yastn.YastnError,
                       match="Indicated axes of two tensors have different number of meta-fused legs or sub-fusions order."):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
        bf = b.fuse_legs(axes=(0, (1, 2, 3)), mode='meta')
        _ = yastn.tensordot(af, bf, axes=(0, 0), conj=(1, 0))
    with pytest.raises(yastn.YastnError,
                       match="Indicated axes of two tensors have different number of hard-fused legs or sub-fusions order."):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        bf = b.fuse_legs(axes=(0, (1, 2, 3)), mode='hard')
        _ = yastn.tensordot(af, bf, axes=(0, 0), conj=(1, 0))
    with pytest.raises(yastn.YastnError,
                       match="Signatures of hard-fused legs do not match."):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        bf = b.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        _ = yastn.tensordot(af, bf, axes=(0, 0), conj=(1, 0))
    with pytest.raises(yastn.YastnError,
                       match="Bond dimensions of fused legs do not match."):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        bf = b.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        _ = yastn.tensordot(af, bf, axes=((1,), (1,)), conj=(1, 0))
    with pytest.raises(yastn.YastnError,
                       match="Bond dimensions do not match."):
        af = a.fuse_legs(axes=(1, 3, (0, 2)), mode='hard')
        bf = b.fuse_legs(axes=(1, 3, (0, 2)), mode='hard')
        _ = yastn.tensordot(af, bf, axes=((1, 2), (1, 2)), conj=(1, 0))
    with pytest.raises(yastn.YastnError,
                       match=re.escape("Outer product with diagonal tensor not supported. Use yastn.diag() first.")):
        c = yastn.rand(config=config_U1, isdiag=True, t=(-1, 0, 1), D=(1, 2, 3))
        _ = yastn.tensordot(c, a, axes=((),()))
    with pytest.raises(yastn.YastnError,
                       match=re.escape("Tensordot policy not recognized. It should be 'fuse_to_matrix', 'fuse_contracted', or 'no_fusion'.")):
        config = config_U1._replace(tensordot_policy='something')
        c = a._replace(config=config)
        _ = yastn.tensordot(c, b, axes=((1, 2), (0, 1)))


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--tensordot_policy", "fuse_to_matrix"])
    pytest.main([__file__, "-vs", "--durations=0", "--tensordot_policy", "fuse_contracted"])
    pytest.main([__file__, "-vs", "--durations=0", "--tensordot_policy", "no_fusion"])
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--tensordot_policy", "fuse_to_matrix"])
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--tensordot_policy", "fuse_contracted"])
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--tensordot_policy", "no_fusion"])
