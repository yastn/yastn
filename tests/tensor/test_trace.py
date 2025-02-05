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
""" yastn.trace() """
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def trace_vs_numpy(a, axes):
    """ Compares yastn.trace vs dense operations in numpy. Return traced yastn tensor. """
    if isinstance(axes[0], int):
        axes = ((axes[0],), (axes[1],))
    out = tuple(i for i in range(a.ndim) if i not in axes[0] + axes[1])

    if not (a.isdiag or len(axes[0]) == 0):
        ma = a.fuse_legs(axes=axes+out, mode='meta')
        tDin = {0: ma.get_legs(1).conj(), 1: ma.get_legs(0).conj()}
        na = ma.to_numpy(tDin)  # to_numpy() with 2 matching axes to be traced
    else:
        na = a.to_numpy() # no trace is axes=((),())

    nat = np.trace(na) if len(axes[0]) > 0 else na
    c = yastn.trace(a, axes)

    assert c.is_consistent()
    if len(axes[0]) > 0:
        assert a.are_independent(c)

    legs_out = {nn: a.get_legs(ii) for nn, ii in enumerate(out)}
    # trace might have removed some charges on remaining legs
    nc = c.to_numpy(legs=legs_out) # for comparison they have to be filled in.
    assert np.linalg.norm(nc - nat) < tol
    return c


def test_trace_basic(config_kwargs):
    """ test trace for different symmetries. """
    # dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)

    empty = yastn.Tensor(config_dense, s=(1, -1))
    assert yastn.trace(empty).item() == 0

    a = yastn.ones(config=config_dense, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))
    b = trace_vs_numpy(a, axes=(0, 2))
    c = trace_vs_numpy(b, axes=(1, 0))
    assert pytest.approx(c.item(), rel=tol) == 10.

    a = yastn.eye(config=config_dense, D=5)
    b = trace_vs_numpy(a, axes=((), ()))
    assert yastn.norm(a - b) < tol
    c = trace_vs_numpy(a, axes=(0, 1))
    assert pytest.approx(c.item(), rel=tol) == 5.

    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg1 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3))
    leg2 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(4, 5))
    leg3 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(6, 7))
    a = yastn.ones(config=config_U1, legs=[leg1.conj(), leg2.conj(), leg3.conj(), leg3, leg2, leg1])
    b = trace_vs_numpy(a, axes=(0, 5))
    b = trace_vs_numpy(b, axes=((), ())) # no trace
    b = trace_vs_numpy(b, axes=(3, 0))
    b = trace_vs_numpy(b, axes=(0, 1))
    assert pytest.approx(b.item(), rel=tol) == 5 * 9 * 13
    b = trace_vs_numpy(a, axes=((1, 5), (4, 0)))
    b = trace_vs_numpy(b, axes=(1, 0))
    assert pytest.approx(b.item(), rel=tol) == 5 * 9 * 13

    leg = yastn.Leg(config_U1, s=1, t=(1, 2, 3), D=(3, 4, 5))
    a = yastn.eye(config=config_U1, legs=leg)
    b = trace_vs_numpy(a, axes=((), ())) # no trace
    b = trace_vs_numpy(a, axes=(0, 1))

    a = yastn.ones(config=config_U1, s=(-1, -1, 1),
                  t=[(1,), (1,), (2,)], D=[(2,), (2,), (2,)])
    b = trace_vs_numpy(a, axes=(0, 2))
    assert b.norm() < tol  # == 0

    # Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    leg1 = yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(6, 4, 9, 6))
    leg2 = yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(20, 16, 25, 20))
    a = yastn.randC(config=config_Z2xU1, legs=[leg1.conj(), leg2.conj(), leg2, leg1])
    b = trace_vs_numpy(a, axes=(0, 3))
    b = trace_vs_numpy(b, axes=(1, 0))
    c = a.trace(axes=((0, 1), (3, 2)))
    assert pytest.approx(c.item(), rel=tol) == b.item()

    leg = yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (1, 1), (0, 2)), D=(2, 2, 2))
    a = yastn.eye(config=config_Z2xU1, legs=leg)
    b = trace_vs_numpy(a, axes=((), ())) # no trace
    assert yastn.norm(a - b) < tol
    b = trace_vs_numpy(a, axes=(0, 1))
    assert pytest.approx(b.item(), rel=tol) == 6


def test_trace_fusions(config_kwargs):
    """ test trace of meta-fused and hard-fused tensors. """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(1, 2, 3))
    leg2 = yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6))

    # meta-fusion

    a = yastn.randR(config=config_U1, legs=[leg1.conj(), leg2, leg1, leg2.conj(), leg1, leg2.conj()])
    af = yastn.fuse_legs(a, axes=((1, 2), (3, 0), (4, 5)), mode='meta')
    b = trace_vs_numpy(a, axes=((1, 2), (3, 0)))
    bf = trace_vs_numpy(af, axes=(0, 1)).unfuse_legs(axes=0)
    assert yastn.norm(bf - b) < tol

    # hard-fusion

    a = yastn.randC(config=config_U1, legs=[leg1.conj(), leg2, leg1, leg2.conj(), leg1, leg2.conj()])
    af = yastn.fuse_legs(a, axes=((1, 2), (3, 0), (4, 5)), mode='hard')
    b = trace_vs_numpy(a, axes=((1, 2), (3, 0)))
    bf = trace_vs_numpy(af, axes=(0, 1)).unfuse_legs(axes=0)
    assert yastn.norm(bf - b) < tol

    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 1), D=(1, 2)),
            yastn.Leg(config_U1, s=1, t=(-1, 2), D=(4, 6)),
            yastn.Leg(config_U1, s=1, t=(-1, 1), D=(1, 2)),
            yastn.Leg(config_U1, s=-1, t=(1, 2), D=(5, 6)),
            yastn.Leg(config_U1, s=1, t=(1, 2), D=(3, 4))]
    a = yastn.rand(config=config_U1, legs=legs)
    af = yastn.fuse_legs(a, axes=((1, 2), (3, 0), 4), mode='hard')
    b = trace_vs_numpy(a, axes=((1, 2), (3, 0)))
    bf = trace_vs_numpy(af, axes=(0, 1))
    assert yastn.norm(bf - b) < tol

    a = yastn.Tensor(config=config_U1, s=(1, -1, 1, 1, -1, -1))
    a.set_block(ts=(1, 2, 0, 2, 1, 0), Ds=(2, 3, 4, 3, 2, 4), val='rand')
    a.set_block(ts=(2, 1, 1, 1, 2, 1), Ds=(6, 5, 4, 5, 6, 4), val='rand')
    a.set_block(ts=(3, 2, 1, 2, 2, 2), Ds=(1, 3, 4, 3, 6, 2), val='rand')
    af = yastn.fuse_legs(a, axes=((0, 1), 2, (4, 3), 5), mode='hard')
    b = trace_vs_numpy(a, axes=((0, 1), (4, 3)))
    bf = trace_vs_numpy(af, axes=(0, 2))
    assert yastn.norm(bf - b) < tol

    aff = yastn.fuse_legs(af, axes=((0, 1), (2, 3)), mode='hard')
    b = trace_vs_numpy(a, axes=((0, 1, 2), (4, 3, 5)))
    bff = trace_vs_numpy(aff, axes=(0, 1))
    assert yastn.norm(bff - b) < tol

    a = yastn.Tensor(config=config_U1, s=(1, -1, 1, -1, 1, -1, 1, -1))
    a.set_block(ts=(1, 1, 2, 2, 0, 0, 1, 1), Ds=(1, 1, 2, 2, 4, 4, 1, 1), val='rand')
    a.set_block(ts=(2, 1, 1, 2, 0, 0, 1, 1), Ds=(2, 1, 1, 2, 4, 4, 1, 1), val='rand')
    a.set_block(ts=(3, 1, 1, 2, 1, 1, 0, 1), Ds=(3, 1, 1, 2, 1, 1, 4, 1), val='rand')
    af = yastn.fuse_legs(a, axes=((0, 2), (1, 3), (4, 6), (5, 7)), mode='hard')
    aff = yastn.fuse_legs(af, axes=((0, 2), (1, 3)), mode='hard')

    b = trace_vs_numpy(a, axes=((0, 2, 4, 6), (1, 3, 5, 7)))
    bf = trace_vs_numpy(af, axes=((0, 2), (1, 3)))
    bff = trace_vs_numpy(aff, axes=(0, 1))
    assert yastn.norm(b - bf) < tol
    assert yastn.norm(b - bff) < tol

    b2 = trace_vs_numpy(a, axes=((0, 2), (1, 3)))
    bf2 = trace_vs_numpy(af, axes=(0, 1))
    bf2 = bf2.unfuse_legs(axes=(0, 1)).transpose(axes=(0, 2, 1, 3))
    assert yastn.norm(b2 - bf2) < tol


def test_trace_exceptions(config_kwargs):
    """ test trigerring some expections """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    t1, D1, D2 = (0, 1), (2, 3), (4, 5)
    a = yastn.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                t=[t1, t1, t1, t1, t1, t1], D=[D1, D2, D2, D2, D2, D2])
    with pytest.raises(yastn.YastnError,
                       match='Axis outside of tensor ndim.'):
        a.trace(axes=(0, 6))
    with pytest.raises(yastn.YastnError,
                       match="The same axis in axes"):
        a.trace(axes=((0, 1, 2), (2, 3, 4)))  # The same axis in axes[0] and axes[1]
    with pytest.raises(yastn.YastnError,
                       match="Repeated axis in axes"):
        a.trace(axes=((1, 2), (3, 3)))  # Repeated axis in axes[0] or axes[1].
    with pytest.raises(yastn.YastnError,
                       match="indicate different number of legs."):
        a.trace(axes=((0, 1, 2), (3, 4)))  # axes[0] and axes[1] indicate different number of legs.
    with pytest.raises(yastn.YastnError,
                       match="Signatures do not match."):
        a.trace(axes=((1, 3), (2, 4)))
    with pytest.raises(yastn.YastnError,
                       match="Bond dimensions do not match."):
        a.trace(axes=((0, 1, 2), (3, 4, 5)))
    with pytest.raises(yastn.YastnError,
                       match="Indicated axes of two tensors have different number of meta-fused legs or sub-fusions order."):
        b = a.fuse_legs(axes=((0, 1, 2), (3, 4), 5), mode='meta')
        b.trace(axes=(0, 1))
    with pytest.raises(yastn.YastnError,
                       match="Signatures do not match."):
        b = a.fuse_legs(axes=(0, (1, 3), (2, 4), 5), mode='meta')
        b.trace(axes=(1, 2))
    with pytest.raises(yastn.YastnError,
                       match="Indicated axes of two tensors have different number of hard-fused legs or sub-fusions order."):
        b = a.fuse_legs(axes=((0, 1, 2), (3, 4), 5), mode='hard')
        b.trace(axes=(0, 1))
    with pytest.raises(yastn.YastnError,
                       match="Signatures of hard-fused legs do not match."):
        b = a.fuse_legs(axes=(0, (1, 3), (4, 5), 2), mode='hard')
        b.trace(axes=(1, 2))


@pytest.mark.skipif("'torch' not in config.getoption('--backend')", reason="Uses torch.autograd.gradcheck().")
def test_trace_backward(config_kwargs):
    import torch

    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg0 =  yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(5, 6, 7))
    a = yastn.rand(config=config_U1, legs=[leg0, leg0, leg0.conj(), leg0.conj()])

    target_block = (0, 0, 0, 0)
    target_block_size = a[target_block].size()

    def test_f(block):
        a[target_block] = block
        b = yastn.trace(a, axes=(1, 2))
        return b.norm()

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(), requires_grad=True),)
    assert torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4) #, check_undefined_grad=False)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
    # pytest.main([__file__, "-vs", "--durations=0"])
