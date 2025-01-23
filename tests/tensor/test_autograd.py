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
""" basic autograd operations """
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name

no_numpy_test = pytest.mark.skipif("'np' in config.getoption('--backend')",
                                   reason="numpy backend does not support autograd")


@no_numpy_test
def test_requires_grad(config_kwargs):
    #
    # create a random U1 symmetric tensor. By default, such tensor
    # does not have autograd active. Activate it
    #
    config = yastn.make_config(sym='U1', **config_kwargs)
    leg1 = yastn.Leg(config, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    leg2 = yastn.Leg(config, s=1, t=(-1, 1, 2), D=(2, 4, 5))
    a = yastn.rand(config=config, legs=[leg1, leg1.conj(), leg2.conj(), leg2])
    assert not a.requires_grad
    a.requires_grad_(True)

    #
    # verify, that outputs of functions operating on tensor a return
    # tensors, which also have autograd active
    b = yastn.rand(config=config, legs=[leg1, leg1.conj()])
    c = yastn.tensordot(a, b, axes=((2, 3), (0, 1)))
    assert c.requires_grad


@no_numpy_test
def test_clone_copy(config_kwargs):
    #
    # create random U1 symmetric tensor and flag it for autograd
    #
    config = yastn.make_config(sym='U1', **config_kwargs)
    leg1 = yastn.Leg(config, s=1, t=(-1, 0, 1), D=(2, 3, 4))
    leg2 = yastn.Leg(config, s=1, t=(-1, 1, 2), D=(2, 4, 5))
    a = yastn.rand(config=config, legs=[leg1, leg1.conj(), leg2.conj(), leg2])
    a.requires_grad_(True)

    #
    # Clone the tensor a resulting in a new, numerically identical, tensor b.
    # However, tensors a and b do not share data - their blocks are independent.
    # Further operations on b would be correctly differentiated when computing gradients
    # with respect to a.
    b = a.clone()
    assert b.requires_grad
    assert yastn.are_independent(a, b)

    #
    # Tensor tracked by autograd can be "detached" from the computational
    # graph. This might be useful, if one wishes to perform some computations
    # with the tensor outside of autograd.
    # The original and detached tensor still share data (blocks).
    c = a.detach()
    assert not c.requires_grad
    assert not yastn.are_independent(a, c)

    #
    # Copy of tensor is both detached from the computational graph
    # and does not share data with the original
    d = a.copy()
    assert not d.requires_grad
    assert yastn.are_independent(a, d)


@no_numpy_test
def test_grad_dense(config_kwargs):
    config = yastn.make_config(sym='none', **config_kwargs)
    H = yastn.Tensor(config=config, s=(-1, -1, 1, 1))
    H.set_block(Ds=(2, 2, 2, 2), val='zeros')
    inds= [(0, 0, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)]
    vals= [0.25, -0.25, 0.5, 0.5, -0.25, 0.25]
    for t, val in zip(inds, vals):
        H[()][t] = val
    v = yastn.Tensor(config=config, s=(-1, -1))
    v.set_block(Ds=(2, 2), val='zeros')
    inds_v = [(0, 0), (0, 1), (1, 0), (1, 1)]
    vals_v = [0.1, 0.9, -0.9, 0.1]
    for t, val in zip(inds_v, vals_v):
        v[()][t] = val

    assert v.requires_grad is False
    v.requires_grad_()
    assert v.requires_grad is True

    Hv = yastn.tensordot(H, v, ((2, 3), (0, 1)))
    vHv = yastn.tensordot(v, Hv, ((0, 1), (0, 1)), conj=(1, 0))
    vv = yastn.vdot(v,v)
    vHv_vv = vHv / vv
    loss = vHv_vv.to_number()

    loss.backward()
    expected_grad= [[0.12046400951814398, -0.013384889946460365],
                    [0.013384889946460365, 0.12046400951814398]]
    g = yastn.Tensor(config=config, s=(-1, -1))
    g.set_block(Ds=(2, 2), val=expected_grad)

    for t in inds_v:
        assert pytest.approx(v.grad()[()][t].item(), rel=tol) == g[()][t].item()


@no_numpy_test
def test_grad_U1(config_kwargs):
    config = yastn.make_config(sym='U1', **config_kwargs)
    H = yastn.Tensor(config=config, s=(-1, -1, 1, 1))
    ta = [(-1, -1, -1, -1), (-1, 1, -1, 1), (-1, 1, 1, -1), (1, -1, -1, 1), (1, -1, 1, -1), (1, 1, 1, 1)]
    ba = [0.25, -0.25, 0.5, 0.5, -0.25, 0.25]
    for t, b in zip(ta, ba):
        H.set_block(ts=t, Ds=(1, 1, 1, 1), val=b)
    v = yastn.Tensor(config=config, s=(-1, -1, 1))
    tv = [(-1, -1, -2), (-1, 1, 0), (1, -1, 0), (1, 1, 2)]
    bv = [0.1, 0.9, -0.9, 0.1]
    for t, b in zip(tv, bv):
        v.set_block(ts=t, Ds=(1, 1, 1), val=b)

    assert v.requires_grad is False
    v.requires_grad_()
    assert v.requires_grad is True

    Hv = yastn.tensordot(H, v, ((2, 3), (0, 1)))
    vHv = yastn.tensordot(v, Hv, ((0, 1, 2), (0, 1, 2)), conj=(1, 0))
    vv = yastn.vdot(v,v)
    vHv_vv = vHv / vv
    loss = vHv_vv.to_number()

    loss.backward()
    expected_grad = [0.12046400951814398, -0.013384889946460365, 0.013384889946460365, 0.12046400951814398]
    for t, g in zip(tv, expected_grad):
        assert pytest.approx(v.grad()[t].item(), rel=tol) == g


if __name__ == '__main__':
    # pytest.main([__file__, "-vs", "--durations=0"])
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
