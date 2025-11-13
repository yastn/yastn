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
""" fill_tensor (which is called in: rand, zeros, ones), yastn.to_numpy() """
import pytest
import yastn


def run_to_dict_simple(a):
    r1d, meta = yastn.split_data_and_meta(a.to_dict(level=0))
    b = yastn.Tensor.from_dict(yastn.combine_data_and_meta(r1d, meta))
    assert yastn.norm(a - b) < 1e-12


def run_to_dict_with_meta(a, a_reference) :
    _, meta = yastn.split_data_and_meta(a_reference.to_dict(level=0))
    r1d, _ = yastn.split_data_and_meta(a.to_dict(level=0, meta=meta))
    c = yastn.Tensor.from_dict(yastn.combine_data_and_meta(r1d, meta))
    assert yastn.norm(a - c) < 1e-12


def test_to_dict_basic(config_kwargs):
    # 3d dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    a = yastn.rand(config=config_dense, s=(-1, 1, 1), D=(1, 2, 3))
    run_to_dict_simple(a)

    # 0d U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.ones(config=config_U1) # s=() # t=(), D=()
    run_to_dict_simple(a)

    # diagonal Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    leg = yastn.Leg(config_Z2xU1, s=1, t=((0, -1), (1, 0), (0, 1)), D=(2, 3, 4))
    a = yastn.rand(config=config_Z2xU1, isdiag=True, legs=leg)
    run_to_dict_simple(a)
    run_to_dict_with_meta(a, 2 * a)


def test_to_dict_embed(config_kwargs):
    """ test embedding zeros to match another tensor """
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
    a.set_block(ts=(2, 0, 1, 1), Ds=(1, 2, 3, 4))
    a.set_block(ts=(0, 1, 0, -1), Ds=(5, 6, 7, 8))
    a.set_block(ts=(1, -1, 2, 0), Ds=(2, 1, 2, 1))
    af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
    aff = af.fuse_legs(axes=[(0, 1)], mode='meta')
    #
    # creates tensor matching a filling in all possible matching blocks with zeros.
    a_ref = yastn.zeros(config=a.config, legs=a.get_legs())
    af_ref = a_ref.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
    aff_ref = af_ref.fuse_legs(axes=[(0, 1)], mode='meta')
    #
    run_to_dict_simple(a)
    # here have to add zero blocks
    run_to_dict_with_meta(a, a_ref)
    run_to_dict_with_meta(af, af_ref)
    run_to_dict_with_meta(aff, aff_ref)


@pytest.mark.skipif("'torch' not in config.getoption('--backend')", reason="Uses torch.autograd.gradcheck().")
def test_mask_backward(config_kwargs):
    import torch

    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    config_U1.backend.random_seed(seed=0)

    a = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
    a.set_block(ts=(2, 0, 1, 1), Ds=(1, 2, 3, 4))
    a.set_block(ts=(0, 1, 0, -1), Ds=(5, 6, 7, 8))
    a.set_block(ts=(1, -1, 2, 0), Ds=(2, 1, 2, 1))
    #
    # creates tensor with all posible blocks consistent with the legs of a.
    a_ref = yastn.zeros(config=a.config, legs=a.get_legs())
    af_ref = a_ref.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
    _, meta = yastn.split_data_and_meta(a_ref.to_dict(level=0))
    _, metaf = yastn.split_data_and_meta(af_ref.to_dict(level=0))

    target_block = (2, 0, 1, 1)
    target_block_size = a[target_block].size()

    def test_f(block):
        a[target_block] = block
        r1d, _ = yastn.split_data_and_meta(a.to_dict(level=0, meta=meta))
        c = yastn.Tensor.from_dict(yastn.combine_data_and_meta(r1d, meta))
        res = c.norm()
        return res

    def test_ff(block):
        a[target_block] = block
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        r1d, _ = yastn.split_data_and_meta(af.to_dict(level=0, meta=metaf))
        c = yastn.Tensor.from_dict(yastn.combine_data_and_meta(r1d, meta))
        res = c.norm()
        return res

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(), requires_grad=True),)
    assert torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4, check_undefined_grad=False)

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(), requires_grad=True),)
    assert torch.autograd.gradcheck(test_ff, op_args, eps=1e-6, atol=1e-4, check_undefined_grad=False)


def test_to_dict_exceptions(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
    a.set_block(ts=(2, 0, 1, 1), Ds=(1, 2, 3, 4))
    _, a_meta = yastn.split_data_and_meta(a.to_dict(level=0))

    ad = yastn.eye(config=config_U1, legs=yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3)))
    _, ad_meta = yastn.split_data_and_meta(ad.to_dict(level=0))

    with pytest.raises(yastn.YastnError,
                      match="Tensor is inconsistent with meta: Signatures do not match."):
        b = yastn.Tensor(config=config_U1, s=(1, 1, 1, 1))
        _ = b.to_dict(meta=a_meta)
    with pytest.raises(yastn.YastnError,
                       match="Tensor is inconsistent with meta: Tensor charges do not match."):
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1), n=1)
        _ = b.to_dict(meta=a_meta)
    with pytest.raises(yastn.YastnError,
                       match="Tensor is inconsistent with meta: Cannot add diagonal tensor to non-diagonal tensor."):
        b = yastn.Tensor(config=config_U1, s=(1, -1))
        _ = b.to_dict(meta=ad_meta)
    with pytest.raises(yastn.YastnError,
                       match="Tensor is inconsistent with meta: Tensors have different number of legs."):
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
        b = b.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
        _ = b.to_dict(meta=a_meta)
    with pytest.raises(yastn.YastnError,
                       match="Tensor is inconsistent with meta."):
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
        b.set_block(ts=(1, -1, 2, 0), Ds=(2, 3, 3, 4))
        _ = b.to_dict(meta=a_meta)
    with pytest.raises(yastn.YastnError,
                       match="Tensor is inconsistent with meta."):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        _, af_meta = yastn.split_data_and_meta(af.to_dict())
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
        b.set_block(ts=(1, -1, 2, 0), Ds=(2, 3, 3, 4))
        bf = b.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        _ = bf.to_dict(meta=af_meta)
    with pytest.raises(yastn.YastnError,
                       match="Tensor is inconsistent with meta: Bond dimensions do not match."):
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
        b.set_block(ts=(2, 0, 1, 1), Ds=(2, 3, 3, 4))
        _ = b.to_dict(meta=a_meta)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
