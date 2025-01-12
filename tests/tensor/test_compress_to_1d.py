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


def run_simple_compress(a):
    r1d, meta = a.compress_to_1d()
    b = yastn.decompress_from_1d(r1d, meta=meta)
    assert yastn.norm(a - b) < 1e-12


def run_compress_with_meta(a, a_reference) :
    _, meta = a_reference.compress_to_1d()  # read meta here
    r1d, _ = a.compress_to_1d(meta)  # use meta here
    c = yastn.decompress_from_1d(r1d, meta=meta)
    assert yastn.norm(a - c) < 1e-12


def test_compress_to_1d_basic(config_kwargs):
    # 3d dense
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    a = yastn.rand(config=config_dense, s=(-1, 1, 1), D=(1, 2, 3))
    run_simple_compress(a)

    # 0d U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.ones(config=config_U1) # s=() # t=(), D=()
    run_simple_compress(a)

    # diagonal Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    leg = yastn.Leg(config_Z2xU1, s=1, t=((0, -1), (1, 0), (0, 1)), D=(2, 3, 4))
    a = yastn.rand(config=config_Z2xU1, isdiag=True, legs=leg)
    run_simple_compress(a)
    run_compress_with_meta(a, 2 * a)


def test_compress_to_1d_embed(config_kwargs):
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
    run_simple_compress(a)
    # here have to add zero blocks
    run_compress_with_meta(a, a_ref)
    run_compress_with_meta(af, af_ref)
    run_compress_with_meta(aff, aff_ref)


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
    _, meta = a_ref.compress_to_1d()  # read meta here
    _, metaf = af_ref.compress_to_1d()  # read meta here

    target_block = (2, 0, 1, 1)
    target_block_size = a[target_block].size()

    def test_f(block):
        a[target_block] = block
        r1d, _ = yastn.compress_to_1d(a, meta)
        c = yastn.decompress_from_1d(r1d, meta=meta)
        res = c.norm()
        return res

    def test_ff(block):
        a[target_block] = block
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        r1d, _ = yastn.compress_to_1d(af, metaf)
        c = yastn.decompress_from_1d(r1d, meta=meta)
        res = c.norm()
        return res

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(), requires_grad=True),)
    assert torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4, check_undefined_grad=False)

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(), requires_grad=True),)
    assert torch.autograd.gradcheck(test_ff, op_args, eps=1e-6, atol=1e-4, check_undefined_grad=False)


def test_compress_to_1d_exceptions(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
    a.set_block(ts=(2, 0, 1, 1), Ds=(1, 2, 3, 4))
    _, a_meta = a.compress_to_1d()

    ad = yastn.eye(config=config_U1, legs=yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3)))
    _, ad_meta = ad.compress_to_1d()

    with pytest.raises(yastn.YastnError,
                       match="Tensor signature do not match meta."):
        b = yastn.Tensor(config=config_U1, s=(1, 1, 1, 1))
        _ = b.compress_to_1d(a_meta)
    with pytest.raises(yastn.YastnError,
                       match="Tensor charge do not match meta."):
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1), n=1)
        _ = b.compress_to_1d(a_meta)
    with pytest.raises(yastn.YastnError,
                       match="Tensor diagonality do not match meta."):
        b = yastn.Tensor(config=config_U1, s=(1, -1))
        _ = b.compress_to_1d(ad_meta)
    with pytest.raises(yastn.YastnError,
                       match="Tensor meta-fusion structure do not match meta."):
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
        b = b.fuse_legs(axes=((0, 1), (2, 3)), mode='meta')
        _ = b.compress_to_1d(a_meta)
    with pytest.raises(yastn.YastnError,
                       match="Tensor has blocks that do not appear in meta."):
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
        b.set_block(ts=(1, -1, 2, 0), Ds=(2, 3, 3, 4))
        _ = b.compress_to_1d(a_meta)
    with pytest.raises(yastn.YastnError,
                       match="Tensor fused legs do not match meta."):
        af = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        _, af_meta = af.compress_to_1d()
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
        b.set_block(ts=(1, -1, 2, 0), Ds=(2, 3, 3, 4))
        bf = b.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
        _ = bf.compress_to_1d(af_meta)
    with pytest.raises(yastn.YastnError,
                       match="Bond dimensions do not match meta."):
        b = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
        b.set_block(ts=(2, 0, 1, 1), Ds=(2, 3, 3, 4))
        _ = b.compress_to_1d(a_meta)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
    # pytest.main([__file__, "-vs", "--durations=0"])
