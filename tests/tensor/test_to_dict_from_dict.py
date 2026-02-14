# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
""" Tensor.to_dict() Tensor.from_dict() yastn.save_to_hdf5() yastn.load_from_hdf5(). """
import os
import numpy as np
import pytest
import yastn


def are_identical_tensors(a, b):
    da, db = a.__dict__, b.__dict__
    assert da.keys() == db.keys()
    for k in da:
        if k != '_data':
            assert da[k] == db[k]
    assert type(a.data) == type(b.data)
    assert np.allclose(a.to_numpy(), b.to_numpy())


@pytest.mark.parametrize("resolve_transpose", [True,False])
def test_to_from_dict(resolve_transpose, config_kwargs):
    config = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config, s=1, t=(0, 1, 2), D= (3, 5, 2)),
            yastn.Leg(config, s=-1, t=(0, 1, 3), D= (1, 2, 3)),
            yastn.Leg(config, s=1, t=(-1, 0, 1), D= (2, 3, 4)),
            yastn.Leg(config, s=1, t=(-1, 0, 1), D= (4, 3, 2))]

    a = yastn.rand(config, legs=legs)
    a = a.fuse_legs(axes=(0, (1, 2), 3), mode='hard')
    a = a.fuse_legs(axes=(0, (1, 2)), mode='meta')

    for level, ind in zip([0, 1, 2], [False, False, True]):
        d = a.to_dict(level=level,resolve_transpose=resolve_transpose)
        b = yastn.Tensor.from_dict(d)
        assert b.is_consistent()

        data, meta = yastn.split_data_and_meta(d)
        dd = yastn.combine_data_and_meta(data, meta)
        c = yastn.Tensor.from_dict(dd)
        d = yastn.from_dict(dd)
        assert d.is_consistent()
        #
        are_identical_tensors(a, b)
        are_identical_tensors(a, c)
        are_identical_tensors(a, d)
        assert yastn.are_independent(a, b) == ind
        #
        # the result below depends on the backend: TODO
        # assert yastn.are_independent(b, c) == ind  # for numpy
        # # assert yastn.are_independent(b, c) == False  # for torch
        # torch.as_tensor and numpy.array have different behavior in creating a copy


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
    # here have to add zero blocks
    check_dict_with_meta(a, a_ref)
    check_dict_with_meta(af, af_ref)
    check_dict_with_meta(aff, aff_ref)


def check_dict_with_meta(a, a_reference) :
    _, meta = yastn.split_data_and_meta(a_reference.to_dict(level=0))
    r1d, _ = yastn.split_data_and_meta(a.to_dict(level=0, meta=meta))
    c = yastn.Tensor.from_dict(yastn.combine_data_and_meta(r1d, meta))
    assert yastn.norm(a - c) < 1e-12


@pytest.mark.skipif("'torch' not in config.getoption('--backend')", reason="Uses torch.autograd.gradcheck().")
def test_to_dict_backward(config_kwargs):
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


def check_to_numpy(a1, config):
    """ save/load to numpy and tests consistency."""
    d1 = a1.to_dict()
    a2 = 2 * a1  # second tensor to be saved
    d2 = a2.save_to_dict()
    data = {'tensor1': d1, 'tensor2': d2}  # two tensors to be saved
    np.save('tmp.npy', data)
    ldata = np.load('tmp.npy', allow_pickle=True).item()
    os.remove('tmp.npy')

    b1 = yastn.from_dict(ldata['tensor1'], config=config)
    b2 = yastn.load_from_dict(d=ldata['tensor2'], config=config)

    assert all(yastn.norm(a - b) < 1e-12 for a, b in [(a1, b1), (a2, b2)])
    assert all(b.is_consistent for b in (b1, b2))
    assert all(yastn.are_independent(a, b) for a, b in [(a1, b1), (a2, b2)])
    are_identical_tensors(a1, b1)
    are_identical_tensors(a2.consume_transpose(), b2)


def check_to_hdf5(a, *args):
    """ Test if two Tensor-s have the same values. """
    h5py = pytest.importorskip("h5py")
    try:
        os.remove("tmp.h5")
    except OSError:
        pass
    with h5py.File('tmp.h5', 'w') as f:
        a.save_to_hdf5(f, './')
    with h5py.File('tmp.h5', 'r') as f:
        b = yastn.load_from_hdf5(a.config, f, './')
    os.remove("tmp.h5")
    b.is_consistent()
    assert yastn.are_independent(a, b)
    are_identical_tensors(a.consume_transpose(), b)


@pytest.mark.parametrize("test_f", [check_to_numpy, check_to_hdf5])
def test_save_load(config_kwargs, test_f):
    """ test exporting tensor to native python data-structure,
        that allows robust saving/loading with np.save/load."""
    config_dense = yastn.make_config(sym='none', **config_kwargs)
    a = yastn.rand(config=config_dense)  # s=() i.e. a scalar
    assert a.size == 1
    test_f(a, config_dense)

    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=1, t=(0, 1, 2), D= (3, 5, 2)),
            yastn.Leg(config_U1, s=-1, t=(0, 1, 3), D= (1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D= (2, 3, 4))]

    a = yastn.rand(config=config_U1, legs=legs)
    test_f(a, config_U1)

    a = yastn.randC(config=config_U1, legs=legs, n=1)
    test_f(a, config_U1) # here a is complex

    a = yastn.rand(config=config_U1, isdiag=True, legs=legs[0])
    test_f(a, config_U1)

    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    legs = [yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(1, 2, 3, 4)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, -2), (0, 2)), D=(2, 1)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, -2), (0, 0), (0, 2), (1, -2), (1, 0), (1, 2)), D=(2, 3, 5, 4, 1, 6))]
    a = yastn.ones(config=config_Z2xU1, legs=legs, n=(0, -2))
    test_f(a, config_Z2xU1)

    a = yastn.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    a = a.fuse_legs(axes=((0, 2), 1, (4, 3), 5), mode='hard')
    a = a.fuse_legs(axes=((0, 2), 1, 3), mode='hard')
    a = a.fuse_legs(axes=((0, 2), 1), mode='meta')
    test_f(a, config_U1)


def test_to_dict_exceptions(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)

    a = yastn.Tensor(config=config_U1, s=(-1, 1, 1, 1))
    a.set_block(ts=(2, 0, 1, 1), Ds=(1, 2, 3, 4))
    _, a_meta = yastn.split_data_and_meta(a.to_dict(level=0))

    ad = yastn.eye(config=config_U1, legs=yastn.Leg(config_U1, s=1, t=(0, 1), D=(2, 3)))
    _, ad_meta = yastn.split_data_and_meta(ad.to_dict(level=0))

    with pytest.raises(yastn.YastnError,
                       match="Symmetry rule in config does not match the one in stored in d."):
        config_Z2 = yastn.make_config(sym='Z2', **config_kwargs)
        d = a.to_dict()
        yastn.from_dict(d, config=config_Z2)
    with pytest.raises(yastn.YastnError,
                       match="Fermionic statistics in config does not match the one in stored in d."):
        config_U1_fermionic = yastn.make_config(sym='U1', fermionic=True, **config_kwargs)
        d = a.to_dict()
        yastn.from_dict(d, config=config_U1_fermionic)
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
