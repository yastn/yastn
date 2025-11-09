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

tol = 1e-12  #pylint: disable=invalid-name

torch_test = pytest.mark.skipif("'torch' not in config.getoption('--backend')",
                                reason="Uses torch.autograd.gradcheck().")


def _test_tensordot_grad(a,b,axes):
    refcfg_args= a.config._asdict()
    refcfg_args["backend"]= "torch"
    refcfg= yastn.make_config(**refcfg_args)

    ref_a= a.copy()
    ref_a.config= refcfg
    ref_b= b.copy()
    ref_b.config= refcfg

    ref_a.requires_grad_(True)
    ref_b.requires_grad_(True)
    ref_ab= yastn.tensordot(ref_a, ref_b, axes=axes)
    ref_cost_f= ref_ab.norm()
    ref_cost_f.backward()

    a.requires_grad_(True)
    b.requires_grad_(True)
    if not (a.grad()._data is None): a.grad()._data.zero_() 
    if not (b.grad()._data is None): b.grad()._data.zero_()
    ab= yastn.tensordot(a, b, axes=axes)
    cost_f= ab.norm()
    cost_f.backward()

    assert (ref_cost_f.item() - cost_f.item()) < tol
    assert np.allclose( ref_a.grad().to_numpy(), a.grad().to_numpy(), rtol=1e-05, atol=1e-08 )
    assert np.allclose( ref_b.grad().to_numpy(), b.grad().to_numpy(), rtol=1e-05, atol=1e-08 )


@torch_test
@pytest.mark.parametrize("dtype",["float64","complex128","float32","complex64"])
def test_tensordot_fuse_hard_backward_0(config_kwargs,dtype):
    import torch
    torch.manual_seed(1)
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    
    ref_cfg= config_kwargs
    ref_cfg["backend"]= "torch"
    refcfg_U1 = yastn.make_config(sym='U1', **ref_cfg)

    config_U1.backend.random_seed(seed=0)
    t1 = (-1, 0, 1)
    D1 = (1, 1, 1)
    #

    a = yastn.rand(config=config_U1, s=(-1, 1, 1),
                t=(t1, t1, t1), D=(D1, D1, D1,), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(1, 1, 1),
                t=(t1, t1, t1), D=(D1, D1, D1), dtype=dtype)
    
    _test_tensordot_grad(a,b.conj(),((2, 1,), (1, 2)))

@torch_test
@pytest.mark.parametrize("dtype",["float64","complex128","float32","complex64"])
def test_tensordot_fuse_hard_backward_01(config_kwargs,dtype):
    import torch
    torch.manual_seed(1)
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    
    ref_cfg= config_kwargs
    ref_cfg["backend"]= "torch"
    refcfg_U1 = yastn.make_config(sym='U1', **ref_cfg)

    config_U1.backend.random_seed(seed=0)
    t1 = (0, 1)
    D1= (1,1)
    #

    a = yastn.rand(config=config_U1, s=(-1, 1, ),
                t=(t1, t1), D=(D1, D1,), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(-1, 1,),
                t=(t1, t1,), D=(D1, D1), dtype=dtype)
    
    _test_tensordot_grad(a,b, ((1,), (0,)))


@torch_test
@pytest.mark.parametrize("dtype",["float64","complex128","float32","complex64"])
def test_tensordot_fuse_hard_backward_02(config_kwargs,dtype):
    import torch
    torch.manual_seed(1)
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    
    ref_cfg= config_kwargs
    ref_cfg["backend"]= "torch"
    refcfg_U1 = yastn.make_config(sym='U1', **ref_cfg)

    config_U1.backend.random_seed(seed=0)
    t1 = (0,)
    D1= (1,)
    #

    a = yastn.rand(config=config_U1, s=(-1, 1, ),
                t=(t1, t1), D=(D1, D1,), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(-1, 1,),
                t=(t1, t1,), D=(D1, D1), dtype=dtype)
    
    _test_tensordot_grad(a,b, ((1,), (0,)))


@torch_test
@pytest.mark.parametrize("dtype",["float64","complex128","float32","complex64"])
def test_tensordot_fuse_hard_backward_1(config_kwargs,dtype):
    import torch
    torch.manual_seed(1)
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    
    ref_cfg= config_kwargs
    ref_cfg["backend"]= "torch"
    refcfg_U1 = yastn.make_config(sym='U1', **ref_cfg)

    config_U1.backend.random_seed(seed=0)
    t1, t2 = (-1, 0, 1), (-1, 0, 1)
    D1, D2 = (2, 2, 2), (2, 2, 2)
    #

    a = yastn.rand(config=config_U1, s=(-1, 1, ),
                t=(t1, t1), D=(D1, D1,), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(1, 1,),
                t=(t1, t1,), D=(D1, D1), dtype=dtype)
    
    _test_tensordot_grad(a,b.conj(), ((1,), (0,)))


@torch_test
def test_tensordot_fuse_hard_backward_12(config_kwargs):
    import torch
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    
    ref_cfg= config_kwargs
    ref_cfg["backend"]= "torch"
    refcfg_U1 = yastn.make_config(sym='U1', **ref_cfg)

    config_U1.backend.random_seed(seed=0)
    t1= (-1, 0, 1)
    D1= (2, 2, 2)
    #
    dtype = 'float64'
    a = yastn.rand(config=config_U1, s=(-1, 1, ),
                t=(t1, t1), D=(D1, D1,), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(1, 1,),
                t=(t1, t1,), D=(D1, D1), dtype=dtype)
    
    _test_tensordot_grad(a,b,axes=(0,0))

@torch_test
def test_tensordot_fuse_hard_backward_13(config_kwargs):
    import torch
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    
    ref_cfg= config_kwargs
    ref_cfg["backend"]= "torch"
    refcfg_U1 = yastn.make_config(sym='U1', **ref_cfg)

    config_U1.backend.random_seed(seed=0)
    t1= (-1, 0, 1)
    D1= (2, 2, 2)
    #
    dtype = 'float64'
    a = yastn.rand(config=config_U1, s=(-1, 1, ),
                t=(t1, t1), D=(D1, D1,), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(1, -1,),
                t=(t1, t1,), D=(D1, D1), dtype=dtype)
    
    _test_tensordot_grad(a,b,axes=(1,1))


@torch_test
def test_tensordot_fuse_hard_backward_2(config_kwargs):
    import torch
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    
    ref_cfg= config_kwargs
    ref_cfg["backend"]= "torch"
    refcfg_U1 = yastn.make_config(sym='U1', **ref_cfg)

    config_U1.backend.random_seed(seed=0)
    t1, t2 = (-1, 0, 1), (-1, 0, 1),
    D1, D2 = (1, 1, 1), (2, 2, 2),
    #
    dtype = 'float64'
    a = yastn.rand(config=config_U1, s=(-1, 1, 1),
                t=(t1, t1, t2), D=(D1, D2, D1), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(1, 1, 1),
                t=(t1, t1, t2), D=(D1, D2, D1), dtype=dtype)
    
    _test_tensordot_grad(a,b.conj(),axes=((2,1), (0,1)))

@torch_test
def test_tensordot_fuse_hard_backward_22(config_kwargs):
    import torch
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    
    ref_cfg= config_kwargs
    ref_cfg["backend"]= "torch"
    refcfg_U1 = yastn.make_config(sym='U1', **ref_cfg)

    config_U1.backend.random_seed(seed=0)
    t1 = (-1, 0, 1)
    D1, D2 = (1, 1, 1), (2, 2, 2),
    #
    dtype = 'float64'
    a = yastn.rand(config=config_U1, s=(-1, 1, 1),
                t=(t1, t1, t1), D=(D1, D2, D1), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(1, -1, 1),
                t=(t1, t1, t1), D=(D1, D2, D1), dtype=dtype)
    
    _test_tensordot_grad(a,b,axes=((0,1), (0,1)))

@torch_test
def test_tensordot_fuse_hard_backward_23(config_kwargs):
    import torch
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    
    ref_cfg= config_kwargs
    ref_cfg["backend"]= "torch"
    refcfg_U1 = yastn.make_config(sym='U1', **ref_cfg)

    config_U1.backend.random_seed(seed=0)
    t1, t2 = (-1, 0, 1), (-1, 0, 1),
    D1, D2 = (2, 2, 2), (2, 2, 2),
    #
    dtype = 'float64'
    a = yastn.rand(config=config_U1, s=(-1, 1, 1),
                t=(t1, t1, t2), D=(D1, D2, D1), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(1, -1, 1),
                t=(t1, t1, t2), D=(D1, D2, D1), dtype=dtype)
    
    _test_tensordot_grad(a,b,axes=((0,1), (0,1)))


@torch_test
@pytest.mark.parametrize("dtype",["float64","complex128","float32","complex64"])
def test_tensordot_fuse_hard_backward_3(config_kwargs,dtype):
    import torch
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    
    ref_cfg= config_kwargs
    ref_cfg["backend"]= "torch"
    refcfg_U1 = yastn.make_config(sym='U1', **ref_cfg)

    config_U1.backend.random_seed(seed=0)
    t1, t2 = (-1, 0, 1), (-1, 0, 1),
    D1, D2 = (2, 2, 2), (2, 2, 2),
    
    a = yastn.rand(config=config_U1, s=(-1, 1, 1),
                t=(t1, t1, t2), D=(D1, D2, D2), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(1, 1, 1),
                t=(t1, t1, t2), D=(D1, D2, D2), dtype=dtype)
    
    _test_tensordot_grad(a,b.conj(),axes=((2,1),(1,0)))

@torch_test
def test_tensordot_fuse_hard_backward_4(config_kwargs):
    import torch
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    
    ref_cfg= config_kwargs
    ref_cfg["backend"]= "torch"
    refcfg_U1 = yastn.make_config(sym='U1', **ref_cfg)

    config_U1.backend.random_seed(seed=0)
    t1, t2, t3 = (-1, 0, 1), (-2, 0, 2), (-3, 0, 3)
    D1, D2, D3 = (2, 2, 2), (2, 2, 2), (2, 2, 2)
    #
    dtype = 'float64'
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t1, t1, t2, t2, t3, t3), D=(D1, D2, D2, D1, D1, D2), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t2, t2, t3, t3, t1, t1), D=(D2, D3, D1, D3, D1, D2), dtype=dtype)
    
    axes= ((1, 5, 2, 3, ), (1, 4, 2, 0))
    _test_tensordot_grad(a,b.conj(),axes=axes)

@torch_test
def test_tensordot_fuse_hard_Z2xU1(config_kwargs):
    """ test tensordot for different symmetries. """
    # Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    t1 = [(0, -1), (0, 1), (1, -1), (1, 1)]
    t2 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yastn.rand(config=config_Z2xU1, s=(-1, 1, 1, -1),
                  t=(t1, t1, t1, t1),
                  D=((1, 2, 2, 4), (9, 4, 3, 2), (5, 6, 7, 8), (7, 8, 9, 10)))
    b = yastn.rand(config=config_Z2xU1, s=(1, -1, 1),
                  t=(t1, t1, t2),
                  D=((1, 2, 2, 4), (9, 4, 3, 2), (5, 6, 7, 8,)))

    _test_tensordot_grad(a,b, axes=((0, 1), (0, 1)))
    _test_tensordot_grad(b,a, axes=((1, 0), (1, 0)))

    # corner cases;
    a = yastn.rand(config=config_Z2xU1, s=(-1, 1),
                  t=(t1, t1), D=((1, 2, 3, 4), (2, 3, 4, 5)))
    b = yastn.rand(config=config_Z2xU1, s=(-1, 1),
                  t=(t2, t2), D=((1, 2, 3, 4), (2, 3, 4, 5)))
    #
    # no matching charges
    with pytest.raises(RuntimeError,
                       match="element 0 of tensors does not require grad and does not have a grad_f"):
        _test_tensordot_grad(b,a, axes=((1,), (0)))


@torch_test
@pytest.mark.parametrize("dtype",["float64", "complex128","float32","complex64"])
def test_tensordot_fuse_hard_gradcheck(config_kwargs,dtype):
    import torch
    # U1
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)

    config_U1.backend.random_seed(seed=0)
    t1, t2, t3 = (-1, 0, 1), (-2, 0, 2), (-3, 0, 3)
    D1, D2, D3 = (1, 2, 2), (2, 2, 2), (2, 2, 2)
    #

    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t1, t1, t2, t2, t3, t3), D=(D1, D2, D2, D1, D1, D2), dtype=dtype)
    b = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1, 1),
                t=(t2, t2, t3, t3, t1, t1), D=(D2, D3, D1, D3, D1, D2), dtype=dtype)
    fb = yastn.fuse_legs(b, axes=(0, (4, 3, 1), (5, 2)), mode='hard')
    ffb = yastn.fuse_legs(fb, axes=(0, (2, 1)), mode='hard')

    target_block = (0, 0, 0, 0, 0, 0)
    target_block_size = a[target_block].size()

    def test_f_native(block):
        a.set_block(ts=target_block, val=block)
        ab = yastn.tensordot(a, b.conj(), axes=((1, 2, 3, 4, 5), (1, 2, 3, 4, 5)))
        ab = ab.norm()
        return ab

    def test_f_fused(block):
        a.set_block(ts=target_block, val=block)
        fa = yastn.fuse_legs(a, axes=(0, (4, 3, 1), (5, 2)), mode='hard')
        ffa = yastn.fuse_legs(fa, axes=(0, (2, 1)), mode='hard')
        ffab = yastn.tensordot(ffa.conj(), ffb, axes=(1, 1))
        ffab = ffab.norm()
        return ffab

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(), requires_grad=True),)
    assert torch.autograd.gradcheck(test_f_native, op_args, eps=1e-6, atol=1e-4)

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(), requires_grad=True),)
    assert torch.autograd.gradcheck(test_f_fused, op_args, eps=1e-6, atol=1e-4)


@torch_test
@pytest.mark.parametrize("dtype",["float64", "complex128","float32","complex64"])
def test_tensordot_gradcheck(config_kwargs,dtype):
    import torch

    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    config_U1.backend.random_seed(seed=0)
    
    a = yastn.rand(config=config_U1, s=(-1, -1, 1, 1),
                t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                D=[(2, 3), (4, 5), (4, 3), (2, 1)], dtype=dtype)
    b1 = yastn.rand(config=config_U1, s=(1, 1, -1, -1),  # charges match exactly
                t=[(0, 1), (0, 1), (0, 1), (0, 1)],
                D=[(2, 3), (4, 5), (4, 3), (2, 1)], dtype=dtype)
    b2 = yastn.rand(config=config_U1, s=(1, 1, -1, -1),  # some block mismatches
                t=[(0, 2), (1, 2), (0, 1, 2), (0, 1, 2)],
                D=[(2, 3), (5, 6), (4, 3, 4), (2, 1, 3)], dtype=dtype)
    b3 = yastn.rand(config=config_U1, s=(1, 1, -1, -1),  # no matching blocks in a @ b
                t=[(0, 2), (-1, 2), (-1,  2), (0, 1, 2)],
                D=[(2, 3), (5, 6), (4, 4), (2, 1, 3)], dtype=dtype)

    for b in [b1, b2, b3]:
        target_block = (0, 1, 1, 0)
        target_block_size = a[target_block].size()

        def test_f(block):
            a.set_block(ts=target_block, val=block)
            ab = yastn.tensordot(a, b, axes=((1, 2), (1, 2)))  # 2 outgoing legs are a problem
            ab = ab.norm()
            return ab

        op_args = (torch.randn(target_block_size, dtype=a.get_dtype(), requires_grad=True),)
        assert torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--tensordot_policy", "fuse_to_matrix"])
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--tensordot_policy", "fuse_contracted"])
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--tensordot_policy", "no_fusion"])
