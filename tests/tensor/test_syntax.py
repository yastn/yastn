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
""" List supported operations on  yastn.Tensor (not all arguments are shown). """
import pytest
import yastn


def test_syntax_tensor_creation_operations(config_kwargs):
    #
    # Initialize several rank-4 tensors, with the following signature
    #             ___
    #  (-) 0--<--| a |--<--1 (+)
    #  (+) 2-->--|___|-->--3 (-)
    #
    # The signatures can be interpreted as tensor legs being directed: ingoing for (+)
    # or outgoing for (-).
    #
    # The symmetry, U1, is specified in config_U1. We specify the charge
    # sectors on all legs by tuple t, with its first member, t[0], defining charge
    # sectors on the first leg, t[1] on second leg and so on.
    # The corresponding dimensions of each charge sector are specified by tuple D
    # with analogous structure as t.
    #
    # Then, upon creation, all blocks which respect charge conservation will be
    # initialized and filled with either random numbers, ones, or zeros in the examples
    # below.
    #
    # The dtype of the tensor elements as well as the device on which its data
    # reside is given in config_U1.
    #
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg1 = yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(1, 2, 3))
    leg2 = yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6))
    leg3 = yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9))
    leg4 = yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))

    a = yastn.rand(config=config_U1, legs=[leg1, leg2, leg3, leg4])
    b = yastn.ones(config=config_U1, legs=[leg1, leg2, leg3, leg4])
    c = yastn.zeros(config=config_U1, legs=[leg1, leg2, leg3, leg4])

    #
    # The identity tensor behaves as rank-2 tensor with automatic signature (1, -1)
    # or (-1, 1). It is enough to provide charge sectors and their dimensions
    # for single leg, the data for other leg is inferred automatically.
    #
    e = yastn.eye(config=config_U1,legs=leg1)

def test_syntax_create_empty_tensor_and_fill(config_kwargs):
    #
    # Initialize empty rank-4 tensor, with the following signature
    #             ___
    #  (-) 0--<--| a |--<--1 (+)
    #  (+) 2-->--|___|-->--3 (-)
    #
    # The symmetry, U1, is specified in config_U1.
    #
    # Then, initialize some blocks with random values. The charges
    # of the block, ts, are given as a tuple with the length identical to
    # the rank of the tensor. Similarly, the dimensions of the block Ds.
    #
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    d = yastn.Tensor(config=config_U1, s=(-1, 1, 1, -1))
    d.set_block(ts=(1, -1, 2, 0), Ds=(2, 4, 9, 2), val='rand')
    d.set_block(ts=(2, 0, 2, 0), Ds=(3, 3, 9, 2), val='rand')

    #
    # Once the dimension is assigned to charge sector on a leg of the tensor
    # attempt to create block with different dimension will raise an error.
    # In the example above sector with charge 2 on 3rd leg has dimension 9.
    #
    # Attempting to create new block with different dimension for the same
    # sector 2 on 3rd leg throws an error.
    #
    with pytest.raises(yastn.YastnError,
                       match="Inconsistent assignment of bond dimension to some charge."):
        d.set_block(ts=(2, 1, 2, 1), Ds=(3, 3, 10, 2), val='rand')


def test_syntax_basic_algebra(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9)),
            yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))]

    a = yastn.rand(config=config_U1, legs=legs)

    #
    # Tensor can be multiplied by scalar
    #
    tensor = a / 2
    tensor = 2. * a
    tensor = a * 2.

    #
    # Tensors can be added or subtracted assuming their structure is
    # compatible.
    #
    b = yastn.ones(config=config_U1, legs=legs)
    tensor = a + b
    tensor = a - b

    #
    # Attempting to add/subtract two tensors with different total charge,
    # or different dimension of a common charge sector raises exception
    #
    legs[0] = yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(7, 2, 3))
    c = yastn.ones(config=config_U1, legs=legs)

    with pytest.raises(yastn.YastnError,
                       match="Bond dimensions do not match."):
        tensor = a + c

    #
    # Sometimes, a composite operation is faster than the serial execution of
    # individual operations. This happens for linear combination of multiple tensors
    #
    tensor = yastn.add(a, b, a, b, amplitudes=(1, -1, 2, 1))
    tensor = yastn.add(a, b, a, b)  # all amplitudes equall to one.

    #
    # element-wise exponentiation, absolute value, reciprocal i.e. x -> 1/x,
    # square root and its reciprocal x -> 1/sqrt(x)
    #
    tensor = a.exp(step=1)
    tensor = yastn.exp(a, step=1)

    tensor = abs(a)

    tensor = a.reciprocal(cutoff=1e-12)
    tensor = yastn.reciprocal(a, cutoff=1e-12)

    tensor = abs(a).sqrt()
    tensor = yastn.sqrt(abs(a))

    tensor = abs(a).rsqrt(cutoff=1e-12)
    tensor = yastn.rsqrt(abs(a), cutoff=1e-12)


def test_syntax_tensor_export_import_operations(config_kwargs):
    #
    # First, we crate a random U1 symmetric tensor
    # Such tensor is stored as dict of non-zero blocks, indexed by charges
    #
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(7, 8, 9))]

    a = yastn.rand(config=config_U1, legs=legs)
    #
    # Tensors can be serialized directly into basic Python dictionary
    #
    dictionary = a.to_dict()
    tensor = yastn.from_dict(d=dictionary, config=config_U1)  # providing config is optional
    #
    # By applying methods split_data_and_meta and combine_data_and_meta
    # we can further serialize symmetric tensors into 1-D vector,
    # holding raw-data of blocks and dictionary, meta, which holds
    # the symmetric structure of the tensors.
    #
    vector, meta = yastn.split_data_and_meta(a.to_dict(level=0), squeeze=True)
    # assure that tensor structures is embeded in provided meta
    vector, meta = yastn.split_data_and_meta(a.to_dict(level=0, meta=meta), squeeze=True)
    tensor = yastn.Tensor.from_dict(yastn.combine_data_and_meta(vector, meta))



def test_syntax_block_access(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(7, 8, 9))]

    a = yastn.rand(config=config_U1, legs=legs)

    #
    # Directly access block with charges (1, 2, 1).
    #
    a[(1, 2, 1)]
    assert a[(1, 2, 1)].shape == (3, 6, 8)

    #
    # Cannot access non-existing block.
    #
    with pytest.raises(yastn.YastnError,
                       match="Tensor does not have the block specified by key."):
        a[(0, 3, 3)]


def test_syntax_block_tensors(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6))]

    a = yastn.rand(config=config_U1, legs=legs)
    b = yastn.ones(config=config_U1, legs=legs)
    c = yastn.zeros(config=config_U1, legs=legs)
    d = yastn.zeros(config=config_U1, legs=legs)

    # block tensors
    tensor1 = yastn.block({(1, 1): a, (1, 2): b}, common_legs=(0,))
    tensor2 = yastn.block({(1, 1): c, (2, 1): d}, common_legs=(0,))

    result1 = yastn.tensordot(tensor1, tensor2.conj(), axes=((1, 2), (2, 1)))

    result2 = yastn.tensordot(a, c.conj(), axes=((1, 2), (2, 1))) + \
                yastn.tensordot(b, d.conj(), axes=((1, 2), (2, 1)))

    # new tensor filled with ones, matching structure of selected legs -- to be used for e.g. dot
    assert yastn.norm(result1 - result2) < 1e-12


def test_syntax_contraction(config_kwargs):
    # create a set of U1-symmetric tensors
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg1 = yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(1, 2, 3))
    leg2 = yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6))
    leg3 = yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9))
    leg4 = yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))

    a = yastn.rand(config=config_U1, legs=[leg1, leg2, leg3, leg4])
    b = yastn.ones(config=config_U1, legs=[leg1, leg2, leg3, leg4])
    c = yastn.rand(config=config_U1, legs=[leg4.conj(), leg3, leg2.conj()])

    # Contract a and b by two indices. The a tensor is conjugated, which
    # reverses the signature on its indices
    #       __           _                ___
    #  0->-|a*|->-1 1->-|b|->-0 =    0->-|a*b|->-0->2
    #  3->-|__|->-2 2->-|_|->-3   1<-3->-|___|->-3
    #
    # The order of the indices on the resulting tensor is as follows:
    # First, the outgoing indices of a (the first argument to tensordot), then
    # the outgoing indices of tensor b
    tensor = yastn.tensordot(a.conj(), b, axes=((1, 2), (1, 2)))

    # Tensordot can also be invoked also as a function of the tensor itself.
    # Conjugating tensors can be also invoked using conj parameter in tensordot.
    #
    tensor = a.tensordot(b, axes=((1, 2), (1, 2)), conj=(1, 0))

    # If no axes are specified, the outer product of two tensors is returned
    tensor = yastn.tensordot(c, c, axes=((), ()) )
    assert tensor.get_rank() == 6


    # A shorthand notation for the specific contraction
    #      _           _             __
    # 0-<-|a|-<-2     |c|-<-1 = 0-<-|ac|-<-2
    # 1->-|_|->-3 0->-|_|->-2   1->-|  |-<-1->3
    #                               |__|->-2->4
    t0 = yastn.tensordot(a, c, axes=(a.ndim - 1, 0))
    #
    # is the @ operator. For rank-2 tensor it is thus equivalent to matrix multiplication
    t1 = a @ c
    assert yastn.norm(t0 - t1) < 1e-12
    #
    # Utility functions simplifying execution of contractions
    t2 = yastn.ncon([a, c], ((-0, -1, -2, 1), (1, -3, -4)))
    t3 = yastn.einsum('ijkx,xlm->ijklm', a, c)
    assert yastn.norm(t0 - t2) < 1e-12
    assert yastn.norm(t0 - t3) < 1e-12


    # Another special case of tensor contraction is a dot product of vectorized tensors.
    #  __           _
    # |a*|-<-0 0-<-|b| = scalar
    # |  |->-1 1->-| |
    # |  |->-2 2->-| |
    # |__|-<-3 3-<-|_|
    tensor = a.conj().tensordot(b, axes=((0, 1, 2, 3), (0, 1, 2, 3)))
    assert isinstance(tensor,yastn.Tensor)
    #
    # Such single element symmetric Tensor can be converted to a single-element
    # tensor of the backend type, or even further to python scalar.
    number = tensor.to_number()
    python_scalar = tensor.item()
    assert isinstance(python_scalar,float)

    # A shorthand function for computing dot products is vdot
    number = yastn.vdot(a, b)
    number = a.vdot(b)

    # Trace over certain indices can be computed using identically named function.
    # In this case, a2_ijil = a2_jl
    a2 = yastn.tensordot(a.conj(), a, axes=((0, 1), (0, 1)))
    tensor = a2.trace(axes=(0, 2))
    assert tensor.get_rank()==2
    #
    # More pairs of indices can be traced at once a_ijij = scalar
    tensor = a2.trace(axes=((0, 1), (2, 3)))
    number = tensor.to_number()


def test_syntax_noDocs(config_kwargs):
    # initialization
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    # with config that is not imported as usually
    if config_U1.backend.BACKEND_ID == 'np':
        cfg_U1 = yastn.make_config(sym=yastn.sym.sym_U1, backend=yastn.backend.backend_np, default_device=config_U1.default_device)
    elif config_U1.backend.BACKEND_ID == 'torch':
        cfg_U1 = yastn.make_config(sym=yastn.sym.sym_U1, backend=yastn.backend.backend_torch, default_device=config_U1.default_device)
    elif config_U1.backend.BACKEND_ID == 'torch_cpp':
        cfg_U1 = yastn.make_config(sym=yastn.sym.sym_U1, backend=yastn.backend.backend_torch_cpp, default_device=config_U1.default_device)
    else:
        raise RuntimeError('Unsupported backend')

    legs = [yastn.Leg(cfg_U1, s=-1, t=(-1, 1, 0), D=(1, 2, 3)),
            yastn.Leg(cfg_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(cfg_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9)),
            yastn.Leg(cfg_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))]
    a = yastn.rand(config=cfg_U1, legs=legs)
    b = yastn.ones(config=config_U1, legs=legs)

    # conj - documented example in test_conj.py
    tensor = a.conj()
    tensor = yastn.conj(a)
    tensor = a.conj_blocks()
    tensor = yastn.conj_blocks(a)
    tensor = a.flip_signature()
    tensor = yastn.flip_signature(a)

    # coping/cloning - documented example in test_autograd.py
    tensor = a.copy()
    tensor = yastn.copy(a)
    tensor = a.clone()
    tensor = yastn.clone(a)
    tensor = a.detach()
    tensor = yastn.detach(a)

    # to
    tensor = a.to(device='cpu')
    tensor = a.to(dtype='complex128')
    # get info
    a.print_properties()
    a.get_rank()
    a.size
    a.get_tensor_charge()
    a.get_signature()
    str(a)
    a.get_blocks_charge()
    a.get_blocks_shape()
    a.get_shape()
    a.shape
    a.get_shape(axes=2)
    a.get_dtype()
    a.dtype

    # leg retrival
    legs = a.get_legs()
    leg = a.get_legs(axes=2)  # legs[2] = leg
    print(leg.tD) # dict od charges with dimensions spanning the leg
    print(leg)

    # output dense
    array = a.to_dense()
    array = a.to_numpy()
    ls = {1: b.get_legs(axes=1)}
    array = a.to_dense(legs=ls)  # on selected legs, enforce to include charges read in previous line
    tensor = a.to_nonsymmetric()

    # permute - documented example in test_transpose.py
    tensor = a.transpose(axes=(2, 3, 0, 1))
    tensor = yastn.transpose(a, axes=(2, 3, 0, 1))

    tensor = a.moveaxis(source=2, destination=3)
    tensor = yastn.moveaxis(a, source=2, destination=3)

    a2 = yastn.tensordot(a.conj(), a, axes=((0, 1), (0, 1)))
    # linalg / split
    U, S, V = yastn.linalg.svd(a, axes=((0, 1), (2, 3)))
    U, S, V = yastn.svd(a, axes=((0, 1), (2, 3)))
    U, S, V = a.svd(axes=((0, 1), (2, 3)))
    U, S, V = yastn.svd_with_truncation(a, axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2)  # here with truncation
    try:
        U, S, V = yastn.svd(a, axes=((0, 1), (2, 3)), D_block=2, policy='lowrank')
        U, S, V = a.svd(axes=((0, 1), (2, 3)), D_block=2, policy='lowrank')
    except NameError:
        pass

    Q, R = yastn.linalg.qr(a, axes=((0, 1), (2, 3)))
    Q, R = yastn.qr(a, axes=((0, 1), (2, 3)))
    Q, R = a.qr(axes=((0, 1), (2, 3)))

    D, U = yastn.linalg.eigh(a2, axes=((0, 1), (2, 3)))
    D, U = yastn.eigh_with_truncation(a2, axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2)  # here with truncation
    D, U = a2.eigh_with_truncation(axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2)  # here with truncation

    # utils
    entropy = yastn.entropy(S ** 2)

    # linalg
    number = a.norm()
    number = yastn.norm(a, p='fro')  # p = 'for', i.e. frobenius is default
    number = yastn.linalg.norm(a, p='inf')

    number = yastn.norm(a - b)
    number = yastn.norm(a - b)
    number = yastn.linalg.norm(a - b)
    number = yastn.norm(a - b)

    # fuse
    tensor = a.fuse_legs(axes=(0, (1, 3), 2))
    tensor = tensor.unfuse_legs(axes=1)

    tensor = yastn.fuse_legs(a, axes=(0, (1, 3), 2))
    tensor = yastn.unfuse_legs(tensor, axes=(0, (1, 3), 2))

    # block
    tensor = yastn.block({(0, 0): a, (0, 1): b, (1, 0): b}, common_legs=(1, 2))

    # tests
    a.is_consistent()
    a.are_independent(b)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
