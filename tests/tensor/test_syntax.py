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
"""List supported operations on yastn.Tensor (not all arguments are shown)."""
import pytest
import yastn


def test_syntax_tensor_creation_operations(config_kwargs):
    #
    # Initialize several rank-4 tensors, with the following signature
    #             ___
    #  (-) 0--<--| a |--<--1 (+)
    #  (+) 2-->--|___|-->--3 (-)
    #
    # The signatures can be interpreted as tensor legs being directed:
    # ingoing for (+) or outgoing for (-).
    #
    # The symmetry, U1, is specified in config_U1.
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    #
    # `t` gives charge sectors and `D` gives dimensions for each sector.
    leg1 = yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(1, 2, 3))
    leg2 = yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6))
    leg3 = yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9))
    leg4 = yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))

    # Upon creation, all blocks that respect charge conservation are
    # initialized and filled with either random numbers, ones, or zeros.
    a = yastn.rand(config=config_U1, legs=[leg1, leg2, leg3, leg4])
    b = yastn.ones(config=config_U1, legs=[leg1, leg2, leg3, leg4])
    c = yastn.zeros(config=config_U1, legs=[leg1, leg2, leg3, leg4])

    # Diagonal tensors infer the second leg from the provided leg.
    d = yastn.rand(config=config_U1, legs=leg1, isdiag=True)
    e = yastn.eye(config=config_U1, legs=leg1)


def test_syntax_create_empty_tensor_and_fill(config_kwargs):
    #
    # Create an empty rank-4 tensor with this signature.
    #             ___
    #  (-) 0--<--| a |--<--1 (+)
    #  (+) 2-->--|___|-->--3 (-)
    #
    # Then fill selected blocks with random values. The block charges,
    # ts, are given as a tuple with one entry per tensor leg. The block
    # dimensions, Ds, follow the same order.
    #
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    d = yastn.Tensor(config=config_U1, s=(-1, 1, 1, -1))
    d.set_block(ts=(1, -1, 2, 0), Ds=(2, 4, 9, 2), val='rand')
    d.set_block(ts=(2, 0, 2, 0), Ds=(3, 3, 9, 2), val='rand')

    # Reusing a charge sector with a different dimension raises an error.
    with pytest.raises(yastn.YastnError,
                       match="Provided Ds is not consistent with " \
                             "dimensions of existing legs."):
        d.set_block(ts=(2, 1, 2, 1), Ds=(3, 3, 10, 2), val='rand')


def test_syntax_basic_algebra(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9)),
            yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))]

    a = yastn.rand(config=config_U1, legs=legs)

    # Scalar operations.
    tensor = a / 2
    tensor = 2. * a
    tensor = a * 2.

    # Addition and subtraction require compatible tensor structure.
    b = yastn.ones(config=config_U1, legs=legs)
    tensor = a + b
    tensor = a - b

    # Incompatible block dimensions prevent tensor addition.
    legs[0] = yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(7, 2, 3))
    c = yastn.ones(config=config_U1, legs=legs)

    with pytest.raises(yastn.YastnError,
                       match="Bond dimensions of some charges do not match."):
        tensor = a + c

    # Composite linear combinations can be more efficient than repeated addition.
    tensor = yastn.add(a, b, a, b, amplitudes=(1, -1, 2, 1))
    tensor = yastn.add(a, b, a, b)  # all amplitudes equal to one.

    # Norm computations.
    number = a.norm()
    number = yastn.norm(a, p='inf')  # infinite norm. p = 'fro' is the default.

    # Elementwise unary ops and both method/top-level forms.
    tensor = a.exp(step=1)
    tensor = yastn.exp(a, step=1)

    tensor = abs(a)

    tensor = a.reciprocal(cutoff=1e-12)
    tensor = yastn.reciprocal(a, cutoff=1e-12)

    tensor = abs(a).sqrt()
    tensor = yastn.sqrt(abs(a))

    tensor = abs(a).rsqrt(cutoff=1e-12)
    tensor = yastn.rsqrt(abs(a), cutoff=1e-12)

    tensor = a.real()
    tensor = a.imag()


def test_syntax_tensor_export_import_operations(config_kwargs):
    # Serialize and deserialize a symmetric tensor.
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(7, 8, 9))]

    a = yastn.rand(config=config_U1, legs=legs)

    dictionary = a.to_dict()
    tensor = yastn.from_dict(d=dictionary, config=config_U1)
    # `config` can override the config stored in the serialized dictionary.

    # Split into raw block data and metadata, then recombine.
    vector, meta = yastn.split_data_and_meta(a.to_dict(level=0), squeeze=True)
    # Ensure that the tensor structure is embedded in the provided metadata.
    vector, meta = yastn.split_data_and_meta(a.to_dict(level=0, meta=meta), squeeze=True)
    tensor = yastn.Tensor.from_dict(yastn.combine_data_and_meta(vector, meta))


def test_syntax_block_access(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(7, 8, 9))]

    a = yastn.rand(config=config_U1, legs=legs)

    # Access an existing block by its charge key and verify its shape.
    assert a[(1, 2, 1)].shape == (3, 6, 8)

    # Modify the block in place.
    a[(1, 2, 1)] = a[(1, 2, 1)] * 2

    # Accessing a missing block raises YastnError.
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
    c = yastn.rand(config=config_U1, legs=legs)
    d = yastn.ones(config=config_U1, legs=legs)

    # Build blocked tensors sharing common axis 0 and compare contractions.
    tensor1 = yastn.block({(1, 1): a, (1, 2): b}, common_legs=(0,))
    tensor2 = yastn.block({(1, 1): c, (2, 1): d}, common_legs=(0,))

    result1 = yastn.tensordot(tensor1, tensor2.conj(), axes=((1, 2), (2, 1)))

    result2 = yastn.tensordot(a, c.conj(), axes=((1, 2), (2, 1))) + \
              yastn.tensordot(b, d.conj(), axes=((1, 2), (2, 1)))

    assert yastn.norm(result1 - result2) < 1e-12


def test_syntax_contraction(config_kwargs):
    # Create a set of U1-symmetric tensors
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    leg1 = yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(1, 2, 3))
    leg2 = yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6))
    leg3 = yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9))
    leg4 = yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))

    a = yastn.rand(config=config_U1, legs=[leg1, leg2, leg3, leg4])
    b = yastn.ones(config=config_U1, legs=[leg1, leg2, leg3, leg4])
    c = yastn.rand(config=config_U1, legs=[leg4.conj(), leg3, leg2.conj()])

    # Contract a and b over two indices. The tensor a is conjugated, which
    # reverses the signature on its indices.
    #       __           _                ___
    #  0->-|a*|->-1 1->-|b|->-0 =    0->-|a*b|->-0->2
    #  3->-|__|->-2 2->-|_|->-3   1<-3->-|___|->-3
    #
    # The order of the indices on the resulting tensor is as follows:
    # First, the outgoing indices of a (the first argument to tensordot), then
    # the outgoing indices of tensor b
    tensor = yastn.tensordot(a.conj(), b, axes=((1, 2), (1, 2)))

    # Alternative tensordot invocation and explicit conjugation via `conj`.
    tensor = a.tensordot(b, axes=((1, 2), (1, 2)), conj=(1, 0))

    # Empty axis lists produce the outer product.
    tensor = yastn.tensordot(c, c, axes=((), ()) )
    assert tensor.get_rank() == 6


    # A shorthand notation for the specific contraction
    #      _           _             __
    # 0-<-|a|-<-2     |c|-<-1 = 0-<-|ac|-<-2
    # 1->-|_|->-3 0->-|_|->-2   1->-|  |-<-1->3
    #                               |__|->-2->4
    t0 = yastn.tensordot(a, c, axes=(a.ndim - 1, 0))
    t1 = a @ c
    assert yastn.norm(t0 - t1) < 1e-12

    # Equivalent contractions using `ncon` and `einsum`.
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
    assert isinstance(tensor, yastn.Tensor)

    # Convert scalar tensor to backend number or Python scalar.
    number = tensor.to_number()
    python_scalar = tensor.item()
    assert isinstance(python_scalar, float)

    # `vdot` computes the conjugate dot product.
    number = yastn.vdot(a, b)
    number = a.vdot(b)

    # Trace specified index pairs.
    a2 = yastn.tensordot(a.conj(), a, axes=((0, 1), (0, 1)))
    tensor = a2.trace(axes=(0, 2))
    assert tensor.get_rank() == 2
    tensor = a2.trace(axes=((0, 1), (2, 3)))
    number = tensor.to_number()


def test_syntax_other(config_kwargs):
    # Initialization
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    # Create config using backend-specific symbols rather than imported aliases.
    if config_U1.backend.BACKEND_ID == 'np':
        cfg_U1 = yastn.make_config(sym=yastn.sym.sym_U1, backend=yastn.backend.backend_np, default_device=config_U1.default_device)
    elif config_U1.backend.BACKEND_ID == 'torch':
        cfg_U1 = yastn.make_config(sym=yastn.sym.sym_U1, backend=yastn.backend.backend_torch, default_device=config_U1.default_device)
    elif config_U1.backend.BACKEND_ID == 'torch_cutensor':
        cfg_U1 = yastn.make_config(sym=yastn.sym.sym_U1, backend=yastn.backend.backend_torch_cutensor, default_device=config_U1.default_device)
    else:
        raise RuntimeError('Unsupported backend')

    legs = [yastn.Leg(cfg_U1, s=-1, t=(-1, 1, 0), D=(1, 2, 3)),
            yastn.Leg(cfg_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(cfg_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9)),
            yastn.Leg(cfg_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))]
    a = yastn.rand(config=cfg_U1, legs=legs)
    b = yastn.ones(config=config_U1, legs=legs)

    # Copy/clone/detach API variants.
    tensor = a.copy()
    tensor = a.clone()
    tensor = a.detach()
    tensor = a.shallow_copy()

    # Device and dtype conversion.
    tensor = a.to(device='cpu')
    tensor = a.to(dtype='complex128')

    # Tensor inspection APIs.
    a.print_properties()
    a.print_blocks_shape()
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
    a.nblocks

    # Leg retrieval
    legs = a.get_legs()
    leg = a.get_legs(axes=2)  # legs[2] = leg
    print(leg.tD)  # dict of charges with dimensions for the leg
    print(leg)

    # Convert to dense and numpy forms.
    array = a.to_dense()
    array = a.to_numpy()
    ls = {1: b.get_legs(axes=1)}
    array = a.to_dense(legs=ls)  # on selected legs, enforce charges in ls
    tensor = a.to_nonsymmetric()

    # Decompositions and truncation.
    U, S, V = yastn.linalg.svd(a, axes=((0, 1), (2, 3)))
    mask = yastn.truncation_mask(S, D_total=2)
    U = yastn.apply_mask(mask, U, axes=2)

    a2 = yastn.tensordot(a.conj(), a, axes=((0, 1), (0, 1)))
    D, U = yastn.linalg.eigh(a2, axes=((0, 1), (2, 3)))
    D, U = yastn.eigh_with_truncation(a2, axes=((0, 1), (2, 3)), D_total=5, tol=1e-12, D_block=2)  # here with truncation

    U, S, V = yastn.eig(a2, axes=((0, 1), (2, 3)))

    # Utility functions.
    entropy = yastn.entropy(S ** 2)

    # Diagonal matrix creation and reconstruction.
    S_matrix = yastn.diag(S)
    S_diag = yastn.diag(S_matrix)

    # Comparison based on existing blocks (extra zero blocks make a difference)
    assert yastn.allclose(S, S_diag)

    # Add and remove a trivial leg.
    tensor = a.add_leg(axis=-1, s=-1, t=(0,))
    tensor = tensor.remove_leg(axis=-1)

    # Fermionic swap gate.
    tensor = yastn.swap_gate(a, axes=((0, 1), (2, 3)))

    # Remove zero or random blocks.
    tensor = a.remove_zero_blocks()
    tensor = a.remove_random_blocks(number=1, keep_legs=True)

    # Consistency checks.
    a.is_consistent()
    a.are_independent(b)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
