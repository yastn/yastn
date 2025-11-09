from typing import Sequence
import torch
from torch import Tensor

__all__ = ["tensordot_bs",]


# def tensordot_bs(a: Tensor, b: Tensor) -> Tensor:
#     return torch.ops.cublocksparse.tensordot_bs.default(a, b)

# def tensordot_bs(a: Tensor, 
#                  b: Tensor, 
#                  a_blocks : Sequence[Sequence[int]],     # indices of non-zero blocks wrt. to extents of a
#                  a_D_per_mode : Sequence[Sequence[int]], # this defines block-structure of a
#                  nout_a : Sequence[int],                 # modes of a that are output
#                  nin_a : Sequence[int],                  # modes of a that are contracted
#                  b_blocks : Sequence[Sequence[int]],     # indices of non-zero blocks wrt. to extents of b
#                  b_D_per_mode : Sequence[Sequence[int]], # this defines block-structure of b
#                  nout_b : Sequence[int],                 # modes of b that are output
#                  nin_b : Sequence[int],                  # modes of b that are contracted
#                  c_size : int,                           # size of result
#                  c_blocks : Sequence[Sequence[int]],     # indices of non-zero blocks wrt. to extents of c
#                  c_D_per_mode : Sequence[Sequence[int]]  # this defines block-structure of c
def tensordot_bs(a: Tensor, 
                 b: Tensor, 
                 a_blocks : Sequence[int],     # indices of non-zero blocks wrt. to extents of a
                 a_offsets : Sequence[int], # offsets of non-zero blocks wrt. to extents of a
                 a_strides : Sequence[int], # strides of non-zero blocks wrt. to ext
                 a_D_per_mode : Tensor, # this defines block-structure of a
                 nout_a : Sequence[int],                 # modes of a that are output
                 nin_a : Sequence[int],                  # modes of a that are contracted
                 b_blocks : Sequence[int],     # indices of non-zero blocks wrt. to extents of b
                 b_offsets : Sequence[int], # offsets of non-zero blocks wrt. to extents of a
                 b_strides : Sequence[int], # strides of non-zero blocks wrt. to ext
                 b_D_per_mode : Tensor, # this defines block-structure of b
                 nout_b : Sequence[int],                 # modes of b that are output
                 nin_b : Sequence[int],                  # modes of b that are contracted
                 c_size : int,                           # size of result
                 c_blocks : Sequence[int],     # indices of non-zero blocks wrt. to extents of c
                 c_offsets : Sequence[int], # offsets of non-zero blocks wrt. to extents of a
                 c_strides : Sequence[int], # strides of non-zero blocks wrt. to ext
                 c_D_per_mode : Tensor  # this defines block-structure of c
) -> Tensor:
    """
    Perform block-sparse contraction of a with b and store the block-sparse result in c
    """
    return torch.ops.cublocksparse.tensordot_bs.default(
        a, b, 
        a_blocks, a_offsets, a_strides, a_D_per_mode, nout_a, nin_a,
        b_blocks, b_offsets, b_strides, b_D_per_mode, nout_b, nin_b,
        c_size, c_blocks, c_offsets, c_strides, c_D_per_mode
    )


def _backward(ctx, grad_c):
    """
    A_a,in B_in,b = C_ab => 
        dA_a,in = dC_ab B_b,in 
        dB_in,b = A_in,a dC_ab 

    dA, dB= dC . B^T, A^T . dC
    """
    a, b = ctx.saved_tensors
    a_size, a_blocks, a_offsets, a_strides, a_D_per_mode, nout_a, nin_a= ctx.struct_a
    b_size, b_blocks, b_offsets, b_strides, b_D_per_mode, nout_b, nin_b= ctx.struct_b
    c_size, c_blocks, c_offsets, c_strides, c_D_per_mode= ctx.struct_c
    grad_a, grad_b = None, None

    if ctx.needs_input_grad[0]:
        # dA_a,in = dC_ab B_b,in 
        #
        # reorder struct a to nout_a, nin_a
        ind_a= nout_a+nin_a # order of original indices on dA_a,in 
        rank_a= len(ind_a)
        a_blocks_p= sum( [ [ a_blocks[i:i+rank_a][c] for c in nout_a+nin_a ] for i in range(0, len(a_blocks), rank_a) ] ,[] )
        a_D_per_mode_p= torch.as_tensor([ a_D_per_mode[i,:].tolist() for i in nout_a+nin_a ],dtype=torch.int64,device='cpu')
        a_strides_p= sum( [ [ a_strides[i:i+rank_a][c] for c in nout_a+nin_a ] for i in range(0, len(a_strides), rank_a) ], [] ) #sorted(range(rank_a), key=(nout_a+nin_a).__getitem__)

        grad_a = tensordot_bs(
            grad_c, b.conj(),
            c_blocks, c_offsets, c_strides, c_D_per_mode, tuple(range(len(nout_a))), tuple(range(len(nout_a),len(nout_a)+len(nout_b))), # len(nin_a)=len(nin_b)
            b_blocks, b_offsets, b_strides, b_D_per_mode, nin_b, nout_b,
            a_size, a_blocks_p, a_offsets, a_strides_p, a_D_per_mode_p
        )

    if ctx.needs_input_grad[1]:
        # dB_in,b = A_in,a dC_ab 
        # 
        # reorder struct b to nin_b, nout_b
        rank_b= len(nin_b)+len(nout_b)
        b_blocks_p= sum( [ [ b_blocks[i:i+rank_b][c] for c in nin_b+nout_b ] for i in range(0, len(b_blocks), rank_b) ] ,[] )
        b_D_per_mode_p= torch.as_tensor([ b_D_per_mode[i,:].tolist() for i in nin_b+nout_b ],dtype=torch.int64,device='cpu')
        b_strides_p= sum( [ [ b_strides[i:i+rank_b][c] for c in nin_b+nout_b ] for i in range(0, len(b_strides), rank_b) ], [] )

        # b_offsets, b_strides are for original b where nin_b and nout_b were mixed
        # The order of b_offsets is tied to block ordering convention

        grad_b = tensordot_bs(
            a.conj(), grad_c,
            a_blocks, a_offsets, a_strides, a_D_per_mode, nin_a, nout_a,
            c_blocks, c_offsets, c_strides, c_D_per_mode, tuple(range(len(nout_a),len(nout_a)+len(nout_b))), tuple(range(len(nout_a))),
            b_size, b_blocks_p, b_offsets, b_strides_p, b_D_per_mode_p
        )
    
    return grad_a, grad_b, *((None,)*17)


def _setup_context(ctx, inputs, output):
    a, b, \
    a_blocks, a_offsets, a_strides, a_D_per_mode, nout_a, nin_a, \
    b_blocks, b_offsets, b_strides, b_D_per_mode, nout_b, nin_b, \
    c_size, c_blocks, c_offsets, c_strides, c_D_per_mode = inputs

    ctx.struct_a= a.size(0), a_blocks, a_offsets, a_strides, a_D_per_mode, nout_a, nin_a
    ctx.struct_b= b.size(0), b_blocks, b_offsets, b_strides, b_D_per_mode, nout_b, nin_b
    ctx.struct_c= c_size, c_blocks, c_offsets, c_strides, c_D_per_mode
    
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This code adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "cublocksparse::tensordot_bs", _backward, setup_context=_setup_context)