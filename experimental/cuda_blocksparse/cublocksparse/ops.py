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