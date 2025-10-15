import torch
# from torch.testing._internal.common_utils import TestCase
from unittest import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import cublocksparse
from torch import Tensor
from typing import Tuple


class TestMyMulAdd(TestCase):
    def sample_inputs(self, device, *, dtype=torch.float64, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, dtype=dtype, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, dtype=dtype, device=device, requires_grad=False)

        return [
            [make_tensor(20), make_tensor(20)],
        ]
    
    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            result = cublocksparse.ops.tensordot_bs(*args)
            # torch.testing.assert_close(result, expected)

    # def test_correctness_cpu(self):
    #     self._test_correctness("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda")

    # def _test_gradients(self, device):
    #         torch.testing.assert_close(result, expected)

    # def test_gradients_cpu(self):
    #     self._test_gradients("cpu")

    # @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    # def test_gradients_cuda(self):
    #     self._test_gradients("cuda")

    def _opcheck(self, device):
        # Use opcheck to check for incorrect usage of operator registration APIs
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            opcheck(torch.ops.cublocksparse.tensordot_bs.default, args)

    # def test_opcheck_cpu(self):
    #     self._opcheck("cpu")

    # TODO relevant for full integration with PyTorch engine
    # @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    # def test_opcheck_cuda(self):
    #     self._opcheck("cuda")

class TestCuBlockSparse(TestCase):
    def sample_inputs(self, device, *, dtype=torch.float64, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, dtype=dtype, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, dtype=dtype, device=device, requires_grad=False)

        # TODO for NSYM>1 extra nesting
        def _blocksparse_coords(struct_t, t_per_mode):
            return [ [ t_per_mode[i].index( (c,) ) for i,c in enumerate(row) ] for row in struct_t ]

        # per block info
        a_struct_t= ((-1, -1, -1, -1), (-1, -1, 1, 1), (-1, -1, 2, 2), (-1, 1, -1, 1), (-1, 2, -1, 2), (1, -1, 1, -1), (1, 1, -1, -1), (1, 1, 1, 1), (1, 1, 2, 2), (1, 2, 1, 2), (2, -1, 2, -1), (2, 1, 2, 1), (2, 2, -1, -1), (2, 2, 1, 1), (2, 2, 2, 2))
        a_struct_D= ((1, 4, 7, 10), (1, 4, 8, 11), (1, 4, 9, 12), (1, 5, 7, 11), (1, 6, 7, 12), (2, 4, 8, 10), (2, 5, 7, 10), (2, 5, 8, 11), (2, 5, 9, 12), (2, 6, 8, 12), (3, 4, 9, 10), (3, 5, 9, 11), (3, 6, 7, 10), (3, 6, 8, 11), (3, 6, 9, 12))
        b_struct_t= ((-1, -1, 0), (1, 1, 0), (1, 2, 1), (2, 1, -1), (2, 2, 0))
        b_struct_D= ((1, 4, 7), (2, 5, 7), (2, 6, 11), (3, 5, 10), (3, 6, 7))
        c_struct_t= ((-1, -1, 0), (1, 1, 0), (1, 2, 1), (2, 1, -1), (2, 2, 0))
        c_struct_D= ((7, 10, 7), (8, 11, 7), (8, 12, 11), (9, 11, 10), (9, 12, 7))

        # global info
        a_t_per_mode= [
            ((-1,), (1,), (2,)),
            ((-1,), (1,), (2,)),
            ((-1,), (1,), (2,)),
            ((-1,), (1,), (2,)),
        ]
        a_D_per_mode= [
            (1, 2, 3),
            (4, 5, 6),
            (7, 8, 9),
            (10, 11, 12),
        ]

        b_t_per_mode= [
            ((-1,), (1,), (2,)),
            ((-1,), (1,), (2,)),
            ((-1,), (0,), (1,))
        ]
        b_D_per_mode= [
            (1, 2, 3),
            (4, 5, 6),
            (10, 7, 11),
        ]
        
        # c_t_per_mode, c_D_per_mode can be obtained from per_mode info of a and b
        nout_a, nin_a = (2,3), (0,1)
        nin_b, nout_b = (0,1), (2,)

        c_t_per_mode= [ a_t_per_mode[i] for i in nout_a ] + [ b_t_per_mode[i] for i in nout_b ]
        c_D_per_mode= [ a_D_per_mode[i] for i in nout_a ] + [ b_D_per_mode[i] for i in nout_b ]

        # for compliance with PyTorch Custom OPS API
        # a) as int64 tensors
        # a_coords = torch.tensor(_blocksparse_coords(a_struct_t, a_t_per_mode), dtype=torch.int64, device=device)
        # b_coords = torch.tensor(_blocksparse_coords(b_struct_t, b_t_per_mode), dtype=torch.int64, device=device)
        # c_coords = torch.tensor(_blocksparse_coords(c_struct_t, c_t_per_mode), dtype=torch.int64, device=device)
        # b) as lists of lists of indices flattened to 1D (which is expected format for cutensor's block-sparse API)
        a_coords = sum(_blocksparse_coords(a_struct_t, a_t_per_mode), start=[])
        b_coords = sum(_blocksparse_coords(b_struct_t, b_t_per_mode), start=[])
        c_coords = sum(_blocksparse_coords(c_struct_t, c_t_per_mode), start=[])

        def _pad_and_convert(D_per_mode):
            max_len = max(len(sublist) for sublist in D_per_mode)
            padded_a_D_per_mode = [list(sublist) + [-1] * (max_len - len(sublist))
                               for sublist in D_per_mode]
            return torch.tensor(padded_a_D_per_mode, dtype=torch.int64, device='cpu')
        T_a_D_per_mode= _pad_and_convert(a_D_per_mode)
        T_b_D_per_mode= _pad_and_convert(b_D_per_mode)
        T_c_D_per_mode= _pad_and_convert(c_D_per_mode)

        a_size= 13758
        b_size= 506
        c_size= 3908

        return [
            [ make_tensor(a_size), make_tensor(b_size), 
              a_coords, T_a_D_per_mode, nout_a, nin_a, b_coords, T_b_D_per_mode, nout_b, nin_b,
              c_size, c_coords, T_c_D_per_mode ],
        ]
    
    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            import pdb; pdb.set_trace()
            result = cublocksparse.ops.tensordot_bs(*args)
            # torch.testing.assert_close(result, expected)

    # def test_correctness_cpu(self):
    #     self._test_correctness("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda")