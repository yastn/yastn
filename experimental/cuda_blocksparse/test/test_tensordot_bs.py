import torch
# from torch.testing._internal.common_utils import TestCase
from unittest import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import cublocksparse
from torch import Tensor
from typing import Tuple


class TestCuBlockSparse(TestCase):
    def sample_inputs(self, device, *, dtype=torch.float64, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, dtype=dtype, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, dtype=dtype, device=device, requires_grad=False)

        a_size= 13758
        b_size= 506
        c_size= 3908

        a_coords = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2, 2, 2]
        a_offsets = (0, 560, 1264, 4546, 5246, 6126, 7206, 9438, 10923, 12183, 13767)
        a_strides = [280, 70, 10, 1, 352, 88, 11, 1, 432, 108, 12, 1, 350, 70, 10, 1, 440, 88, 11, 1, 540, 108, 12, 1, 576, 96, 12, 1, 495, 99, 11, 1, 420, 70, 10, 1, 528, 88, 11, 1, 648, 108, 12, 1]
        T_a_D_per_mode = torch.as_tensor([(2, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)], dtype=torch.int64, device='cpu')
        nout_a = (2, 3)
        nin_a = (0, 1)
        b_coords = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 0, 2, 2, 1]
        b_offsets = (0, 56, 126, 258, 408)
        b_strides = [28, 7, 1, 35, 7, 1, 66, 11, 1, 50, 10, 1, 42, 7, 1]
        T_b_D_per_mode = torch.as_tensor([(2, 2, 3), (4, 5, 6), (10, 7, 11)], dtype=torch.int64, device='cpu')
        nout_b = (2,)
        nin_b = (0,1)
        c_coords = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 0, 2, 2, 1]
        c_offsets = (0, 490, 1106, 2162, 3152)
        c_strides = [70, 7, 1, 77, 7, 1, 132, 11, 1, 110, 10, 1, 84, 7, 1]
        T_c_D_per_mode = torch.as_tensor([(7, 8, 9), (10, 11, 12), (10, 7, 11)], dtype=torch.int64, device='cpu')

        return [
            [ make_tensor(a_size), make_tensor(b_size), 
              a_coords, a_offsets, a_strides, T_a_D_per_mode, nout_a, nin_a, 
              b_coords, b_offsets, b_strides, T_b_D_per_mode, nout_b, nin_b,
              c_size, c_coords, c_offsets, c_strides, T_c_D_per_mode ],
            [ make_tensor(a_size), make_nondiff_tensor(b_size), 
              a_coords, a_offsets, a_strides, T_a_D_per_mode, nout_a, nin_a, 
              b_coords, b_offsets, b_strides, T_b_D_per_mode, nout_b, nin_b,
              c_size, c_coords, c_offsets, c_strides, T_c_D_per_mode ],
            [ make_nondiff_tensor(a_size), make_tensor(b_size), 
              a_coords, a_offsets, a_strides, T_a_D_per_mode, nout_a, nin_a, 
              b_coords, b_offsets, b_strides, T_b_D_per_mode, nout_b, nin_b,
              c_size, c_coords, c_offsets, c_strides, T_c_D_per_mode ],
        ]
    
    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            result = cublocksparse.ops.tensordot_bs(*args)
            # expected = reference_impl(*args)
            # torch.testing.assert_close(result, expected)

    # def test_correctness_cpu(self):
    #     self._test_correctness("cpu")

    def _test_gradients(self, device):
        samples = self.sample_inputs(device, requires_grad=True)
        for args in samples:
            diff_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
            out = cublocksparse.ops.tensordot_bs(*args)
            grad_out = torch.randn_like(out)
            result = torch.autograd.grad(out, diff_tensors, grad_out)

            # out = reference_impl(*args)
            # expected = torch.autograd.grad(out, diff_tensors, grad_out)
            # torch.testing.assert_close(result, expected)

    # def test_gradients_cpu(self):
    #     self._test_gradients("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_gradients_cuda(self):
        self._test_gradients("cuda")

    def _opcheck(self, device):
        # Use opcheck to check for incorrect usage of operator registration APIs
        samples = self.sample_inputs(device, requires_grad=True)
        samples.extend(self.sample_inputs(device, requires_grad=False))
        for args in samples:
            opcheck(torch.ops.cublocksparse.tensordot_bs.default, args)

    # def test_opcheck_cpu(self):
    #     self._opcheck("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")