""" yast.vdot yast.move_leg"""
import unittest
import numpy as np
import pytest
import yast
try:
    from .configs import config_dense, config_U1, config_Z2xU1
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name

def run_move_leg(a, ad, source, destination, result):
    newa = a.move_leg(source=source, destination=destination)
    assert newa.to_numpy().shape == result
    assert newa.get_shape() == result
    assert np.moveaxis(ad, source=source, destination=destination).shape == result
    assert newa.is_consistent()
    assert a.are_independent(newa)


def run_transpose(a, ad, axes, result):
    newa = a.transpose(axes=axes)
    assert newa.to_numpy().shape == result
    assert newa.get_shape() == result
    assert np.transpose(ad, axes=axes).shape == result
    assert newa.is_consistent()
    assert a.are_independent(newa)


class TestSyntaxTranspose(unittest.TestCase):

    def test_transpose_syntax(self):
        #
        # define rank-6 U(1)-symmetric tensor
        #
        a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(2, 3), (4, 5), (6, 7), (6, 5), (4, 3), (2, 1)])

        #
        # for each leg its dense dimension is given by the sum of dimensions
        # of individual sectors. Hence, the (dense) shape of this tensor
        # is (2+3, 4+5, 6+7, 6+5, 4+3, 2+1)
        assert a.get_shape() ==  (5, 9, 13, 11, 7, 3)

        #
        # permute the legs of the tensor and check the shape is changed
        # accordingly
        b = a.transpose(axes=(0,2,4,1,3,5))
        assert b.get_shape() == (5,13,7,9,11,3)

        #
        # sometimes, instead of writing explicit permutation of all legs
        # it is more convenient to only specify pairs of legs to switched.
        # In this example, we reverse the permutation done previously thus
        # ending up with tensor numerically identical to a.
        #
        c = b.move_leg(source=(1,2), destination=(2,4))
        assert c.get_shape() == a.get_shape()
        assert yast.norm(a-c)<tol


def test_transpose_basic():
    """ test transpose for different symmetries. """
    # dense
    a = yast.ones(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    assert a.get_shape() == (2, 3, 4, 5)
    ad = a.to_numpy()
    run_transpose(a, ad, axes=(1, 3, 2, 0), result=(3, 5, 4, 2))
    run_move_leg(a, ad, source=1, destination=-1, result=(2, 4, 5, 3))
    run_move_leg(a, ad, source=(1, 3), destination=(1, 0), result=(5, 3, 2, 4))
    run_move_leg(a, ad, source=(3, 1), destination=(0, 1), result=(5, 3, 2, 4))
    run_move_leg(a, ad, source=(3, 1), destination=(1, 0), result=(3, 5, 2, 4))
    run_move_leg(a, ad, source=(1, 3), destination=(0, 1), result=(3, 5, 2, 4))

    # U1
    a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(2, 3), (4, 5), (6, 7), (6, 5), (4, 3), (2, 1)])
    ad = a.to_numpy()
    assert a.get_shape() == (5, 9, 13, 11, 7, 3)
    run_transpose(a, ad, axes=(1, 2, 3, 0, 5, 4), result=(9, 13, 11, 5, 3, 7))
    run_move_leg(a, ad, source=1, destination=4, result=(5, 13, 11, 7, 9, 3))
    run_move_leg(a, ad, source=(2, 0), destination=(0, 2), result=(13, 9, 5, 11, 7, 3))
    run_move_leg(a, ad, source=(2, -1, 0), destination=(-1, 2, -2), result=(9, 11, 3, 7, 5, 13))

    # Z2xU1
    t1 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.ones(config=config_Z2xU1, s=(-1, -1, 1, 1),
                  t=[t1, t1, t1, t1],
                  D=[(7, 3, 4, 5), (5, 4, 3, 2), (3, 4, 5, 6), (1, 2, 3, 4)])
    assert a.get_shape() == (19, 14, 18, 10)
    ad = a.to_numpy()
    run_transpose(a, ad, axes=(1, 2, 3, 0), result=(14, 18, 10, 19))
    run_move_leg(a, ad, source=-1, destination=-3, result=(19, 10, 14, 18))


def test_transpose_inplace():
    a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    assert a.get_shape() == (3, 5, 7, 9, 11, 13)

    a.transpose(axes=(5, 4, 3, 2, 1, 0), inplace=True)
    assert a.get_shape() == (13, 11, 9, 7, 5, 3)
    a.move_leg(source=2, destination=3, inplace=True)
    assert a.get_shape() == (13, 11, 7, 9, 5, 3)
    a.moveaxis(source=2, destination=3, inplace=True)  # moveaxis is alias for move_leg
    assert a.get_shape() == (13, 11, 9, 7, 5, 3)



def test_transpose_diag():
    a = yast.eye(config=config_U1, t=(-1, 0, 2), D=(2, 2 ,4))
    at = a.transpose(axes=(1, 0))
    assert yast.tensordot(a, at, axes=((0, 1), (0, 1))).item() == 8.
    assert yast.vdot(a, at, conj=(0, 0)).item() == 8.
    a.transpose(axes=(1, 0), inplace=True)
    assert yast.vdot(a, at).item() == 8.


def test_transpose_exceptions():
    """ test handling expections """
    a = yast.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    with pytest.raises(yast.YastError):
        _ = a.transpose(axes=(0, 1, 3, 5))  # Provided axes do not match tensor ndim.
    with pytest.raises(yast.YastError):
        _ = a.transpose(axes=(0, 1, 1, 2, 2, 3))  # Provided axes do not match tensor ndim.

@pytest.mark.skipif(config_dense.backend.BACKEND_ID=="numpy", reason="numpy backend does not support autograd")
def test_transpose_backward():
    import torch

    # U1
    a= yast.rand(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(2, 3), (4, 5), (6, 7), (6, 5), (4, 3), (2, 1)])
    b= a.transpose(axes=(1, 2, 3, 0, 5, 4))
    target_block=(0,1,0,0,1,0)
    target_block_size= a[target_block].size()

    def test_f(block):
        a.set_block(ts=target_block, val=block)
        tmp_a= a.transpose(axes=(1, 2, 3, 0, 5, 4))
        ab= b.vdot(tmp_a)
        return ab

    op_args = (torch.randn(target_block_size, dtype=a.get_dtype(),requires_grad=True), )
    test = torch.autograd.gradcheck(test_f, op_args, eps=1e-6, atol=1e-4)
    assert test

if __name__ == '__main__':
    test_transpose_basic()
    test_transpose_inplace()
    test_transpose_diag()
    test_transpose_exceptions()
    test_transpose_backward()
    unittest.main()
