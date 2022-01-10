""" yast.conj() """
import unittest
import numpy as np
import yast
try:
    from .configs import config_Z2, config_Z2xU1
except ImportError:
    from configs import config_Z2, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name

class TestConj_Z2(unittest.TestCase):

    def test_conj_1(self):
        a = yast.rand(config=config_Z2, s=(1, 1, 1, -1, -1, -1), n=1,
                      t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                      D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
        b = a.conj()
        assert a.struct.n == b.struct.n
        na = a.to_numpy()
        nb = b.to_numpy()
        assert np.linalg.norm(na - nb) < tol

class TestConj_Z2xU1(unittest.TestCase):

    def test_conj_Z2xU1(self):
        #
        # create random complex-valued symmetric tensor with symmetry Z2 x U(1)
        #
        a = yast.rand(config=config_Z2_U1, s=(1, -1), n=(1, 1),
                      t=[[(0, 2), (1, 1), (0, 2)], [(0, 1), (0, 0), (1, 1)]],
                      D=[[1, 2, 3], [4, 5, 6]], dtype="complex128")

        #
        # conjugate tensor a: verify that signature and total charge
        # has been reversed.
        #
        b = a.conj()
        assert b.get_tensor_charge() == (1, -1)
        assert b.get_signature() == (-1, 1)

        #
        # Interpreting these tensors a,b as vectors, following contraction
        #  _           _
        # | |-<-0 0-<-| |
        # |a|->-1 1->-|b|
        #
        # is equivalent to computing the square of Frobenius norm of a.
        # Result is chargeless single-element tensor equivalent to scalar.
        #
        norm_F = yast.tensordot(a, b, axes=((0, 1), (0, 1)))
        assert norm_F.get_tensor_charge() == (0, 0)
        assert norm_F.get_signature() == ()
        assert abs(a.norm()**2 - norm_F.real().to_number()) < tol

        #
        # only complex-conjugate elements of the blocks of tensor a, leaving 
        # the structure i.e. signature and total charge intact.
        #
        c = a.conj_blocks()
        assert c.get_tensor_charge() == a.get_tensor_charge()
        assert c.get_signature() == a.get_signature()

        #
        # flip signature of the tensor c and its total charge, but do not
        # complex-conjugate elements of its block
        # 
        d = c.flip_signature()
        assert d.get_tensor_charge() == b.get_tensor_charge()
        assert d.get_signature() == b.get_signature()
        
        #
        # conj() is equivalent to flip_signature().conj_blocks() (or in the 
        # opposite order). Hence, tensor b and tensor d should be numerically 
        # identical
        #
        assert yast.norm_diff(b,d)<tol

if __name__ == '__main__':
    unittest.main()
