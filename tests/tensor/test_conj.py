""" yast.conj()  yast.flip_signature() """
import unittest
import numpy as np
import yast
try:
    from .configs import config_Z2, config_Z2xU1
except ImportError:
    from configs import config_Z2, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name

def conj_vs_numpy(a, expected_n):
    """ run conj() flip_signature() and a few tests. """
    b = a.conj()
    c = a.flip_signature()
    d = a.conj_blocks()
    assert all(x.is_consistent() for x in (b, c, d))

    assert all(x.struct.n == expected_n for x in (b, c))
    assert a.struct.n == d.struct.n

    assert all(sa + sb == 0 for sa, sb in zip(a.struct.s, b.struct.s))
    assert all(sa + sc == 0 for sa, sc in zip(a.struct.s, c.struct.s))
    assert a.struct.s == d.struct.s

    na, nb, nc, nd = a.to_numpy(), b.to_numpy(), c.to_numpy(), d.to_numpy()
    assert np.linalg.norm(na.conj() - nb) < tol
    assert np.linalg.norm(na - nc) < tol
    assert np.linalg.norm(na.conj() - nd) < tol
    assert abs(yast.vdot(a, a).item() - yast.vdot(b, a, conj=(0, 0)).item()) < tol
    assert abs(yast.vdot(a, a).item() - yast.vdot(d, c, conj=(0, 0)).item()) < tol

    assert yast.norm(a - b.conj()) < tol
    assert yast.norm(a - c.flip_signature()) < tol
    assert yast.norm(a - d.conj_blocks()) < tol


def test_conj_basic():
    """ test conj for different symmerties """
    # U1
    a = yast.randC(config=config_Z2, s=(1, 1, 1, -1, -1, -1), n=1,
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    conj_vs_numpy(a, expected_n=(1,))

    a = yast.rand(config=config_Z2xU1, s=(1, -1), n=(1, 2),
                  t=[[(0, 0), (1, 1), (0, 2)], [(0, 1), (0, 0), (1, 1)]],
                  D=[[1, 2, 3], [4, 5, 6]])
    conj_vs_numpy(a, expected_n=(1, -2))


def test_conj_hard_fusion():
    a = yast.randC(config=config_Z2, s=(1, -1, 1, -1, 1, -1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    a.fuse_legs(axes=((0, 1), (2, 3), (4, 5)), inplace=True)
    a.fuse_legs(axes=((0, 1), 2), inplace=True)
    b = a.conj()
    c = a.flip_signature()
    d = a.conj_blocks()
    assert all(sa + sb == 0 for sa, sb in zip(a.struct.s, b.struct.s))
    assert all(sa + sc == 0 for sa, sc in zip(a.struct.s, c.struct.s))
    assert a.struct.s == d.struct.s

    assert all(sa + sb == 0 for hfa, hfb in zip(a.hard_fusion, b.hard_fusion) for sa, sb in zip(hfa.s, hfb.s))
    assert all(sa + sc == 0 for hfa, hfc in zip(a.hard_fusion, c.hard_fusion) for sa, sc in zip(hfa.s, hfc.s))
    assert all(hfa.s == hfd.s for hfa, hfd in zip(a.hard_fusion, d.hard_fusion))


class TestConj_Z2xU1(unittest.TestCase):

    def test_conj_Z2xU1(self):
        #
        # create random complex-valued symmetric tensor with symmetry Z2 x U(1)
        #
        a = yast.rand(config=config_Z2xU1, s=(1, -1), n=(1, 2),
                      t=[[(0, 2), (1, 1), (0, 2)], [(0, 1), (0, 0), (1, 1)]],
                      D=[[1, 2, 3], [4, 5, 6]], dtype="complex128")

        #
        # conjugate tensor a: verify that signature and total charge
        # has been reversed.
        #
        b = a.conj()
        assert b.get_tensor_charge() == (1, -2)
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
        assert yast.norm(b-d)<tol

if __name__ == '__main__':
    test_conj_basic()
    test_conj_hard_fusion()
    unittest.main()