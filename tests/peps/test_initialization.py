""" Test operation of peps.Lattice and peps.Peps that inherits Lattice"""
import pytest
import yastn
import yastn.tn.fpeps as fpeps


def test_propuct_peps():
    """ Generate a few lattices veryfing expected output of some functions. """
    ops = yastn.operators.SpinlessFermions(sym='U1')

    geometry = fpeps.SquareLattice(lattice='checkerboard', boundary='infinite')
    psi = fpeps.product_peps(geometry, {(0, 0): ops.vec_n(val=1), (0, 1): ops.vec_n(val=0)})



if __name__ == '__main__':
    test_propuct_peps()
