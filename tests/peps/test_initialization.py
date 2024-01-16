""" Test operation of peps.Lattice and peps.Peps that inherits Lattice"""
import pytest
import yastn
import yastn.tn.fpeps as fpeps


tol = 1e-12

def test_propuct_peps():
    """ Generate a few lattices veryfing expected output of some functions. """
    ops = yastn.operators.SpinlessFermions(sym='U1')

    geometry = fpeps.CheckerboardLattice()
    v0 = ops.vec_n(val=0).add_leg(axis=0, s=-1).fuse_legs(axes=[(0, 1)])
    v1 = ops.vec_n(val=1).add_leg(axis=0, s=-1).fuse_legs(axes=[(0, 1)])
    psi = fpeps.product_peps(geometry, {(0, 0): v0, (0, 1): v1})
    for site in psi.sites():
        legs = psi[site].get_legs()
        for leg in legs:
            assert leg.t == ((0,),)
    #
    #
    geometry = fpeps.SquareLattice(dims=(3, 2), boundary='obc')
    psi = fpeps.product_peps(geometry, ops.I().fuse_legs(axes=[(0, 1)]))
    for site in psi.sites():
        legs = psi[site].get_legs()
        for leg in legs:
            assert leg.t == ((0,),)
    #
    #
    with pytest.raises(yastn.YastnError):
        psi = fpeps.product_peps(geometry, {(0, 0): v0})
        # product_peps did not initialize some peps tensor


def test_save_load():
    ops = yastn.operators.Spin1(sym='Z3')
    geometry = fpeps.SquareLattice(dims=(3, 2), boundary='obc')
    psi = fpeps.product_peps(geometry, ops.I().fuse_legs(axes=[(0, 1)]))

    d = psi.save_to_dict()
    psi2 = fpeps.load_from_dict(ops.config, d)

    for site in psi.sites():
        assert (psi2[site] - psi[site]).norm() < tol


if __name__ == '__main__':
    test_propuct_peps()
    test_save_load()
