import yastn
import yastn.tn.fpeps as fpeps

try:
    from .configs import config_dense
except ImportError:
    from configs import config_dense


def atest_apply_gate_onsite():
    """ initialize vacuum state and check the functions apply_gate_onsite and apply_gate_onsite_2s """

    net = fpeps.SquareLattice(dims=(2, 2), boundary='obc')
    ops = yastn.operators.SpinfulFermions(sym='U1xU1',backend=config_dense.backend,default_device=config_dense.default_device)
    c_up, c_dn, cdag_up, cdag_dn = ops.c(spin='u'), ops.c(spin='d'), ops.cp(spin='u'), ops.cp(spin='d')

    # initialize peps in Neel state
    A10 = ops.vec_n(val=(1, 0))
    A01 = ops.vec_n(val=(0, 1))
    psi = fpeps.product_peps(net, {(0, 0): A10, (1, 1): A10, (0, 1): A01, (1, 0): A01})
    #
    for ms in psi.sites():  # asserting each peps tensor has 6 legs in its unfused form
        assert psi[ms].unfuse_legs(axes=(0, 1, 2)).ndim == 6
    #
    # apply annihilation operator of the same polarization. It should create a hole.
    ten10 = psi[0, 0]
    ten00 = fpeps.apply_gate_onsite(ten10, c_up)
    assert ten10.unfuse_legs(axes=2).get_legs(axes=2).t == ((1, 0),)
    assert ten00.unfuse_legs(axes=2).get_legs(axes=2).t == ((0, 0),)
    #
    # annihilate site (0, 0).
    vac = fpeps.apply_gate_onsite(psi[0, 0], c_dn)
    assert vac.unfuse_legs(axes=2).get_legs(axes=2).t == ()
    vac = fpeps.apply_gate_onsite(psi[0, 0], cdag_up)
    assert vac.unfuse_legs(axes=2).get_legs(axes=2).t == ()
    #
    # create double occupancy
    doc = fpeps.apply_gate_onsite(psi[0, 0], cdag_dn)
    assert doc.unfuse_legs(axes=2).get_legs(axes=2).t == ((1, 1),)
    #
    # we transfer the spin-up electron from site (0, 0) to (0, 1)
    c1 = c_up.add_leg(s=1).swap_gate(axes=(0, 2))
    c2dag = cdag_up.add_leg(s=-1)
    cc =  yastn.ncon([c1, c2dag], ((-0, -2, 1) , (-1, -3, 1)))
    gate = fpeps.gates.decompose_nn_gate(cc)
    vac = fpeps.apply_gate_onsite(psi[0, 0], gate.G0, dirn='l')
    doc = fpeps.apply_gate_onsite(psi[0, 1], gate.G1, dirn='r')
    assert vac.unfuse_legs(axes=2).get_legs(axes=2).t == ((0, 0),)
    assert doc.unfuse_legs(axes=2).get_legs(axes=2).t == ((1, 1),)
    vac = fpeps.apply_gate_onsite(psi[0, 0], gate.G0, dirn='t')
    doc = fpeps.apply_gate_onsite(psi[1, 0], gate.G1, dirn='b')
    assert vac.unfuse_legs(axes=2).get_legs(axes=2).t == ((0, 0),)
    assert doc.unfuse_legs(axes=2).get_legs(axes=2).t == ((1, 1),)


if __name__ == "__main__":
    atest_apply_gate_onsite()
