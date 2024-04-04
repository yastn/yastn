""" Test the expectation values of spinless fermions with analytical values of fermi sea for finite and infinite lattices """
import yastn
import yastn.tn.fpeps as fpeps

try:
    from .configs import config_U1xU1_R_fermionic as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_U1xU1_R_fermionic as cfg


def test_hopping_gate():
    kwargs = {'backend': cfg.backend, 'default_device': cfg.default_device}

    ops = yastn.operators.SpinlessFermions(sym='U1', **kwargs)
    check_hopping_gate(ops, t=0.5, ds=0.02)

    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **kwargs)
    check_hopping_gate(ops, t=2, ds=0.005)

    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', **kwargs)
    check_hopping_gate(ops, t=1, ds=0.1)


def check_hopping_gate(ops, t, ds):

    I, c, cdag = ops.I(), ops.c(), ops.cp()

    g_hop = fpeps.gates.gate_nn_hopping(t, ds, I, c, cdag)  # nn gate for 2D fermi sea
    O = yastn.ncon([g_hop.A, g_hop.B], [(-0, -2, 1) , (-1, -3, 1)])
    O = O.fuse_legs(axes=((0, 1), (2, 3)))

    c1dag = cdag.add_leg(s=1).swap_gate(axes=(0, 2))
    c2 = c.add_leg(s=-1)
    c1 = c.add_leg(s=1).swap_gate(axes=(1, 2))
    c2dag = cdag.add_leg(s=-1)
    H = - t * yastn.ncon([c1dag, c2], [(-0, -2, 1) , (-1, -3, 1)]) \
        - t * yastn.ncon([c1, c2dag], [(-0, -2, 1) , (-1, -3, 1)])
    H = H.fuse_legs(axes=((0, 1), (2, 3)))
    II = yastn.ncon([I, I], [(-0, -2) , (-1, -3)])
    II = II.fuse_legs(axes=((0, 1), (2, 3)))

    O2 = II + (-ds) * H + (ds ** 2 / 2) * H @ H + (-ds ** 3 / 6) * H @ H @ H + (ds ** 4 / 24) * H @ H @ H @ H

    assert ((O - O2).norm()) < (ds * t) ** 5




if __name__ == '__main__':
    test_hopping_gate()
