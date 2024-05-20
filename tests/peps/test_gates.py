""" Test definitions of two-site gates. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps._gates_auxiliary import fkron

try:
    from .configs import config as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config as cfg


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

    # nearest-neighbor hopping gate
    g_hop = fpeps.gates.gate_nn_hopping(t, ds, I, c, cdag)
    O = yastn.ncon([g_hop.G0, g_hop.G1], [(-0, -2, 1) , (-1, -3, 1)])
    O = O.fuse_legs(axes=((0, 1), (2, 3)))

    # hopping Hamiltonian
    H = - t * fpeps.gates.fkron(cdag, c, sites=(0, 1), merge=True) \
        - t * fpeps.gates.fkron(cdag, c, sites=(1, 0), merge=True)
    H = H.fuse_legs(axes=((0, 1), (2, 3)))

    II = yastn.ncon([I, I], [(-0, -2) , (-1, -3)])
    II = II.fuse_legs(axes=((0, 1), (2, 3)))

    # Hamiltonian exponent from Taylor expansion
    O2 = II + (-ds) * H + (ds ** 2 / 2) * H @ H \
       + (-ds ** 3 / 6) * H @ H @ H + (ds ** 4 / 24) * H @ H @ H @ H

    assert ((O - O2).norm()) < (ds * t) ** 5

def test_heisenberg_gates():
    kwargs = {'backend': cfg.backend, 'default_device': cfg.default_device}

    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', **kwargs)
    check_heisenberg_gates(ops, Jz=2, J=1, Jn = 0.5, ds=0.005)

    ops = yastn.operators.SpinfulFermions_tJ(sym='U1xU1xZ2', **kwargs)
    check_heisenberg_gates(ops, Jz=1, J=1, Jn = 1, ds=0.005)

    ops = yastn.operators.SpinfulFermions(sym='Z2', **kwargs)
    check_heisenberg_gates(ops, Jz=2, J=1, Jn = 0.5, ds=0.005)

    ops = yastn.operators.SpinfulFermions_tJ(sym='Z2', **kwargs)
    check_heisenberg_gates(ops, Jz=1, J=1, Jn = 1, ds=0.005)

    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **kwargs)
    check_heisenberg_gates(ops, Jz=1, J=2, Jn = 0.5, ds=0.005)

    ops = yastn.operators.SpinfulFermions_tJ(sym='U1xU1', **kwargs)
    check_heisenberg_gates(ops, Jz=1, J=1, Jn = 1, ds=0.005)

def check_heisenberg_gates(ops, Jz, J, Jn, ds):

    [I, Sz, Sm, Sp] = [ops.I(), ops.Sz(), ops.Sm(), ops.Sp()]
    nu, nd = ops.n(spin='u'), ops.n(spin='d')

    n = nu + nd

    g_heisenberg = fpeps.gates.gates_Heisenberg_spinful(ds, Jz, J, Jn, Sz, Sp, Sm, n, I)
    O = yastn.ncon([g_heisenberg.G0, g_heisenberg.G1], [(-0, -2, 1) , (-1, -3, 1)])
    O = O.fuse_legs(axes=((0, 1), (2, 3)))

    H = fkron(I, I, sites=(0, 1))
    H = H + J / 2 * (fkron(Sp, Sm, sites=(0, 1)) + fkron(Sp, Sm, sites=(1, 0))) + \
            Jz * fkron(Sz, Sz, sites=(0, 1)) - 0.25 * Jn * fkron(n, n, sites=(0, 1))


    H = H.fuse_legs(axes=((0, 1), (2, 3)))
    II = yastn.ncon([I, I], [(-0, -2) , (-1, -3)])
    II = II.fuse_legs(axes=((0, 1), (2, 3)))

    O2 = II + (-ds) * H + (ds ** 2 / 2) * H @ H + (-ds ** 3 / 6) * H @ H @ H + (ds ** 4 / 24) * H @ H @ H @ H

    assert ((O - O2).norm()) < (ds * max(abs(Jz), abs(J), abs(Jn))) ** 4



def test_gate_raises():
    kwargs = {'backend': cfg.backend, 'default_device': cfg.default_device}
    ops = yastn.operators.SpinlessFermions(sym='U1', **kwargs)

    c, cdag = ops.c(), ops.cp()

    with pytest.raises(yastn.YastnError):
        fpeps.gates.fkron(c, cdag, sites=(0, 2))
        # sites should be equal to (0, 1) or (1, 0)


if __name__ == '__main__':
    test_hopping_gate()
    test_gate_raises()
