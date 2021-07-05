""" yast.set_block """
import pytest
import yast
from .configs import config_U1_fermionic

tol = 1e-12


def calculate_c3cp1(psi):
    """ Calcupate <psi| c_3 cp_1 |psi> for |psi> with 4 legs (0, 1, 2, 3)"""
    c = yast.Tensor(config=config_U1_fermionic, s=(1, 1, -1))
    c.set_block(ts=(1, 0, 1), Ds=(1, 1, 1), val=1)
    cp = yast.Tensor(config=config_U1_fermionic, s=(-1, 1, -1))
    cp.set_block(ts=(1, 1, 0), Ds=(1, 1, 1), val=1)
    # applyting cp1 (with one artificial leg carring charge)
    b = yast.tensordot(cp, psi, axes=(2, 1)).transpose(axes=(2, 0, 1, 3, 4))
    b = yast.tensordot(b, c, axes=(4, 2))
    yast.swap_gate(b, axes=((2, 3), 4), inplace=True)
    b = yast.trace(b, axes=(1, 4))
    return yast.vdot(psi, b)


def test_swap_1():
    a = yast.Tensor(config=config_U1_fermionic, s=(1, 1, 1, 1), n=2)
    a.set_block(ts=(0, 0, 1, 1), Ds=(1, 1, 1, 1), val=0.5)
    a.set_block(ts=(0, 1, 1, 0), Ds=(1, 1, 1, 1), val=0.5)
    a.set_block(ts=(1, 0, 0, 1), Ds=(1, 1, 1, 1), val=0.5)
    a.set_block(ts=(1, 1, 0, 0), Ds=(1, 1, 1, 1), val=0.5)

    assert pytest.approx(yast.norm(a), rel=tol) == 1
    assert pytest.approx(calculate_c3cp1(a), abs=tol) == 0


if __name__ == '__main__':
    test_swap_1()
