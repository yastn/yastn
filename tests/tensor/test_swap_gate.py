""" yast.swap_gate() to introduce fermionic statistics. """
import pytest
from itertools import product
import yast
try:
    from .configs import config_Z2, config_Z2_fermionic
    from .configs import config_U1xU1_fermionic, config_U1xU1xZ2_fermionic
except ImportError:
    from configs import config_Z2, config_Z2_fermionic
    from configs import config_U1xU1_fermionic, config_U1xU1xZ2_fermionic



tol = 1e-12  #pylint: disable=invalid-name

# local space is |0>, u+|0>, d+|0>, u+d+|0> = |00>, |10>, |01>, |11>
# u+, d+ are creation operators for spin-up and spin-down

def calculate_c3cp1(psi):
    """ Calcupate <psi| c_3 cp_1 |psi> for |psi> with 4 legs (0, 1, 2, 3), i.e. 4 fermionic modes"""
    c = yast.Tensor(config=config_Z2_fermionic, s=(1, 1, -1))
    c.set_block(ts=(1, 0, 1), Ds=(1, 1, 1), val=1)
    cp = yast.Tensor(config=config_Z2_fermionic, s=(-1, 1, -1))
    cp.set_block(ts=(1, 1, 0), Ds=(1, 1, 1), val=1)
    # applyting cp1 (with one artificial leg carring charge)
    b = yast.tensordot(cp, psi, axes=(2, 1)).transpose(axes=(2, 0, 1, 3, 4))
    b = yast.tensordot(b, c, axes=(4, 2))

    b_test = yast.swap_gate(b, axes=(2, 4, 3, 4))  # swap_gate between legs 2 and 4, and legs 3 and 4
    yast.swap_gate(b, axes=((2, 3), 4), inplace=True)  # the same here, but with different syntax; and inplace
    assert yast.norm(b - b_test) < tol

    b = yast.trace(b, axes=(1, 4))
    return yast.vdot(psi, b).item()


def test_swap_gate_basic():
    """ apply swap_gate during calculation of <psi| c1 c3 |psi> expectation value, for |psi> with 4 fermionic modes"""
    t1, D1 = (0, 1, 2, 3), (2, 2, 2, 2)
    a = yast.rand(config=config_Z2, s=(1, -1, 1, -1), t=(t1, t1, t1, t1), D=(D1, D1, D1, D1))
    b = a.swap_gate(axes=(0, 1))
    assert a is b  # for config.fermionic = False, swap_gate does nothing.

    psi = yast.Tensor(config=config_Z2_fermionic, s=(1, 1, 1, 1), n=0)
    psi.set_block(ts=(0, 0, 1, 1), Ds=(1, 1, 1, 1), val=0.5)
    psi.set_block(ts=(0, 1, 1, 0), Ds=(1, 1, 1, 1), val=0.5)
    psi.set_block(ts=(1, 0, 0, 1), Ds=(1, 1, 1, 1), val=0.5)
    psi.set_block(ts=(1, 1, 0, 0), Ds=(1, 1, 1, 1), val=0.5)

    assert pytest.approx(yast.norm(psi).item(), rel=tol) == 1
    assert pytest.approx(calculate_c3cp1(psi), abs=tol) == 0


def test_swap_gate_exceptions():
    """ swap_gate raising exceptions """
    t1, D1 = (0, 1, 2, 3), (2, 2, 2, 2)
    a = yast.rand(config=config_Z2_fermionic, s=(1, -1, 1, -1), t=(t1, t1, t1, t1), D=(D1, D1, D1, D1))
    with pytest.raises(yast.YastError):
        a.swap_gate(axes=(0, 1, 2))
        # Odd number of elements in axes. Elements of axes should come in pairs.
    with pytest.raises(yast.YastError):
        a.swap_gate(axes=((0, 1), 0))
        # Cannot swap the same index


def operator_c(s='up', sym='Z2'): # here, first leg acts on Hilbert space
    """ define anihilation operators for spinfull local Hilbert space. """
    if sym == 'Z2' and s == 'up': # charges: 0 <-> (|00>, |11>); <-> (|10>, |01>)
        temp = yast.Tensor(config=config_Z2_fermionic, s=(-1, 1), n=1)
        temp.set_block(ts=(1, 0), Ds=(2, 2), val=[[1, 0], [0, 0]])
        temp.set_block(ts=(0, 1), Ds=(2, 2), val=[[0, 0], [0, 1]])
    if sym == 'Z2' and s == 'down':
        temp = yast.Tensor(config=config_Z2_fermionic, s=(-1, 1), n=1)
        temp.set_block(ts=(1, 0), Ds=(2, 2), val=[[0, 0], [1, 0]])
        temp.set_block(ts=(0, 1), Ds=(2, 2), val=[[0, 0], [-1, 0]])
    if sym == 'U1xU1_ind' and s == 'up': # charges <-> (ocupation up, occupation down, total_parity)
        temp = yast.Tensor(config=config_U1xU1xZ2_fermionic, s=(-1, 1), n=(-1, 0, 1))
        temp.set_block(ts=((1, 0, 1), (0, 0, 0)), Ds=(1, 1), val=1)
        temp.set_block(ts=((1, 1, 0), (0, 1, 1)), Ds=(1, 1), val=1)
    if sym == 'U1xU1_ind' and s == 'down':
        temp = yast.Tensor(config=config_U1xU1xZ2_fermionic, s=(-1, 1), n=(0, -1, 1))
        temp.set_block(ts=((0, 1, 1), (0, 0, 0)), Ds=(1, 1), val=1)
        temp.set_block(ts=((1, 1, 0), (1, 0, 1)), Ds=(1, 1), val=-1)
    if sym == 'U1xU1_dis' and s == 'up':  # charges <-> (ocupation up, occupation down)
        temp = yast.Tensor(config=config_U1xU1_fermionic, s=(-1, 1), n=(-1, 0))
        temp.set_block(ts=((1, 0), (0, 0)), Ds=(1, 1), val=1)
        temp.set_block(ts=((1, 1), (0, 1)), Ds=(1, 1), val=1)
    if sym == 'U1xU1_dis' and s == 'down':
        temp = yast.Tensor(config=config_U1xU1_fermionic, s=(-1, 1), n=(0, -1))
        temp.set_block(ts=((0, 1), (0, 0)), Ds=(1, 1), val=1)
        temp.set_block(ts=((1, 1), (1, 0)), Ds=(1, 1), val=1)
    return temp


def operator_cdag(s='up', sym='Z2'):
    """ define creation operators for spinfull local Hilbert space. """
    if sym == 'Z2' and s == 'up':  # charges: 0 <-> (|00>, |11>); <-> (|10>, |01>)
        temp = yast.Tensor(config=config_Z2_fermionic, s=(-1, 1), n=1)
        temp.set_block(ts=(1, 0), Ds=(2, 2), val=[[0, 0], [0, 1]])
        temp.set_block(ts=(0, 1), Ds=(2, 2), val=[[1, 0], [0, 0]])
    if sym == 'Z2' and s == 'down':
        temp = yast.Tensor(config=config_Z2_fermionic, s=(-1, 1), n=1)
        temp.set_block(ts=(1, 0), Ds=(2, 2), val=[[0, -1], [0, 0]])
        temp.set_block(ts=(0, 1), Ds=(2, 2), val=[[0, 1], [0, 0]])
    if sym == 'U1xU1_ind' and s == 'up':
        temp = yast.Tensor(config=config_U1xU1xZ2_fermionic, s=(-1, 1), n=(1, 0, 1))
        temp.set_block(ts=((0, 0, 0), (1, 0, 1)), Ds=(1, 1), val=1)
        temp.set_block(ts=((0, 1, 1), (1, 1, 0)), Ds=(1, 1), val=1)
    if sym == 'U1xU1_ind' and s == 'down':
        temp = yast.Tensor(config=config_U1xU1xZ2_fermionic, s=(-1, 1), n=(0, 1, 1))
        temp.set_block(ts=((0, 0, 0), (0, 1, 1)), Ds=(1, 1), val=1)
        temp.set_block(ts=((1, 0, 1), (1, 1, 0)), Ds=(1, 1), val=-1)
    if sym == 'U1xU1_dis' and s == 'up':
        temp = yast.Tensor(config=config_U1xU1_fermionic, s=(-1, 1), n=(1, 0))
        temp.set_block(ts=((0, 0), (1, 0)), Ds=(1, 1), val=1)
        temp.set_block(ts=((0, 1), (1, 1)), Ds=(1, 1), val=1)
    if sym == 'U1xU1_dis' and s == 'down':
        temp = yast.Tensor(config=config_U1xU1_fermionic, s=(-1, 1), n=(0, 1))
        temp.set_block(ts=((0, 0), (0, 1)), Ds=(1, 1), val=1)
        temp.set_block(ts=((1, 0), (1, 1)), Ds=(1, 1), val=1)
    return temp


def operator_id(sym='Z2'):
    """ define identity operator for spinfull local Hilbert space."""
    if sym == 'Z2':
        temp = yast.Tensor(config=config_Z2_fermionic, s=(-1, 1), n=0)
        for t in [0, 1]:
            temp.set_block(ts=(t, t), Ds=(2, 2), val=[[1, 0], [0, 1]])
    if sym == 'U1xU1_ind':
        temp = yast.Tensor(config=config_U1xU1xZ2_fermionic, s=(-1, 1), n=(0, 0, 0))
        for t in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
            temp.set_block(ts=(t, t), Ds=(1, 1), val=1)
    if sym == 'U1xU1_dis':
        temp = yast.Tensor(config=config_U1xU1_fermionic, s=(-1, 1), n=(0, 0))
        for t in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            temp.set_block(ts=(t, t), Ds=(1, 1), val=1)
    return temp


def test_operators():
    """ test commutation rules for operators """
    spins = ('up', 'down')

    for sym, inter_sgn in [('Z2', 1), ('U1xU1_ind', 1), ('U1xU1_dis', -1)]:
        c = {s: operator_c(s=s, sym=sym) for s in spins}
        cp = {s: operator_cdag(s=s, sym=sym) for s in spins}
        iden = operator_id(sym=sym)
        ccp = {(s1, s2): yast.tensordot(c[s1], cp[s2], axes=(1, 0)) for s1, s2 in product(spins, spins)}
        cpc = {(s1, s2): yast.tensordot(cp[s1], c[s2], axes=(1, 0)) for s1, s2 in product(spins, spins)}
        assert yast.norm((ccp[('up', 'up')] + cpc[('up', 'up')]) - iden) < tol
        assert yast.norm((ccp[('down', 'down')] + cpc[('down', 'down')]) - iden) < tol
        assert yast.norm((ccp[('up', 'down')] + inter_sgn * cpc[('down', 'up')])) < tol
        assert yast.norm((ccp[('down', 'up')] + inter_sgn * cpc[('up', 'down')])) < tol


if __name__ == '__main__':
    test_swap_gate_basic()
    test_swap_gate_exceptions()
    test_operators()




    # n_loc_up = yast.tensordot(fcdag_up, fc_up, axes = (1, 0))
    # n_loc_down = yast.tensordot(fcdag_down, fc_down, axes = (1, 0))
    # iden = yast.tensordot(fid, fid, axes=((),()))
    # n1_up = yast.tensordot(n_loc_up, fid, axes=((),()))
    # n2_up = yast.tensordot(fid, n_loc_up, axes=((),()))
    # n1_down = yast.tensordot(n_loc_down, fid, axes=((),()))
    # n2_down = yast.tensordot(fid, n_loc_down, axes=((),()))
    # n1n2_up = yast.tensordot(n_loc_up, n_loc_up, axes=((),()))
    # n1n2_down = yast.tensordot(n_loc_down, n_loc_down, axes=((),()))
    # hop_up = yast.tensordot(fcdag_up, fc_up, axes=((),())) + yast.tensordot(fc_up, fcdag_up, axes=((),()))
    # hop_down = yast.tensordot(fcdag_down, fc_down, axes=((),())) + yast.tensordot(fc_down, fcdag_down, axes=((),()))
    # n_int = yast.tensordot(n_loc_up, n_loc_down, axes=(1, 0))

    # def nn_Gate_spinfull_hop(t_up, t_down, beta):
    #     # nn term
    #     U_nn_up =   iden -(1 - np.cosh(-0.25 * t_up * beta)) * (n1_up + n2_up - 2*n1n2_up) - np.sinh(-0.25 * t_up * beta) * hop_up
    #     U_nn_down =  iden - (1 - np.cosh(-0.25 * t_down * beta)) * (n1_down + n2_down - 2*n1n2_down) - np.sinh(-0.25 * t_down * beta) * hop_down

    #     Aup, Xup, Bup = yast.svd(U_nn_up, axes = ((0, 1), (2, 3)), sU = -1, tol = 1e-15)
    #     GA_nn_up = yast.ncon((Aup, Xup.sqrt()), ([-1, -3, 1], [1, -2]))
    #     GB_nn_up = yast.ncon((Xup.sqrt(), Bup), ([-2, 1], [1, -1, -3]))

    #     Adown, Xdown, Bdown = yast.svd(U_nn_down, axes = ((0, 1), (2, 3)), sU = -1, tol = 1e-15)
    #     GA_nn_down = yast.ncon((Adown, Xdown.sqrt()), ([-1, -3, 1], [1, -2]))
    #     GB_nn_down = yast.ncon((Xdown.sqrt(), Bdown), ([-2, 1], [1, -1, -3]))