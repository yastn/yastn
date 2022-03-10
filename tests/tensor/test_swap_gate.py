""" yast.swap_gate() to introduce fermionic statistics. """
from itertools import product
import pytest
import yast
try:
    from .configs import config_U1xU1_fermionic, config_U1xU1xZ2_fermionic, config_Z2_fermionic
except ImportError:
    from configs import config_U1xU1_fermionic, config_U1xU1xZ2_fermionic, config_Z2_fermionic



tol = 1e-12  #pylint: disable=invalid-name


def test_swap_gate_basic():
    """ basic tests of swap_gate """
    t, D = (0, 1), (1, 1)
    a = yast.ones(config=config_Z2_fermionic, s=(1, 1, 1, 1), n=0, t=(t, t, t, t), D=(D, D, D, D))
    assert pytest.approx(sum(a.to_numpy().ravel()), rel=tol) == 8

    b = a.swap_gate(axes=(0, 1))
    assert pytest.approx(sum(b.to_numpy().ravel()), rel=tol) == 4

    c = a.swap_gate(axes=(0, 2))
    c.swap_gate(axes=(1, 2), inplace=True)
    assert pytest.approx(sum(c.to_numpy().ravel()), rel=tol) == 4
    c1 = a.swap_gate(axes=((0, 1), 2))  # swap between group of (0, 1) and 2
    c2 = a.swap_gate(axes=(0, 2, 1, 2)) # swap between 0, 2 and then 1, 2
    assert all(yast.norm(c - x) < tol for x in (c1, c2))

    d = a.swap_gate(axes=((0, 1), (2, 3))) # swap between group of (0, 1) and (2, 3)
    assert pytest.approx(sum(d.to_numpy().ravel()), rel=tol) == 0

    assert pytest.approx(yast.vdot(a, b).item(), rel=tol) == 4
    assert pytest.approx(yast.vdot(a, c).item(), rel=tol) == 4
    assert pytest.approx(yast.vdot(a, d).item(), rel=tol) == 0
    assert pytest.approx(yast.vdot(b, c).item(), rel=tol) == 0
    assert pytest.approx(yast.vdot(b, d).item(), rel=tol) == -4
    assert pytest.approx(yast.vdot(c, d).item(), rel=tol) == 4


def apply_operator(psi, c, site):
    """
    Apply operator c on site of psi.

    Use swap_gate to impose fermionic statistics.
    Fermionic order from first to last site.
    """
    ndim = psi.ndim
    ca = c.add_leg(axis=-1)
    cpsi = yast.tensordot(psi, ca, axes=(site, 1))
    cpsi.move_leg(source=ndim - 1, destination=site, inplace=True)
    cpsi.swap_gate(axes=(tuple(range(site)), ndim), inplace=True)
    cpsi.remove_leg(axis=-1, inplace=True)
    return cpsi


def test_apply_operators():
    """ apply swap_gate during calculation of expectation value such as, e.g. <psi| c_1 cdag_3 |psi>"""
    spins = ('up', 'dn')
    for sym in ['Z2', 'U1xU1_ind', 'U1xU1_dis']:
        ann = {s: operator_spinfull('ann', s=s, sym=sym) for s in spins}
        cre = {s: operator_spinfull('cre', s=s, sym=sym) for s in spins}

        vac = vacum_spinfull(sites=5, sym=sym)
        psi = vac
        for s in spins:
            psi0 = None
            for sites in ((2, 3), (1, 2), (0, 3), (0, 1)):  # sum of fermions created on those sites
                temp = apply_operator(psi, cre[s], sites[1])
                temp = apply_operator(temp, cre[s], sites[0])
                psi0 = temp if psi0 is None else psi0 + temp
            psi = psi0 / psi0.norm()
        # product state between spin-spicies; 2 fermions in each spicies

        for s in spins:
            psi2 = apply_operator(psi, cre[s], site=1)
            psi2 = apply_operator(psi2, ann[s], site=3)  # 0.5 * |12> - 0.5 * |01>
            assert abs(yast.vdot(psi, psi2)) < tol
            assert pytest.approx(yast.vdot(psi2, psi2).item(), rel=tol) == 0.5

            psi3 = apply_operator(psi, ann[s], site=0)
            psi3 = apply_operator(psi3, cre[s], site=2)  # 0.5 * |23> - 0.5 |12>
            assert abs(yast.vdot(psi, psi3)) < tol
            assert pytest.approx(yast.vdot(psi3, psi3).item(), rel=tol) == 0.5

            assert pytest.approx(yast.vdot(psi2, psi3).item(), rel=tol) == -0.25
            psi = apply_operator(psi, cre[s], site=2) # adds particle on site 2 'up' for next loop
            psi = psi / psi.norm()


def test_operators():
    """ test commutation rules for creation and anihilation operators for spinfull fermions. """
    spins, ops = ('up', 'dn'), ('cre', 'ann')
    for sym, inter_sgn in [('Z2', 1), ('U1xU1_ind', 1), ('U1xU1_dis', -1)]:
        c = {op + s: operator_spinfull(op, s=s, sym=sym) for s, op in product(spins, ops)}
        one = operator_spinfull('one', sym=sym)

        # check anti-commutation relations
        assert all(yast.norm(c[op + s] @ c[op + s]) < tol for s, op in product(spins, ops))
        assert all(yast.norm(c['ann' + s] @ c['cre' + s] + c['cre' + s] @ c['ann' + s] - one) < tol for s in spins)

        # anticommutator for indistinguishable; commutator for distinguishable
        assert all(yast.norm(c[op1 + 'up'] @ c[op2 + 'dn'] + inter_sgn * c[op2 + 'dn'] @ c[op1 + 'up']) < tol
                    for op1, op2 in product(ops, ops))


def vacum_spinfull(sites=4, sym='Z2'):
    """ |Psi> as a tensor with proper symmetry information and one axis per site. """
    s = (1,) * sites
    if sym == 'Z2':
        psi = yast.zeros(config=config_Z2_fermionic, s=s, t=[((0,),)] * sites, D=(2,) * sites)
        psi[(0,) * sites][0] = 1.
    if sym == 'U1xU1_ind':
        psi = yast.ones(config=config_U1xU1xZ2_fermionic, s=s, t=[((0, 0, 0),)] * sites, D=(1,) * sites)
    if sym == 'U1xU1_dis':
        psi = yast.ones(config=config_U1xU1_fermionic, s=s, t=[((0, 0),)] * sites, D=(1,) * sites)
    return psi


def operator_spinfull(op='ann', s='up', sym='Z2'):
    """ define operators for spinfull local Hilbert space and various symmetries. """
    if op == 'ann':  # annihilation
        if sym == 'Z2' and s == 'up': # charges: 0 <-> (|00>, |11>); 1 <-> (|10>, |01>)
            temp = yast.Tensor(config=config_Z2_fermionic, s=(1, -1), n=1)
            temp.set_block(ts=(0, 1), Ds=(2, 2), val=[[1, 0], [0, 0]])
            temp.set_block(ts=(1, 0), Ds=(2, 2), val=[[0, 0], [0, 1]])
        if sym == 'Z2' and s == 'dn':
            temp = yast.Tensor(config=config_Z2_fermionic, s=(1, -1), n=1)
            temp.set_block(ts=(0, 1), Ds=(2, 2), val=[[0, 1], [0, 0]])
            temp.set_block(ts=(1, 0), Ds=(2, 2), val=[[0, -1], [0, 0]])
        if sym == 'U1xU1_ind' and s == 'up': # charges <-> (ocupation up, occupation down, total_parity)
            temp = yast.Tensor(config=config_U1xU1xZ2_fermionic, s=(1, -1), n=(-1, 0, 1))
            temp.set_block(ts=((0, 0, 0), (1, 0, 1)), Ds=(1, 1), val=1)
            temp.set_block(ts=((0, 1, 1), (1, 1, 0)), Ds=(1, 1), val=1)
        if sym == 'U1xU1_ind' and s == 'dn':
            temp = yast.Tensor(config=config_U1xU1xZ2_fermionic, s=(1, -1), n=(0, -1, 1))
            temp.set_block(ts=((0, 0, 0), (0, 1, 1)), Ds=(1, 1), val=1)
            temp.set_block(ts=((1, 0, 1), (1, 1, 0)), Ds=(1, 1), val=-1)
        if sym == 'U1xU1_dis' and s == 'up':  # charges <-> (ocupation up, occupation down)
            temp = yast.Tensor(config=config_U1xU1_fermionic, s=(1, -1), n=(-1, 0))
            temp.set_block(ts=((0, 0), (1, 0)), Ds=(1, 1), val=1)
            temp.set_block(ts=((0, 1), (1, 1)), Ds=(1, 1), val=1)
        if sym == 'U1xU1_dis' and s == 'dn':
            temp = yast.Tensor(config=config_U1xU1_fermionic, s=(1, -1), n=(0, -1))
            temp.set_block(ts=((0, 0), (0, 1)), Ds=(1, 1), val=1)
            temp.set_block(ts=((1, 0), (1, 1)), Ds=(1, 1), val=1)
    if op == 'cre':  # creation
        if sym == 'Z2' and s == 'up':  # charges: 0 <-> (|00>, |11>); <-> (|10>, |01>)
            temp = yast.Tensor(config=config_Z2_fermionic, s=(1, -1), n=1)
            temp.set_block(ts=(0, 1), Ds=(2, 2), val=[[0, 0], [0, 1]])
            temp.set_block(ts=(1, 0), Ds=(2, 2), val=[[1, 0], [0, 0]])
        if sym == 'Z2' and s == 'dn':
            temp = yast.Tensor(config=config_Z2_fermionic, s=(1, -1), n=1)
            temp.set_block(ts=(0, 1), Ds=(2, 2), val=[[0, 0], [-1, 0]])
            temp.set_block(ts=(1, 0), Ds=(2, 2), val=[[0, 0], [1, 0]])
        if sym == 'U1xU1_ind' and s == 'up':
            temp = yast.Tensor(config=config_U1xU1xZ2_fermionic, s=(1, -1), n=(1, 0, 1))
            temp.set_block(ts=((1, 0, 1), (0, 0, 0)), Ds=(1, 1), val=1)
            temp.set_block(ts=((1, 1, 0), (0, 1, 1)), Ds=(1, 1), val=1)
        if sym == 'U1xU1_ind' and s == 'dn':
            temp = yast.Tensor(config=config_U1xU1xZ2_fermionic, s=(1, -1), n=(0, 1, 1))
            temp.set_block(ts=((0, 1, 1), (0, 0, 0)), Ds=(1, 1), val=1)
            temp.set_block(ts=((1, 1, 0), (1, 0, 1)), Ds=(1, 1), val=-1)
        if sym == 'U1xU1_dis' and s == 'up':
            temp = yast.Tensor(config=config_U1xU1_fermionic, s=(1, -1), n=(1, 0))
            temp.set_block(ts=((1, 0), (0, 0)), Ds=(1, 1), val=1)
            temp.set_block(ts=((1, 1), (0, 1)), Ds=(1, 1), val=1)
        if sym == 'U1xU1_dis' and s == 'dn':
            temp = yast.Tensor(config=config_U1xU1_fermionic, s=(1, -1), n=(0, 1))
            temp.set_block(ts=((0, 1), (0, 0)), Ds=(1, 1), val=1)
            temp.set_block(ts=((1, 1), (1, 0)), Ds=(1, 1), val=1)
    if op == 'one':  # identity
        if sym == 'Z2':
            temp = yast.Tensor(config=config_Z2_fermionic, s=(1, -1), n=0)
            for t in [0, 1]:
                temp.set_block(ts=(t, t), Ds=(2, 2), val=[[1, 0], [0, 1]])
        if sym == 'U1xU1_ind':
            temp = yast.Tensor(config=config_U1xU1xZ2_fermionic, s=(1, -1), n=(0, 0, 0))
            for t in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
                temp.set_block(ts=(t, t), Ds=(1, 1), val=1)
        if sym == 'U1xU1_dis':
            temp = yast.Tensor(config=config_U1xU1_fermionic, s=(1, -1), n=(0, 0))
            for t in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                temp.set_block(ts=(t, t), Ds=(1, 1), val=1)
    return temp


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


if __name__ == '__main__':
    test_swap_gate_basic()
    test_operators()
    test_apply_operators()
    test_swap_gate_exceptions()
