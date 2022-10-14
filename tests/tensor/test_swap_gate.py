""" yast.swap_gate() to introduce fermionic statistics. """
from itertools import product
import pytest
from tests.tensor.configs import config_dense
import yast
try:
    from .configs import config_U1xU1_fermionic, config_U1xU1xZ2_fermionic, config_Z2_fermionic, config_Z2
except ImportError:
    from configs import config_U1xU1_fermionic, config_U1xU1xZ2_fermionic, config_Z2_fermionic, config_Z2



tol = 1e-12  #pylint: disable=invalid-name


def test_swap_gate_basic():
    """ basic tests of swap_gate """
    leg = yast.Leg(config_Z2_fermionic, t=(0, 1), D=(1, 1))
    a = yast.ones(config=config_Z2_fermionic, legs=[leg, leg, leg, leg], n=0)
    assert pytest.approx(sum(a.to_numpy().ravel()), rel=tol) == 8

    b = a.swap_gate(axes=(0, 1))
    assert pytest.approx(sum(b.to_numpy().ravel()), rel=tol) == 4

    c = a.swap_gate(axes=(0, 2))
    c = c.swap_gate(axes=(1, 2))
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

    leg = yast.Leg(config_Z2, t=(0, 1), D=(1, 1))
    a_bosonic = yast.ones(config=config_Z2, legs=[leg, leg, leg, leg], n=0)
    b_bosonic = a_bosonic.swap_gate(axes=(0, 1))
    assert a_bosonic is b_bosonic


def apply_operator(psi, c, site):
    """
    Apply operator c on site of psi.

    Use swap_gate to impose fermionic statistics.
    Fermionic order from first to last site.
    """
    ndim = psi.ndim
    ca = c.add_leg(axis=-1)
    cpsi = yast.tensordot(psi, ca, axes=(site, 1))
    cpsi = cpsi.move_leg(source=ndim - 1, destination=site)
    cpsi = cpsi.swap_gate(axes=(tuple(range(site)), ndim))
    cpsi = cpsi.remove_leg(axis=-1)
    return cpsi


def test_apply_operators():
    """ 
    Apply swap_gate during calculation of expectation value such as, e.g. <psi| c_1 cdag_3 |psi>.

    Use SpinfulFermions defined in :class:`yast.operators.SpinfulFermions`
    """
    for sym in ['Z2', 'U1xU1xZ2', 'U1xU1']:
        ops = yast.operators.SpinfulFermions(sym=sym, backend=config_dense.backend)
        # pytest switches backends in configs imported in tests

        vac = vacum_spinful(sites=4, ops=ops)
        psi = vac
        for s in ('u', 'd'):
            psi0 = None
            for sites in ((2, 3), (1, 2), (0, 3), (0, 1)):  # sum of fermions created on those sites
                temp = apply_operator(psi, ops.cp(s), sites[1])
                temp = apply_operator(temp, ops.cp(s), sites[0])
                psi0 = temp if psi0 is None else psi0 + temp
            psi = psi0 / psi0.norm()
        # product state between spin-spicies; 2 fermions in each spicies

        for s in ('u', 'd'):
            psi2 = apply_operator(psi, ops.cp(s), site=1)
            psi2 = apply_operator(psi2, ops.c(s), site=3)  # 0.5 * |12> - 0.5 * |01>
            assert abs(yast.vdot(psi, psi2)) < tol
            assert pytest.approx(yast.vdot(psi2, psi2).item(), rel=tol) == 0.5

            psi3 = apply_operator(psi, ops.c(s), site=0)
            psi3 = apply_operator(psi3, ops.cp(s), site=2)  # 0.5 * |23> - 0.5 |12>
            assert abs(yast.vdot(psi, psi3)) < tol
            assert pytest.approx(yast.vdot(psi3, psi3).item(), rel=tol) == 0.5

            assert pytest.approx(yast.vdot(psi2, psi3).item(), rel=tol) == -0.25
            psi = apply_operator(psi, ops.cp(s), site=2) # adds particle on site 2 'up' for next loop
            psi = psi / psi.norm()


def vacum_spinful(sites, ops):
    """ |Psi> as a tensor with proper symmetry information and one axis per site. """
    s = (1,) * sites
    I = ops.I()
    if ops._sym == 'Z2':
        psi = yast.zeros(config=I.config, s=s, t=[((0,),)] * sites, D=(2,) * sites)
        psi[(0,) * sites][(0,) * sites] = 1.  # here first [(0,0,..)] is the block charge -- here outputed as ndim object. 
        # The second [(0,0,..)] is index in the block.
    else: # ops._sym in ('U1xU1xZ2', 'U1xU1'):
        psi = yast.ones(config=I.config, s=s, t=[(I.n,)] * sites, D=(1,) * sites)
    return psi


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
    test_apply_operators()
    test_swap_gate_exceptions()
