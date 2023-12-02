""" yastn.swap_gate() to introduce fermionic statistics. """
import pytest
import yastn
try:
    from .configs import config_dense, config_Z2_fermionic, config_Z2
except ImportError:
    from configs import config_dense, config_Z2_fermionic, config_Z2


tol = 1e-12  #pylint: disable=invalid-name


def test_swap_gate_basic():
    """ basic tests of swap_gate """
    leg = yastn.Leg(config_Z2_fermionic, t=(0, 1), D=(1, 1))
    a = yastn.ones(config=config_Z2_fermionic, legs=[leg, leg, leg, leg], n=0)
    assert pytest.approx(sum(a.to_numpy().ravel()), rel=tol) == 8

    b = a.swap_gate(axes=(0, 1))
    assert pytest.approx(sum(b.to_numpy().ravel()), rel=tol) == 4

    c = a.swap_gate(axes=(0, 2))
    c = c.swap_gate(axes=(1, 2))
    assert pytest.approx(sum(c.to_numpy().ravel()), rel=tol) == 4
    c1 = a.swap_gate(axes=((0, 1), 2))  # swap between group of (0, 1) and 2
    c2 = a.swap_gate(axes=(0, 2, 1, 2)) # swap between 0, 2 and then 1, 2
    assert all(yastn.norm(c - x) < tol for x in (c1, c2))

    d = a.swap_gate(axes=((0, 1), (2, 3))) # swap between group of (0, 1) and (2, 3)
    assert pytest.approx(sum(d.to_numpy().ravel()), rel=tol) == 0

    assert pytest.approx(yastn.vdot(a, b).item(), rel=tol) == 4
    assert pytest.approx(yastn.vdot(a, c).item(), rel=tol) == 4
    assert pytest.approx(yastn.vdot(a, d).item(), rel=tol) == 0
    assert pytest.approx(yastn.vdot(b, c).item(), rel=tol) == 0
    assert pytest.approx(yastn.vdot(b, d).item(), rel=tol) == -4
    assert pytest.approx(yastn.vdot(c, d).item(), rel=tol) == 4

    leg = yastn.Leg(config_Z2, t=(0, 1), D=(1, 1))
    a_bosonic = yastn.ones(config=config_Z2, legs=[leg, leg, leg, leg], n=0)
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
    cpsi = yastn.tensordot(psi, ca, axes=(site, 1))
    cpsi = cpsi.move_leg(source=ndim - 1, destination=site)
    cpsi = cpsi.swap_gate(axes=(tuple(range(site)), ndim))
    cpsi = cpsi.remove_leg(axis=-1)
    return cpsi


def test_apply_operators():
    """
    Apply swap_gate during calculation of expectation value such as, e.g. <psi| c_1 cdag_3 |psi>.

    Use SpinfulFermions defined in :class:`yastn.operators.SpinfulFermions`
    """
    for sym in ['Z2', 'U1xU1xZ2', 'U1xU1']:
        ops = yastn.operators.SpinfulFermions(sym=sym, backend=config_dense.backend, default_device=config_dense.default_device)
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
            assert abs(yastn.vdot(psi, psi2)) < tol
            assert pytest.approx(yastn.vdot(psi2, psi2).item(), rel=tol) == 0.5

            psi3 = apply_operator(psi, ops.c(s), site=0)
            psi3 = apply_operator(psi3, ops.cp(s), site=2)  # 0.5 * |23> - 0.5 |12>
            assert abs(yastn.vdot(psi, psi3)) < tol
            assert pytest.approx(yastn.vdot(psi3, psi3).item(), rel=tol) == 0.5

            assert pytest.approx(yastn.vdot(psi2, psi3).item(), rel=tol) == -0.25
            psi = apply_operator(psi, ops.cp(s), site=2) # adds particle on site 2 'up' for next loop
            psi = psi / psi.norm()


def vacum_spinful(sites, ops):
    """ |Psi> as a tensor with proper symmetry information and one axis per site. """
    s = (1,) * sites
    I = ops.I()
    if ops._sym == 'Z2':
        psi = yastn.zeros(config=I.config, s=s, t=[((0,),)] * sites, D=(2,) * sites)
        psi[(0,) * sites][(0,) * sites] = 1.  # here first [(0,0,..)] is the block charge -- here outputed as ndim object.
        # The second [(0,0,..)] is index in the block.
    else: # ops._sym in ('U1xU1xZ2', 'U1xU1'):
        psi = yastn.ones(config=I.config, s=s, t=[(I.n,)] * sites, D=(1,) * sites)
    return psi


def test_swap_gate_charge():
    """ test is swap_gate(axes, charge) give the same results as swap_gate(axes)"""
    for sym in ['Z2', 'U1xU1xZ2', 'U1xU1']:
        ops = yastn.operators.SpinfulFermions(sym=sym,
                                              backend=config_dense.backend,
                                              default_device=config_dense.default_device)
        for x, y, z in [[ops.cp('u'), ops.cp('d'), ops.c('u')],
                        [ops.cp('u'), ops.c('u'), ops.cp('u')],
                        [ops.n('d'), ops.n('u'), ops.cp('d')],
                        [ops.cp('u'), ops.c('u'), ops.cp('d')]]:

            x1 = x.swap_gate(axes=0, charge=z.n)
            y1 = y.swap_gate(axes=0, charge=z.n)
            xyz1 = yastn.ncon([x1, y1, z], [(-0, -1), (-2, -3), (-4, -5)])

            z = z.add_leg(axis=2)
            xyz2 = yastn.ncon([x, y, z], [(-0, -1), (-2, -3), (-4, -5, -6)])
            xyz2 = xyz2.swap_gate(axes=((0, 2), 6)).remove_leg(axis=6)

            assert (xyz1 - xyz2).norm() < tol


def test_swap_gate_exceptions():
    """ swap_gate raising exceptions """
    t1, D1 = (0, 1, 2, 3), (2, 2, 2, 2)
    a = yastn.rand(config=config_Z2_fermionic, s=(1, -1, 1, -1), t=(t1, t1, t1, t1), D=(D1, D1, D1, D1))
    with pytest.raises(yastn.YastnError):
        a.swap_gate(axes=(0, 1, 2))
        # Odd number of elements in axes. Elements of axes should come in pairs.
    with pytest.raises(yastn.YastnError):
        a.swap_gate(axes=((0, 1), 0))
        # Cannot swap the same index
    with pytest.raises(yastn.YastnError):
        a.swap_gate(axes=(0,), charge=(1, 1))
        # Len of charge (1, 1) does not match sym.NSYM = 1.


if __name__ == '__main__':
    test_swap_gate_basic()
    test_apply_operators()
    test_swap_gate_charge()
    test_swap_gate_exceptions()
