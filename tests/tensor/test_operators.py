""" Predefined operators """
import pytest
from itertools import product
import numpy as np
import yast
try:
    from .configs import config_dense
except ImportError:
    from configs import config_dense

tol = 1e-12  #pylint: disable=invalid-name


def test_spin12():
    """ Generate standard operators in two-dimensional Hilbert space for various symmetries. """
    backend = config_dense.backend  # pytest switches backends in config files for testing

    ops_dense = yast.operators.Spin12(sym='dense', backend=backend)
    ops_Z2 = yast.operators.Spin12(sym='Z2', backend=backend)
    ops_U1 = yast.operators.Spin12(sym='U1', backend=backend)
    rs = (False, False, True) # reverse option for to_numpy/to_dense/to_nonsymmetric

    assert all(ops.config.fermionic == False for ops in (ops_dense, ops_Z2, ops_U1))

    Is = [ops_dense.I(), ops_Z2.I(), ops_U1.I()]
    assert all(np.allclose(I.to_numpy(reverse=r), np.eye(2)) for (I, r) in zip(Is, rs))

    zs = [ops_dense.z(), ops_Z2.z(), ops_U1.z()]
    szs = [ops_dense.sz(), ops_Z2.sz(), ops_U1.sz()]
    assert all(np.allclose(z.to_numpy(reverse=r), np.array([[1, 0], [0, -1]])) for (z, r) in zip(zs, rs))

    xs = [ops_dense.x(), ops_Z2.x()]
    sxs = [ops_dense.sx(), ops_Z2.sx()]
    assert all(np.allclose(x.to_numpy(reverse=r), np.array([[0, 1], [1, 0]])) for (x, r) in zip(xs, rs))

    ys = [ops_dense.y(), ops_Z2.y()]
    sys = [ops_dense.sy(), ops_Z2.sy()]
    assert all(np.allclose(y.to_numpy(reverse=r), np.array([[0, -1j], [1j, 0]])) for (y, r) in zip(ys, rs))

    lss = [{0: I.get_legs(0), 1: I.get_legs(1)} for I in Is]

    sps = [ops_dense.sp(), ops_Z2.sp(), ops_U1.sp()]
    assert all(np.allclose(sp.to_numpy(legs=ls, reverse=r), np.array([[0, 1], [0, 0]])) for sp, ls, r in zip(sps, lss, rs))

    sms = [ops_dense.sm(), ops_Z2.sm(), ops_U1.sm()]
    assert all(np.allclose(Sm.to_numpy(legs=ls, reverse=r), np.array([[0, 0], [1, 0]])) for Sm, ls, r in zip(sms, lss, rs))

    assert all(yast.norm(sx + 1j * sy - sp) < tol for sx, sy, sp in zip(sxs, sys, sps))
    assert all(yast.norm(sx - 1j * sy - sm) < tol for sx, sy, sm in zip(sxs, sys, sms))

    assert all(yast.norm(sp @ sm - sm @ sp - 2 * sz) < tol for sp, sm, sz in zip(sps, sms, szs))
    assert all(yast.norm(sx @ sy - sy @ sx - 1j * sz) < tol for sx, sy, sz in zip(sxs, sys, szs))
    assert all(yast.norm(sz @ sp - sp @ sz - sp) < tol for sz, sp in zip(szs, sps))
    assert all(yast.norm(sz @ sm - sm @ sz + sm) < tol for sz, sm in zip(szs, sms))

    with pytest.raises(yast.YastError):
        _ = ops_U1.x()
        # Cannot define sigma_x operator for U(1) symmetry
    with pytest.raises(yast.YastError):
        _ = ops_U1.y()
        # Cannot define sigma_y operator for U(1) symmetry
    with pytest.raises(yast.YastError):
        yast.operators.Spin12('wrong symmetry')
        # For Spin12 sym should be in ('dense', 'Z2', 'U1').


def test_spin1():
    """ Generate standard operators in 3-dimensional Hilbert space for various symmetries. """
    backend = config_dense.backend  # pytest switches backends in config files for testing

    ops_dense = yast.operators.Spin1(sym='dense', backend=backend)
    ops_Z3 = yast.operators.Spin1(sym='Z3', backend=backend)
    ops_U1 = yast.operators.Spin1(sym='U1', backend=backend)
    rs = (False, False, True) # reverse option for to_numpy/to_dense/to_nonsymmetric

    assert all(ops.config.fermionic == False for ops in (ops_dense, ops_Z3, ops_U1))

    Is = [ops_dense.I(), ops_Z3.I(), ops_U1.I()]
    assert all(np.allclose(I.to_numpy(reverse=r), np.eye(3)) for (I, r) in zip(Is, rs))

    Szs = [ops_dense.sz(), ops_Z3.sz(), ops_U1.sz()]
    assert all(np.allclose(Sz.to_numpy(reverse=r), np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])) for (Sz, r) in zip(Szs, rs))

    Sxs = [ops_dense.sx()]
    assert all(np.allclose(Sx.to_numpy(reverse=r), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)) for (Sx, r) in zip(Sxs, rs))

    Sys = [ops_dense.sy()]
    assert all(np.allclose(Sy.to_numpy(reverse=r), np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)) for (Sy, r) in zip(Sys, rs))

    lss = [{0: I.get_legs(0), 1: I.get_legs(1)} for I in Is]

    Sps = [ops_dense.sp(), ops_Z3.sp(), ops_U1.sp()]
    assert all(np.allclose(Sp.to_numpy(legs=ls, reverse=r), np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]) * np.sqrt(2)) for Sp, ls, r in zip(Sps, lss, rs))

    Sms = [ops_dense.sm(), ops_Z3.sm(), ops_U1.sm()]
    assert all(np.allclose(Sm.to_numpy(legs=ls, reverse=r), np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * np.sqrt(2)) for Sm, ls, r in zip(Sms, lss, rs))

    assert all(yast.norm(Sx + 1j * Sy - Sp) < tol for Sx, Sy, Sp in zip(Sxs, Sys, Sps))
    assert all(yast.norm(Sx - 1j * Sy - Sm) < tol for Sx, Sy, Sm in zip(Sxs, Sys, Sms))

    assert all(yast.norm(Sp @ Sm - Sm @ Sp - 2 * Sz) < tol for Sp, Sm, Sz in zip(Sps, Sms, Szs))
    assert all(yast.norm(Sx @ Sy - Sy @ Sx - 1j * Sz) < tol for Sx, Sy, Sz in zip(Sxs, Sys, Szs))
    assert all(yast.norm(Sz @ Sp - Sp @ Sz - Sp) < tol for Sz, Sp in zip(Szs, Sps))
    assert all(yast.norm(Sz @ Sm - Sm @ Sz + Sm) < tol for Sz, Sm in zip(Szs, Sms))

    with pytest.raises(yast.YastError):
        _ = ops_U1.sx()
        # Cannot define Sx operator for U(1) or Z3 symmetry.
    with pytest.raises(yast.YastError):
        _ = ops_Z3.sy()
        # Cannot define Sy operator for U(1) or Z3 symmetry.
    with pytest.raises(yast.YastError):
        yast.operators.Spin1('wrong symmetry')
        # For Spin1 sym should be in ('dense', 'Z3', 'U1').


def test_spinless_fermions():
    """ Generate standard operators in two-dimensional Hilbert space for various symmetries. """
    backend = config_dense.backend  # pytest switches backends in config files for testing

    ops_Z2 = yast.operators.SpinlessFermions(sym='Z2', backend=backend)
    ops_U1 = yast.operators.SpinlessFermions(sym='U1', backend=backend)

    assert all(ops.config.fermionic == True for ops in (ops_Z2, ops_U1))

    Is = [ops_Z2.I(), ops_U1.I()]
    assert all(np.allclose(I.to_numpy(), np.eye(2)) for I in Is)

    ns = [ops_Z2.n(), ops_U1.n()]
    assert all(np.allclose(n.to_numpy(), np.array([[0, 0], [0, 1]])) for n in ns)

    lss = [{0: I.get_legs(0), 1: I.get_legs(1)} for I in Is]

    cps = [ops_Z2.cp(), ops_U1.cp()]
    assert all(np.allclose(cp.to_numpy(legs=ls), np.array([[0, 0], [1, 0]])) for cp, ls in zip(cps, lss))

    cs = [ops_Z2.c(), ops_U1.c()]
    assert all(np.allclose(c.to_numpy(legs=ls), np.array([[0, 1], [0, 0]])) for c, ls in zip(cs, lss))

    assert all(yast.norm(cp @ c - n) < tol for cp, c, n in zip(cps, cs, ns))

    with pytest.raises(yast.YastError):
        yast.operators.SpinlessFermions('dense')
        # For SpinlessFermions sym should be in ('Z2', 'U1').


def test_spinfull_fermions():
    """ Generate standard operators in two-dimensional Hilbert space for various symmetries. """
    backend = config_dense.backend  # pytest switches backends in config files for testing

    ops_Z2 = yast.operators.SpinfullFermions(sym='Z2', backend=backend)
    ops_U1xU1_ind = yast.operators.SpinfullFermions(sym='U1xU1xZ2', backend=backend)
    ops_U1xU1_dis = yast.operators.SpinfullFermions(sym='U1xU1', backend=backend)

    Is = [ops_Z2.I(), ops_U1xU1_ind.I(), ops_U1xU1_dis.I()]
    assert all(np.allclose(I.to_numpy(), np.eye(4)) for I in Is)

    assert all(ops.config.fermionic == fs for ops, fs in zip((ops_Z2, ops_U1xU1_ind, ops_U1xU1_dis), (True, (False, False, True), True)))

    for ops, inter_sgn in [(ops_Z2, 1), (ops_U1xU1_ind, 1), (ops_U1xU1_dis, -1)]:
        # check anti-commutation relations
        assert all(yast.norm(ops.c(s) @ ops.c(s)) < tol for s in ('u', 'd'))
        assert all(yast.norm(ops.c(s) @ ops.cp(s) + ops.cp(s) @ ops.c(s) - ops.I()) < tol for s in ('u', 'd'))

        # anticommutator for indistinguishable; commutator for distinguishable
        assert yast.norm(ops.c('u') @ ops.cp('d') + inter_sgn * ops.cp('d') @ ops.c('u')) < tol
        assert yast.norm(ops.c('u') @ ops.c('d') + inter_sgn * ops.c('d') @ ops.c('u')) < tol
        assert yast.norm(ops.cp('u') @ ops.cp('d') + inter_sgn * ops.cp('d') @ ops.cp('u')) < tol
        assert yast.norm(ops.cp('u') @ ops.c('d') + inter_sgn * ops.c('d') @ ops.cp('u')) < tol

    with pytest.raises(yast.YastError):
        yast.operators.SpinfullFermions('dense')
        # For SpinlessFermions sym should be in ('Z2', 'U1xU1', 'U1xU1xZ2').
    with pytest.raises(yast.YastError):
        ops_Z2.c(spin='down')
        # spin shoul be equal 'u' or 'd'.
    with pytest.raises(yast.YastError):
        ops_Z2.cp(spin=+1)
    # spin shoul be equal 'u' or 'd'.
    

if __name__ == '__main__':
    test_spin12()
    test_spin1()
    test_spinless_fermions()
    test_spinfull_fermions()
