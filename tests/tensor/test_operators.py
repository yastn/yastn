""" Predefined operators """
import pytest
import numpy as np
import yast

tol = 1e-12  #pylint: disable=invalid-name


def test_spin12():
    """ Generate standard operators in two-dimensional Hilbert space for various symmetries. """
    ops_dense = yast.operators.Spin12(sym='dense')
    ops_Z2 = yast.operators.Spin12(sym='Z2')
    ops_U1 = yast.operators.Spin12(sym='U1')

    Is = [ops_dense.I(), ops_Z2.I(), ops_U1.I()]
    assert all(np.allclose(I.to_numpy(), np.array([[1, 0], [0, 1]])) for I in Is)

    Zs = [ops_dense.Z(), ops_Z2.Z(), ops_U1.Z()]
    assert all(np.allclose(Z.to_numpy(), np.array([[1, 0], [0, -1]])) for Z in Zs)

    Xs = [ops_dense.X(), ops_Z2.X()]
    assert all(np.allclose(X.to_numpy(), np.array([[0, 1], [1, 0]])) for X in Xs)

    Ys = [ops_dense.Y(), ops_Z2.Y()]
    assert all(np.allclose(Y.to_numpy(), np.array([[0, -1j], [1j, 0]])) for Y in Ys)

    lss = [{0: I.get_legs(0), 1: I.get_legs(1)} for I in Is]

    Sps = [ops_dense.Sp(), ops_Z2.Sp(), ops_U1.Sp()]
    assert all(np.allclose(Sp.to_numpy(legs=ls), np.array([[0, 1], [0, 0]])) for Sp, ls in zip(Sps, lss))

    Sms = [ops_dense.Sm(), ops_Z2.Sm(), ops_U1.Sm()]
    assert all(np.allclose(Sm.to_numpy(legs=ls), np.array([[0, 0], [1, 0]])) for Sm, ls in zip(Sms, lss))

    assert all(yast.norm(X + 1j * Y - 2 * Sp) < tol for X, Y, Sp in zip(Xs, Ys, Sps))
    assert all(yast.norm(X - 1j * Y - 2 * Sm) < tol for X, Y, Sm in zip(Xs, Ys, Sms))

    assert all(yast.norm(Sp @ Sm - Sm @ Sp -  Z) < tol for Sp, Sm, Z in zip(Sps, Sms, Zs))
    assert all(yast.norm(X @ Y - Y @ X - 2j * Z) < tol for X, Y, Z in zip(Xs, Ys, Zs))

    with pytest.raises(yast.YastError):
        _ = ops_U1.X()
        # Cannot define sigma_x operator for U(1) symmetry
    with pytest.raises(yast.YastError):
        _ = ops_U1.Y()
        # Cannot define sigma_y operator for U(1) symmetry
    with pytest.raises(yast.YastError):
        yast.operators.Spin12('wrong symmetry')
        # For Spin12 sym should be in ('dense', 'Z2', 'U1').

if __name__ == '__main__':
    test_spin12()
