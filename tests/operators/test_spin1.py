""" Predefined operators """
import pytest
import numpy as np
import yastn
try:
    from .configs import config_dense
except ImportError:
    from configs import config_dense


tol = 1e-12  #pylint: disable=invalid-name


def test_spin1():
    """ Generate standard operators in 3-dimensional Hilbert space for various symmetries. """
    # pytest switches backends and default_device in config files for testing
    backend = config_dense.backend
    default_device = config_dense.default_device

    print(config_dense)
    print(config_dense.backend)
    print(config_dense.default_device)

    ops_dense = yastn.operators.Spin1(sym='dense', backend=backend, default_device=default_device)
    ops_Z3 = yastn.operators.Spin1(sym='Z3', backend=backend, default_device=default_device)
    ops_U1 = yastn.operators.Spin1(sym='U1', backend=backend, default_device=default_device)
    rs = (False, False, True) # reverse option for to_numpy/to_dense/to_nonsymmetric

    assert all(ops.config.fermionic == False for ops in (ops_dense, ops_Z3, ops_U1))

    Is = [ops_dense.I(), ops_Z3.I(), ops_U1.I()]
    assert all(np.allclose(I.to_numpy(reverse=r), np.eye(3)) for (I, r) in zip(Is, rs))
    assert all(I.device[:len(default_device)] == default_device for I in Is)  # for cuda, accept cuda:0 == cuda

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

    assert all(yastn.norm(Sx + 1j * Sy - Sp) < tol for Sx, Sy, Sp in zip(Sxs, Sys, Sps))
    assert all(yastn.norm(Sx - 1j * Sy - Sm) < tol for Sx, Sy, Sm in zip(Sxs, Sys, Sms))

    assert all(yastn.norm(Sp @ Sm - Sm @ Sp - 2 * Sz) < tol for Sp, Sm, Sz in zip(Sps, Sms, Szs))
    assert all(yastn.norm(Sx @ Sy - Sy @ Sx - 1j * Sz) < tol for Sx, Sy, Sz in zip(Sxs, Sys, Szs))
    assert all(yastn.norm(Sz @ Sp - Sp @ Sz - Sp) < tol for Sz, Sp in zip(Szs, Sps))
    assert all(yastn.norm(Sz @ Sm - Sm @ Sz + Sm) < tol for Sz, Sm in zip(Szs, Sms))

    with pytest.raises(yastn.YastnError):
        _ = ops_U1.sx()
        # Cannot define Sx operator for U(1) or Z3 symmetry.
    with pytest.raises(yastn.YastnError):
        _ = ops_Z3.sy()
        # Cannot define Sy operator for U(1) or Z3 symmetry.
    with pytest.raises(yastn.YastnError):
        yastn.operators.Spin1('wrong symmetry')
        # For Spin1 sym should be in ('dense', 'Z3', 'U1').


if __name__ == '__main__':
    test_spin1()
