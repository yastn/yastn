""" Predefined spin-1 operators """
from itertools import chain
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

    ops_dense = yastn.operators.Spin1(sym='dense', backend=backend, default_device=default_device)
    ops_Z3 = yastn.operators.Spin1(sym='Z3', backend=backend, default_device=default_device)
    ops_U1 = yastn.operators.Spin1(sym='U1', backend=backend, default_device=default_device)
    rs = (False, False, True) # reverse option for to_numpy/to_dense/to_nonsymmetric

    assert all(ops.config.fermionic == False for ops in (ops_dense, ops_Z3, ops_U1))

    Is = [ops_dense.I(), ops_Z3.I(), ops_U1.I()]
    legs = [ops_dense.space(), ops_Z3.space(), ops_U1.space()]

    assert all(leg == I.get_legs(axes=0) for (leg, I) in zip(legs, Is))
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

    zp1s = [ops_dense.vec_z(val=+1), ops_Z3.vec_z(val=+1), ops_U1.vec_z(val=+1)]
    z0s = [ops_dense.vec_z(val=0), ops_Z3.vec_z(val=0), ops_U1.vec_z(val=0)]
    zm1s = [ops_dense.vec_z(val=-1), ops_Z3.vec_z(val=-1), ops_U1.vec_z(val=-1)]
    xp1s = [ops_dense.vec_x(val=+1)]
    x0s = [ops_dense.vec_x(val=0)]
    xm1s = [ops_dense.vec_x(val=-1)]
    yp1s = [ops_dense.vec_y(val=+1)]
    y0s = [ops_dense.vec_y(val=0)]
    ym1s = [ops_dense.vec_y(val=-1)]

    assert all(yastn.norm(Sz @ v - v) < tol for Sz, v in zip(Szs, zp1s))
    assert all(yastn.norm(Sz @ v) < tol for Sz, v in zip(Szs, z0s))
    assert all(yastn.norm(Sz @ v + v) < tol for Sz, v in zip(Szs, zm1s))
    assert all(yastn.norm(Sx @ v - v) < tol for Sx, v in zip(Sxs, xp1s))
    assert all(yastn.norm(Sx @ v) < tol for Sx, v in zip(Sxs, x0s))
    assert all(yastn.norm(Sx @ v + v) < tol for Sx, v in zip(Sxs, xm1s))
    assert all(yastn.norm(Sy @ v - v) < tol for Sy, v in zip(Sys, yp1s))
    assert all(yastn.norm(Sy @ v) < tol for Sy, v in zip(Sys, y0s))
    assert all(yastn.norm(Sy @ v + v) < tol for Sy, v in zip(Sys, ym1s))
    assert all(pytest.approx(v.norm(), rel=tol) == 1 for v in chain(zp1s, z0s, zm1s, xp1s, x0s, xm1s, yp1s, y0s, ym1s))


    vecss = [ops_dense.vec_s(), ops_Z3.vec_s(), ops_U1.vec_s()]
    gs = [ops_dense.g(), ops_Z3.g(), ops_U1.g()]
    assert all(vs.s == (-1, 1, -1) and vs.get_shape() == (3, 3, 3) for vs in vecss)
    assert all(g.s == (1, 1) and g.get_shape() == (3, 3) for g in gs)

    S = 1
    for vs, g, I in zip(vecss, gs, Is):
        assert (yastn.ncon((vs, g, vs), ((1, -0, 3), (1, 2), (2, 3, -1))) - S * (S + 1) * I).norm() < tol


    with pytest.raises(yastn.YastnError):
        _ = ops_U1.sx()
        # Cannot define Sx operator for U(1) or Z3 symmetry.
    with pytest.raises(yastn.YastnError):
        _ = ops_Z3.sy()
        # Cannot define Sy operator for U(1) or Z3 symmetry.
    with pytest.raises(yastn.YastnError):
        yastn.operators.Spin1('wrong symmetry')
        # For Spin1 sym should be in ('dense', 'Z3', 'U1').
    with pytest.raises(yastn.YastnError):
        ops_U1.vec_z(val=10)
        # Eigenvalues val should be in (-1, 0, 1).
    with pytest.raises(yastn.YastnError):
        ops_dense.vec_x(val=10)
        # Eigenvalues val should be in (-1, 0, 1) and eigenvectors of Sx are well defined only for dense tensors.
    with pytest.raises(yastn.YastnError):
        ops_Z3.vec_y(val=1)
        # Eigenvalues val should be in (-1, 0, 1) and eigenvectors of Sy are well defined only for dense tensors.


    # used in mps Generator
    d = ops_dense.to_dict()
    (d["I"](3) - ops_dense.I()).norm() < tol  # here 3 is a posible position in the mps
    assert all(k in d for k in ('I', 'sx', 'sy', 'sz', 'sp', 'sm'))


if __name__ == '__main__':
    test_spin1()
