""" Predefined operators """
import pytest
import numpy as np
import yastn
try:
    from .configs import config_dense
except ImportError:
    from configs import config_dense


tol = 1e-12  #pylint: disable=invalid-name


def test_spin12():
    """ Standard operators and some vectors in two-dimensional Hilbert space for various symmetries. """
    # pytest switches backends and default_device in config files for testing
    backend = config_dense.backend
    default_device = config_dense.default_device

    ops_dense = yastn.operators.Spin12(sym='dense', backend=backend, default_device=default_device)
    ops_Z2 = yastn.operators.Spin12(sym='Z2', backend=backend, default_device=default_device)
    ops_U1 = yastn.operators.Spin12(sym='U1', backend=backend, default_device=default_device)
    rs = (False, False, True) # reverse option for to_numpy/to_dense/to_nonsymmetric

    assert all(ops.config.fermionic == False for ops in (ops_dense, ops_Z2, ops_U1))

    Is = [ops_dense.I(), ops_Z2.I(), ops_U1.I()]
    assert all(np.allclose(I.to_numpy(reverse=r), np.eye(2)) for (I, r) in zip(Is, rs))
    assert all(I.device[:len(default_device)] == default_device for I in Is)  # for cuda, accept cuda:0 == cuda

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

    assert all(yastn.norm(sx + 1j * sy - sp) < tol for sx, sy, sp in zip(sxs, sys, sps))
    assert all(yastn.norm(sx - 1j * sy - sm) < tol for sx, sy, sm in zip(sxs, sys, sms))

    assert all(yastn.norm(sp @ sm - sm @ sp - 2 * sz) < tol for sp, sm, sz in zip(sps, sms, szs))
    assert all(yastn.norm(sx @ sy - sy @ sx - 1j * sz) < tol for sx, sy, sz in zip(sxs, sys, szs))
    assert all(yastn.norm(sz @ sp - sp @ sz - sp) < tol for sz, sp in zip(szs, sps))
    assert all(yastn.norm(sz @ sm - sm @ sz + sm) < tol for sz, sm in zip(szs, sms))

    zp1s = [ops_dense.vec_z(val=+1), ops_Z2.vec_z(val=+1), ops_U1.vec_z(val=+1)]
    zm1s = [ops_dense.vec_z(val=-1), ops_Z2.vec_z(val=-1), ops_U1.vec_z(val=-1)]

    assert all(yastn.norm(z @ v - v) < tol for z, v in zip(zs, zp1s))
    assert all(yastn.norm(z @ v + v) < tol for z, v in zip(zs, zm1s))

    with pytest.raises(yastn.YastnError):
        _ = ops_U1.x()
        # Cannot define sigma_x operator for U(1) symmetry
    with pytest.raises(yastn.YastnError):
        _ = ops_U1.y()
        # Cannot define sigma_y operator for U(1) symmetry
    with pytest.raises(yastn.YastnError):
        yastn.operators.Spin12('wrong symmetry')
        # For Spin12 sym should be in ('dense', 'Z2', 'U1').
    with pytest.raises(yastn.YastnError):
        ops_U1.vec_z(val=10)
        # For Spin12 val in vec_z should be in (-1, 1).


if __name__ == '__main__':
    test_spin12()
