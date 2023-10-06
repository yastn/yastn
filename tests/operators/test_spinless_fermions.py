""" Predefined operators """
import pytest
import numpy as np
import yastn
try:
    from .configs import config_dense
except ImportError:
    from configs import config_dense


tol = 1e-12  #pylint: disable=invalid-name


def test_spinless_fermions():
    """ Generate standard operators in two-dimensional Hilbert space for various symmetries. """
    # pytest switches backends and default_device in config files for testing
    backend = config_dense.backend
    default_device = config_dense.default_device

    ops_Z2 = yastn.operators.SpinlessFermions(sym='Z2', backend=backend, default_device=default_device)
    ops_U1 = yastn.operators.SpinlessFermions(sym='U1', backend=backend, default_device=default_device)

    assert all(ops.config.fermionic == True for ops in (ops_Z2, ops_U1))

    Is = [ops_Z2.I(), ops_U1.I()]
    assert all(np.allclose(I.to_numpy(), np.eye(2)) for I in Is)
    assert all(I.device[:len(default_device)] == default_device for I in Is)  # for cuda, accept cuda:0 == cuda

    lss = [{0: I.get_legs(0), 1: I.get_legs(1)} for I in Is]

    ns = [ops_Z2.n(), ops_U1.n()]
    assert all(np.allclose(n.to_numpy(legs=ls), np.array([[0, 0], [0, 1]])) for n, ls in zip(ns, lss))

    cps = [ops_Z2.cp(), ops_U1.cp()]
    assert all(np.allclose(cp.to_numpy(legs=ls), np.array([[0, 0], [1, 0]])) for cp, ls in zip(cps, lss))

    cs = [ops_Z2.c(), ops_U1.c()]
    assert all(np.allclose(c.to_numpy(legs=ls), np.array([[0, 1], [0, 0]])) for c, ls in zip(cs, lss))

    assert all(yastn.norm(cp @ c - n) < tol for cp, c, n in zip(cps, cs, ns))

    n0s = [ops_Z2.vec_n(val=0), ops_U1.vec_n(val=0)]
    n1s = [ops_Z2.vec_n(val=1), ops_U1.vec_n(val=1)]

    assert all(yastn.norm(n @ v) < tol for n, v in zip(ns, n0s))
    assert all(yastn.norm(n @ v - v) < tol for n, v in zip(ns, n1s))
    assert all(yastn.norm(cp @ v0 - v1) < tol for cp, v0, v1 in zip(cps, n0s, n1s))
    assert all(yastn.norm(c @ v1 - v0) < tol for c, v0, v1 in zip(cs, n0s, n1s))
    assert all(yastn.norm(c @ v0) < tol for c, v0 in zip(cs, n0s))
    assert all(yastn.norm(cp @ v1) < tol for cp, v1 in zip(cps, n1s))


    with pytest.raises(yastn.YastnError):
        yastn.operators.SpinlessFermions('dense')
        # For SpinlessFermions sym should be in ('Z2', 'U1').
    with pytest.raises(yastn.YastnError):
        ops_U1.vec_n(val=-1)
        # For SpinlessFermions val in vec_n should be in (0, 1).


if __name__ == '__main__':
    test_spinless_fermions()
