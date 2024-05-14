""" Predefined spinful fermion operators operators """
import pytest
import numpy as np
import yastn
try:
    from .configs import config_dense
except ImportError:
    from configs import config_dense


tol = 1e-12  #pylint: disable=invalid-name


def test_spinful_fermions():
    """ Generate standard operators in two-dimensional Hilbert space for various symmetries. """
    # pytest switches backends and default_device in config files for testing
    backend = config_dense.backend
    default_device = config_dense.default_device

    ops_Z2 = yastn.operators.SpinfulFermions(sym='Z2', backend=backend, default_device=default_device)
    ops_U1 = yastn.operators.SpinfulFermions(sym='U1', backend=backend, default_device=default_device)
    ops_U1xU1_ind = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', backend=backend, default_device=default_device)
    ops_U1xU1_dis = yastn.operators.SpinfulFermions(sym='U1xU1', backend=backend, default_device=default_device)

    Is = [ops_Z2.I(), ops_U1.I(), ops_U1xU1_ind.I(), ops_U1xU1_dis.I()]
    legs = [ops_Z2.space(), ops_U1.space(), ops_U1xU1_ind.space(), ops_U1xU1_dis.space()]

    assert all(leg == I.get_legs(axes=0) for (leg, I) in zip(legs, Is))
    assert all(np.allclose(I.to_numpy(), np.eye(4)) for I in Is)
    assert all(default_device in I.device for I in Is)  # accept 'cuda' in 'cuda:0'

    assert all(ops.config.fermionic == fs for ops, fs in zip((ops_Z2, ops_U1, ops_U1xU1_ind, ops_U1xU1_dis), (True, True, (False, False, True), True)))

    for ops, inter_sgn in [(ops_Z2, 1), (ops_U1, 1), (ops_U1xU1_ind, 1), (ops_U1xU1_dis, -1)]:
        # check anti-commutation relations
        assert all(yastn.norm(ops.c(s) @ ops.c(s)) < tol for s in ('u', 'd'))
        assert all(yastn.norm(ops.c(s) @ ops.cp(s) + ops.cp(s) @ ops.c(s) - ops.I()) < tol for s in ('u', 'd'))
        assert all(yastn.norm(ops.c(s) @ ops.cp(s) + ops.n(s) - ops.I()) < tol for s in ('u', 'd'))

        # anticommutator for indistinguishable; commutator for distinguishable
        assert yastn.norm(ops.c('u') @ ops.cp('d') + inter_sgn * ops.cp('d') @ ops.c('u')) < tol
        assert yastn.norm(ops.c('u') @ ops.c('d') + inter_sgn * ops.c('d') @ ops.c('u')) < tol
        assert yastn.norm(ops.cp('u') @ ops.cp('d') + inter_sgn * ops.cp('d') @ ops.cp('u')) < tol
        assert yastn.norm(ops.cp('u') @ ops.c('d') + inter_sgn * ops.c('d') @ ops.cp('u')) < tol

        v00, v10, v01, v11 = ops.vec_n((0, 0)), ops.vec_n((1, 0)), ops.vec_n((0, 1)), ops.vec_n((1, 1))
        nu, nd = ops.n('u'), ops.n('d')
        assert yastn.norm(nu @ v00) < tol and yastn.norm(nd @ v00) < tol
        assert yastn.norm(nu @ v01) < tol and yastn.norm(nd @ v01 - v01) < tol
        assert yastn.norm(nu @ v10 - v10) < tol and yastn.norm(nd @ v10) < tol
        assert yastn.norm(nu @ v11 - v11) < tol and yastn.norm(nd @ v11 - v11) < tol

    with pytest.raises(yastn.YastnError):
        yastn.operators.SpinfulFermions('dense')
        # For SpinlessFermions sym should be in ('Z2', 'U1xU1', 'U1xU1xZ2').
    with pytest.raises(yastn.YastnError):
        ops_Z2.c(spin='down')
        # spin shoul be equal 'u' or 'd'.
    with pytest.raises(yastn.YastnError):
        ops_Z2.cp(spin=+1)
        # spin shoul be equal 'u' or 'd'.
    with pytest.raises(yastn.YastnError):
        ops_Z2.vec_n(1)
        # Occupations given by val should be (0, 0), (1, 0), (0, 1), or (1, 1).

    d = ops_Z2.to_dict()  # dict used in mps.Generator
    (d["I"](3) - ops_Z2.I()).norm() < tol  # here 3 is a posible position in the mps
    d.keys() == ops_Z2.operators
    assert all(k in d for k in ('I', 'nu', 'cu', 'cpu', 'nd', 'cd', 'cpd'))


if __name__ == '__main__':
    test_spinful_fermions()
