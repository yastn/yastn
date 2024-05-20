""" Predefined spinful fermion operators operators """
import pytest
import numpy as np
import yastn
try:
    from .configs import config_dense
except ImportError:
    from configs import config_dense


tol = 1e-12  #pylint: disable=invalid-name


def test_spinful_fermions_tJ():
    """ Generate standard operators in two-dimensional Hilbert space for tJ model. """
    # pytest switches backends and default_device in config files for testing
    backend = config_dense.backend
    default_device = config_dense.default_device

    ops_U1xU1_Z2_tJ = yastn.operators.SpinfulFermions_tJ('U1xU1xZ2', backend=backend, default_device=default_device)
    ops_U1xU1_tJ = yastn.operators.SpinfulFermions_tJ('U1xU1', backend=backend, default_device=default_device)
    ops_Z2_tJ = yastn.operators.SpinfulFermions_tJ('Z2', backend=backend, default_device=default_device)

    Is = [ops_U1xU1_Z2_tJ.I(), ops_U1xU1_tJ.I(), ops_Z2_tJ.I()]
    legs = [ops_U1xU1_Z2_tJ.space(), ops_U1xU1_tJ.space(), ops_Z2_tJ.space()]

    assert all(leg == I.get_legs(axes=0) for (leg, I) in zip(legs, Is))
    assert all(np.allclose(I.to_numpy(), np.eye(3)) for I in Is)
    assert all(default_device in I.device for I in Is)  # accept 'cuda' in 'cuda:0'
    
    assert (ops_U1xU1_Z2_tJ.config.fermionic) == (False, False, True)
    
    for ops in [ops_Z2_tJ, ops_U1xU1_Z2_tJ, ops_U1xU1_tJ]:
        # check c^2 = 0
        assert all(yastn.norm(ops.c(s) @ ops.c(s)) < tol for s in ('u', 'd'))
        
        # check no double-occupancy
        assert yastn.norm(ops.cp('u') @ ops.cp('d')) < tol
        assert yastn.norm(ops.cp('d') @ ops.cp('u')) < tol
        assert yastn.norm(ops.c('u') @ ops.cp('u') - ops.h()) < tol
        assert yastn.norm(ops.c('d') @ ops.cp('d') - ops.h()) < tol
        
        # check number operators
        assert yastn.norm(ops.cp('u') @ ops.c('u') - ops.n('u')) < tol
        assert yastn.norm(ops.cp('d') @ ops.c('d') - ops.n('d')) < tol
        assert yastn.norm(ops.cp('d') @ ops.c('d') - ops.n('u')) > 1
        assert yastn.norm(ops.cp('u') @ ops.c('u') - ops.n('d')) > 1
        
        # check completeness of projectors
        assert yastn.norm(ops.n('u') + ops.n('d') + ops.h() - ops.I()) < tol
        
        # check commute and anti-commute relation between spin-1/2 operators
        assert yastn.norm(ops.Sp() @ ops.Sm() - ops.Sm() @ ops.Sp() - 2 * ops.Sz()) < tol
        assert yastn.norm(ops.Sz() @ ops.Sp() - ops.Sp() @ ops.Sz() - ops.Sp()) < tol
        assert yastn.norm(ops.Sm() @ ops.Sz() - ops.Sz() @ ops.Sm() - ops.Sm()) < tol
        assert yastn.norm(ops.Sz() @ ops.Sp() + ops.Sp() @ ops.Sz()) < tol
        assert yastn.norm(ops.Sm() @ ops.Sz() + ops.Sz() @ ops.Sm()) < tol
        assert yastn.norm(ops.Sp() @ ops.Sm() + ops.Sm() @ ops.Sp() - ops.n('u') - ops.n('d')) < tol
        
                
        # check basic algebras
        v00, v10, v01 = ops.vec_n((0, 0)), ops.vec_n((1, 0)), ops.vec_n((0, 1))
        nu, nd = ops.n('u'), ops.n('d')
        assert yastn.norm(nu @ v00) < tol and yastn.norm(nd @ v00) < tol
        assert yastn.norm(nu @ v01) < tol and yastn.norm(nd @ v01 - v01) < tol
        assert yastn.norm(nu @ v10 - v10) < tol and yastn.norm(nd @ v10) < tol
        
    d = ops_Z2_tJ.to_dict()  # dict used in mps.Generator
    (d["I"](3) - ops_Z2_tJ.I()).norm() < tol  # here 3 is a posible position in the mps
    d.keys() == ops_Z2_tJ.operators
    assert all(k in d for k in ('I', 'nu', 'cu', 'cpu', 'nd', 'cd', 'cpd', 'Sz', 'Sm', 'Sp', 'h'))

    with pytest.raises(yastn.YastnError):
        yastn.operators.SpinfulFermions('dense')
        # For SpinlessFermions sym should be in ('Z2', 'U1xU1', 'U1xU1xZ2').


if __name__ == '__main__':
    test_spinful_fermions_tJ()
