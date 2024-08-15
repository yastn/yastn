# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

    ops_U1xU1xZ2_tJ = yastn.operators.SpinfulFermions_tJ(sym='U1xU1xZ2', backend=backend, default_device=default_device)
    ops_U1xU1_tJ = yastn.operators.SpinfulFermions_tJ(sym='U1xU1', backend=backend, default_device=default_device)
    # other way to initialize
    config_Z2 = yastn.make_config(fermionic=True, sym="Z2", backend=backend, default_device=default_device)
    ops_Z2_tJ = yastn.operators.SpinfulFermions_tJ(**config_Z2._asdict())
    ops_U1_tJ = yastn.operators.SpinfulFermions_tJ(sym='U1', backend=backend, default_device=default_device)

    Is = [ops_Z2_tJ.I(), ops_U1_tJ.I(), ops_U1xU1xZ2_tJ.I(), ops_U1xU1_tJ.I(), ]
    legs = [ops_Z2_tJ.space(), ops_U1_tJ.space(), ops_U1xU1xZ2_tJ.space(), ops_U1xU1_tJ.space()]

    assert all(leg == I.get_legs(axes=0) for (leg, I) in zip(legs, Is))
    assert all(np.allclose(I.to_numpy(), np.eye(3)) for I in Is)
    assert all(default_device in I.device for I in Is)  # accept 'cuda' in 'cuda:0'
    assert all(ops.config.fermionic == fs for ops, fs in zip((ops_Z2_tJ, ops_U1_tJ, ops_U1xU1_tJ, ops_U1xU1xZ2_tJ), (True, True, True, (False, False, True))))

    for ops in [ops_Z2_tJ, ops_U1_tJ, ops_U1xU1xZ2_tJ, ops_U1xU1_tJ]:
        for s in ('u', 'd'):
            # check c^2 = 0
            assert yastn.norm(ops.c(s) @ ops.c(s)) < tol
            assert yastn.norm(ops.cp(s) @ ops.cp(s)) < tol
            # check number operators
            assert yastn.norm(ops.cp(s) @ ops.c(s) - ops.n(s)) < tol
            assert yastn.norm(ops.c(s) @ ops.cp(s) - ops.h()) < tol

        # check no double-occupancy
        assert yastn.norm(ops.cp('u') @ ops.cp('d')) < tol
        assert yastn.norm(ops.cp('d') @ ops.cp('u')) < tol

        # check completeness of projectors
        assert yastn.norm(ops.n('u') + ops.n('d') + ops.h() - ops.I()) < tol

        # check commute and anti-commute relation between spin-1/2 operators
        assert yastn.norm(ops.Sp() @ ops.Sm() - ops.Sm() @ ops.Sp() - 2 * ops.Sz()) < tol
        assert yastn.norm(ops.Sz() @ ops.Sp() - ops.Sp() @ ops.Sz() - ops.Sp()) < tol
        assert yastn.norm(ops.Sz() @ ops.Sm() - ops.Sm() @ ops.Sz() + ops.Sm()) < tol
        assert yastn.norm(ops.Sz() @ ops.Sp() + ops.Sp() @ ops.Sz()) < tol
        assert yastn.norm(ops.Sm() @ ops.Sz() + ops.Sz() @ ops.Sm()) < tol
        assert yastn.norm(ops.Sp() @ ops.Sm() + ops.Sm() @ ops.Sp() - (ops.n('u') + ops.n('d'))) < tol

        # check basic algebras
        v00, v10, v01 = ops.vec_n((0, 0)), ops.vec_n((1, 0)), ops.vec_n((0, 1))
        assert yastn.norm(ops.n('u') @ v00) < tol
        assert yastn.norm(ops.n('d') @ v00) < tol
        assert yastn.norm(ops.h() @ v00 - v00) < tol

        assert yastn.norm(ops.n('u') @ v01) < tol
        assert yastn.norm(ops.n('d') @ v01 - v01) < tol
        assert yastn.norm(ops.h() @ v01) < tol

        assert yastn.norm(ops.n('u') @ v10 - v10) < tol
        assert yastn.norm(ops.n('d') @ v10) < tol
        assert yastn.norm(ops.h() @ v10) < tol

        assert yastn.norm(ops.c('u') @ v10 - v00) < tol
        assert yastn.norm(ops.c('d') @ v01 - v00) < tol
        assert yastn.norm(ops.cp('u') @ v00 - v10) < tol
        assert yastn.norm(ops.cp('d') @ v00 - v01) < tol


    d = ops_Z2_tJ.to_dict()  # dict used in mps.Generator
    (d["I"](3) - ops_Z2_tJ.I()).norm() < tol  # here 3 is a posible position in the mps
    d.keys() == ops_Z2_tJ.operators
    assert all(k in d for k in ('I', 'nu', 'cu', 'cpu', 'nd', 'cd', 'cpd', 'Sz', 'Sm', 'Sp', 'h'))

    with pytest.raises(yastn.YastnError):
        yastn.operators.SpinfulFermions_tJ(sym='dense')
        # For SpinfulFermions_tJ sym should be in ('Z2', 'U1xU1', 'U1xU1xZ2').
    with pytest.raises(yastn.YastnError):
        yastn.operators.SpinfulFermions_tJ(sym='Z2', fermionic=False)
        # For SpinfulFermions_tJ config.sym does not match config.fermionic.
    with pytest.raises(yastn.YastnError):
        yastn.operators.SpinfulFermions_tJ(sym='U1xU1xZ2', fermionic=True)
        # For SpinfulFermions_tJ config.sym does not match config.fermionic.
    with pytest.raises(yastn.YastnError):
        ops_Z2_tJ.c(spin='down')
        # spin should be equal 'u' or 'd'.
    with pytest.raises(yastn.YastnError):
        ops_Z2_tJ.cp(spin=+1)
        # spin should be equal 'u' or 'd'.
    with pytest.raises(yastn.YastnError):
        ops_Z2_tJ.n(spin=+1)
        # spin should be equal 'u' or 'd'.
    with pytest.raises(yastn.YastnError):
        ops_Z2_tJ.vec_n(1)
        # For SpinfulFermions_tJ val in vec_n should be (0, 0), (1, 0), or (0, 1).


if __name__ == '__main__':
    test_spinful_fermions_tJ()
