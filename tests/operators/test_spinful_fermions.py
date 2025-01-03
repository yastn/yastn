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
""" Predefined spinful fermion operators operators. """
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_spinful_fermions(config_kwargs):
    """ Generate standard operators in two-dimensional Hilbert space for various symmetries. """
    ops_Z2 = yastn.operators.SpinfulFermions(sym='Z2', **config_kwargs)
    ops_U1xU1_ind = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', **config_kwargs)
    ops_U1xU1_dis = yastn.operators.SpinfulFermions(sym='U1xU1', **config_kwargs)
    # other way to initialize
    config_U1 = yastn.make_config(fermionic=True, sym="U1", **config_kwargs)
    ops_U1 = yastn.operators.SpinfulFermions(**config_U1._asdict())

    Is = [ops_Z2.I(), ops_U1.I(), ops_U1xU1_ind.I(), ops_U1xU1_dis.I()]
    legs = [ops_Z2.space(), ops_U1.space(), ops_U1xU1_ind.space(), ops_U1xU1_dis.space()]

    assert all(leg == I.get_legs(axes=0) for (leg, I) in zip(legs, Is))
    assert all(np.allclose(I.to_numpy(), np.eye(4)) for I in Is)

    assert all(ops.config.fermionic == fs for ops, fs in zip((ops_Z2, ops_U1, ops_U1xU1_ind, ops_U1xU1_dis), (True, True, (False, False, True), True)))

    for ops, inter_sgn in [(ops_Z2, 1), (ops_U1, 1), (ops_U1xU1_ind, 1), (ops_U1xU1_dis, -1)]:
        for s in ('u', 'd'):
            # occupation operators
            assert yastn.norm(ops.cp(s) @ ops.c(s) - ops.n(s)) < tol

            # check anti-commutation relations
            assert yastn.norm(ops.c(s) @ ops.c(s)) < tol
            assert yastn.norm(ops.cp(s) @ ops.cp(s)) < tol
            assert yastn.norm(ops.c(s) @ ops.cp(s) + ops.cp(s) @ ops.c(s) - ops.I()) < tol

        # anticommutator for indistinguishable; commutator for distinguishable
        assert yastn.norm(ops.c('u') @ ops.cp('d') + inter_sgn * ops.cp('d') @ ops.c('u')) < tol
        assert yastn.norm(ops.c('u') @ ops.c('d') + inter_sgn * ops.c('d') @ ops.c('u')) < tol
        assert yastn.norm(ops.cp('u') @ ops.cp('d') + inter_sgn * ops.cp('d') @ ops.cp('u')) < tol
        assert yastn.norm(ops.cp('u') @ ops.c('d') + inter_sgn * ops.c('d') @ ops.cp('u')) < tol

        # check commute and anti-commute relation between spin-1/2 operators
        assert yastn.norm(ops.Sp() @ ops.Sm() - ops.Sm() @ ops.Sp() - 2 * ops.Sz()) < tol
        assert yastn.norm(ops.Sz() @ ops.Sp() - ops.Sp() @ ops.Sz() - ops.Sp()) < tol
        assert yastn.norm(ops.Sz() @ ops.Sm() - ops.Sm() @ ops.Sz() + ops.Sm()) < tol
        assert yastn.norm(ops.Sz() @ ops.Sp() + ops.Sp() @ ops.Sz()) < tol
        assert yastn.norm(ops.Sm() @ ops.Sz() + ops.Sz() @ ops.Sm()) < tol
        #
        # |ud>; |11> = cu+ cd+ |00>;
        # cu |11> =  |01>; cu |10> = |00>
        # cd |11> = -|10>; cd |01> = |00>
        v00, v10, v01, v11 = ops.vec_n((0, 0)), ops.vec_n((1, 0)), ops.vec_n((0, 1)), ops.vec_n((1, 1))
        nu, nd = ops.n('u'), ops.n('d')
        assert yastn.norm(nu @ v00) < tol
        assert yastn.norm(nd @ v00) < tol

        assert yastn.norm(nu @ v01) < tol
        assert yastn.norm(nd @ v01 - v01) < tol

        assert yastn.norm(nu @ v10 - v10) < tol
        assert yastn.norm(nd @ v10) < tol

        assert yastn.norm(nu @ v11 - v11) < tol
        assert yastn.norm(nd @ v11 - v11) < tol

        assert yastn.norm(ops.cp('u') @ ops.cp('d') @ v00 - v11) < tol
        assert yastn.norm(ops.c('u') @ v10 - v00) < tol
        assert yastn.norm(ops.c('u') @ v11 - v01) < tol
        assert yastn.norm(ops.c('d') @ v01 - v00) < tol
        assert yastn.norm(ops.c('d') @ v11 + inter_sgn * v10) < tol

        assert yastn.norm(ops.cp('u') @ v00 - v10) < tol
        assert yastn.norm(ops.cp('u') @ v01 - v11) < tol
        assert yastn.norm(ops.cp('d') @ v00 - v01) < tol
        assert yastn.norm(ops.cp('d') @ v10 + inter_sgn * v11) < tol


    with pytest.raises(yastn.YastnError):
        yastn.operators.SpinfulFermions(sym='dense')
        # For SpinfulFermions sym should be in ('Z2', 'U1', 'U1xU1', 'U1xU1xZ2').
    with pytest.raises(yastn.YastnError):
        yastn.operators.SpinfulFermions(sym='U1', fermionic=False)
        # For SpinfulFermions config.sym does not match config.fermionic.
    with pytest.raises(yastn.YastnError):
        yastn.operators.SpinfulFermions(sym='U1xU1xZ2', fermionic=True)
        # For SpinfulFermions config.sym does not match config.fermionic.
    with pytest.raises(yastn.YastnError):
        ops_Z2.c(spin='down')
        # spin should be equal 'u' or 'd'.
    with pytest.raises(yastn.YastnError):
        ops_Z2.cp(spin=+1)
        # spin should be equal 'u' or 'd'.
    with pytest.raises(yastn.YastnError):
        ops_Z2.n(spin=+1)
        # spin should be equal 'u' or 'd'.
    with pytest.raises(yastn.YastnError):
        ops_Z2.vec_n(1)
        # Occupations given by val should be (0, 0), (1, 0), (0, 1), or (1, 1).

    d = ops_Z2.to_dict()  # dict used in mps.Generator
    (d["I"](3) - ops_Z2.I()).norm() < tol  # here 3 is a posible position in the mps
    d.keys() == ops_Z2.operators
    assert all(k in d for k in ('I', 'nu', 'cu', 'cpu', 'nd', 'cd', 'cpd'))


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
