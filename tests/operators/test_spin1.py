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
""" Predefined spin-1 operators. """
from itertools import chain
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_spin1(config_kwargs):
    """ Generate standard operators in 3-dimensional Hilbert space for various symmetries. """
    ops_dense = yastn.operators.Spin1(sym='dense', **config_kwargs)
    ops_Z3 = yastn.operators.Spin1(sym='Z3', **config_kwargs)
    # other way to initialize
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    ops_U1 = yastn.operators.Spin1(**config_U1._asdict())

    opss = [ops_dense, ops_Z3, ops_U1]

    assert all(ops.config.fermionic == False for ops in (ops_dense, ops_Z3, ops_U1))

    Is = [ops_dense.I(), ops_Z3.I(), ops_U1.I()]
    legs = [ops_dense.space(), ops_Z3.space(), ops_U1.space()]

    assert all(leg == I.get_legs(axes=0) for (leg, I) in zip(legs, Is))
    assert all(np.allclose(ops.I().to_numpy(key=ops.key()), np.eye(3)) for ops in opss)
    assert all(np.allclose(ops.sz().to_numpy(key=ops.key()), np.diag([1, 0, -1])) for ops in opss)

    lss = [dict(enumerate(ops.I().get_legs())) for ops in opss]
    assert all(np.allclose(ops.sp().to_numpy(legs=ls, key=ops.key()), np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]) * np.sqrt(2)) for ops, ls in zip(opss, lss))
    assert all(np.allclose(ops.sm().to_numpy(legs=ls, key=ops.key()), np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * np.sqrt(2)) for ops, ls in zip(opss, lss))

    # dense only
    assert np.allclose(ops_dense.sx().to_numpy(), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2))
    assert np.allclose(ops_dense.sy().to_numpy(), np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2))
    assert (1j * ops_dense.sy() - ops_dense.isy()).norm() < tol
    assert (ops_dense.sx() + 1j * ops_dense.sy() - ops_dense.sp()).norm() < tol
    assert (ops_dense.sx() - 1j * ops_dense.sy() - ops_dense.sm()).norm() < tol

    assert all(yastn.norm(ops.sp() @ ops.sm() - ops.sm() @ ops.sp() - 2 * ops.sz()) < tol for ops in opss)
    assert all(yastn.norm(ops.sz() @ ops.sp() - ops.sp() @ ops.sz() - ops.sp()) < tol for ops in opss)
    assert all(yastn.norm(ops.sz() @ ops.sm() - ops.sm() @ ops.sz() + ops.sm()) < tol for ops in opss)

    sz_vecs = [(ops.sz(), ops.vec_z(val=val), val) for ops in opss for val in (+1, 0, -1)]
    assert all(yastn.norm(O @ v - val * v) < tol for O, v, val in sz_vecs)
    sx_vecs = [(ops_dense.sx(), ops_dense.vec_x(val=val), val) for val in (+1, 0, -1)]
    assert all(yastn.norm(O @ v - val * v) < tol for O, v, val in sx_vecs)
    sy_vecs = [(ops_dense.sy(), ops_dense.vec_y(val=val), val) for val in (+1, 0, -1)]
    assert all(yastn.norm(O @ v - val * v) < tol for O, v, val in sy_vecs)
    assert all(abs(v.norm() - 1) < tol for _, v, _ in chain(sz_vecs, sx_vecs, sy_vecs))

    vecss = [ops.vec_s() for ops in opss]
    gs = [ops.g() for ops in opss]
    assert all(vs.s == (-1, 1, -1) and vs.get_shape() == (3, 3, 3) for vs in vecss)
    assert all(g.s == (1, 1) and g.get_shape() == (3, 3) for g in gs)
    #
    S = 1
    for vs, g, I in zip(vecss, gs, Is):
        assert (yastn.ncon((vs, g, vs), ((1, -0, 3), (1, 2), (2, 3, -1))) - S * (S + 1) * I).norm() < tol

    with pytest.raises(yastn.YastnError):
        _ = ops_U1.sx()
        # Cannot define Sx operator for U1 or Z3 symmetry.
    with pytest.raises(yastn.YastnError):
        _ = ops_Z3.sy()
        # Cannot define Sy operator for U1 or Z3 symmetry.
    with pytest.raises(yastn.YastnError):
        _ = ops_Z3.isy()
        # Cannot define sy operator for U1 or Z3 symmetry.
    with pytest.raises(yastn.YastnError):
        yastn.operators.Spin1(sym='wrong symmetry')
        # For Spin1 sym should be in ('dense', 'Z3', 'U1').
    with pytest.raises(yastn.YastnError):
        yastn.operators.Spin1(sym='U1', fermionic=True)
        # For Spin1 config.fermionic should be False.
    with pytest.raises(yastn.YastnError):
        yastn.operators.Spin1(sym='U1xU1')
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
    pytest.main([__file__, "-vs", "--durations=0"])
