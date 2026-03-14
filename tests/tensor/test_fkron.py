# Copyright 2026 The YASTN Authors. All Rights Reserved.
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
""" Test definitions of two-site gates. """
import re
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_fkron(config_kwargs):
    for sym in ['Z2', 'U1', 'U1xU1xZ2', 'U1xU1']:
        for opsclass in [yastn.operators.SpinfulFermions,
                         yastn.operators.SpinfulFermions_tJ]:
            ops = opsclass(sym=sym, **config_kwargs)

            Sm1Sp0 = yastn.fkron(ops.Sp(), ops.Sm(), sites=(0, 1))  # Sp_0 Sm_1
            Sp0Sm1 = yastn.fkron(ops.Sm(), ops.Sp(), sites=(1, 0))  # Sm_1 Sp_0
            assert yastn.norm(Sm1Sp0 - Sp0Sm1) < tol  # [Sp_0, Sm_1] = 0

            for s in ['u', 'd']:
                c1cp0 = yastn.fkron(ops.cp(s), ops.c(s), sites=(0, 1))  # c+_0 c_1
                cp0c1 = yastn.fkron(ops.c(s), ops.cp(s), sites=(1, 0))  # c_1 c+_0
                assert yastn.norm(cp0c1 + c1cp0) < tol  # {cp_0, c_1} = 0

    for sym, sgn in zip(['Z2', 'U1', 'U1xU1xZ2', 'U1xU1'], [-1, -1, -1, 1]):   # for U1xU1, cu and cd commute
        ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
        v0110 = yastn.ncon([ops.vec_n((0, 1)), ops.vec_n((1, 0))], [[-0], [-1]])
        v1100 = yastn.ncon([ops.vec_n((1, 1)), ops.vec_n((0, 0))], [[-0], [-1]])
        v0011 = yastn.ncon([ops.vec_n((0, 0)), ops.vec_n((1, 1))], [[-0], [-1]])
        psi = v1100 + v0110 + v0011

        for s in ['u', 'd']:
            # |nu_0 nd_0 nu_1, nd_1 >, where the convention is
            # |1111> = cu+_0 cd+_0 cu+_1 cd+_1 |0000>, i.e.,
            # sites 0 is before 1 in fermionic order
            op = yastn.fkron(ops.cp(s), ops.c(s), sites=(0, 1))  # cs+_0 cs_1
            phi = yastn.ncon([op, psi], [[-0, 1, -1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) - sgn / 3) < tol

            op = yastn.fkron(ops.c(s), ops.cp(s), sites=(1, 0))  # cs_1 cs+_0
            phi = yastn.ncon([op, psi], [[-0, 1, -1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) + sgn / 3) < tol

            op = yastn.fkron(ops.c(s), ops.cp(s), sites=(0, 1))  # cs_0 cs+_1
            phi = yastn.ncon([op, psi], [[-0, 1, -1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) + sgn / 3) < tol

            op = yastn.fkron(ops.cp(s), ops.c(s), sites=(1, 0))  # cs+_1 cs_0
            phi = yastn.ncon([op, psi], [[-0, 1, -1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) - sgn / 3) < tol

            op = yastn.fkron(ops.c(s), ops.cp(s), sites=(0, 1), application_order=(0, 1))  # cs+_1 cs_0
            phi = yastn.ncon([op, psi], [[-0, 1, -1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) - sgn / 3) < tol

            op = yastn.fkron(ops.c(s), ops.cp(s), sites=(1, 0), application_order=(1, 0))  # cs_1 cs+_0
            phi = yastn.ncon([op, psi], [[-0, 1, -1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) + sgn / 3) < tol

            op = yastn.fkron(ops.c(s), ops.cp(s), sites=(1, 0), application_order=(0, 1))  # cs+_0 cs_1
            phi = yastn.ncon([op, psi], [[-0, 1, -1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) - sgn / 3) < tol


def test_fkron_raises(config_kwargs):
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    c, cdag = ops.c(), ops.cp()
    with pytest.raises(yastn.YastnError,
                       match=re.escape('sites should be a permutation of 0, 1, ..., len(operators) - 1.')):
        yastn.fkron(c, cdag, sites=(0, 2))

    with pytest.raises(yastn.YastnError,
                       match=re.escape('application_order should be a permutation of 0, 1, ..., len(operators) - 1.')):
        yastn.fkron(c, cdag, application_order=(3,))



if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
