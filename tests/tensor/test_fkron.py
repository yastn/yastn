# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
