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
""" Test definitions of two-site gates. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps._gates_auxiliary import fkron

tol = 1e-12  #pylint: disable=invalid-name


def test_fkron(config_kwargs):
    for sym in ['Z2', 'U1', 'U1xU1xZ2', 'U1xU1']:
        for opsclass in [yastn.operators.SpinfulFermions,
                         yastn.operators.SpinfulFermions_tJ]:
            ops = opsclass(sym=sym, **config_kwargs)

            Sm1Sp0 = fkron(ops.Sp(), ops.Sm(), sites=(0, 1))  # Sp_0 Sm_1
            Sp0Sm1 = fkron(ops.Sm(), ops.Sp(), sites=(1, 0))  # Sm_1 Sp_0
            assert yastn.norm(Sm1Sp0 - Sp0Sm1) < tol  # [Sp_0, Sm_1] = 0

            for s in ['u', 'd']:
                c1cp0 = fkron(ops.cp(s), ops.c(s), sites=(0, 1))  # c+_0 c_1
                cp0c1 = fkron(ops.c(s), ops.cp(s), sites=(1, 0))  # c_1 c+_0
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
            op = fkron(ops.cp(s), ops.c(s), sites=(0, 1))  # cs+_0 cs_1
            phi = yastn.ncon([op, psi], [[-0, 1, -1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) - sgn / 3) < tol

            op = fkron(ops.c(s), ops.cp(s), sites=(1, 0))  # cs_1 cs+_0
            phi = yastn.ncon([op, psi], [[-0, 1, -1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) + sgn / 3) < tol

            op = fkron(ops.c(s), ops.cp(s), sites=(0, 1))  # cs_0 cs+_1
            phi = yastn.ncon([op, psi], [[-0, 1, -1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) + sgn / 3) < tol

            op = fkron(ops.cp(s), ops.c(s), sites=(1, 0))  # cs+_1 cs_0
            phi = yastn.ncon([op, psi], [[-0, 1, -1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) - sgn / 3) < tol


def check_1site_gate_taylor(gate, I, H, ds, eps):
    """ check gate vs Taylor expansion of the exp(-ds * H). """
    O = gate.G[0]
    O2 = I + (-ds) * H + (ds ** 2 / 2) * H @ H
    O2 = O2 + (-ds ** 3 / 6) * H @ H @ H + (ds ** 4 / 24) * H @ H @ H @ H
    assert ((O - O2).norm()) < abs(eps) ** 5


def check_2site_gate_taylor(gate, I, H, ds, eps):
    """ check gate vs Taylor expansion of the exp(-ds * H). """
    O = yastn.ncon(gate.G, [(-0, -1, 1) , (-2, -3, 1)])
    O = O.fuse_legs(axes=((0, 2), (1, 3)))

    H = H.fuse_legs(axes=((0, 2), (1, 3)))
    II = yastn.ncon([I, I], [(-0, -1), (-2, -3)])
    II = II.fuse_legs(axes=((0, 2), (1, 3)))

    O2 = II + (-ds) * H + (ds ** 2 / 2) * H @ H
    O2 = O2 + (-ds ** 3 / 6) * H @ H @ H + (ds ** 4 / 24) * H @ H @ H @ H
    assert ((O - O2).norm()) < abs(eps) ** 5


def test_hopping_gate(config_kwargs):
    """ test fpeps.gates.gate_nn_hopping. """
    def check_hopping_gate(ops, t, ds):
        I, c, cdag = ops.I(), ops.c(), ops.cp()
        gate = fpeps.gates.gate_nn_hopping(t, ds, I, c, cdag)
        H = - t * fpeps.gates.fkron(cdag, c, sites=(0, 1)) \
            - t * fpeps.gates.fkron(cdag, c, sites=(1, 0))
        check_2site_gate_taylor(gate, I, H, ds, ds * t)

    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    check_hopping_gate(ops, t=0.5, ds=0.02)
    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **config_kwargs)
    check_hopping_gate(ops, t=2, ds=0.005)
    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', **config_kwargs)
    check_hopping_gate(ops, t=1, ds=0.1)


def test_tJ_gate(config_kwargs):
    """ test fpeps.gates.gate_nn_tJ. """
    def check_tJ_gate(ops, J, tu, td, muu0, muu1, mud0, mud1, ds):
        I, cu, cpu, cd, cpd = ops.I(), ops.c(spin='u'), ops.cp(spin='u'), ops.c(spin='d'), ops.cp(spin='d')
        Sz, Sp, Sm = ops.Sz(), ops.Sp(), ops.Sm()
        nu, nd = ops.n(spin='u'), ops.n(spin='d')
        gate = fpeps.gates.gate_nn_tJ(J, tu, td, muu0, muu1, mud0, mud1, ds, I, cu, cpu, cd, cpd)
        H = - tu * fpeps.gates.fkron(cpu, cu, sites=(0, 1)) \
            - tu * fpeps.gates.fkron(cpu, cu, sites=(1, 0)) \
            - td * fpeps.gates.fkron(cpd, cd, sites=(0, 1)) \
            - td * fpeps.gates.fkron(cpd, cd, sites=(1, 0)) \
            + (J / 2) * fpeps.gates.fkron(Sp, Sm, sites=(0, 1)) \
            + (J / 2) * fpeps.gates.fkron(Sm, Sp, sites=(0, 1)) \
            + J * fpeps.gates.fkron(Sz, Sz, sites=(0, 1)) \
            - (J / 4) * fpeps.gates.fkron((nu + nd), (nu + nd), sites=(0, 1)) \
            - muu0 * fpeps.gates.fkron(nu, I, sites=(0, 1)) \
            - muu1 * fpeps.gates.fkron(I, nu, sites=(0, 1)) \
            - mud0 * fpeps.gates.fkron(nd, I, sites=(0, 1)) \
            - mud1 * fpeps.gates.fkron(I, nd, sites=(0, 1))
        check_2site_gate_taylor(gate, I, H, ds, ds * max(map(abs, (J, tu, td, muu0, muu1, mud0, mud1))))

    ops = yastn.operators.SpinfulFermions_tJ(sym='U1', **config_kwargs)
    J, tu, td, muu0, muu1, mud0, mud1 = 1, 0.5, 0.7, 0.1, 0.2, 0.3, 0.4
    check_tJ_gate(ops, J, tu, td, muu0, muu1, mud0, mud1, ds=0.02)
    ops = yastn.operators.SpinfulFermions_tJ(sym='U1xU1', **config_kwargs)
    J, tu, td, muu0, muu1, mud0, mud1 = -0.5, -0.5, 0.5, 0.2, 0.3, 0.1, 0.1
    check_tJ_gate(ops, J, tu, td, muu0, muu1, mud0, mud1, ds=0.01)
    ops = yastn.operators.SpinfulFermions_tJ(sym='U1xU1xZ2', **config_kwargs)
    J, tu, td, muu0, muu1, mud0, mud1 = 0.5, 0.4, 0.2, 0.1, -0.2, -0.3, -0.1
    check_tJ_gate(ops, J, tu, td, muu0, muu1, mud0, mud1, ds=0.1)


def test_Ising_gate(config_kwargs):
    """ test fpeps.gates.gate_nn_Ising. """
    def check_Ising_gate(ops, J, ds):
        I, sp, sm = ops.I(), ops.sp(), ops.sm()
        X = sp + sm
        gate = fpeps.gates.gate_nn_Ising(J, ds, I, X)
        H = J * fpeps.gates.fkron(X, X, sites=(0, 1))
        check_2site_gate_taylor(gate, I, H, ds, ds * J)

    ops = yastn.operators.Spin12(sym='Z2', **config_kwargs)
    check_Ising_gate(ops, J=1, ds=0.02)
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    check_Ising_gate(ops, J=-2, ds=0.05)


def test_occupation_gate(config_kwargs):
    """ test fpeps.gates.gate_local_occupation. """
    def check_occupation_gate(ops, mu, ds):
        I, n = ops.I(), ops.n()
        gate = fpeps.gates.gate_local_occupation(mu, ds, I, n)
        H = - mu * n
        check_1site_gate_taylor(gate, I, H, ds, ds * abs(mu))

    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **config_kwargs)
    check_occupation_gate(ops, 2, ds=0.02)
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    check_occupation_gate(ops, -0.1, ds=0.05)


def test_field_gate(config_kwargs):
    """ test fpeps.gates.gate_local_field. """
    def check_field_gate(I, X, h, ds):
        gate = fpeps.gates.gate_local_field(h, ds, I, X)
        H = - h * X
        check_1site_gate_taylor(gate, I, H, ds, ds * abs(h))

    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    check_field_gate(ops.I(), ops.x(), 2, ds=0.02)
    ops = yastn.operators.Spin12(sym='Z2', **config_kwargs)
    check_field_gate(ops.I(), ops.z(), -0.5, ds=0.1)


def test_Coulomb_gate(config_kwargs):
    """ test fpeps.gates.gate_local_Coulomb. """
    def check_Coulomb_gate(ops, muu, mud, U, ds):
        I, nu, nd = ops.I(), ops.n(spin='u'), ops.n(spin='d')
        gate = fpeps.gates.gate_local_Coulomb(muu, mud, U, ds, I, nu, nd)
        H = U * (nu - I / 2) @ (nd - I / 2) - muu * nu - mud * nd - (U / 4) * I
        check_1site_gate_taylor(gate, I, H, ds, ds * max(map(abs, (U, muu, mud))))

    ops = yastn.operators.SpinfulFermions(sym='Z2', **config_kwargs)
    check_Coulomb_gate(ops, 0, 0, 2, ds=0.02)
    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **config_kwargs)
    check_Coulomb_gate(ops, 0.3, 0.2, -2, ds=0.2)


def test_gate_raises(config_kwargs):
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    c, cdag = ops.c(), ops.cp()
    with pytest.raises(yastn.YastnError):
        fpeps.gates.fkron(c, cdag, sites=(0, 2))
        # 'Sites should be equal to (0, 1) or (1, 0).'

if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
