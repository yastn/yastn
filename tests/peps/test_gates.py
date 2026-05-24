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

tol = 1e-12  #pylint: disable=invalid-name


def check_1site_gate_taylor(gate, I, H, ds, eps):
    """ check gate vs Taylor expansion of the exp(-ds * H). """
    O = gate.G[0]

    O2 = I + (-ds) * H + (ds ** 2 / 2) * H @ H
    O2 = O2 + (-ds ** 3 / 6) * H @ H @ H + (ds ** 4 / 24) * H @ H @ H @ H
    assert ((O - O2).norm()) < abs(eps) ** 5

    G3 = fpeps.gates.gate_local_exp(ds, I, H)
    O3 = G3.G[0]
    assert ((O - O3).norm()) < tol


def check_2site_gate_taylor(gate, I, H, ds, eps):
    """ check gate vs Taylor expansion of the exp(-ds * H). """
    G = fpeps.gates.gate_nn_exp(ds, I, H)
    O = yastn.ncon(gate.G, [(-0, -1, 1) , (-2, -3, 1)])
    O = O.fuse_legs(axes=((0, 2), (1, 3)))

    H = H.fuse_legs(axes=((0, 2), (1, 3)))
    II = yastn.ncon([I, I], [(-0, -1), (-2, -3)])
    II = II.fuse_legs(axes=((0, 2), (1, 3)))

    O2 = II + (-ds) * H + (ds ** 2 / 2) * H @ H
    O2 = O2 + (-ds ** 3 / 6) * H @ H @ H + (ds ** 4 / 24) * H @ H @ H @ H
    assert ((O - O2).norm()) < abs(eps) ** 5

    O3 = yastn.ncon(G.G, [(-0, -1, 1) , (-2, -3, 1)])
    O3 = O3.fuse_legs(axes=((0, 2), (1, 3)))
    assert ((O - O3).norm()) < tol


def test_hopping_gate(config_kwargs):
    """ test fpeps.gates.gate_nn_hopping. """
    def check_hopping_gate(ops, t, ds):
        I, c, cdag = ops.I(), ops.c(), ops.cp()
        gate = fpeps.gates.gate_nn_hopping(t, ds, I, c, cdag)
        H = - t * yastn.fkron(cdag, c, sites=(0, 1)) \
            - t * yastn.fkron(cdag, c, sites=(1, 0))
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
        H = - tu * yastn.fkron(cpu, cu, sites=(0, 1)) \
            - tu * yastn.fkron(cpu, cu, sites=(1, 0)) \
            - td * yastn.fkron(cpd, cd, sites=(0, 1)) \
            - td * yastn.fkron(cpd, cd, sites=(1, 0)) \
            + (J / 2) * yastn.fkron(Sp, Sm, sites=(0, 1)) \
            + (J / 2) * yastn.fkron(Sm, Sp, sites=(0, 1)) \
            + J * yastn.fkron(Sz, Sz, sites=(0, 1)) \
            - (J / 4) * yastn.fkron((nu + nd), (nu + nd), sites=(0, 1)) \
            - muu0 * yastn.fkron(nu, I, sites=(0, 1)) \
            - muu1 * yastn.fkron(I, nu, sites=(0, 1)) \
            - mud0 * yastn.fkron(nd, I, sites=(0, 1)) \
            - mud1 * yastn.fkron(I, nd, sites=(0, 1))
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
        H = J * yastn.fkron(X, X, sites=(0, 1))
        check_2site_gate_taylor(gate, I, H, ds, ds * J)

    ops = yastn.operators.Spin12(sym='Z2', **config_kwargs)
    check_Ising_gate(ops, J=1, ds=0.02)
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    check_Ising_gate(ops, J=-2, ds=0.05)


def test_Heisenberg_gate(config_kwargs):
    """ test fpeps.gates.gate_nn_Heisenberg. """
    def check_Heisenberg_gate(ops, J, ds):
        I, sp, sm = ops.I(), ops.sp(), ops.sm()
        sx, sy, sz = ops.sx(), ops.sy(), ops.sz()
        gate = fpeps.gates.gate_nn_Heisenberg(J, ds, I, sz, sp, sm)
        H = J * yastn.fkron(sx, sx, sites=(0, 1)) \
          + J * yastn.fkron(sy, sy, sites=(0, 1)) \
          + J * yastn.fkron(sz, sz, sites=(0, 1))

        check_2site_gate_taylor(gate, I, H, ds, ds * J)

    ops = yastn.operators.Spin12(sym='Z2', **config_kwargs)
    check_Heisenberg_gate(ops, J=1, ds=0.02)
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    check_Heisenberg_gate(ops, J=-2, ds=0.05)


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


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
