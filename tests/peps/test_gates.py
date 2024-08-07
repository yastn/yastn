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
from yastn.tn.fpeps._gates_auxiliary import fkron, gate_product_operator

try:
    from .configs import config as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config as cfg


tol = 1e-12

def test_fkron():
    kwargs = {'backend': cfg.backend, 'default_device': cfg.default_device}

    for sym in ['Z2', 'U1xU1', 'U1xU1xZ2']:  # TODO: add 'U1' in tJ
        for opsclass in [yastn.operators.SpinfulFermions,
                         yastn.operators.SpinfulFermions_tJ]:
            ops = opsclass(sym=sym, **kwargs)

            Sm1Sp0 = fkron(ops.Sp(), ops.Sm(), sites=(0, 1))  # Sp_0 Sm_1
            Sp0Sm1 = fkron(ops.Sm(), ops.Sp(), sites=(1, 0))  # Sm_1 Sp_0
            assert yastn.norm(Sm1Sp0 - Sp0Sm1) < tol  # [Sp_0, Sm_1] = 0

            for s in ['u', 'd']:
                c1cp0 = fkron(ops.cp(s), ops.c(s), sites=(0, 1))  # c+_0 c_1
                cp0c1 = fkron(ops.c(s), ops.cp(s), sites=(1, 0))  # c_1 c+_0
                assert yastn.norm(cp0c1 + c1cp0) < tol  # {cp_0, c_1} = 0

    for sym, sgn in zip(['Z2', 'U1xU1', 'U1xU1xZ2'], [-1, 1, -1]):   # for U1xU1, cu and cd commute
        ops = yastn.operators.SpinfulFermions(sym=sym, **kwargs)
        # |nu_0 nd_0 nu_1, nd_1 >, where convention is
        # |1111> = cu+_0 cd+_0 cu+_1 cd+_1 |0000>
        v0110 = yastn.ncon([ops.vec_n((0, 1)), ops.vec_n((1, 0))], [[-0], [-1]])
        v1100 = yastn.ncon([ops.vec_n((1, 1)), ops.vec_n((0, 0))], [[-0], [-1]])
        v0011 = yastn.ncon([ops.vec_n((0, 0)), ops.vec_n((1, 1))], [[-0], [-1]])
        psi = v1100 + v0110 + v0011

        for s in ['u', 'd']:
            op = fkron(ops.cp(s), ops.c(s), sites=(0, 1))  # cs+_0 cs_1
            phi = yastn.ncon([op, psi], [[-0, -1, 1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) - sgn / 3) < tol

            op = fkron(ops.c(s), ops.cp(s), sites=(1, 0))  # cs_1 cs+_0
            phi = yastn.ncon([op, psi], [[-0, -1, 1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) + sgn / 3) < tol

            op = fkron(ops.c(s), ops.cp(s), sites=(0, 1))  # cs_0 cs+_1
            phi = yastn.ncon([op, psi], [[-0, -1, 1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) + sgn / 3) < tol

            op = fkron(ops.cp(s), ops.c(s), sites=(1, 0))  # cs+_1 cs_0
            phi = yastn.ncon([op, psi], [[-0, -1, 1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) - sgn / 3) < tol

            # now we test reversed fermionic order between sites,
            # effectively, this means |1111> = cu+_1 cd+_1 cu+_0 cd+_0 |0000>
            # vector representation of psi is the same as above, i.e., it has positive amplitudes.

            op = gate_product_operator(ops.cp(s), ops.c(s), l_ordered=True, f_ordered=False, merge=True)  # cs+_0 cs_1
            phi = yastn.ncon([op, psi], [[-0, -1, 1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) - 1 / 3) < tol

            op = gate_product_operator(ops.c(s), ops.cp(s), l_ordered=False, f_ordered=True, merge=True)  # cs_1 cs+_0
            phi = yastn.ncon([op, psi], [[-0, -1, 1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) + 1 / 3) < tol

            op = gate_product_operator(ops.c(s), ops.cp(s), l_ordered=True, f_ordered=False, merge=True)  # cs_0 cs+_1
            phi = yastn.ncon([op, psi], [[-0, -1, 1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) + 1 / 3) < tol

            op = gate_product_operator(ops.cp(s), ops.c(s), l_ordered=False, f_ordered=True, merge=True)  # cs+_1 cs_0
            phi = yastn.ncon([op, psi], [[-0, -1, 1, 2], [1, 2]])
            assert abs(yastn.vdot(phi, psi) / yastn.vdot(psi, psi) - 1 / 3) < tol


def test_hopping_gate():
    kwargs = {'backend': cfg.backend, 'default_device': cfg.default_device}

    ops = yastn.operators.SpinlessFermions(sym='U1', **kwargs)
    check_hopping_gate(ops, t=0.5, ds=0.02)

    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **kwargs)
    check_hopping_gate(ops, t=2, ds=0.005)

    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', **kwargs)
    check_hopping_gate(ops, t=1, ds=0.1)


def check_hopping_gate(ops, t, ds):

    I, c, cdag = ops.I(), ops.c(), ops.cp()

    # nearest-neighbor hopping gate
    g_hop = fpeps.gates.gate_nn_hopping(t, ds, I, c, cdag)
    O = yastn.ncon([g_hop.G0, g_hop.G1], [(-0, -2, 1) , (-1, -3, 1)])
    O = O.fuse_legs(axes=((0, 1), (2, 3)))

    # hopping Hamiltonian
    H = - t * fpeps.gates.fkron(cdag, c, sites=(0, 1), merge=True) \
        - t * fpeps.gates.fkron(cdag, c, sites=(1, 0), merge=True)
    H = H.fuse_legs(axes=((0, 1), (2, 3)))

    II = yastn.ncon([I, I], [(-0, -2) , (-1, -3)])
    II = II.fuse_legs(axes=((0, 1), (2, 3)))

    # Hamiltonian exponent from Taylor expansion
    O2 = II + (-ds) * H + (ds ** 2 / 2) * H @ H \
       + (-ds ** 3 / 6) * H @ H @ H + (ds ** 4 / 24) * H @ H @ H @ H

    assert ((O - O2).norm()) < (ds * t) ** 5

def test_heisenberg_gates():
    kwargs = {'backend': cfg.backend, 'default_device': cfg.default_device}

    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2', **kwargs)
    check_heisenberg_gates(ops, Jz=2, J=1, Jn = 0.5, ds=0.005)

    ops = yastn.operators.SpinfulFermions_tJ(sym='U1xU1xZ2', **kwargs)
    check_heisenberg_gates(ops, Jz=1, J=1, Jn = 1, ds=0.005)

    ops = yastn.operators.SpinfulFermions(sym='Z2', **kwargs)
    check_heisenberg_gates(ops, Jz=2, J=1, Jn = 0.5, ds=0.005)

    ops = yastn.operators.SpinfulFermions_tJ(sym='Z2', **kwargs)
    check_heisenberg_gates(ops, Jz=1, J=1, Jn = 1, ds=0.005)

    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **kwargs)
    check_heisenberg_gates(ops, Jz=1, J=2, Jn = 0.5, ds=0.005)

    ops = yastn.operators.SpinfulFermions_tJ(sym='U1xU1', **kwargs)
    check_heisenberg_gates(ops, Jz=1, J=1, Jn = 1, ds=0.005)

def check_heisenberg_gates(ops, Jz, J, Jn, ds):

    [I, Sz, Sm, Sp] = [ops.I(), ops.Sz(), ops.Sm(), ops.Sp()]
    nu, nd = ops.n(spin='u'), ops.n(spin='d')

    n = nu + nd

    g_heisenberg = fpeps.gates.gates_Heisenberg_spinful(ds, Jz, J, Jn, Sz, Sp, Sm, n, I)
    O = yastn.ncon([g_heisenberg.G0, g_heisenberg.G1], [(-0, -2, 1) , (-1, -3, 1)])
    O = O.fuse_legs(axes=((0, 1), (2, 3)))

    H = fkron(I, I, sites=(0, 1))
    H = H + J / 2 * (fkron(Sp, Sm, sites=(0, 1)) + fkron(Sp, Sm, sites=(1, 0))) + \
            Jz * fkron(Sz, Sz, sites=(0, 1)) - 0.25 * Jn * fkron(n, n, sites=(0, 1))


    H = H.fuse_legs(axes=((0, 1), (2, 3)))
    II = yastn.ncon([I, I], [(-0, -2) , (-1, -3)])
    II = II.fuse_legs(axes=((0, 1), (2, 3)))

    O2 = II + (-ds) * H + (ds ** 2 / 2) * H @ H + (-ds ** 3 / 6) * H @ H @ H + (ds ** 4 / 24) * H @ H @ H @ H

    assert ((O - O2).norm()) < (ds * max(abs(Jz), abs(J), abs(Jn))) ** 4



def test_gate_raises():
    kwargs = {'backend': cfg.backend, 'default_device': cfg.default_device}
    ops = yastn.operators.SpinlessFermions(sym='U1', **kwargs)

    c, cdag = ops.c(), ops.cp()

    with pytest.raises(yastn.YastnError):
        fpeps.gates.fkron(c, cdag, sites=(0, 2))
        # sites should be equal to (0, 1) or (1, 0)




if __name__ == '__main__':
    test_hopping_gate()
    test_gate_raises()
    test_fkron()
