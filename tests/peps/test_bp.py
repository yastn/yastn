# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
""" Test PEPS measurments with MpsBoundary in a product state. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps

tol = 1e-12  #pylint: disable=invalid-name


def test_iterate_measure_product(config_kwargs):
    """ Initialize a product PEPS and perform a set of measurment. """
    ops = yastn.operators.Spin1(sym='Z3', **config_kwargs)

    # initialized PEPS in a product state
    g = fpeps.SquareLattice(dims=(4, 3), boundary='obc')
    sites = g.sites()
    vals = [0, 1, -1, 1, 1, 0, 1, -1, -1, -1, 1, -1]
    vals = dict(zip(sites, vals))
    vectors = {s: ops.vec_z(val=v) for s, v in vals.items()}
    psi = fpeps.product_peps(g, vectors)

    env = fpeps.EnvBP(psi, init='eye')
    info = env.iterate_(max_sweeps=5, diff_tol=1e-10)
    assert info.converged
    run_save_load(env)
    #
    #  measure_1site
    #
    sz = ops.sz()
    ez = env.measure_1site(sz)
    assert all(abs(v - ez[s]) < tol for s, v in vals.items())
    #
    #  sample
    #
    vecs = {v: ops.vec_z(val=v) for v in [-1, 0, 1]}
    out = env.sample(xrange=(1, 4), yrange=(1, 3), number=8, projectors=vecs)
    assert all(all(x == vals[k] for x in v) for k, v in out.items())


def run_save_load(env):
    # test save, load

    config = env.psi.config
    d = env.save_to_dict()

    env_save = fpeps.load_from_dict(config, d)

    for site in env.sites():
        for dirn in  ['t', 'l', 'b', 'r']:
            ten0 = getattr(env[site], dirn)
            ten1 = getattr(env_save[site], dirn)

            assert yastn.are_independent(ten0, ten1)
            assert (ten0 - ten1).norm() < 1e-14


def test_iterate_measure_2x1(config_kwargs):
    """ Initialize a product PEPS of 1x2 cells and perform a set of measurment. """

    for dims in [(1, 2), (2, 1)]:
        g = fpeps.SquareLattice(dims=dims, boundary='infinite')

        for sym in ['Z2', 'U1', 'U1xU1xZ2']:
            ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
            v0110 = yastn.ncon([ops.vec_n((0, 1)), ops.vec_n((1, 0))], [[-0], [-1]])
            v1100 = yastn.ncon([ops.vec_n((1, 1)), ops.vec_n((0, 0))], [[-0], [-1]])
            v0011 = yastn.ncon([ops.vec_n((0, 0)), ops.vec_n((1, 1))], [[-0], [-1]])
            state = v1100 + v0110 + v0011

            r0, r1 = yastn.qr(state, sQ=-1, Qaxis=0)

            # auxilliary leg;
            # in general, left/top tensor r0 requires a swap_gate between physical and ancilla lags.
            r0 = r0.add_leg(axis=-1, s=-1).swap_gate(axes=(1, 2)).fuse_legs(axes=(0, (1, 2)))
            # right/bottom tensor r1 swap_gate is effectively trivial
            r1 = r1.add_leg(axis=-1, s=-1).swap_gate(axes=((0, 1), 2)).fuse_legs(axes=(0, (1, 2)))

            r0 = r0.add_leg(axis=0, s=-1)  # t
            r0 = r0.add_leg(axis=1, s=1)  # l
            r0 = r0.add_leg(axis=2, s=1)  if dims == (1, 2) else r0.add_leg(axis=3, s=-1)  # b or r

            r1 = r1.add_leg(axis=0, s=-1)  if dims == (1, 2) else r1.add_leg(axis=1, s=1)  # t or l
            r1 = r1.add_leg(axis=2, s=1)  # b
            r1 = r1.add_leg(axis=3, s=-1)  # r

            psi = fpeps.Peps(g, tensors=dict(zip(g.sites(), [r0, r1])))

            env = fpeps.EnvBP(psi, init='eye')
            # no need to converge ctmrg_ in this example
            env.iterate_(max_sweeps=5)

            val = -1 / 3  #  result of <psi | cs_0+ cs_1 | psi>
            for s in ['u', 'd']:
                bond = [*g.sites()]
                assert abs(env.measure_nn(ops.cp(s), ops.c(s), bond=bond) - val) < tol
                assert abs(env.measure_nn(ops.c(s), ops.cp(s), bond=bond[::-1]) - (-val)) < tol
                assert abs(env.measure_nn(ops.c(s), ops.cp(s), bond=bond) - (-val.conjugate())) < tol
                assert abs(env.measure_nn(ops.cp(s), ops.c(s), bond=bond[::-1]) - (val.conjugate())) < tol

            s0, s1 = g.sites()
            assert abs(env.measure_1site(ops.n('u'), site=s0) - 1 / 3) < tol
            assert abs(env.measure_1site(ops.n('d'), site=s0) - 2 / 3) < tol
            assert abs(env.measure_1site(ops.n('u'), site=s1) - 2 / 3) < tol
            assert abs(env.measure_1site(ops.n('d'), site=s1) - 1 / 3) < tol

            # example that is testing auxlliary leg swap-gate in the initialization
            # it has non-trivial charge carried by auxlliary leg
            v0111 = yastn.ncon([ops.vec_n((0, 1)), ops.vec_n((1, 1))], [[-0], [-1]])
            v1101 = yastn.ncon([ops.vec_n((1, 1)), ops.vec_n((0, 1))], [[-0], [-1]])

            phase = (-1 - 1j) / 2 ** 0.5
            val = (1 + 1j) / 2 ** 1.5
            # for state = v1101 + phase * v0111
            # <state| cu_0+ cu |state > == val
            for state in [v1101 + phase * v0111, v1100 + phase * v0110]:
                for nU in [True, False]:  # testing non-zero charge in r0 and r1
                    # split 2-site state
                    r0, ss, r1 = yastn.svd(state, sU=-1, Uaxis=0, nU=nU)
                    r1 = ss @ r1

                    # adding legs to form PEPS tensors
                    # aux leg of left tensor. here with a swap gate between physical and ancilla legs
                    r0 = r0.add_leg(axis=-1, s=-1).swap_gate(axes=(1, 2)).fuse_legs(axes=(0, (1, 2)))
                    r0 = r0.add_leg(axis=0, s=-1)  # t
                    r0 = r0.add_leg(axis=1, s=1)  # l
                    r0 = r0.add_leg(axis=2, s=1) if dims == (1, 2) else r0.add_leg(axis=3, s=-1)  # b or r

                    # aux leg of right tensor. We put a swap gate, though it is trivial
                    r1 = r1.add_leg(axis=-1, s=-1).swap_gate(axes=((0, 1), 2)).fuse_legs(axes=(0, (1, 2)))
                    r1 = r1.add_leg(axis=0, s=-1) if dims == (1, 2) else r1.add_leg(axis=1, s=1)  # t or l
                    r1 = r1.add_leg(axis=2, s=1)  # b
                    r1 = r1.add_leg(axis=3, s=-1)  # r

                    psi = fpeps.Peps(g, tensors=dict(zip(g.sites(), [r0, r1])))

                    env = fpeps.EnvBP(psi)
                    # no need to converge ctmrg_ in this example, but we can do it anyway
                    env.iterate_(max_sweeps=2)
                    bond = [*g.sites()]
                    assert abs(env.measure_nn(ops.cp('u'), ops.c('u'), bond=bond) - val) < tol
                    assert abs(env.measure_nn(ops.c('u'), ops.cp('u'), bond=bond[::-1]) - (-val)) < tol
                    assert abs(env.measure_nn(ops.c('u'), ops.cp('u'), bond=bond) - (-val.conjugate())) < tol
                    assert abs(env.measure_nn(ops.cp('u'), ops.c('u'), bond=bond[::-1]) - (val.conjugate())) < tol

                    assert abs(env.measure_1site(ops.n('u'), site=(0, 0)) - 0.5) < tol
                    assert abs(env.measure_1site(ops.n('d'), site=(0, 0)) - 1.0) < tol


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
