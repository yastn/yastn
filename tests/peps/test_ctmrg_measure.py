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
""" Test PEPS measurments with MpsBoundary in a product state. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps.envs.rdm import measure_rdm_1site, measure_rdm_nn, measure_rdm_2x2

tol = 1e-12  #pylint: disable=invalid-name


def run_ctm_save_load_copy(env):
    # test save, load, copy, clone,

    config = env.psi.config
    d = env.save_to_dict()

    env_save = fpeps.load_from_dict(config, d)
    env_copy = env.copy()
    env_clone = env.clone()
    env_shallow = env.shallow_copy()

    for site in env.sites():
        for dirn in  ['tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r']:
            ten0 = getattr(env[site], dirn)
            ten1 = getattr(env_save[site], dirn)
            ten2 = getattr(env_copy[site], dirn)
            ten3 = getattr(env_clone[site], dirn)
            ten4 = getattr(env_shallow[site], dirn)

            assert yastn.are_independent(ten0, ten1)
            assert yastn.are_independent(ten0, ten2)
            assert yastn.are_independent(ten0, ten3)
            assert ten0 is ten4

            assert (ten0 - ten1).norm() < 1e-14
            assert (ten0 - ten2).norm() < 1e-14
            assert (ten0 - ten3).norm() < 1e-14
            assert (ten0 - ten4).norm() < 1e-14


@pytest.mark.parametrize("boundary", ["obc", "infinite"])
def test_ctmrg_measure_product(config_kwargs, boundary):
    """ Initialize a product PEPS and perform a set of measurment. """
    ops = yastn.operators.Spin1(sym='Z3', **config_kwargs)

    # initialized PEPS in a product state
    g = fpeps.SquareLattice(dims=(4, 3), boundary=boundary)
    sites = g.sites()
    vals = [0, 1, -1, 1, 1, 0, 1, -1, -1, -1, 1, -1]
    vals = dict(zip(sites, vals))
    occs = {s: ops.vec_z(val=v) for s, v in vals.items()}
    psi = fpeps.product_peps(g, occs)

    env = fpeps.EnvCTM(psi, init='eye')
    #
    #  measure_1site
    #
    sz = ops.sz()
    I= ops.I()
    ez = env.measure_1site(sz)
    assert all(abs(v - ez[s]) < tol for s, v in vals.items())
    ez_rdm1x1= {s: measure_rdm_1site(s, psi, env, sz) for s in sites}
    assert all(abs(v - ez[s]) < tol for s, v in ez_rdm1x1.items())

    ezz = env.measure_nn(sz, sz)
    for (s1, s2), v in ezz.items():
        s1, s2 = map(g.site2index, (s1, s2))
        assert abs(vals[s1] * vals[s2] - v) < tol
    for s0 in [(0,0), (1,0), (0,1), (1,1)]:
        for dirn in ['h', 'v']:
            val= measure_rdm_nn(s0,dirn,psi,env, (sz, sz))
            assert abs( val - ezz[(s0, psi.nn_site(s0, "r" if dirn=="h" else "b"))] )<tol

    s_list= [ [((0, 1), (1, 0)), ((0,0),(I, sz, sz, I))],
        [((2, 1), (2, 2)), ((2,1),(sz,I,sz,I))], [((1, 1), (2, 1)), ((1,1),(sz,sz,I,I)) ],
    ]
    for s_elem in s_list:
        s1,s2= s_elem[0]
        v = env.measure_2x2(sz, sz, sites=(s1, s2))
        assert abs(vals[s1] * vals[s2] - v) < tol
        s0, op= s_elem[1]
        v_rdm = measure_rdm_2x2(s0,psi,env,op)
        assert abs(v - v_rdm) < tol


    s1, s2, s3 = (1, 2), (2, 1), (2, 2)
    v = env.measure_2x2(sz, sz, sz, sites=(s1, s2, s3))
    assert abs(vals[s1] * vals[s2] * vals[s3] - v) < tol
    v_rdm= measure_rdm_2x2((1,1),psi,env,(I,sz,sz,sz))
    assert abs(v - v_rdm) < tol

    if boundary != 'obc':
        s1, s2, s3 = (3, 2), (4, 1), (4, 2)
        v = env.measure_2x2(sz, sz, sz, sites=(s1, s2, s3))
        s1, s2, s3 = map(g.site2index, (s1, s2, s3))
        assert abs(vals[s1] * vals[s2] * vals[s3] - v) < tol
    #
    #  measure_line
    #
    for s1, s2, s3 in [((1, 0), (1, 2), (1, 1)), ((0, 2), (2, 2), (3, 2))]:
        v = env.measure_line(sz, sz, sz, sites=(s1, s2, s3))
        assert abs(vals[s1] * vals[s2] * vals[s3] - v) < tol

    if boundary != 'obc':
        for s1, s2 in [((2, 0), (2, 5)), ((0, 1), (6, 1))]:
            v = env.measure_line(sz, sz, sites=(s1, s2))
            s1, s2 = map(g.site2index, (s1, s2))
            assert abs(vals[s1] * vals[s2] - v) < tol
    #
    #  sample
    #
    vecs = {v: ops.vec_z(val=v) for v in [-1, 0, 1]}
    out = env.sample(xrange=(1, 4), yrange=(1, 3), number=8, projectors=vecs)
    assert all(all(x == vals[k] for x in v) for k, v in out.items())

    if boundary != 'obc':
        out = env.sample(xrange=(3, 7), yrange=(-1, 2), number=5, projectors=vecs)
        assert all(all(x == vals[g.site2index(k)] for x in v) for k, v in out.items())

    #
    #  measure_2site
    #
    out = env.measure_2site(sz, sz, xrange=(1, 4), yrange=(0, 3))
    assert all(abs(vals[s0] * vals[s1] - v) < tol for (s0, s1), v in out.items())

    if boundary != 'obc':
        out = env.measure_2site(sz, sz, xrange=(3, 7), yrange=(-1, 2))
        assert all(abs(vals[g.site2index(s0)] * vals[g.site2index(s1)] - v) < tol for (s0, s1), v in out.items())
    #
    # save, copy, ...
    #
    run_ctm_save_load_copy(env)

    with pytest.raises(yastn.YastnError):
        env.measure_2x2(sz, sz, sz, sites=((0, 0), (1, 1)))
        # Number of operators and sites should match.
    with pytest.raises(yastn.YastnError):
        env.measure_2x2(sz, sz, sites=((0, 0), (1, 2)))
        # Sites do not form a 2x2 window.
    with pytest.raises(yastn.YastnError):
        env.measure_line(sz, sz, sz, sites=((0, 0), (1, 0)))
        # Number of operators and sites should match.
    with pytest.raises(yastn.YastnError):
        env.measure_line(sz, sz, sites=((0, 0), (1, 2)))
        # Sites should form a horizontal or vertical line.


@pytest.mark.parametrize("env_init", ["eye", "dl"])
def test_ctmrg_measure_2x1(config_kwargs, env_init):
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

            env = fpeps.EnvCTM(psi, init=env_init)
            # no need to converge ctmrg_ in this example
            env.ctmrg_(opts_svd = {"D_total": 3}, max_sweeps=2)

            val = -1 / 3  #  result of <psi | cs_0+ cs_1 | psi>
            for s in ['u', 'd']:
                bond = [*g.sites()]
                assert abs(env.measure_nn(ops.cp(s), ops.c(s), bond=bond) - val) < tol
                assert abs(env.measure_nn(ops.c(s), ops.cp(s), bond=bond[::-1]) - (-val)) < tol
                assert abs(env.measure_nn(ops.c(s), ops.cp(s), bond=bond) - (-val.conjugate())) < tol
                assert abs(env.measure_nn(ops.cp(s), ops.c(s), bond=bond[::-1]) - (val.conjugate())) < tol
                dirn, l_ordered= g.nn_bond_type(bond)
                if l_ordered:
                    assert abs(measure_rdm_nn(bond[0],dirn,psi,env,(ops.cp(s), ops.c(s))) - val) < tol
                    assert abs(measure_rdm_nn(bond[0],dirn,psi,env,(ops.c(s), ops.cp(s))) - (-val.conjugate())) < tol


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

                    env = fpeps.EnvCTM(psi, init=env_init)
                    # no need to converge ctmrg_ in this example, but we can do it anyway
                    # env.ctmrg_(opts_svd = {"D_total": 2}, max_sweeps=2)
                    bond = [*g.sites()]
                    assert abs(env.measure_nn(ops.cp('u'), ops.c('u'), bond=bond) - val) < tol
                    assert abs(env.measure_nn(ops.c('u'), ops.cp('u'), bond=bond[::-1]) - (-val)) < tol
                    assert abs(env.measure_nn(ops.c('u'), ops.cp('u'), bond=bond) - (-val.conjugate())) < tol
                    assert abs(env.measure_nn(ops.cp('u'), ops.c('u'), bond=bond[::-1]) - (val.conjugate())) < tol
                    dirn, l_ordered= g.nn_bond_type(bond)
                    if l_ordered:
                        assert abs(measure_rdm_nn(bond[0],dirn,psi,env,(ops.cp('u'), ops.c('u'))) - val) < tol
                        assert abs(measure_rdm_nn(bond[0],dirn,psi,env,(ops.c('u'), ops.cp('u'))) - (-val.conjugate())) < tol


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
