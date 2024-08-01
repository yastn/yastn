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
import yastn.tn.mps as mps
try:
    from .configs import config as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config as cfg


tol = 1e-12

def mean(xs):
    return sum(xs) / len(xs)


@pytest.mark.parametrize("boundary", ["obc", "infinite"])
def test_ctmrg_measure_product(boundary):
    """ Initialize a product PEPS and perform a set of measurment. """

    ops = yastn.operators.Spin1(sym='Z3', backend=cfg.backend, default_device=cfg.default_device)

    # initialized PEPS in a product state
    g = fpeps.SquareLattice(dims=(4, 3), boundary=boundary)
    sites = g.sites()
    vals = [0, 1, -1, 1, 1, 0, 1, -1, -1, -1, 1, -1]
    vals = dict(zip(sites, vals))
    occs = {s: ops.vec_z(val=v) for s, v in vals.items()}
    psi = fpeps.product_peps(g, occs)

    env = fpeps.EnvCTM(psi, init='eye')

    sz = ops.sz()
    ez = env.measure_1site(sz)
    assert all(abs(v - ez[s]) < tol for s, v in vals.items())

    ezz = env.measure_nn(sz, sz)
    for (s1, s2), v in ezz.items():
        s1, s2 = map(g.site2index, (s1, s2))
        assert abs(vals[s1] * vals[s2] - v) < tol

    for s1, s2 in [((0, 1), (1, 0)), ((2, 1), (2, 2)), ((1, 1), (2, 1))]:
        v = env.measure_2x2(sz, sz, sites=(s1, s2))
        assert abs(vals[s1] * vals[s2] - v) < tol

    s1, s2, s3 = (1, 2), (2, 1), (2, 2)
    v = env.measure_2x2(sz, sz, sz, sites=(s1, s2, s3))
    assert abs(vals[s1] * vals[s2] * vals[s3] - v) < tol

    if boundary != 'obc':
        s1, s2, s3 = (3, 2), (4, 1), (4, 2)
        v = env.measure_2x2(sz, sz, sz, sites=(s1, s2, s3))
        s1, s2, s3 = map(g.site2index, (s1, s2, s3))
        assert abs(vals[s1] * vals[s2] * vals[s3] - v) < tol

    for s1, s2, s3 in [((1, 0), (1, 2), (1, 1)), ((0, 2), (2, 2), (3, 2))]:
        v = env.measure_line(sz, sz, sz, sites=(s1, s2, s3))
        assert abs(vals[s1] * vals[s2] * vals[s3] - v) < tol

    if boundary != 'obc':
        for s1, s2 in [((2, 0), (2, 5)), ((0, 1), (6, 1))]:
            v = env.measure_line(sz, sz, sites=(s1, s2))
            s1, s2 = map(g.site2index, (s1, s2))
            assert abs(vals[s1] * vals[s2] - v) < tol

    with pytest.raises(yastn.YastnError):
        env.measure_2x2(sz, sz, sz, sites=((0, 0), (1, 1)))
        # Number of operators and sites should match.
    with pytest.raises(yastn.YastnError):
        env.measure_2x2(sz, sz, sites=((0, 0), (1, 2)))
        # Sites do not form a 2x2 window.
    with pytest.raises(yastn.YastnError):
        env.measure_2x2(sz, sz, sites=((0, 0), (0, 0)))
        # Sites should not repeat.
    with pytest.raises(yastn.YastnError):
        env.measure_line(sz, sz, sz, sites=((0, 0), (1, 0)))
        # Number of operators and sites should match.
    with pytest.raises(yastn.YastnError):
        env.measure_line(sz, sz, sites=((0, 0), (1, 2)))
        # Sites should form a horizontal or vertical line.
    with pytest.raises(yastn.YastnError):
        env.measure_line(sz, sz, sites=((0, 0), (0, 0)))
        # Sites should not repeat.



def test_ctmrg_measure_fermionic():
    """ Initialize a product PEPS of 1x2 cells and perform a set of measurment. """

    for dims in [(1, 2), (2, 1)]:
        g = fpeps.SquareLattice(dims=dims, boundary='infinite')
        ops = yastn.operators.SpinfulFermions(sym='Z2')

        v0000 = yastn.ncon([ops.vec_n((0, 0)), ops.vec_n((0, 0))], [[-0], [-1]])
        v0110 = yastn.ncon([ops.vec_n((0, 1)), ops.vec_n((1, 0))], [[-0], [-1]])
        v1100 = yastn.ncon([ops.vec_n((1, 1)), ops.vec_n((0, 0))], [[-0], [-1]])
        state = v1100 + v0110

        val_no = yastn.vdot(state, state)
        cu = ops.c('u').add_leg(axis=2)
        tmp = yastn.ncon([cu, state], [[-1, 1, -2], [-0, 1]])
        tmp = tmp.swap_gate(axes=(0, 2))
        tmp = tmp.remove_leg(axis=2)
        tmp = yastn.ncon([ops.cp('u'), tmp], [[-0, 1], [1, -1]])
        val_ex = yastn.vdot(state, tmp)
        assert abs(val_ex / val_no + 0.5) < tol


        o1100 = fpeps.fkron(ops.cp('u') @ ops.cp('d'), ops.I())
        o0110 = fpeps.fkron(ops.cp('d'), ops.cp('u'))
        v1100 = yastn.ncon([o1100, v0000], [(-0, -1, 1, 2), (1, 2)])
        v0110 = yastn.ncon([o0110, v0000], [(-0, -1, 1, 2), (1, 2)])
        state = v1100 + v0110

        r0, r1 = yastn.qr(state, sQ=-1, Qaxis=0)

        r0 = r0.add_leg(axis=0, s=-1)
        r0 = r0.add_leg(axis=1, s=1)
        if dims == (1, 2):
            r0 = r0.add_leg(axis=2, s=1)
            r1 = r1.add_leg(axis=0, s=-1)
        else:  # dims == (2, 1):
            r0 = r0.add_leg(axis=3, s=-1)
            r1 = r1.add_leg(axis=1, s=1)
        r1 = r1.add_leg(axis=2, s=1)
        r1 = r1.add_leg(axis=3, s=-1)

        psi = fpeps.Peps(g, tensors=dict(zip(g.sites(), [r0, r1])))
        env = fpeps.EnvCTM(psi, init='eye')

        info = env.ctmrg_(opts_svd = {"D_total": 2}, max_sweeps=5)
        print(info)

        val = env.measure_nn(ops.cp('u'), ops.c('u'), bond=g.sites())
        print(val)

        val = env.measure_1site(ops.n('u'))
        print(val)
        val = env.measure_1site(ops.n('d'))
        print(val)



if __name__ == '__main__':
    test_ctmrg_measure_product(boundary='obc')
    test_ctmrg_measure_product(boundary='infinite')
    test_ctmrg_measure_fermionic()
