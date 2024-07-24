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


@pytest.mark.parametrize("boundary", ["obc", "cylinder"])
def test_ctmrg_measure(boundary):
    """ Initialize a product PEPS and perform a set of measurment. """

    ops = yastn.operators.Spin1(sym='Z3', backend=cfg.backend, default_device=cfg.default_device)

    # initialized PEPS in a product state
    geometry = fpeps.SquareLattice(dims=(4, 3), boundary=boundary)
    sites = geometry.sites()
    vals = [0, 1, -1, 1, 1, 0, 1, -1, -1, -1, 1, -1]
    vals = dict(zip(sites, vals))
    occs = {s: ops.vec_z(val=v) for s, v in vals.items()}
    psi = fpeps.product_peps(geometry, occs)

    env = fpeps.EnvCTM(psi, init='eye')

    sz = ops.sz()
    ez = env.measure_1site(sz)
    assert all(abs(v - ez[s]) < tol for s, v in vals.items())

    ezz = env.measure_nn(sz, sz)
    for (s1, s2), v in ezz.items():
        s1, s2 = map(geometry.site2index, (s1, s2))
        assert abs(vals[s1] * vals[s2] - v) < tol

    for s1, s2 in [((0, 1), (1, 0)), ((2, 1), (2, 2)), ((1, 1), (2, 1))]:
        v = env.measure_2x2(sz, sz, sites=(s1, s2))
        assert abs(vals[s1] * vals[s2] - v) < tol

    s1, s2, s3 = (1, 2), (2, 1), (2, 2)
    v = env.measure_2x2(sz, sz, sz, sites=(s1, s2, s3))
    assert abs(vals[s1] * vals[s2] * vals[s3] - v) < tol

    with pytest.raises(yastn.YastnError):
        env.measure_2x2(sz, sz, sz, sites=((0, 0), (1, 1)))
        # Number of operators and sites should match.

    with pytest.raises(yastn.YastnError):
        env.measure_2x2(sz, sz, sites=((0, 0), (1, 2)))
        # Sites do not form a 2x2 window.

if __name__ == '__main__':
    test_ctmrg_measure(boundary='obc')
    test_ctmrg_measure(boundary='infinite')
