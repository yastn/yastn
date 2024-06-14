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


def test_window_shapes():
    """ Initialize a product PEPS and perform a set of measurment. """

    # initialized PEPS with mixed bond dimensions
    geometry = fpeps.SquareLattice(dims=(2, 3), boundary='infinite')
    psi = fpeps.Peps(geometry)
    psi[0, 0] = yastn.rand(cfg, s=(-1, 1, 1, -1, 1), D=(2, 3, 4, 5, 2))
    psi[1, 0] = yastn.rand(cfg, s=(-1, 1, 1, -1, 1), D=(4, 1, 2, 4, 2))
    psi[0, 1] = yastn.rand(cfg, s=(-1, 1, 1, -1, 1), D=(3, 5, 5, 2, 2))
    psi[1, 1] = yastn.rand(cfg, s=(-1, 1, 1, -1, 1), D=(5, 4, 3, 1, 2))
    psi[0, 2] = yastn.rand(cfg, s=(-1, 1, 1, -1, 1), D=(2, 2, 3, 3, 2))
    psi[1, 2] = yastn.rand(cfg, s=(-1, 1, 1, -1, 1), D=(3, 1, 2, 1, 2))

    opts_svd = {'D_total': 10, 'tol': 1e-10}
    env_ctm = fpeps.EnvCTM(psi, init='rand')  # in the product state no need for update
    env_ctm.update_(opts_svd=opts_svd)

    env = fpeps.EnvWindow(env_ctm, xrange=(0, 1), yrange=(0, 6))
    env['l', 0]


    # tv = env.transfer_mpo(2, dirn='v')
    # th = env.transfer_mpo(2, dirn='h')

    # print(len(tv))
    # for i in range(len(tv)):
    #     print(tv[i])
    # print(len(th))
    # for i in range(len(th)):
    #     print(th[i])


def test_window():
    """ Initialize a product PEPS and perform a set of measurment. """

    ops = yastn.operators.Spin1(sym='Z3', backend=cfg.backend, default_device=cfg.default_device)

    # initialized PEPS in a product state
    geometry = fpeps.SquareLattice(dims=(4, 3), boundary='infinite')
    sites = geometry.sites()
    vals = [1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1]
    vals = dict(zip(sites, vals))
    occs = {s: ops.vec_z(val=v) for s, v in vals.items()}
    psi = fpeps.product_peps(geometry, occs)

    opts_svd = {'D_total': 2, 'tol': 1e-10}
    env_ctm = fpeps.EnvCTM(psi, init='eye')  # in the product state no need for update

    env = fpeps.EnvWindow(env_ctm, xlim=(0, 2), ylim=(0, 3))

    tv = env.transfer_mpo(2, dirn='v')
    th = env.transfer_mpo(2, dirn='h')

    print(len(tv))
    for i in range(len(tv)):
        print(tv[i])
    print(len(th))
    for i in range(len(th)):
        print(th[i])

if __name__ == '__main__':
    test_window_shapes()
    # test_window()
