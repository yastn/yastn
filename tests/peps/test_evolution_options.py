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
""" Real-time evolution of spinless fermions on a cylinder. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps

try:
    from .configs import config as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config as cfg


def test_evolution():
    """ Simulate purification of a small finite system to test evolution options and output. """
    #
    Nx, Ny = 2, 2
    geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary='obc')
    #
    # prepare evolution gates
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    I = ops.I()
    t, dbeta = 1.0, 0.1
    g_hop = fpeps.gates.gate_nn_hopping(t, dbeta / 2, I, ops.c(), ops.cp())
    gates = fpeps.gates.distribute(geometry, gates_nn=g_hop)
    #
    #
    # time-evolve initial state
    #
    opts_svd = {"D_total": 4, 'tol': 1e-14}
    steps = 4
    old_Delta = 1
    for initialization, le, lp in [("SVD", 2, 1), ("EAT", 2, 2), ("EAT_SVD", 4, 3)]:
        # initialized product state
        psi = fpeps.product_peps(geometry, ops.I())
        env = fpeps.EnvNTU(psi, which='NN')
        infoss = []
        for _ in range(steps):
            infos = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization=initialization)
            infoss.append(infos)

        assert len(infoss[0]) == 2 * ((Nx - 1) * Ny + (Ny - 1) * Nx)
        info = infoss[3][0]
        assert len(info.truncation_errors) == le and len(info.pinv_cutoffs) == lp
        assert abs(info.truncation_error - info.truncation_errors[1]) < 1e-12

        Delta = fpeps.accumulated_truncation_error(infoss)
        # accumulated truncation error should be <= between 'SVD', 'EAT', 'EAT_SVD'
        assert Delta < old_Delta + 1e-8
        old_Delta = Delta

    with pytest.raises(yastn.YastnError):
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization='none')
        # initialization='none' not recognized. Should contain 'SVD' or 'EAT'.

if __name__ == '__main__':
    test_evolution()
