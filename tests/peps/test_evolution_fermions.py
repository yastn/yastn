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
""" Test evolution of PEPS using an example of quantum Fourier transform. """
import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.mps as mps

tol = 1e-12  #pylint: disable=invalid-name


def test_peps_evolution_hopping(config_kwargs):
    """ Simulate spinless fermions hopping on a small finite system. """
    #
    geometry = fpeps.SquareLattice(dims=(3, 4), boundary='obc')
    bonds = geometry.bonds()
    sites = geometry.sites()
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', default_dtype='complex128', **config_kwargs)
    vec = {0: ops.vec_n(val=0), 1: ops.vec_n(val=1)}
    occ0 = dict(zip(sites, [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1]))
    occ1 = dict(zip(sites, [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]))
    #
    psi0 = fpeps.product_peps(geometry, {k: vec[v] for k, v in occ0.items()})
    psi1 = fpeps.product_peps(geometry, {k: vec[v] for k, v in occ1.items()})
    psi = psi0 + psi1
    #
    i2s = dict(enumerate(sites))
    s2i = {s: i for i, s in i2s.items()}  # 1d order of sites for mps
    #
    gates_nn = []
    gates_local = []
    mu = 0.1
    D, dt = 8, 0.05
    # gt = fpeps.gates.gate_nn_hopping(t, 1j * dt / 2, I, ops.c(spin=spin), ops.cp(spin=spin))
    # gates_nn.append(gt._replace(sites=bond))
    # gt = fpeps.gates.gate_local_occupation(mu, 1j * dt / 2, I, ops.n(spin=spin))
    # gates = fpeps.Gates(nn=gates_nn, local=gates_local)
    # #
    # initialized product state



if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
