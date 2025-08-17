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


def peps34_to_tensor(psi):
    """
    Turn PEPS 2x3 into a single tensor.
    """
    A00 = psi[0, 0].unfuse_legs(axes=(0, 1)).remove_leg(axis=1).remove_leg(axis=0)
    A10 = psi[1, 0].unfuse_legs(axes=(0, 1)).remove_leg(axis=1)
    A20 = psi[2, 0].unfuse_legs(axes=(0, 1)).remove_leg(axis=2).remove_leg(axis=1)

    A01 = psi[0, 1].unfuse_legs(axes=(0, 1)).remove_leg(axis=0)
    A11 = psi[1, 1].unfuse_legs(axes=(0, 1))
    A21 = psi[2, 1].unfuse_legs(axes=(0, 1)).remove_leg(axis=2)

    A02 = psi[0, 2].unfuse_legs(axes=(0, 1)).remove_leg(axis=0)
    A12 = psi[1, 2].unfuse_legs(axes=(0, 1))
    A22 = psi[2, 2].unfuse_legs(axes=(0, 1)).remove_leg(axis=2)

    A03 = psi[0, 3].unfuse_legs(axes=(0, 1)).remove_leg(axis=3).remove_leg(axis=0)
    A13 = psi[1, 3].unfuse_legs(axes=(0, 1)).remove_leg(axis=3)
    A23 = psi[2, 3].unfuse_legs(axes=(0, 1)).remove_leg(axis=3).remove_leg(axis=2)

    psit = yastn.ncon([A00, A10, A20, A01, A11, A21, A02, A12, A22, A03, A13, A23],
                      [[1, 2, -0], [1, 3, -1], [2, 4, 5, -2],
                       [4, 3, 6, -3], [5, 7, -4], [7, 6, -5]])

    if psit.get_legs(axes=0).is_fused():
        psit = psit.unfuse_legs(axes=(0, 1, 2, 3, 4, 5))

    return psit


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

    psi0 = fpeps.product_peps(geometry, {k: vec[v] for k, v in occ0.items()})
    psi1 = fpeps.product_peps(geometry, {k: vec[v] for k, v in occ1.items()})

    psi = psi0 + psi1

    i2s = dict(enumerate(sites))
    s2i = {s: i for i, s in i2s.items()}  # 1d order of sites for mps



    D, dt = 8, 0.05
    steps = round(tf / dt)
    dt = tf / steps

    I = ops.I()
    gates_nn = []
    gates_local = []
    for spin in 'ud':
        for bond, t in Js.items():
            gt = fpeps.gates.gate_nn_hopping(t, 1j * dt / 2, I, ops.c(spin=spin), ops.cp(spin=spin))
            gates_nn.append(gt._replace(sites=bond))
        for site, mu in ms.items():
            gt = fpeps.gates.gate_local_occupation(mu, 1j * dt / 2, I, ops.n(spin=spin))
            gates_local.append(gt._replace(sites=(site,)))
    gates = fpeps.Gates(nn=gates_nn, local=gates_local)
    #
    # initialized product state
    psi = fpeps.product_peps(geometry, {s: ops.vec_n(val=(occs['u'][s], occs['d'][s])) for s in sites})
    #
    # time-evolve initial state
    env = fpeps.EnvNTU(psi, which='NN')
    opts_svd = {"D_total": D, 'tol': 1e-12}
    for step in range(steps):
        print(f"t = {(step + 1) * dt:0.3f}" )
        fpeps.evolution_step_(env, gates, opts_svd=opts_svd)

    opts_svd_mps = {'D_total': D, 'tol': 1e-10}
    env = fpeps.EnvBoundaryMPS(psi, opts_svd=opts_svd_mps, setup='lr')
    for spin in 'ud':
        print(f"{spin=}")
        occf = env.measure_1site(ops.n(spin=spin))
        for k, v in sorted(occf.items()):
            print(f"{k}, {v.real:0.7f}, {Cf[spin][s2i[k], s2i[k]].real:0.7f}, {v.real - Cf[spin][s2i[k], s2i[k]].real:0.2e}")
            # assert abs(v - Cf[spin][s2i[k], s2i[k]]) < 5e-4


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
