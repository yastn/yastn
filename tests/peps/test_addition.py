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
""" Test fpeps.add and __add__ """
import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.mps as mps

tol = 1e-12  #pylint: disable=invalid-name


def test_add_peps(config_kwargs):
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)

    geometry = fpeps.SquareLattice(dims=(2, 3), boundary='obc')
    vec = {-1: ops.vec_x(val=-1), 1: ops.vec_x(val=1)}
    occ0 = {(0, 0): -1, (0, 1):  1, (0, 2): 1, (1, 0):  1, (1, 1): -1, (1, 2): -1}
    occ1 = {(0, 0): -1, (0, 1): -1, (0, 2): 1, (1, 0): -1, (1, 1):  1, (1, 2):  1}
    #
    psi0 = fpeps.product_peps(geometry, {k: vec[v] for k, v in occ0.items()})
    psi1 = fpeps.product_peps(geometry, {k: vec[v] for k, v in occ1.items()})
    #
    psi = psi0 + psi1
    env = fpeps.EnvCTM(psi, init='rand')
    env.iterate_(opts_svd={"D_total": 2}, max_sweeps=5, corner_tol=1e-4)
    evs = env.measure_1site(ops.x())
    assert psi[0, 0].unfuse_legs(axes=(0, 1)).shape == (1, 1, 2, 2, 2)
    assert psi[1, 2].unfuse_legs(axes=(0, 1)).shape == (2, 2, 1, 1, 2)
    assert all(abs(v.item() - (0.5 * occ0[k] + 0.5 * occ1[k])) < tol for k, v in evs.items())
    #
    psi = fpeps.add(psi0, psi1, amplitudes=(3, 4))
    env = fpeps.EnvCTM(psi, init='rand')
    env.iterate_(opts_svd={"D_total": 2}, max_sweeps=5, corner_tol=1e-4)
    evs = env.measure_1site(ops.x())
    assert all(abs(v.item() - (0.36 * occ0[k] + 0.64 * occ1[k])) < tol for k, v in evs.items())
    #
    geometry2 = fpeps.SquareLattice(dims=(2, 3), boundary='infinite')
    psi2 = fpeps.product_peps(geometry2, {k: vec[v] for k, v in occ0.items()})
    psi3 = fpeps.product_peps(geometry2, {k: vec[v] for k, v in occ1.items()})
    psi = psi2 + psi3
    assert psi[0, 0].unfuse_legs(axes=(0, 1)).shape == (2, 2, 2, 2, 2)
    assert psi[1, 2].unfuse_legs(axes=(0, 1)).shape == (2, 2, 2, 2, 2)
    #
    with pytest.raises(yastn.YastnError,
                       match='All added states should have the same geometry.'):
        psi1 + psi2
    with pytest.raises(yastn.YastnError,
                       match='All added states should be Peps-s.'):
        psi1 + None
    with pytest.raises(yastn.YastnError,
                       match='Number of Peps-s to add must be equal to the number of coefficients in amplitudes.'):
        fpeps.add(psi0, psi1, amplitudes=(1, 2, 3))


def test_to_tensor(config_kwargs):
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    #
    geometry = fpeps.SquareLattice(dims=(2, 3), boundary='obc')
    vec = {0: ops.vec_n(val=0), 1: ops.vec_n(val=1)}
    occ0 = {(0, 0): 0, (0, 1): 0, (0, 2): 1, (1, 0): 0, (1, 1): 1, (1, 2): 1}
    occ1 = {(0, 0): 0, (0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 0, (1, 2): 0}
    #
    psi0 = fpeps.product_peps(geometry, {k: vec[v] for k, v in occ0.items()})
    gates = [fpeps.gates.Gate([((-1) ** occ0[(1, 0)]) * ops.cp().add_leg(axis=2, s=-1), ops.c().add_leg(axis=2, s=1)], [(1, 0), (1, 1)]),
             fpeps.gates.Gate([((-1) ** occ0[(0, 1)]) * ops.cp().add_leg(axis=2, s=-1), ops.c().add_leg(axis=2, s=1)], [(0, 1), (0, 2)]),
             fpeps.gates.Gate([((-1) ** occ0[(0, 2)]) * ops.cp().add_leg(axis=2, s=-1), ops.c().add_leg(axis=2, s=1)], [(0, 2), (1, 2)])]
    # moving charges is done to retain original auxiliary legs for subsequent addition
    # there are swap-gates with auxiliary charges (if there was originally charge 1 on left site)
    # that is why there is "-" in the last gate to ofset it
    #
    psi1 = psi0.shallow_copy()
    for gate in gates:
        fpeps.apply_gate_(psi1, gate)
    #
    env = fpeps.EnvBP(psi1)
    env.iterate_(max_sweeps=5, diff_tol=1e-12)
    evs = env.measure_1site(ops.n())
    assert all(abs(evs[k] - occ1[k]) < tol for k in evs)
    #
    psi = (psi0 + psi1).to_tensor()
    # remove auxiliary legs
    for k in [11, 9, 7, 5, 3, 1]:
         psi = psi.remove_leg(axis=k)
    #
    phi0 = mps.product_mps([vec[occ0[k]] for k in geometry.sites()])
    phi1 = mps.product_mps([vec[occ1[k]] for k in geometry.sites()])
    phi = (phi0 + phi1).to_tensor()
    #
    assert (psi - phi).norm() < tol
    #
    # for infinite system, raise an error
    #
    geometry = fpeps.SquareLattice(dims=(2, 3), boundary='infinite')
    psi = fpeps.product_peps(geometry, {k: vec[v] for k, v in occ0.items()})
    with pytest.raises(yastn.YastnError,
                       match='to_tensor\(\) works only for a finite Peps.'):
        psi = psi.to_tensor()


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
