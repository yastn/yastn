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
import re
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
    assert psi[0, 0].shape == (1, 1, 2, 2, 2)
    assert psi[1, 2].shape == (2, 2, 1, 1, 2)
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
    assert psi[0, 0].shape == (2, 2, 2, 2, 2)
    assert psi[1, 2].shape == (2, 2, 2, 2, 2)
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


def mpo_cpc(ops, N, p0, p1):
    I, cp, c = ops.I(), ops.cp(), ops.c()
    return mps.generate_mpo(I, [mps.Hterm(1, [p0, p1], [cp, c])], N=N)


def test_to_tensor(config_kwargs):
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    #
    Nx, Ny = 2, 3
    geometry = fpeps.SquareLattice(dims=(Nx, Ny), boundary='obc')
    vec = {0: ops.vec_n(val=0), 1: ops.vec_n(val=1)}
    occ0 = {(0, 0): 0, (0, 1): 0, (0, 2): 1, (1, 0): 0, (1, 1): 1, (1, 2): 1}
    occ1 = {(0, 0): 0, (0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 0, (1, 2): 0}
    #
    psi0 = fpeps.product_peps(geometry, {k: vec[v] for k, v in occ0.items()})
    #
    # psi1 will have the same ancilas as psi0
    sitess = [((1, 0), (1, 1)), ((0, 1), (0, 2)), ((0, 2), (1, 2))]
    gates = [fpeps.Gate(mpo_cpc(ops, 2, 0, 1), sites) for sites in sitess]
    psi1 = psi0.shallow_copy()
    for gate in gates:
        psi1.apply_gate_(gate)
    #
    # test occupancy in psi1
    env = fpeps.EnvBP(psi1)
    env.iterate_(max_sweeps=5, diff_tol=1e-12)
    evs = env.measure_1site(ops.n())
    assert all(abs(evs[k] - occ1[k]) < tol for k in evs)
    #
    psi = (psi0 + psi1).to_tensor()
    for k in [11, 9, 7, 5, 3, 1]:  # remove auxiliary legs
        psi = psi.remove_leg(axis=k)
    #
    phi0 = mps.product_mps([vec[occ0[k]] for k in geometry.sites()])
    phi1 = mps.product_mps([vec[occ1[k]] for k in geometry.sites()])
    # verify with gates applied to phi0
    s2i = {s: k for k, s in enumerate(geometry.sites())}  # mapping to mps sites
    gates = [mpo_cpc(ops, Nx * Ny, s2i[s0], s2i[s1]) for s0, s1 in sitess]
    phi2 = phi0.shallow_copy()
    for gate in gates:
        phi2 = gate @ phi2
    assert (phi1 - phi2).norm() < tol
    #
    phi = (phi0 + phi1).to_tensor()
    #
    assert (psi - phi).norm() < tol
    #
    # for infinite system, raise an error
    #
    geometry = fpeps.SquareLattice(dims=(2, 3), boundary='infinite')
    psi = fpeps.product_peps(geometry, {k: vec[v] for k, v in occ0.items()})
    with pytest.raises(yastn.YastnError,
                       match=re.escape('to_tensor() works only for a finite PEPS.')):
        psi = psi.to_tensor()


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
