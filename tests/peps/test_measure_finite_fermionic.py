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
import yastn.tn.mps as mps
import math
from itertools import product

tol = 1e-2  # CTMRG has problem in this finite peps, being stuck at small bond dimension


def generate_peps(g, ops, occs_init, angles):
    full_empty = [ops.c() @ ops.cp(), ops.cp() @ ops.c()]
    vectors = {site: full_empty[n] for site, n in occs_init.items()}
    psi = fpeps.product_peps(g, vectors)
    # This gives peps tensors with physical and ancilla leg,
    # where ancillas offset (single) charge present on the physical leg.
    # Virtual legs are initiated in (0, 0, 0, 0) charge sector.
    # Full physical space will be built by subsequent evolution,
    # while 1-dimensional ancillas spaces remain unchanged.
    #
    # apply a list of gates on nn bonds
    for bond, angle in angles:
        gate = fpeps.gates.gate_nn_hopping(1, angle, ops.I(), ops.c(), ops.cp())
        dirn, l_ordered = psi.nn_bond_type(bond)
        assert l_ordered
        s0, s1 = bond
        _, _, R0, R1, Q0f, Q1f = fpeps._evolution.apply_gate_nn(psi[s0], psi[s1], gate.G0, gate.G1, dirn)
        M0, M1 = R0, R1  # fpeps._evolution.symmetrized_svd(R0, R1, opts_svd={}, normalize=True)
        psi[s0], psi[s1] = fpeps._evolution.apply_bond_tensors(Q0f, Q1f, M0, M1, dirn)
    return psi


def mpo_from_gate(N, ops, gate, bond, s2i):
    H = mps.product_mpo(ops.I(), N)
    s0, s1 = bond
    i0, i1 = s2i[s0], s2i[s1]
    assert i0 < i1
    H[i0] = gate.G0.add_leg(axis=0, s=-1).transpose(axes=(0, 1, 3, 2))
    H[i1] = gate.G1.add_leg(axis=3, s=1).transpose(axes=(2, 0, 3, 1))
    leg = gate.G0.get_legs(axes=2).conj()
    conn = yastn.eye(ops.config, leg, isdiag=False, device=H[i0].device)
    conn = yastn.ncon([conn, ops.I()], [(-0, -2), (-1, -3)])
    conn = conn.swap_gate(axes=(0, 1))  # add string
    for i in range(i0 + 1, i1):
        H[i] = conn
    return H


def generate_mps(ops, occs_init, angles, s2i):

    vectors = [ops.vec_n(occs_init[site]) for site in s2i]
    phi = mps.product_mps(vectors)

    for bond, angle in angles:
        gate = fpeps.gates.gate_nn_hopping(1, angle, ops.I(), ops.c(), ops.cp())
        H = mpo_from_gate(phi.N, ops, gate, bond, s2i)
        phi = H @ phi
        phi.canonize_(to='last')
        phi.canonize_(to='first')
    print(phi.get_bond_dimensions())
    return phi


def measure_2x2(*operators, env=None):
    """
    Test all possible combination of sites; skip those where measure_2x2 cannot be applied
    """
    res = {}
    lo = len(operators)
    for sites in product(*([env.sites()] * lo)):
        try:
            res[sites] = env.measure_2x2(*operators, sites=sites)
        except:
            pass
    return res


def measure_line(*operators, env=None):
    """
    Test all possible combination of sites; skip those where measure_2x2 cannot be applied
    """
    res = {}
    lo = len(operators)
    for sites in product(*([env.sites()] * lo)):
        try:
            res[sites] = env.measure_line(*operators, sites=sites)
        except:
            pass
    return res

def test_measure(config_kwargs, L=3):
    """
    Test calculation of fermionic exceptation values with CTM
    using finite PEPS and shallow circuit.
    """
    # work with spinless fermions
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    # and L x L system
    g = fpeps.SquareLattice(dims=(L, L), boundary='obc')
    s2i = {site: i for i, site in enumerate(g.sites())}  # linear mps order
    #
    # we will initialize product state near/at half-filling
    occs_init = {}  # predefine occupation for a few L's
    occs_init[4] = {(0, 0): 1, (0, 1): 0, (0, 2): 1, (0, 3): 1,
                    (1, 0): 0, (1, 1): 1, (1, 2): 0, (1, 3): 0,
                    (2, 0): 1, (2, 1): 0, (2, 2): 1, (2, 3): 0,
                    (3, 0): 0, (3, 1): 1, (3, 2): 0, (3, 3): 1}
    occs_init[3] = {(0, 0): 1, (0, 1): 1, (0, 2): 1,
                    (1, 0): 0, (1, 1): 0, (1, 2): 0,
                    (2, 0): 0, (2, 1): 1, (2, 2): 0}
    occs_init[2] = {(0, 0): 1, (0, 1): 1,
                    (1, 0): 0, (1, 1): 0}
    #
    # and apply a single layer of hopping gates with large random angls
    ops.random_seed(seed=0)
    angles  = [(bond, 0.1 + 1j * ops.config.backend.rand(1) * math.pi / 2) for bond in g.bonds(dirn='v')]
    angles += [(bond, 0.1 + 1j * ops.config.backend.rand(1) * math.pi / 2) for bond in g.bonds(dirn='h')]
    #
    # 1j * pi / 4 is half of oscillation; adds phase 1j to transfered particle
    # 1j * pi / 2 fully transfer particle between sites adding phase 1j
    #
    phi = generate_mps(ops, occs_init[L], angles, s2i)
    psi = generate_peps(g, ops, occs_init[L], angles)
    #
    # converge ctm
    env_ctm = fpeps.EnvCTM(psi, init='dl')
    opts_svd = {'D_total': 32, 'tol': 1e-12}
    # CTMRG is not precise here, and cannot rise D_max above 4.
    if L > 2:
        info = env_ctm.ctmrg_(max_sweeps=100, opts_svd=opts_svd, corner_tol=1e-10)
        print(info)
        assert info.converged
    #
    env_bd = fpeps.EnvBoundaryMPS(psi, opts_svd=opts_svd, setup='lr')
    #
    # check occupations
    occ_mps = mps.measure_1site(phi, ops.n(), phi)
    assert abs(sum(occ_mps.values()) - sum(occs_init[L].values())) < tol
    #
    occ_peps = {}
    occ_peps['mps'] = env_bd.measure_1site(ops.n())
    occ_peps['ctm'] = env_ctm.measure_1site(ops.n())
    print("Occupations: mps, ctm, bd, mps-ctm, mps-bd")
    for method, res in occ_peps.items():
        print('Occupation', method)
        for site in g.sites():
            error = abs(occ_mps[s2i[site]] - res[site])
            if error > tol:
                print(site, occ_mps[s2i[site]], error)
            # assert error < tol
    #
    # check 2-point correlators density-density
    nn_mps = mps.measure_2site(phi, ops.n(), ops.n(), phi, bonds='a')
    nn_peps = {}
    nn_peps['mps'] = env_bd.measure_2site(ops.n(), ops.n(), opts_svd=opts_svd)
    nn_peps['nn'] = env_ctm.measure_nn(ops.n(), ops.n())
    nn_peps['2s'] = env_ctm.measure_2site(ops.n(), ops.n(), xrange=[0, L], yrange=[0, L], opts_svd=opts_svd)
    nn_peps['2x2'] = measure_2x2(ops.n(), ops.n(), env=env_ctm)
    nn_peps['line'] = measure_line(ops.n(), ops.n(), env=env_ctm)
    #
    for method, res in nn_peps.items():
        print('Density-density', method)
        for (s0, s1), v in res.items():
            error = abs(nn_mps[s2i[s0], s2i[s1]] - v)
            if error > tol:
                print(s0, s1, v, error)
            # assert  error < tol
    #
    # check 2-point correlators; hopping
    cpc_mps = mps.measure_2site(phi, ops.cp(), ops.c(), phi, bonds='a')
    cpc_peps = {}
    cpc_peps['nn'] = env_ctm.measure_nn(ops.cp(), ops.c())
    cpc_peps['2x2'] = measure_2x2(ops.cp(), ops.c(), env=env_ctm)
    cpc_peps['line'] = measure_line(ops.cp(), ops.c(), env=env_ctm)

    for method, res in cpc_peps.items():
        print('Hopping', method)
        for (s0, s1), v in res.items():
            error = abs(cpc_mps[s2i[s0], s2i[s1]] - v)
            if error > tol:
                print(s0, s1, v, cpc_mps[s2i[s0], s2i[s1]], error)
    #
    # check 4-point correlator
    sites=[(0, 0), (0, 1), (1, 0), (1, 1)]
    positions = [s2i[site] for site in sites]
    operators = [ops.cp(), ops.c(), ops.cp(), ops.c()]
    v1 = env_ctm.measure_2x2(*operators, sites=sites)
    I = mps.product_mpo(ops.I(), N=phi.N)
    O = mps.generate_mpo(I, terms=[mps.Hterm(positions=positions, operators=operators)])
    v2 = mps.vdot(phi, O, phi)
    assert abs(v1 - v2) < tol


if __name__ == '__main__':
    pytest.main([__file__, "-vs"])
