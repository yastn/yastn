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
# from yastn.tn.fpeps.envs.rdm import measure_rdm_1site, measure_rdm_nn, measure_rdm_2x2
import yastn.tn.mps as mps

tol = 1e-6  #pylint: disable=invalid-name


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
                    (3, 0): 1, (3, 1): 0, (3, 2): 0, (3, 3): 1}
    occs_init[3] = {(0, 0): 1, (0, 1): 0, (0, 2): 1,
                    (1, 0): 0, (1, 1): 1, (1, 2): 1,
                    (2, 0): 1, (2, 1): 0, (2, 2): 0}
    occs_init[2] = {(0, 0): 1, (0, 1): 0,
                    (1, 0): 0, (1, 1): 1}
    #
    # and apply a single layer of hopping gates with large random angls
    ops.random_seed(seed=0)
    angles = [(bond, (0.1 + 1j) * ops.config.backend.rand(1))
              for bond in g.bonds()]
    angles = angles + angles[::-1]
    #
    phi = generate_mps(ops, occs_init[L], angles, s2i)
    psi = generate_peps(g, ops, occs_init[L], angles)
    #
    # converge ctm
    env_ctm = fpeps.EnvCTM(psi, init='dl')
    opts_svd = {'D_total': 64, 'tol': 1e-14}
    info = env_ctm.ctmrg_(max_sweeps=200, opts_svd=opts_svd, corner_tol=1e-6)
    print(info)
    assert info.converged
    #
    env_bd = fpeps.EnvBoundaryMPS(psi, opts_svd=opts_svd, setup='lr')
    #
    # check occupations
    occ_bd = env_bd.measure_1site(ops.n())
    occ_mps = mps.measure_1site(phi, ops.n(), phi)
    occ_ctm = env_ctm.measure_1site(ops.n())
    for site in g.sites():
        assert abs(occ_mps[s2i[site]] - occ_ctm[site]) < tol
        assert abs(occ_mps[s2i[site]] - occ_bd[site]) < tol
    assert abs(sum(occ_ctm.values()) - sum(occs_init[L].values())) < tol
    #
    # check 2-point correlators
    nn_mps = mps.measure_2site(phi, ops.n(), ops.n(), phi, bonds='a')
    nn_bd = env_bd.measure_2site(ops.n(), ops.n(), opts_svd=opts_svd)
    nn_ctm = env_bd.measure_2site(ops.n(), ops.n(), opts_svd=opts_svd)

    for (s0, s1), v in nn_bd.items():
        assert abs((nn_mps[s2i[s0], s2i[s1]] - v) / v) < tol
    for (s0, s1), v in nn_ctm.items():
        assert abs((nn_mps[s2i[s0], s2i[s1]] - v) / v) < tol
    #
    # check 2-point correlators
    cpc_mps = mps.measure_2site(phi, ops.cp(), ops.c(), phi, bonds='a')
    cpc_ctm = env_ctm.measure_nn(ops.cp(), ops.c())

    for (s0, s1), v in cpc_ctm.items():
        # print(s0, s1, v, cpc_mps[s2i[s0], s2i[s1]])
        # print(s0, s1, v, cpc_mps[s2i[s1], s2i[s0]])
        assert abs(cpc_mps[s2i[s0], s2i[s1]] - v) < tol

    for s0 in g.sites():
        for s1 in g.sites():
            try:
                v = env_ctm.measure_2x2(ops.cp(), ops.c(), sites=[s0, s1])
                print(s0, s1, v, cpc_mps[s2i[s0], s2i[s1]])
                print(s0, s1, v, cpc_mps[s2i[s1], s2i[s0]])
                assert abs(cpc_mps[s2i[s0], s2i[s1]] - v) < tol
            except:
                pass


if __name__ == '__main__':
    pytest.main([__file__, "-vs"])
