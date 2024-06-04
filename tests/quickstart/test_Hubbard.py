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
""" Test Quickstart example """

def test_quickstart_hubbard():

    import yastn
    import yastn.tn.fpeps as fpeps

    t = 1
    mu = 0
    U = 10
    beta = 0.05  # inverse temperature

    ops = yastn.operators.SpinfulFermions(sym='U1xU1')
    I = ops.I()
    c_up, cdag_up, n_up = ops.c('u'), ops.cp('u'), ops.n('u')
    c_dn, cdag_dn, n_dn = ops.c('d'), ops.cp('d'), ops.n('d')

    geometry = fpeps.CheckerboardLattice()
    psi = fpeps.product_peps(geometry=geometry, vectors=I)

    db = 0.01  # Trotter step size
    steps = round((beta / 2) / db)
    db = (beta / 2) / steps

    g_hop_u = fpeps.gates.gate_nn_hopping(t, db/2, I, c_up, cdag_up)
    g_hop_d = fpeps.gates.gate_nn_hopping(t, db/2, I, c_dn, cdag_dn)
    g_loc = fpeps.gates.gate_local_Coulomb(mu, mu, U, db/2, I, n_up, n_dn)
    gates = fpeps.gates.distribute(geometry,
                                    gates_nn=[g_hop_u, g_hop_d],
                                    gates_local=g_loc)

    env = fpeps.EnvNTU(psi, which='NN')

    D = 8  # bond dimenson

    opts_svd = {'D_total': D, 'tol': 1e-12}
    infoss = []
    for step in range(1, steps + 1):
        print(f"beta_purification = {step * db:0.3f}" )
        infos = fpeps.evolution_step_(env, gates, opts_svd=opts_svd)
        infoss.append(infos)
    Delta = fpeps.accumulated_truncation_error(infoss)
    print(f"Accumulated mean truncation error: {Delta:0.6f}")


    env_ctm = fpeps.EnvCTM(psi, init='rand')
    chi = 5 * D
    opts_svd_ctm = {'D_total': chi, 'tol': 1e-10}

    mean = lambda data: sum(data) / len(data)

    energy_old, tol_exp = 0, 1e-7
    for i in range(50):
        #
        env_ctm.update_(opts_svd=opts_svd_ctm)  # single CMTRG sweep
        #
        # calculate energy expectation value
        #
        ev_nn = env_ctm.measure_1site((n_up - I / 2) @ (n_dn - I / 2))
        # calculate for all unique sites; {site: value}
        #
        ev_cdagc_up = env_ctm.measure_nn(cdag_up, c_up)
        ev_cdagc_dn = env_ctm.measure_nn(cdag_dn, c_dn)
        # calculate for all unique bonds; {bond: value}
        #
        energy = U * mean([*ev_nn.values()])  # mean over all sites
        energy += -4 * t * mean([*ev_cdagc_up.values()])
        energy += -4 * t * mean([*ev_cdagc_dn.values()])
        #
        print(f"Energy per site after iteration {i}: ", energy)
        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

    ev_n_up = env_ctm.measure_1site(n_up)
    ev_n_dn = env_ctm.measure_1site(n_dn)
    print("Occupation spin up: ", mean([*ev_n_up.values()]))
    print("Occupation spin dn: ", mean([*ev_n_dn.values()]))

    print("Kinetic energy per bond")
    print("spin up electrons: ", 2 * mean([*ev_cdagc_up.values()]))
    print("spin dn electrons: ", 2 * mean([*ev_cdagc_dn.values()]))

    print("Average double occupancy ", mean([*ev_nn.values()]))

    sz = 0.5 * (n_up - n_dn)  # Sz operator
    ev_szsz = env_ctm.measure_nn(sz, sz)
    print("Average NN spin-spin correlator ", mean([*ev_szsz.values()]))


if __name__ == '__main__':
    test_quickstart_hubbard()
