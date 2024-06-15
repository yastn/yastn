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
""" Test quickstart example Hubbard. """

# The script below can be benchmarked against reference values obtained from METTS on a cylinder 
# implemented with ITensor library. Keys are: (observable, beta).
metts_values = {("energy", 2.0): -2.80404, ("energy", 4.0): -2.87870, ("energy", 6.0): -2.91614, 
                ("ev_double", 2.0): 0.02817, ("ev_double", 4.0): 0.03384, ("ev_double", 6.0): 0.03649, 
                ("ev_SzSz", 2.0): -0.04167, ("ev_SzSz", 4.0):-0.07500, ("ev_SzSz", 6.0): -0.09099}


def test_quickstart_hubbard(D, beta=0.5):

    import yastn
    import yastn.tn.fpeps as fpeps
    from tqdm import tqdm  # progressbar

    t = 1
    mu = 0
    U = 10
    # beta = 0.5  # inverse temperature

    ops = yastn.operators.SpinfulFermions(sym='U1xU1')
    I = ops.I()
    c_up, cdag_up, n_up = ops.c('u'), ops.cp('u'), ops.n('u')
    c_dn, cdag_dn, n_dn = ops.c('d'), ops.cp('d'), ops.n('d')

    geometry = fpeps.CheckerboardLattice()
    psi = fpeps.product_peps(geometry=geometry, vectors=I)

    db = 0.01  # Trotter step size
    # making sure we have integer number of steps to target beta / 2
    steps = round((beta / 2) / db)
    db = (beta / 2) / steps

    g_hop_u = fpeps.gates.gate_nn_hopping(t, db / 2, I, c_up, cdag_up)
    g_hop_d = fpeps.gates.gate_nn_hopping(t, db / 2, I, c_dn, cdag_dn)
    g_loc = fpeps.gates.gate_local_Coulomb(mu, mu, U, db/2, I, n_up, n_dn)
    gates = fpeps.gates.distribute(geometry,
                                   gates_nn=[g_hop_u, g_hop_d],
                                   gates_local=g_loc)

    env = fpeps.EnvNTU(psi, which='NN')

    D = 12  # bond dimenson

    opts_svd = {'D_total': D, 'tol': 1e-12}
    infoss = []

    for _ in tqdm(range(1, steps + 1)):
        infos = fpeps.evolution_step_(env, gates, opts_svd=opts_svd)
        # The state psi is contained in env;
        # evolution_step_ updates psi in place.
        #
        infoss.append(infos)

    Delta = fpeps.accumulated_truncation_error(infoss)
    print(f"Accumulated truncation error {Delta:0.5f}")

    env_ctm = fpeps.EnvCTM(psi, init='eye')
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
        # calculate for all unique sites; {site: value}
        ev_nn = env_ctm.measure_1site((n_up - I / 2) @ (n_dn - I / 2))
        ev_nn = mean([*ev_nn.values()])  # mean over all sites
        #
        # calculate for all unique bonds; {bond: value}
        ev_cdagc_up = env_ctm.measure_nn(cdag_up, c_up)
        ev_cdagc_dn = env_ctm.measure_nn(cdag_dn, c_dn)
        ev_cdagc_up = mean([*ev_cdagc_up.values()]) # mean over bonds
        ev_cdagc_dn = mean([*ev_cdagc_dn.values()])
        #
        energy = -4 * t * (ev_cdagc_up + ev_cdagc_dn) + U * ev_nn
        #
        print(f"Energy per site after iteration {i}: {energy:0.8f}")
        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

    ev_n_up = env_ctm.measure_1site(n_up)
    ev_n_dn = env_ctm.measure_1site(n_dn)
    ev_n_up = mean([*ev_n_up.values()])
    ev_n_dn = mean([*ev_n_dn.values()])
    print(f"Occupation spin up: {ev_n_up:0.8f}")
    print(f"Occupation spin dn: {ev_n_dn:0.8f}")

    print("Kinetic energy per bond")
    print(f"spin up electrons: {2 * ev_cdagc_up:0.6f}")
    print(f"spin dn electrons: {2 * ev_cdagc_dn:0.6f}")

    ev_double = env_ctm.measure_1site(n_up @ n_dn)
    ev_double = mean([*ev_double.values()])
    print(f"Average double occupancy: {ev_double:0.6f}")

    Sz = 0.5 * (n_up - n_dn)   # Sz operator
    ev_SzSz = env_ctm.measure_nn(Sz, Sz)
    ev_SzSz = mean([*ev_SzSz.values()])
    print(f"Average NN spin-spin correlator: {ev_SzSz:0.6f}")




if __name__ == '__main__':
    test_quickstart_hubbard(D=12, beta=0.5)
