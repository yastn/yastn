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
import pytest
#
# The script below can be benchmarked against reference values obtained from METTS
# on a 4 x 16 cylinder implemented with ITensor library.
# Values are accompanied with statistical errors from
# Monte Carlo sampling in METTS
#
#            (observable, beta): (value, statistical err.)
metts_values = {("energy", 2.0): (-2.8040, 0.0026),
                ("energy", 4.0): (-2.8787, 0.0014),
                ("energy", 6.0): (-2.9161, 0.0009),
                ("double_occ", 2.0): (0.0282, 0.0001),
                ("double_occ", 4.0): (0.0338, 0.0001),
                ("double_occ", 6.0): (0.0365, 0.0001),
                ("SzSz_NN", 2.0): (-0.0417, 0.0018),
                ("SzSz_NN", 4.0): (-0.0750, 0.0013),
                ("SzSz_NN", 6.0): (-0.0910, 0.0013)}
#
# PEPS results
#
# beta=2.0
#             D=12      D=16      D=20
# energy     -2.7803   -2.8002   -2.8002
# double_occ  0.0302    0.0295    0.0294
# SzSz_NN    -0.0382   -0.0415   -0.0412
#
# beta=4.0
# energy     -2.8677   -2.8715   -2.8707
# double_occ  0.0337    0.0339    0.0338
# SzSz_NN    -0.0703   -0.0712   -0.0703
#
# beta=6.0
# energy     -2.8979   -2.8989
# double_occ  0.0354    0.0354
# SzSz_NN    -0.0848   -0.0842
#
#
@pytest.mark.skipif("not config.getoption('quickstarts')")
def test_quickstart_hubbard(D=12, betas=[0.5]):

    import yastn
    import yastn.tn.fpeps as fpeps
    from tqdm import tqdm  # progressbar

    t = 1
    mu = 0
    U = 10

    ops = yastn.operators.SpinfulFermions(sym='U1xU1')
    I = ops.I()
    c_up, cdag_up, n_up = ops.c('u'), ops.cp('u'), ops.n('u')
    c_dn, cdag_dn, n_dn = ops.c('d'), ops.cp('d'), ops.n('d')

    geometry = fpeps.CheckerboardLattice()
    psi = fpeps.product_peps(geometry=geometry, vectors=I)

    beta0, infoss = 0, []
    for beta in betas:
        db = 0.01  # Trotter step size
        # making sure we have integer number of steps to target beta / 2
        steps = round(((beta-beta0) / 2) / db)
        db = ((beta-beta0) / 2) / steps
        beta0 = beta
        #
        g_hop_u = fpeps.gates.gate_nn_hopping(t, db / 2, I, c_up, cdag_up)
        g_hop_d = fpeps.gates.gate_nn_hopping(t, db / 2, I, c_dn, cdag_dn)
        g_loc = fpeps.gates.gate_local_Coulomb(mu, mu, U, db/2, I, n_up, n_dn)
        gates = fpeps.gates.distribute(geometry,
                                    gates_nn=[g_hop_u, g_hop_d],
                                    gates_local=g_loc)
        #
        env = fpeps.EnvNTU(psi, which='NN')
        opts_svd = {'D_total': D, 'tol': 1e-12}
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

        ctm = env_ctm.ctmrg_(opts_svd=opts_svd_ctm,
                             iterator_step=1,
                             max_sweeps=50)  # generator

        energy_old, tol_exp = 0, 1e-7
        for info in ctm:
            # single CMTRG sweep as iterator_step=1 in the ctm generator
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
            print(f"Energy per site after iteration {info.sweeps}: {energy:0.8f}")
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

        double_occ = env_ctm.measure_1site(n_up @ n_dn)
        double_occ = mean([*double_occ.values()])
        print(f"Average double occupancy: {double_occ:0.6f}")

        Sz = 0.5 * (n_up - n_dn)   # Sz operator
        SzSz_NN = env_ctm.measure_nn(Sz, Sz)
        SzSz_NN = mean([*SzSz_NN.values()])
        print(f"Average NN spin-spin correlator: {SzSz_NN:0.6f}")


if __name__ == '__main__':
    test_quickstart_hubbard(D=12, betas=[0.5])
