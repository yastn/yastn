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
""" Test quickstart example Kibble-Zurek. """
import pytest

@pytest.mark.skipif("not config.getoption('quickstarts')")
def test_quickstart_KZ(config_kwargs):
    """ test quickstart Kibble-Zurek example"""

    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm  # progressbar
    import yastn
    import yastn.tn.mps as mps
    import yastn.tn.fpeps as peps
    from yastn.tn.fpeps.gates import gate_nn_Ising, gate_local_field
    #
    # Employ PEPS lattice geometry for sites and nearest-neighbor bonds
    Lx, Ly = 4, 4  # lattice size
    geometry = peps.SquareLattice(dims=(Lx, Ly), boundary='obc')
    sites = geometry.sites()
    #
    # Draw random couplings from uniform distribution in [-1, 1].
    np.random.seed(seed=0)
    Jij = {k: 2 * np.random.rand() - 1 for k in geometry.bonds()}
    #
    # Define quench protocol
    fXX = lambda s : math.sin((s - 0.5) * math.pi) + 1
    fZ  = lambda s : 1 - math.sin((s - 0.5) * math.pi)
    ta = 2.0  # annealing time
    dt = 0.04  # time step; make it smaller to decrease errors
    steps = round(ta / dt)
    dt = ta / steps
    #
    # Load operators. Problem has Z2 symmetry, which we impose.
    ops = yastn.operators.Spin12(sym='Z2', **config_kwargs)
    #
    print("PEPS simulations")
    #
    def gates_Ising(Jij, fXX, fZ, s, dt, sites, ops):
        """ Trotter gates at time s. """
        nn, local = [], []
        # time-step is 1j * dt / 2, as trotterized evolution
        # is completed by its adjoint for 2nd order method.
        dt2 = 1j * dt / 2
        for bond, J in Jij.items():
            gt = gate_nn_Ising(J * fXX(s), dt2, ops.I(), ops.x(), bond)
            nn.append(gt)
        for site in sites:
            gt = gate_local_field(fZ(s), dt2, ops.I(), ops.z(), site)
            local.append(gt)
        return peps.Gates(nn=nn, local=local)
    #
    # Initialize system in the product ground state at s=0.
    psi = peps.product_peps(geometry=geometry, vectors=ops.vec_z(val=1))
    #
    # simulation parameters
    D = 6  # PEPS bond dimension
    opts_svd_ntu = {"D_total": D}
    env = peps.EnvNTU(psi, which='NN+')
    #
    # execute time evolution
    infoss, t = [], 0
    for _ in tqdm(range(steps)):
        t += dt / 2
        gates = gates_Ising(Jij, fXX, fZ, t / ta, dt, sites, ops)
        infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_ntu)
        # The state psi is contained in env;
        # evolution_step_ updates psi in place.
        infoss.append(infos)
        t += dt / 2

    Delta = peps.accumulated_truncation_error(infoss, statistics='mean')
    print(f"Accumulated mean truncation error {Delta:0.5f}")
    #
    print("Calculating correlations")
    #
    opts_svd_env = {'D_total': 4 * D}
    opts_var_env = {"max_sweeps": 8,
                    "overlap_tol": 1e-5,
                    "Schmidt_tol": 1e-5}
    #
    # setting-up environment
    env_mps = peps.EnvBoundaryMps(psi,
                                  opts_svd=opts_svd_env,
                                  opts_var=opts_var_env, setup='lr')
    #
    # Calculating 1-site <Z_i> for all sites
    Ez_peps = env_mps.measure_1site(ops.z())
    #
    # Calculating 2-site <X_i X_j> for all pairs i <= j
    Exx_peps = env_mps.measure_2site(ops.x(), ops.x(),
                                     opts_svd=opts_svd_env,
                                     opts_var=opts_var_env)
    #
    print("MPS simulations")
    #
    # map between sites and linear MPS ordering.
    s2i = {s: i for i, s in enumerate(sites)}
    b2i = lambda s1, s2: tuple(sorted([s2i[s1], s2i[s2]]))
    #
    # define Hamiltonian MPO
    I = mps.product_mpo(ops.I(), Lx * Ly)  # identity MPO
    #
    termsXX = []
    for (s1, s2), J in Jij.items():
        termXX = mps.Hterm(J, [s2i[s1], s2i[s2]], [ops.x(), ops.x()])
        termsXX.append(termXX)
    HXX = mps.generate_mpo(I, termsXX)
    #
    termsZ = [mps.Hterm(-1, [i], [ops.z()]) for i in range(Lx * Ly)]
    HZ = mps.generate_mpo(I, termsZ)
    #
    # MPO contributions in H(t) will be added up.
    H = lambda t: [HXX * fXX(t / ta), HZ * fZ(t / ta)]
    #
    # Initial state. TDVP is unstable starting in a product state
    # There may be many strategies to mitigate it.
    # Here, it is sufficient to start with a product state obtained
    # via DMRG with artificially enlarged bond dimension.
    psi = mps.random_mps(I, D_total=16)  # initialize with D=16
    mps.dmrg_(psi, H(0), method='1site', max_sweeps=8, Schmidt_tol=1e-8)
    #
    # time-evolution parameters
    opts_expmv = {'hermitian': True, 'tol': 1e-12}
    opts_svd = {'tol': 1e-6, 'D_total': 64}  # max MPS bond dimension
    evol = mps.tdvp_(psi, H, times=(0, ta),
                     method='12site', dt=dt, order='2nd',
                     opts_svd=opts_svd, opts_expmv=opts_expmv,
                     progressbar=True)
    #
    # run evolution
    # evol is a generator with one (final) snapshot to reach
    next(evol)
    #
    # calculate expectation values
    Ez_mps = mps.measure_1site(psi, ops.z(), psi)
    Exx_mps = mps.measure_2site(psi, ops.x(), ops.x(), psi, bonds="<=")

    Z1 = np.array([Ez_peps[st].real for st in sites])
    Z2 = np.array([Ez_mps[s2i[st]].real for st in sites])
    error_Z = np.linalg.norm(Z1 - Z2) / np.linalg.norm(Z1)
    print(f"Relative difference of PEPS vs MPS in Z magnetization: {error_Z:0.5f}")

    dist = lambda s1, s2: np.linalg.norm([s1[0]-s2[0], s1[1]-s2[1]])
    rs = np.array([dist(s1, s2) for (s1, s2) in Exx_peps])
    XX1 = np.array([*Exx_peps.values()]).real
    XX2 = np.array([Exx_mps[b2i(*bond)] for bond in Exx_peps.keys()]).real
    error_XX = np.linalg.norm(XX1 - XX2) / np.linalg.norm(XX1)
    print(f"Relative difference of PEPS vs MPS in XX correlations: {error_XX:0.5f}")
    #
    assert error_XX < 5e-3
    assert error_Z < 5e-3
    #
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    ax[0].scatter(rs, XX1, label='PEPS',
                  marker='+', color='r')
    ax[0].scatter(rs, XX2, label='MPS',
                  marker='o', color='b', facecolors='none')
    ax[0].set_ylim([-1.05, 1.05])
    ax[0].set_xlabel(r"distance $||i - j||$")
    ax[0].set_ylabel(r"two-point correlations $\langle X_i X_j \rangle$")
    ax[0].legend()
    ax[1].scatter(np.arange(len(Z1)), Z1, label='PEPS',
                    marker='+', color='r')
    ax[1].scatter(np.arange(len(Z1)), Z2, label='MPS',
                    marker='o', color='b', facecolors='none')
    ax[1].set_xlabel(r"linear site index i")
    ax[1].set_ylabel(r"transverse magnetization $\langle Z_i \rangle$")
    ax[1].set_ylim([-1.05, 1.05])
    fig.suptitle(f"{Lx}x{Ly} lattice; annealing_time = {ta:0.1f}")
    # fig.show()
    # fig.savefig(f"corr_{Lx}x{Ly}_{ta=}.png")


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--quickstarts"])
