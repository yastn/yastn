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
""" test dmrg """
import numpy as np
import pytest
import yastn.tn.mps as mps
import yastn
try:
    from .configs import config_dense as cfg
except ImportError:
    from configs import config_dense as cfg
# pytest modifies cfg to inject different backends and devices during tests


def run_dmrg(phi, H, O_occ, E_target, occ_target, opts_svd, tol):
    r"""
    Run mps.dmrg_ to find the ground state and
    a few low-energy states of the Hamiltonian H.
    Verify resulting energies against known reference solutions.
    """
    #
    # DMRG can look for solutions in a space
    # orthogonal to provided MPSs. We start with empty list,
    # project = [], and keep adding to it previously found eigenstates.
    # This allows us to find the ground state
    # and a few consequative lowest-energy eigenstates.
    #
    project, states = [], []
    for ref_eng, ref_occ  in zip(E_target, occ_target):
        #
        # We find a state and check that its energy and total occupation
        # matches the expected reference values.
        #
        # We copy initial random MPS psi, as
        # yastn.dmrg_ modifies provided input state in place.
        #
        psi = phi.shallow_copy()
        #
        # We set up dmrg_ to terminate iterations
        # when energy is converged within some tolerance.
        #
        out = mps.dmrg_(psi, H, project=project, method='2site',
                        energy_tol=tol / 10, max_sweeps=20, opts_svd=opts_svd)
        #
        # Output of _dmrg is a nametuple with information about the run,
        # including the final energy.
        # Occupation number has to be calcuted using measure_mpo.
        #
        eng = out.energy
        occ  = mps.measure_mpo(psi, O_occ, psi)
        #
        # Print the result:
        #
        print(f"2site DMRG; energy: {eng:{1}.{8}} / {ref_eng:{1}.{8}}; "
              + f"occupation: {occ:{1}.{8}} / {ref_occ}")
        assert abs(eng - ref_eng) < tol * 100
        assert abs(occ - ref_occ) < tol
        #
        # We furthe iterate with '1site' DMRG
        # and stricter convergence criterion.
        #
        out = mps.dmrg_(psi, H, project=project, method='1site',
                        Schmidt_tol=tol / 10, max_sweeps=20)

        eng = mps.measure_mpo(psi, H, psi)
        occ = mps.measure_mpo(psi, O_occ, psi)
        print(f"1site DMRG; energy: {eng:{1}.{8}} / {ref_eng:{1}.{8}}; "
              + f"occupation: {occ:{1}.{8}} / {ref_occ}")
        # test that energy outputed by dmrg is correct
        assert abs(eng - ref_eng) < tol
        assert abs(occ - ref_occ) < tol
        assert abs(eng - out.energy) < tol  # test dmrg_ output information
        #
        # Finally, we add the found state psi to the list of states
        # to be projected out in the next step of the loop.
        #
        penalty = 100
        project.append((penalty, psi))
        states.append(psi)
    return states


@pytest.mark.parametrize("kwargs", [{'config': cfg}])
def test_dmrg(kwargs):
    dmrg_XX_model_dense(**kwargs, tol=1e-6)
    dmrg_XX_model_Z2(**kwargs, tol=1e-6)
    dmrg_XX_model_U1(**kwargs, tol=1e-6)


def dmrg_XX_model_dense(config=None, tol=1e-6):
    """
    Initialize random MPS of dense tensors and
    runs a few sweeps of DMRG with the Hamiltonian of XX model.
    """
    # Knowing the exact solution we can compare it DMRG result.
    # In this test we will consider sectors of different occupations.
    #
    # In this example we use yastn.Tensor with no symmetry imposed.
    #
    # Here, the Hamiltonian for N = 7 sites
    # is obtained with the automatic generator.
    #
    N = 7
    #
    opts_config = {} if config is None else \
                {'backend': config.backend,
                 'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.Spin12(sym='dense', **opts_config)
    generate = mps.Generator(N=N, operators=ops)
    parameters = {"t": 1.0, "mu": 0.2,
                  "rN": range(N),
                  "rNN": [(i, i+1) for i in range(N - 1)]}
    H_str = "\sum_{i,j \in rNN} t ( sp_{i} sm_{j} + sp_{j} sm_{i} )"
    H_str += " + \sum_{j \in rN} mu sp_{j} sm_{j}"
    H = generate.mpo_from_latex(H_str, parameters)
    #
    # and MPO to measure occupation:
    #
    O_occ = generate.mpo_from_latex("\sum_{j \in rN} sp_{j} sm_{j}",
                                  parameters)
    #
    # Known energies and occupations of low-energy eigenstates.
    #
    Eng_gs = [-3.427339492125, -3.227339492125, -2.861972627395]
    occ_gs = [3, 4, 2]
    #
    # To standardize this test we fix a seed for random MPS we use.
    #
    generate.random_seed(seed=0)
    #
    # Set options for truncation for '2site' method of mps.dmrg_.
    #
    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}
    #
    # Finally run DMRG starting from random MPS psi
    #
    psi = generate.random_mps(D_total=Dmax)
    #
    # A single run for a ground state can be done using:
    # mps.dmrg_(psi, H, method=method,energy_tol=tol/10,
    #           max_sweeps=20, opts_svd=opts_svd)
    #
    # We create a subfunction run_dmrg to explain how to
    # target some sectors for occupation. This is not necessary
    # but we do it for the sake of clarity and testing.
    #
    psis = run_dmrg(psi, H, O_occ, Eng_gs, occ_gs, opts_svd, tol)
    #
    # _dmrg can be executed as a generator to monitor states
    # between dmrg sweeps. This is done by providing `iterator_step`.
    #
    psi = generate.random_mps(D_total=Dmax)
    for step in mps.dmrg_(psi, H, method='1site',
                          max_sweeps=20, iterator_step=2):
        assert step.sweeps % 2 == 0  # stop every second sweep here
        occ = mps.measure_mpo(psi, O_occ, psi)  # measure occupation
        # here breaks if it is close to the known result.
        if abs(occ.item() - occ_gs[0]) < tol:
            break
    assert step.sweeps < 20
    # Here, dmrg stopped based on convergence, not on max_sweeps.


def dmrg_XX_model_Z2(config=None, tol=1e-6):
    """
    Initialize random MPS of Z2 tensors and tests mps.dmrg_ vs known results.
    """
    opts_config = {} if config is None else \
            {'backend': config.backend,
            'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.SpinlessFermions(sym='Z2', **opts_config)
    generate = mps.Generator(N=7, operators=ops)
    generate.random_seed(seed=0)
    N, Dmax  = 7, 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}

    Eng_occ_target = {
        0: ([-3.227339492125, -2.861972627395, -2.461972627395],
            [4, 2, 4]),
        1: ([-3.427339492125, -2.661972627395, -2.261972627395],
            [3, 3, 5])}
    H_str = "\sum_{i,j \in rNN} t (cp_{i} c_{j} + cp_{j} c_{i})"
    H_str += " + \sum_{j\in rN} mu cp_{j} c_{j}"
    parameters = {"t": 1.0, "mu": 0.2,
                  "rN": range(N),
                  "rNN": [(i, i+1) for i in range(N - 1)]}
    H = generate.mpo_from_latex(H_str, parameters)
    O_occ = generate.mpo_from_latex("\sum_{j\in rN} cp_{j} c_{j}", parameters)

    for parity, (E_target, occ_target) in Eng_occ_target.items():
        psi = generate.random_mps(D_total=Dmax, n=parity)
        run_dmrg(psi, H, O_occ, E_target, occ_target, opts_svd, tol)


def dmrg_XX_model_U1(config=None, tol=1e-6):
    """
    Initialize random MPS of U(1) tensors and tests _dmrg vs known results.
    """
    opts_config = {} if config is None else \
        {'backend': config.backend,
        'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.SpinlessFermions(sym='U1', **opts_config)
    generate = mps.Generator(N=7, operators=ops)
    generate.random_seed(seed=0)

    N = 7
    H_str = "\sum_{i,j \in rNN} t (cp_{i} c_{j} + cp_{j} c_{i})"
    H_str += " + \sum_{j\in rN} mu cp_{j} c_{j}"
    parameters = {"t": 1.0, "mu": 0.2,
                  "rN": range(N),
                  "rNN": [(i, i+1) for i in range(N - 1)]}
    H = generate.mpo_from_latex(H_str, parameters)
    O_occ = generate.mpo_from_latex("\sum_{j\in rN} cp_{j} c_{j}", parameters)

    Eng_sectors = {
        2: [-2.861972627395, -2.213125929752], #  -1.779580427103],
        3: [-3.427339492125, -2.661972627395], #  -2.013125929752],
        4: [-3.227339492125, -2.461972627395], #  -1.813125929752]
        }

    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}

    for occ_sector, E_target in Eng_sectors.items():
        psi = generate.random_mps(D_total=Dmax, n=occ_sector)
        occ_target = [occ_sector] * len(E_target)
        run_dmrg(psi, H, O_occ, E_target, occ_target, opts_svd, tol)


def test_dmrg_XX_model_U1_sum_of_Mpos(config=cfg, tol=1e-6):
    """
    Initialize random MPS of U(1) tensors and tests _dmrg vs known results.
    """
    opts_config = {} if config is None else \
        {'backend': config.backend,
        'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.SpinlessFermions(sym='U1', **opts_config)
    generate = mps.Generator(N=7, operators=ops)
    generate.random_seed(seed=0)

    N = 7
    H_str_nn = "\sum_{i,j \in rNN} (cp_{i} c_{j} + cp_{j} c_{i})"
    H_str_n  = "\sum_{j \in rN} cp_{j} c_{j}"
    parameters = {"rN": list(range(N)),
                  "rNN": [(i, i+1) for i in range(N - 1)]}
    H_nn = generate.mpo_from_latex(H_str_nn, parameters)
    H_n = generate.mpo_from_latex(H_str_n, parameters)
    O_occ = generate.mpo_from_latex("\sum_{j \in rN} cp_{j} c_{j}", parameters)

    Eng_sectors = {
        2: [-2.861972627395, -2.213125929752], #  -1.779580427103],
        3: [-3.427339492125, -2.661972627395], #  -2.013125929752],
        4: [-3.227339492125, -2.461972627395], #  -1.813125929752]
        }

    Dmax = 8
    opts_svd = {'tol': 1e-8, 'D_total': Dmax}

    t, mu = 1.0, 0.2
    H = [t * H_nn, mu * H_n]

    for occ_sector, E_target in Eng_sectors.items():
        psi = generate.random_mps(D_total=Dmax, n=occ_sector)
        occ_target = [occ_sector] * len(E_target)
        run_dmrg(psi, H, O_occ, E_target, occ_target, opts_svd, tol)


def test_dmrg_Ising_PBC_Z2(config=cfg, tol=1e-4):
    """
    Initialize random MPS of U(1) tensors and tests _dmrg vs known results.
    """
    opts_config = {} if config is None else \
        {'backend': config.backend,
        'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.Spin12(sym='Z2', **opts_config)
    N = 8
    I = mps.product_mpo(ops.I(), N)

    terms = []
    for n in range(N):
        terms.append(mps.Hterm(-1, [n], [ops.z()]))
        terms.append(mps.Hterm(-1, [n, (n + 1) % N], [ops.x(), ops.x()]))

    # OBC MPO
    H0 = mps.generate_mpo(I, terms)
    P0 = mps.product_mpo(ops.z(), N)

    H1 = mps.Mpo(N, periodic=True)
    P1 = mps.Mpo(N, periodic=True)
    dn = N // 2
    for n in range(N):
        H1[(n + dn) % N] = H0[n].copy()
        P1[(n + dn) % N] = P0[n].copy()

    Eng_sectors = {
        0: [-10.2516617910, -8.6909392148], #  -7.2490195708, -7.2490195708, -7.2490195708, -7.2490195708, -6.1454220537],
        1: [-10.0546789842, -8.5239452547], #  -8.5239452547, -7.2262518595, -7.2262518595, -6.9932115253, -6.3591608542],
        }

    Dmax = 16
    opts_svd = {'tol': 1e-10, 'D_total': Dmax}

    for H, P in [(H0, P0), (H1, P1)]:
        for parity, E_target in Eng_sectors.items():
            psi = mps.random_mps(I, D_total=Dmax, n=(parity,)).canonize_(to='first')
            parity_target = [(-1) ** parity] * len(E_target)
            psis = run_dmrg(psi, H, P, E_target, parity_target, opts_svd, tol / 100)
            for EE, psi in zip(E_target, psis):
                psi1 = mps.zipper(H, psi, opts_svd=opts_svd, normalize=False)
                mps.compression_(psi1, [H, psi], method='2site', max_sweeps=4, opts_svd=opts_svd, normalize=False)
                mps.compression_(psi1, [H, psi], method='1site', max_sweeps=10, normalize=False)
                assert (EE * psi - psi1).norm() < tol
                #
                # evolve in real time  # test Heff0 in PBC
                tf = 0.1
                next(mps.tdvp_(psi1, H, method='1site', times=(0, tf), dt=0.05, normalize=False))
                print((EE * psi * (np.exp(-1j * EE * tf)) - psi1).norm())
                assert (EE * psi * (np.exp(-1j * EE * tf)) - psi1).norm() < tol


def test_dmrg_raise(config=cfg):
    opts_config = {} if config is None else \
        {'backend': config.backend,
        'default_device': config.default_device}
    ops = yastn.operators.Spin12(sym='dense', **opts_config)
    I = mps.product_mpo(ops.I(), N=7)
    H = mps.random_mpo(I, D_total=5)
    psi = mps.random_mpo(I, D_total=4)

    with pytest.raises(yastn.YastnError):
        mps.dmrg_(psi, H, method='1site', Schmidt_tol=-1)
        #  DMRG: Schmidt_tol has to be positive or None.
    with pytest.raises(yastn.YastnError):
        mps.dmrg_(psi, H, method='1site', energy_tol=-1)
        # DMRG: energy_tol has to be positive or None.
    with pytest.raises(yastn.YastnError):
        mps.dmrg_(psi, H, method='one-site')
        # DMRG: dmrg method one-site not recognized.
    with pytest.raises(yastn.YastnError):
        mps.dmrg_(psi, H, method='2site')
        # DMRG: provide opts_svd for 2site method.


if __name__ == "__main__":
    test_dmrg_raise()
    test_dmrg({'config': cfg})
    test_dmrg_XX_model_U1_sum_of_Mpos()
    test_dmrg_Ising_PBC_Z2()
