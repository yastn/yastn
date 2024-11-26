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
""" test tdvp """
import numpy as np
import pytest
import yastn
import yastn.tn.mps as mps

try:
    from test_generate_mpo import mpo_hopping_Hterm
except ModuleNotFoundError:
    from .test_generate_mpo import mpo_hopping_Hterm


@pytest.mark.parametrize('sym', ['U1', 'Z2'])
def test_tdvp_sudden_quench(config_kwargs, sym, tol=1e-10):
    """
    Simulate a sudden quench of a free-fermionic (hopping) model.
    Compare observables versus known reference results.
    """
    N, n = 6, 3  # Consider a system of 6 modes and 3 particles.
    #
    # Load operators
    #
    ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)
    ops.random_seed(seed=0)
    #
    # Hopping matrix, functions consuming it use only upper-triangular part.
    #
    J0 = [[1,   0.5j, 0,    0.3,  0.1,  0   ],
          [0,  -1,    0.5j, 0,    0.3,  0.1 ],
          [0,   0,    1,    0.5j, 0,    0.3 ],
          [0,   0,    0,   -1,    0.5j, 0   ],
          [0,   0,    0,    0,    1,    0.5j],
          [0,   0,    0,    0,    0,   -1   ]]
    #
    # Generate corresponding MPO using the function from previous examples
    #
    H0 = mpo_hopping_Hterm(config_kwargs, sym, J0)
    #
    # Find the ground state using DMRG
    # Bond dimension Dmax = 8 for N = 6 is large enough
    # to avoid truncation errors in the test
    #
    Dmax = 8
    opts_svd = {'tol': 1e-15, 'D_total': Dmax}
    #
    # Run DMRG for the ground state
    # global ground state is for n=3, i.e., works for Z2 and U1
    #
    I = mps.product_mpo(ops.I(), N)
    n_psi = n % 2 if sym=='Z2' else n  # for U1; charge of MPS
    psi = mps.random_mps(I, D_total=Dmax, n=n_psi)
    #
    out = mps.dmrg_(psi, H0, method='2site', max_sweeps=2, opts_svd=opts_svd)
    out = mps.dmrg_(psi, H0, method='1site', max_sweeps=10,
                    energy_tol=1e-14, Schmidt_tol=1e-14)
    #
    # Get reference results for the ground state and check mps
    #
    C0ref, E0ref = gs_correlation_matrix_exact(J0, n)  # defined below
    assert abs(out.energy - E0ref) < tol
    #
    # Calculate correlation matrix for MPS and test vs exact reference
    #
    C0psi = correlation_matrix_from_mps(psi, ops, tol)  # defined below
    assert np.allclose(C0ref, C0psi, rtol=tol)
    #
    # Sudden quench with a new Hamiltonian
    #
    J1 = [[-1,   0.5,   0,  -0.3, 0.1, 0  ],
          [ 0,   1  ,   0.5, 0,  -0.3, 0.1],
          [ 0,   0  ,  -1,   0.5, 0,  -0.3],
          [ 0,   0  ,   0,   1,   0.5, 0  ],
          [ 0,   0  ,   0,   0,  -1,   0.5],
          [ 0,   0  ,   0,   0,   0,   1  ]]
    H1 = mpo_hopping_Hterm(config_kwargs, sym, J1)
    #
    # Run time evolution and calculate correlation matrix at two snapshots
    #
    times = (0, 0.25, 0.6)
    #
    # Parameters for expmv in tdvp_,
    # 'ncv' is an initial guess for the size of Krylov space.
    # It gets updated at each site/bond during evolution.
    #
    opts_expmv = {'hermitian': True, 'ncv': 5, 'tol': 1e-12}
    #
    for method in ('1site', '2site', '12site'):  # test various methods
        # shallow_copy is sufficient to retain the initial state
        phi = psi.shallow_copy()
        for step in mps.tdvp_(phi, H1, times=times, method=method, dt=0.125,
                              opts_svd=opts_svd, opts_expmv=opts_expmv):
            #
            # Calculate correlation matrix from mps.
            # Compare with exact reference results defined below.
            #
            Cphi = correlation_matrix_from_mps(phi, ops, tol)
            Cref = evolve_correlation_matrix_exact(C0ref, J1, step.tf)
            assert np.allclose(Cref, Cphi, rtol=tol)


def correlation_matrix_from_mps(psi, ops, tol):
        """
        Calculate correlation matrix for MPS psi.
        """
        assert abs(psi.norm() - 1) < tol  # check normalization
        cpc = mps.measure_2site(psi, ops.cp(), ops.c(), psi, bonds='<=>') # all
        C = np.zeros((psi.N, psi.N), dtype=np.complex128)
        for (n1, n2), v in cpc.items():
            C[n2, n1] = v
        return C


def gs_correlation_matrix_exact(J, n):
    """
    Correlation matrix for the ground state
    of n particles with hopping Hamiltonian matrix J.
    C[m, n] = <c_n^dag c_m>
    """
    J = np.triu(J, 0) + np.triu(J, 1).T.conj()
    D, V = np.linalg.eigh(J)
    Egs = np.sum(D[:n])
    C0 = np.zeros(len(D))
    C0[:n] = 1
    C = V @ np.diag(C0) @ V.T.conj()
    return C, Egs


def evolve_correlation_matrix_exact(Ci, J, t):
    """
    Evolve correlation matrix C by time t with hopping Hamiltonian J.
    Diagonal elements of J array is on-site potential and
    upper triangular terms of J are hopping amplitudes.
    """
    J = np.triu(J, 0) + np.triu(J, 1).T.conj()
    # U = expm(1j * t * J)
    D, V = np.linalg.eigh(J)
    U = V @ np.diag(np.exp(1j * t * D)) @ V.T.conj()
    Cf = U.conj().T @ Ci @ U
    return Cf


@pytest.mark.parametrize('sym', ['U1'])
def test_tdvp_sudden_quench_mpo_sum(config_kwargs, sym, tol=1e-10):
    """
    Simulate a sudden quench of a free-fermionic (hopping) model.
    Compare observables versus known reference results.
    """
    N, n = 6, 3
    ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)
    ops.random_seed(seed=0)

    J0 = [[1,   0.5j, 0,    0.3,  0.1,  0   ],
          [0,  -1,    0.5j, 0,    0.3,  0.1 ],
          [0,   0,    1,    0.5j, 0,    0.3 ],
          [0,   0,    0,   -1,    0.5j, 0   ],
          [0,   0,    0,    0,    1,    0.5j],
          [0,   0,    0,    0,    0,   -1   ]]
    H0 = mpo_hopping_Hterm(config_kwargs, sym, J0)

    Dmax = 8
    opts_svd = {'tol': 1e-15, 'D_total': Dmax}
    I = mps.product_mpo(ops.I(), N)
    n_psi = n % 2 if sym=='Z2' else n # for U1; charge of MPS
    psi = mps.random_mps(I, D_total=Dmax, n=n_psi)
    out = mps.dmrg_(psi, H0, method='2site', max_sweeps=2, opts_svd=opts_svd)
    out = mps.dmrg_(psi, H0, method='1site', max_sweeps=10,
                    energy_tol=1e-14, Schmidt_tol=1e-14)
    #
    C0ref, E0ref = gs_correlation_matrix_exact(J0, n)
    assert abs(out.energy - E0ref) < tol
    #
    # verify MPS vs reference
    #
    C0psi = correlation_matrix_from_mps(psi, ops, tol)
    assert np.allclose(C0ref, C0psi, rtol=tol)
    #
    # Sudden quench with a new Hamiltonian, here as sum of Mpos
    #
    J1 = np.asarray(
        [[-1,   0.5,   0,  -0.3, 0.1, 0  ],
         [ 0,   1  ,   0.5, 0,  -0.3, 0.1],
         [ 0,   0  ,  -1,   0.5, 0,  -0.3],
         [ 0,   0  ,   0,   1,   0.5, 0  ],
         [ 0,   0  ,   0,   0,  -1,   0.5],
         [ 0,   0  ,   0,   0,   0,   1  ]])
    J1s = [np.zeros_like(J1) for _ in range(J1.shape[0])]
    for i in range(len(J1s)):
        J1s[i][:, i]= J1[:, i]
    H1 = [mpo_hopping_Hterm(config_kwargs, sym, col) for col in J1s]
    #
    # Run time evolution and calculate correlation matrix at two snapshots
    #
    times = (0, 0.25, 0.6)
    #
    # Parameters for expmv in tdvp_,
    # 'ncv' is an initial guess for the size of Krylov space.
    # It gets updated at each site/bond during evolution.
    #
    opts_expmv = {'hermitian': True, 'ncv': 5, 'tol': 1e-12}
    #
    for method in ('12site', ):   # '1site', '2site', test various methods
        # shallow_copy is sufficient to retain the initial state
        phi = psi.shallow_copy()
        for step in mps.tdvp_(phi, H1, times=times, method=method, dt=0.125,
                              opts_svd=opts_svd, opts_expmv=opts_expmv):
            #
            # Calculate correlation matrix from mps
            # and exact reference
            #
            Cphi = correlation_matrix_from_mps(phi, ops, tol)
            Cref = evolve_correlation_matrix_exact(C0ref, J1, step.tf)
            #
            # Compare results with references
            #
            assert np.allclose(Cref, Cphi, rtol=tol)

@pytest.mark.parametrize('sym', ['Z2'])
def test_tdvp_sudden_quench_Heisenberg(config_kwargs, sym, tol=1e-5):
    """
    Simulate a sudden quench of a free-fermionic (hopping) model.
    Compare observables versus known reference results.

    Here we employ the Heisenberg picture to evolve an operator of interest.
    Test constructor [-H, H.on_bra()]
    """
    N, n = 6, 3  # Consider a system of 6 modes and 3 particles.
    #
    # Load operators
    #
    ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)
    ops.random_seed(seed=0)
    #
    # Hopping matrix, it is hermitized inside functions consuming it.
    #
    J0 = [[1,   0.5j, 0,    0.3,  0.1,  0   ],
          [0,  -1,    0.5j, 0,    0.3,  0.1 ],
          [0,   0,    1,    0.5j, 0,    0.3 ],
          [0,   0,    0,   -1,    0.5j, 0   ],
          [0,   0,    0,    0,    1,    0.5j],
          [0,   0,    0,    0,    0,   -1   ]]
    H0 = mpo_hopping_Hterm(config_kwargs, sym, J0)
    # assert (H0 - H0.H).norm() < tol * H0.norm()  # flip signature of virtual legs?
    #
    Dmax = 8
    opts_svd = {'tol': 1e-15, 'D_total': Dmax}
    I = mps.product_mpo(ops.I(), N)
    n_psi = n % 2 if sym=='Z2' else n # for U1; charge of MPS
    psi = mps.random_mps(I, D_total=Dmax, n=n_psi)
    #
    out = mps.dmrg_(psi, H0, method='2site', max_sweeps=2, opts_svd=opts_svd)
    out = mps.dmrg_(psi, H0, method='1site', max_sweeps=10,
                    energy_tol=1e-14, Schmidt_tol=1e-14)
    #
    C0ref, E0ref = gs_correlation_matrix_exact(J0, n)
    assert abs(out.energy - E0ref) < tol
    #
    # Sudden quench with a new Hamiltonian
    #
    J1 = [[-1,   0.5,   0,  -0.3, 0.1, 0  ],
          [ 0,   1  ,   0.5, 0,  -0.3, 0.1],
          [ 0,   0  ,  -1,   0.5, 0,  -0.3],
          [ 0,   0  ,   0,   1,   0.5, 0  ],
          [ 0,   0  ,   0,   0,  -1,   0.5],
          [ 0,   0  ,   0,   0,   0,   1  ]]
    H1 = mpo_hopping_Hterm(config_kwargs, sym, J1)
    #
    # C[m, n] = <c_n^dag c_m>
    pps = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]  # (m, n)
    terms = [mps.Hterm(1, pp[::-1], (ops.cp(), ops.c())) for pp in pps]
    O = mps.generate_mpo(I, terms)
    #
    e0ref = sum(C0ref[pp] for pp in pps)
    e0 = mps.vdot(psi, O, psi)
    assert abs(e0 - e0ref) < tol * abs(e0ref)
    #
    OO = O.shallow_copy()
    opts_expmv = {'hermitian': False, 'ncv': 5, 'tol': 1e-12}
    opts_svd = {'tol': 1e-15, 'D_total': 64}

    HH = [-H1, H1.on_bra()]
    for step in mps.tdvp_(OO, HH, times=(0, 0.00001, 0.00002, 0.00005, 0.0001, 0.002, 0.005, 0.01, 0.05),
                          method='2site', dt=0.01, normalize=False,
                          opts_svd=opts_svd, opts_expmv=opts_expmv):
        Cref = evolve_correlation_matrix_exact(C0ref, J1, step.tf)
        e1ref = sum(Cref[pp] for pp in pps)
        e1 = mps.vdot(psi, OO, psi)
        assert abs(e1 - e1ref) < tol * abs(e1ref)

    for step in mps.tdvp_(OO, HH, times=(0.05, 0.1, 0.2),
                          method='12site', dt=0.01, normalize=False,
                          opts_svd=opts_svd, opts_expmv=opts_expmv):
        Cref = evolve_correlation_matrix_exact(C0ref, J1, step.tf)
        e1ref = sum(Cref[pp] for pp in pps)
        e1 = mps.vdot(psi, OO, psi)
        assert abs(e1 - e1ref) < tol * abs(e1ref)


@pytest.mark.parametrize('sym', ['Z2', 'dense'])
def test_tdvp_KZ_quench(config_kwargs, sym):
    """
    Simulate a slow quench across a quantum critical point in
    a small transverse field Ising chain with periodic boundary conditions.
    Compare with exact reference results.
    """
    #
    N = 10  # Consider a system of 10 sites
    #
    # Load spin-1/2 operators
    #
    ops = yastn.operators.Spin12(sym=sym, **config_kwargs)
    ops.random_seed(seed=0)
    #
    # Hterm-s to generate H = -sum_i X_i X_{i+1} - g * Z_i
    #
    I = mps.product_mpo(ops.I(), N)  # identity MPO
    termsXX = [mps.Hterm(-1, [i, (i + 1) % N], [ops.x(), ops.x()]) for i in range(N)]
    HXX = mps.generate_mpo(I, termsXX)
    termsZ = [mps.Hterm(-1, i, ops.z()) for i in range(N)]
    HZ = mps.generate_mpo(I, termsZ)
    #
    # Kibble-Zurek quench across a critical point at gc = 1
    # tauQ is the quench time
    #
    tauQ, gc = 1, 1
    ti, tf = -tauQ, tauQ  # evolve from gi = 2 to gf = 0
    H = lambda t: [HXX, (gc - t / tauQ) * HZ]  # linear quench
    #
    # Analytical reference expectation values measured
    # at g = 1 and g = 0 (for tauQ=1, gi=2 and N=10)
    #
    XXex = {1: 0.470182292934, 0: 0.738769410121}
    Zex  = {1: 0.776255260472, 0: 0.163491011822}
    Egs = -2.127120881869  # the ground state energy at gi = 2
    #
    # Start with the ground state at gi = 2
    # Run DMRG to get the initial ground state
    #
    Dmax = 10
    psi = mps.random_mps(I, D_total=Dmax)
    out = mps.dmrg_(psi, H(ti), method='2site', max_sweeps=2,
                    opts_svd={'tol': 1e-6, 'D_total': Dmax})
    out = mps.dmrg_(psi, H(ti), method='1site', max_sweeps=10,
                    energy_tol=1e-12, Schmidt_tol=1e-12)
    #
    # Test ground state energy versus reference
    #
    assert abs(out.energy / N - Egs) < 1e-7
    assert psi.get_bond_dimensions() == (1, 2, 4, 8, 10, 10, 10, 8, 4, 2, 1)
    #
    # Slow quench to gf = 0
    # Sets up tdvp_ parameters; allows the growth of the bond dimension.
    #
    Dmax = 16
    opts_expmv = {'hermitian': True, 'tol': 1e-12}
    opts_svd = {'tol': 1e-6, 'D_total': Dmax}
    #
    for step in mps.tdvp_(psi, H, times=(ti, 0, tf),
                          method='12site',dt=0.04, order='2nd',
                          opts_svd=opts_svd, opts_expmv=opts_expmv):
        #
        # tdvp_() always gives an iterator
        # Calculate expectation values at snapshots
        #
        EZ = mps.measure_1site(psi, ops.z(), psi)
        EXX = mps.measure_2site(psi, ops.x(), ops.x(), psi, bonds='r1p')  # periodic nn
        #
        # Compare them with the exact result
        #
        gg = round(gc - step.tf / tauQ)  # g at the snapshot
        assert all(abs(EXX[k, (k + 1) % N] - XXex[gg]) < 1e-4 for k in range(N))
        assert all(abs(EZ[k] - Zex[gg]) < 1e-4 for k in range(N))
        #
        # Bond dimension was updated
        #
        bd_ref = (1, 2, 4, 8, 16, 16, 16, 8, 4, 2, 1)
        assert psi.get_bond_dimensions() == bd_ref


def test_tdvp_raise(config_kwargs):
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    ops.random_seed(seed=0)
    I = mps.product_mpo(ops.I(), N=3)
    H = mps.random_mpo(I, D_total=3)
    psi = mps.random_mpo(I, D_total=2)

    with pytest.raises(yastn.YastnError):
        next(mps.tdvp_(psi, H, method='one-site'))
        # TDVP: tdvp method one-site not recognized
    with pytest.raises(yastn.YastnError):
        next(mps.tdvp_(psi, H, method='2site'))
        # TDVP: provide opts_svd for 2site method.
    with pytest.raises(yastn.YastnError):
        next(mps.tdvp_(psi, H, dt=0.))
        # TDVP: dt should be positive.
    with pytest.raises(yastn.YastnError):
        next(mps.tdvp_(psi, H, times=(1, 0)))
        # TDVP: times should be an ascending tuple.
    with pytest.raises(yastn.YastnError):
        next(mps.tdvp_(psi, H, order='1st'))
        # TDVP: order should be in ('2nd', '4th')

    out = next(mps.tdvp_(psi, H, dt=0.1, times=0.1))
    # times=0.1 => times=(0, 0.1)
    assert out.ti == 0 and out.tf == 0.1 and out.steps == 1


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
