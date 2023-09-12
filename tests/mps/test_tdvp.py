""" dmrg tested on XX model. """
import itertools
import logging
import pytest
import numpy as np
import yastn.tn.mps as mps
import yastn
try:
    from .configs import config_dense as cfg
    # pytest modifies cfg to inject different backends and devices during tests
    from .test_generator import mpo_hopping_Hterm
    from .exact_references.free_fermions import gs_correlation_matrix, evolve_correlation_matrix
except ImportError:
    from configs import config_dense as cfg
    from test_generator import mpo_hopping_Hterm
    from exact_references.free_fermions import gs_correlation_matrix, evolve_correlation_matrix

tol = 1e-10

def correlation_matrix(psi, gen):
    """ Calculate correlation matrix for Mps psi  C[m,n] = <c_n^dag c_m>"""
    assert pytest.approx(psi.norm().item(), rel=tol) == 1

    # first approach: directly act with c operators on state psi
    cns = [mps.generate_mpo(gen.I(), [mps.Hterm(1, [n], [gen._ops.c()])]) for n in range(gen.N)]
    ps = [cn @ psi for cn in cns]
    C = np.zeros((gen.N, gen.N), dtype=np.complex128)
    for m in range(gen.N):
        for n in range(gen.N):
            C[m, n] = mps.vdot(ps[n], ps[m])

    # second approach: use measure_1site() and measure_2site()
    occs = mps.measure_1site(psi, gen._ops.n(), psi)
    cpc = mps.measure_2site(psi, gen._ops.cp(), gen._ops.c(), psi)
    C2 = np.zeros((gen.N, gen.N), dtype=np.complex128)
    for n, v in occs.items():
        C2[n, n] = v
    for (n1, n2), v in cpc.items():
        C2[n2, n1] = v
        C2[n1, n2] = v.conj()
    assert np.allclose(C, C2)
    return C


def test_tdvp_hermitian():
    """
    Simulate a sudden quench of a free-fermionic (hopping) model.
    Compare observables versus known references.
    """
    N, n = 6, 3  # consider a system of 6 modes and 3 particles
    #
    # load operators
    operators = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)
    generate = mps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)
    #
    # hopping matrix, it is hermitized inside functions consuming it.
    J0 = [[1, 0.5j, 0, 0.3, 0.1, 0], [0, -1, 0.5j, 0, 0.3, 0.1], [0, 0, 1, 0.5j, 0, 0.3], [0, 0, 0, -1, 0.5j, 0], [0, 0, 0, 0, 1, 0.5j], [0, 0, 0, 0, 0, -1]]
    #
    # generate corresponding mpo using function from test_generate
    H0 = mpo_hopping_Hterm(N, J0, sym="U1", config=cfg)
    #
    # find ground state using dmrg
    Dmax = 8  # we will have exact Mps
    opts_svd = {'tol': 1e-15, 'D_total': Dmax}  # with no truncation in 2site methods
    psi = generate.random_mps(D_total=Dmax, n=n)
    out = mps.dmrg_(psi, H0, method='2site', max_sweeps=2, opts_svd=opts_svd)
    out = mps.dmrg_(psi, H0, method='1site', energy_tol=1e-14, Schmidt_tol=1e-14, max_sweeps=10)
    #
    # get reference results for ground state and check mps
    C0ref, E0ref = gs_correlation_matrix(J0, n)
    assert pytest.approx(out.energy.item(), rel=1e-14) == E0ref
    assert np.allclose(C0ref, correlation_matrix(psi, generate), rtol=1e-12)
    #
    # sudden quench with a new Hamiltonian
    J1 = [[-1, 0.5, 0, -0.3, 0.1, 0], [0, 1, 0.5, 0, -0.3, 0.1], [0, 0, -1, 0.5, 0, -0.3], [0, 0, 0, 1, 0.5, 0], [0, 0, 0, 0, -1, 0.5], [0, 0, 0, 0, 0, 1]]
    H1 = mpo_hopping_Hterm(N, J1, sym="U1", config=cfg)
    #
    # run time evolution; calculate correlation matrix at 2 snapshots
    times = (0, 0.25, 0.6)
    # parameters for expmv in tdvp_,
    # 'ncv' is an initial guess for the size of Krylov space. It gets updated at each site/bond during evolution.
    opts_expmv = {'hermitian': True, 'ncv': 5, 'tol': 1e-12}
    for method in ('1site', '2site', '12site'):  # test various methods
        psi1 = psi.shallow_copy()  # shallow_copy is sufficient to retain initial state
        for step in mps.tdvp_(psi1, H1, times=times, method=method, dt=0.125, opts_svd=opts_svd, opts_expmv=opts_expmv):
            C1 = correlation_matrix(psi1, generate)  # calculate correlation matrix from mps
            C1ref = evolve_correlation_matrix(C0ref, J1, step.tf)  # exact reference
            assert np.linalg.norm(C1ref - C1) < 1e-12  # compare results with references


def test_tdvp_time_dependent():
    """
    Simulate a slow quench across a quantum critical point in a transverse Ising model.
    Use a small chain with periodic boundary conditions and compare with exact reference.
    """
    # load spin-1/2 operators
    operators = yastn.operators.Spin12(sym='Z2', backend=cfg.backend, default_device=cfg.default_device)
    #
    N = 10  # consider a system of 10 sites
    generate = mps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)
    #
    # We take the Hamiltonian with periodic boundary conditions.
    parameters = {'NNsites' : tuple((n, (n + 1) % N) for n in range(N)), 'sites' : tuple(range(N))}
    HZZ = generate.mpo_from_latex("\sum_{i,j \in NNsites} x_{i} x_{j}", parameters)  # sum XX
    HX = generate.mpo_from_latex("\sum_{i \in sites} z_{i} ", parameters)  # sum Z
    #
    # Kibble-Zurek quench across a critical point at gc; tauQ is quench time.
    tauQ, gc = 1, 1
    ti, tf = -tauQ, tauQ  #evolve from gi = 2 to gf = 0
    g = lambda t : gc - t / tauQ
    H = lambda t : -1 * HZZ - g(t) * HX
    #
    # analytical reference expectation values measured at g = 1 and g = 0 (for tauQ = 1, gi = 2 and N=10)
    ZZex = {1: 0.4701822929348489514, 0: 0.7387694101217603240}
    Xex  = {1: 0.7762552604723193039, 0: 0.1634910118224495344}
    Egs = -2.127120881869593  # at gi = 2
    #
    # start with ground state at gi = 2
    Dmax = 10
    psi = generate.random_mps(D_total=Dmax, n=0)
    out = mps.dmrg_(psi, H(ti), method='2site', max_sweeps=2, opts_svd={'tol': 1e-6, 'D_total': Dmax})
    out = mps.dmrg_(psi, H(ti), method='1site', energy_tol=1e-12, Schmidt_tol=1e-12, max_sweeps=10)
    assert pytest.approx(out.energy.item() / N, rel=1e-7) == Egs
    assert psi.get_bond_dimensions() == (1, 2, 4, 8, 10, 10, 10, 8, 4, 2, 1)
    #
    # slow quench to gf = 0
    opts_expmv = {'hermitian': True, 'tol': 1e-12} # parameters for expmv in tdvp_
    opts_svd = {'tol': 1e-6, 'D_total': 16}  # parameters to grow bond dimension during evolution
    for step in mps.tdvp_(psi, H, times=(ti, 0, tf), method='12site', dt=0.125, order='4th', opts_svd=opts_svd, opts_expmv=opts_expmv):
        EZZ = mps.vdot(psi, HZZ, psi) / N  # calculate <ZZ>
        EX = mps.vdot(psi, HX, psi) / N  # calculate <X>
        gg = round(g(step.tf))  # g at the snapshot
        assert pytest.approx(EX.item(), rel=1e-5) == Xex[gg]  # compare with exact result
        assert pytest.approx(EZZ.item(), rel=1e-5) == ZZex[gg]
        assert psi.get_bond_dimensions() == (1, 2, 4, 8, 16, 16, 16, 8, 4, 2, 1)


if __name__ == "__main__":
    test_tdvp_hermitian()
    test_tdvp_time_dependent()
