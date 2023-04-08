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
    cns = [mps.generate_mpo(gen.I(), [mps.Hterm(1, [n], [gen._ops.c()])]) for n in range(gen.N)]
    ps = [cn @ psi for cn in cns]
    C = np.zeros((gen.N, gen.N), dtype=np.complex128)
    for m in range(gen.N):
        for n in range(gen.N):
            C[m, n] = mps.vdot(ps[n], ps[m])
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
    print(H1.get_bond_dimensions())
    #
    # run time evolution; calculate correlation matrix at 2 snapshots
    times = (0, 0.25, 0.6)
    opts_expmv = {'hermitian': True, 'ncv': 5, 'tol': 1e-12} # parameters for expmv in tdvp_
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
    # load operators
    operators = yastn.operators.Spin12(sym='Z2', backend=cfg.backend, default_device=cfg.default_device)
    #
    N = 12  # consider a system of 12 sites
    generate = mps.Generator(N=N, operators=operators)
    generate.random_seed(seed=0)
    parameters = {'NNsites' : tuple((n, (n + 1) % N) for n in range(N)), 'sites' : tuple(range(N))}
    HZZ = generate.mpo_from_latex("\sum_{i,j \in NNsites} z_{i} z_{j}", parameters)
    HX = generate.mpo_from_latex("\sum_{i \in sites} x_{i} ", parameters)
    
    print(HZZ.get_bond_dimensions())
    print(HX.get_bond_dimensions())



if __name__ == "__main__":
    test_tdvp_hermitian()
    test_tdvp_time_dependent()
