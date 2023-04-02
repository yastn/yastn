import pytest
import yastn
import yastn.tn.mps as mps


def run_evolution():
    ops = yastn.operators.Spin12(sym='Z2')
    H_ZZ = "\sum_{j,k \in rangeNN} (sz_{j} sz_{k}) + \sum_{i \in rangeN} mu cp_{i} c_{i}"
    H_X = "\sum_{j,k \in rangeNN} (sz_{j} sz_{k}) + \sum_{i \in rangeN} mu cp_{i} c_{i}"


# def test_generator_mpo():
#     for sym in ['Z2', 'U1']:
#         ops = yastn.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
#         for t in [0,0.2, -0.3]:
#             for mu in [0.2, -0.3]:
#                 for N in [2,3]:
#                     example_mapping = (\
#                                         {i: i for i in range(N)},\
#                                         {str(i): i for i in range(N)},\
#                                         {(str(i), 'A'): i for i in range(N)},\
#                     )
#                     example_parameters = (\
#                         {"t": t * np.ones((N,N)), "mu": mu, "rangeN": [i for i in range(N)], "rangeNN": zip([i for i in range(N-1)], [i for i in range(1,N)])},\
#                         {"t": t * np.ones((N,N)), "mu": mu, "rangeN": [str(i) for i in range(N)], "rangeNN": zip([str(i) for i in range(N-1)], [str(i) for i in range(1,N)])},\
#                         {"t": t * np.ones((N,N)), "mu": mu, "rangeN": [(str(i),'A') for i in range(N)], "rangeNN": zip([(str(i),'A') for i in range(N-1)], [(str(i),'A') for i in range(1,N)])},\
#                     )
#                     for (emap, eparam) in zip(example_mapping, example_parameters):
#                         generate = mps.Generator(N, ops, map=emap)
#                         generate.random_seed(seed=0)
                        
#                         H_ref = mpo_nn_hopping(generate.config, N=N, t=t, mu=mu)
#                         H = generate.mpo_from_latex(H_str, eparam)

#                         psi = generate.random_mps(D_total=8, n=0) + generate.random_mps( D_total=8, n=1)
                        
#                         x_ref = mps.measure_mpo(psi, H_ref, psi).item()
#                         x = mps.measure_mpo(psi, H, psi).item()
#                         assert abs(x_ref - x) < tol

#                         psi.canonize_(to='first')
#                         psi.canonize_(to='last')
#                         x_ref = mps.measure_mpo(psi, H_ref, psi).item()
#                         x = mps.measure_mpo(psi, H, psi).item()
#                         assert abs(x_ref - x) < tol

# def test_mpo_from_latex():
    
#     # the model is random with handom hopping and on-site energies. sym is symmetry for tensors we will use
#     sym, N = 'U1', 3
    
#     # generate set of basic ops for the model we want to work with
#     ops = yastn.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
    
#     # generate data for random Hamiltonian
#     amplitudes1 = np.random.rand(N, N)
#     param1 = amplitudes1 - np.diag(np.diag(amplitudes1))
#     param2 = np.diag(amplitudes1)
    
#     # use this map which is used for naming the sites in MPO
#     # maps between iteractors and MPO
#     emap = {i: i for i in range(N)}
    
#     # create a generator initialized for emap mapping
#     generate = mps.Generator(N, ops, map=emap)
#     generate.random_seed(seed=0)
    
#     # define parameters for automatic generator and Hamiltonian in a latex-like form
#     eparam ={"t": param1, "mu": param2, "rangeN": range(N)}
#     h_input = "\sum_{j\in rangeN} \sum_{k\in rangeN} t_{j,k} (cp_{j} c_{k} + cp_{k} c_{j}) + \
#             \sum_{j\in rangeN} mu_{j} cp_{j} c_{j}"
    
#     # generate MPO from latex-like input
#     h_str = generate.mpo_from_latex(h_input, eparam)

#     # generate Hamiltonian manually
#     man_input = []
#     for j, val in enumerate(param2):
#         man_input.append(mps.Hterm(val, (j, j,), (ops.cp(), ops.c(),)))
    
#     for j, row in enumerate(param1):
#         for k, val in enumerate(row):
#             man_input.append(mps.Hterm(val, (j, k,), (ops.cp(), ops.c(),)))
#             man_input.append(mps.Hterm(val, (k, j,), (ops.cp(), ops.c(),)))
#     h_man = mps.generate_mpo(generate.I(), man_input)
    
#     # test the result by comparing expectation value for a steady state.
#     # use random seed to generate mps
#     generate.random_seed(seed=0)

#     # generate mps and compare overlaps
#     psi = generate.random_mps(D_total=8, n=0) + generate.random_mps( D_total=8, n=1)
#     x_man = mps.measure_mpo(psi, h_man, psi).item()
#     x_str = mps.measure_mpo(psi, h_str, psi).item()
    
#     assert abs(x_man - x_str) < tol


# def test_mpo_from_templete():
    
#     # the model is random with handom hopping and on-site energies. sym is symmetry for tensors we will use
#     sym, N = 'U1', 3
    
#     # generate set of basic ops for the model we want to work with
#     ops = yastn.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
    
#     # generate data for random Hamiltonian
#     amplitudes1 = np.random.rand(N, N)
#     amplitudes1 = 0.5 * (amplitudes1 + amplitudes1.transpose())
    
#     # use this map which is used for naming the sites in MPO
#     # maps between iteractors and MPO
#     emap = {i: i for i in range(N)}
    
#     # create a generator initialized for emap mapping
#     generate = mps.Generator(N, ops, map=emap)
#     generate.random_seed(seed=0)
    
#     # define parameters for automatic generator and Hamiltonian in a latex-like form
#     eparam ={"A": amplitudes1, "rangeN": range(N)}
#     h_input = "\sum_{j\in rangeN} \sum_{k\in rangeN} A_{j,k} cp_{j} c_{k}"
    
#     # generate MPO from latex-like input
#     h_str = generate.mpo_from_latex(h_input, eparam)

#     # generate Hamiltonian manually
#     man_input = []
#     for n0 in emap.keys():
#         for n1 in emap.keys():
#             man_input.append(\
#                 mps.single_term((('A',n0,n1), ('cp',n0), ('c',n1),)))
#     h_man = generate.mpo_from_templete(man_input, eparam)
    
#     # test the result by comparing expectation value for a steady state.
#     # use random seed to generate mps
#     generate.random_seed(seed=0)

#     # generate mps and compare overlaps
#     psi = generate.random_mps(D_total=8, n=0) + generate.random_mps( D_total=8, n=1)
#     x_man = mps.measure_mpo(psi, h_man, psi).item()
#     x_str = mps.measure_mpo(psi, h_str, psi).item()
    
#     assert abs(x_man - x_str) < tol

# def mps_basis_ex(config):
#     plus = yastn.Tensor(config=config, s=[1])
#     plus.set_block(val=[0, 1],Ds=(2,))
#     minus = yastn.Tensor(config=config, s=[1])
#     minus.set_block(val=[1, 0],Ds=(2,))
#     return plus, minus

# def mpo_basis_ex(config):
#     cpc = yastn.Tensor(config=config, s=[1, -1])
#     cpc.set_block(val=[[0,0],[0,1]],Ds=(2,2,))
#     ccp = yastn.Tensor(config=config, s=[1, -1])
#     ccp.set_block(val=[[1,0],[0,0]],Ds=(2,2,))
#     I = yastn.Tensor(config=config, s=[1, -1])
#     I.set_block(val=[[1,0],[0,1]],Ds=(2,2,))
#     return cpc, ccp, I

# def test_generator_mps():
#     N = 3
    
#     cpc, ccp, I = mpo_basis_ex(cfg)
    
#     ops = yastn.operators.General({'cpc': lambda j: cpc, 'ccp': lambda j: ccp, 'I': lambda j: I})
        
#     emap = {str(i): i for i in range(N)}
    
#     generate = mps.Generator(N, ops, map=emap)
#     generate.random_seed(seed=0)
    
#     # generate from LaTeX-like instruction
#     A = np.random.rand(2)
#     psi_str = "A_{0} Plus_{0} Plus_{1} Plus_{2} + A_{1} Minus_{0} Minus_{1} Minus_{2}"
#     plus, minus = mps_basis_ex(cfg)

#     psi_ltx = generate.mps_from_latex(psi_str, \
#         vectors = {'Plus': lambda j: plus, 'Minus': lambda j: minus}, \
#         parameters = {'A': A})
    
#     psi_tmpl = generate.mps_from_templete(
#         [mps.single_term((('A','0'),('Plus','0'),('Plus','1'),('Plus','2'))), \
#         mps.single_term((('A','1'),('Minus','0'),('Minus','1'),('Minus','2')))], \
#         vectors = {'Plus': lambda j: plus, 'Minus': lambda j: minus}, \
#         parameters = {'A': A})

#     psi = generate.random_mps(D_total=8)
#     assert mps.measure_overlap(psi_tmpl, psi) == mps.measure_overlap(psi_ltx, psi)

if __name__ == "__main__":
    run_evolution()
