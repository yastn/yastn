import pytest
import numpy as np
import yast
import yast.tn.mps as mps
try:
    from .configs import config_dense as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config_dense as cfg

tol = 1e-12


####### MPO for XX model ##########
def mpo_XX_model_dense(config, N, t, mu):
    # Initialize MPO tensor by tensor. Example for NN-hopping model
    # TODO ref ?

    # Define basic rank-2 blocks (matrices) of on-site tensors
    #
    cp = np.array([[0, 0], [1, 0]])
    c = np.array([[0, 1], [0, 0]])
    nn = np.array([[0, 0], [0, 1]])
    ee = np.array([[1, 0], [0, 1]])
    oo = np.array([[0, 0], [0, 0]])

    # Build empty MPO for system of N sites
    # 
    H = mps.Mpo(N)

    # Depending on the site position, define elements of on-site tensor
    #
    for n in H.sweep(to='last'):  # empty tensors
        if n == H.first:
            tmp = np.block([[mu * nn, t * cp, t * c, ee]])
            tmp = tmp.reshape((1, 2, 4, 2))
            Ds = (1, 2, 2, 4)
        elif n == H.last:
            tmp = np.block([[ee], [c], [cp], [mu * nn]])
            tmp = tmp.reshape((4, 2, 1, 2))
            Ds = (4, 2, 2, 1)
        else:
            tmp = np.block([[ee, oo, oo, oo],
                            [c, oo, oo, oo],
                            [cp, oo, oo, oo],
                            [mu * nn, t * cp, t * c, ee]])
            tmp = tmp.reshape((4, 2, 4, 2))
            Ds = (4, 2, 2, 4)
        # tmp = np.transpose(tmp, (0, 1, 3, 2))
        #
        # We chose signature convention for indices of the MPO tensor as follows
        #         
        #          | 
        #          V(+1)
        #          | 
        # (+1) ->-|T|->-(-1)
        #          |
        #          V(-1)
        #          |
        #
        on_site_t = yast.Tensor(config=config, s=(1, 1, -1, -1))
        on_site_t.set_block(val=tmp, Ds=Ds)

        # Set n-th on-site tensor of MPO
        H[n]= on_site_t
    return H


def mpo_XX_model_Z2(config, N, t, mu):
    # Initialize MPO tensor by tensor. Example for NN-hopping model, 
    # using explicit Z2 symmetry of the model.
    # TODO ref ?

    # Build empty MPO for system of N sites
    #
    H = mps.Mpo(N)

    # Depending on the site position, define elements of on-site tensor
    #
    for n in H.sweep(to='last'):
        #
        # Define empty yast.Tensor as n-th on-site tensor
        # We chose signature convention for indices of the MPO tensor as follows
        #         
        #          | 
        #          V(+1)
        #          | 
        # (+1) ->-|T|->-(-1)
        #          |
        #          V(-1)
        #          |
        #
        H[n] = yast.Tensor(config=config, s=[1, 1, -1, -1], n=0)
        
        # set blocks, indexed by tuple of Z2 charges, of on-site tensor at n-th position
        #
        if n == H.first:
            H[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 2, 1))
            H[n].set_block(ts=(0, 1, 0, 1), val=[mu, 1], Ds=(1, 1, 2, 1))
            H[n].set_block(ts=(0, 0, 1, 1), val=[t, 0], Ds=(1, 1, 2, 1))
            H[n].set_block(ts=(0, 1, 1, 0), val=[0, t], Ds=(1, 1, 2, 1))
        elif n == H.last:
            H[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H[n].set_block(ts=(0, 1, 0, 1), val=[1, mu], Ds=(2, 1, 1, 1))
            H[n].set_block(ts=(1, 1, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H[n].set_block(ts=(1, 0, 0, 1), val=[0, 1], Ds=(2, 1, 1, 1))
        else:
            H[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 2, 1))
            H[n].set_block(ts=(0, 1, 0, 1), val=[[1, 0], [mu, 1]], Ds=(2, 1, 2, 1))
            H[n].set_block(ts=(0, 0, 1, 1), val=[[0, 0], [t, 0]], Ds=(2, 1, 2, 1))
            H[n].set_block(ts=(0, 1, 1, 0), val=[[0, 0], [0, t]], Ds=(2, 1, 2, 1))
            H[n].set_block(ts=(1, 1, 0, 0), val=[[1, 0], [0, 0]], Ds=(2, 1, 2, 1))
            H[n].set_block(ts=(1, 0, 0, 1), val=[[0, 0], [1, 0]], Ds=(2, 1, 2, 1))
    return H


def mpo_XX_model_U1(config, N, t, mu):
    # Initialize MPO tensor by tensor. Example for NN-hopping model, 
    # using explicit U1 symmetry of the model.
    # TODO ref ?

    # Build empty MPO for system of N sites
    #
    H = mps.Mpo(N)

    # Depending on the site position, define elements of on-site tensor
    #
    for n in H.sweep(to='last'):
        #
        # Define empty yast.Tensor as n-th on-site tensor
        # We chose signature convention for indices of the MPO tensor as follows
        #         
        #          | 
        #          V(+1)
        #          | 
        # (+1) ->-|T|->-(-1)
        #          |
        #          V(-1)
        #          |
        #
        H.A[n] = yast.Tensor(config=config, s=[1, 1, -1, -1], n=0)

        # set blocks, indexed by tuple of U1 charges, of on-site tensor at n-th position
        #
        if n == H.first:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 2, 1))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[mu, 1], Ds=(1, 1, 2, 1))
            H.A[n].set_block(ts=(0, 0, -1, 1), val=[t], Ds=(1, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[t], Ds=(1, 1, 1, 1))
        elif n == H.last:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[1, mu], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(-1, 1, 0, 0), val=[1], Ds=(1, 1, 1, 1))
            H.A[n].set_block(ts=(1, 0, 0, 1), val=[1], Ds=(1, 1, 1, 1))
        else:
            H.A[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 2, 1))
            H.A[n].set_block(ts=(0, 1, 0, 1), val=[[1, 0], [mu, 1]], Ds=(2, 1, 2, 1))
            H.A[n].set_block(ts=(0, 0, -1, 1), val=[0, t], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(0, 1, 1, 0), val=[0, t], Ds=(2, 1, 1, 1))
            H.A[n].set_block(ts=(-1, 1, 0, 0), val=[1, 0], Ds=(1, 1, 2, 1))
            H.A[n].set_block(ts=(1, 0, 0, 1), val=[1, 0], Ds=(1, 1, 2, 1))
    return H


def mpo_XX_model(config, N, t, mu):
    if config.sym.SYM_ID == 'dense':
        return mpo_XX_model_dense(config, N, t, mu)
    elif config.sym.SYM_ID == 'Z2':
        return mpo_XX_model_Z2(config, N, t, mu)
    elif config.sym.SYM_ID == 'U1':
        return mpo_XX_model_U1(config, N, t, mu)


def test_generator_mps():
    N = 10
    D_total = 16
    bds = (1,) + (D_total,) * (N - 1) + (1,)

    for sym, nn in (('Z2', (0,)), ('Z2', (1,)), ('U1', (N // 2,))):
        ops = yast.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        generate = mps.Generator(N, ops)
        I = generate.I()
        assert pytest.approx(mps.measure_overlap(I, I).item(), rel=tol) == 2 ** N
        O = I @ I + (-1 * I)
        assert pytest.approx(mps.measure_overlap(O, O).item(), abs=tol) == 0
        psi = generate.random_mps(D_total=D_total, n = nn)
        assert psi[psi.last].get_legs(axis=2).t == (nn,)
        assert psi[psi.first].get_legs(axis=0).t == ((0,) * len(nn),)
        bds = psi.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])


def test_generator_mpo():
    # uniform chain with nearest neighbor hopping
    # notation:
    # * in the sum there are all elements which are connected by multiplication, so \sum_{.} -1 ... shuold be \sum_{.} (-1) ...
    # * 1j is an imaginary number
    # * multiple sums are supported so you can write \sum_{.} \sum_{.} ...
    # * multiplication of the sum is allowed but '*' or bracket is needed.
    #   ---> this is an artifact of allowing space=' ' to be equivalent to multiplication
    #   E.g.1, 2 \sum... can be written as 2 (\sum...) or 2 * \sum... or (2) * \sum...
    #   E.g.2, \sum... \sum.. write as \sum... * \sum... or (\sum...) (\sum...)
    #   E.g.4, -\sum... is supported and equivalent to (-1) * \sum...
    H_str = "\sum_{j,k \in rangeNN} t_{j,k} (cp_{j} c_{k}+cp_{k} c_{j}) + \sum_{i \in rangeN} mu cp_{i} c_{i}"
    for sym in ['Z2', 'U1']:
        ops = yast.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        for t in [0,0.2, -0.3]:
            for mu in [0.2, -0.3]:
                for N in [3,5]:
                    example_mapping = (\
                                        {i: i for i in range(N)},\
                                        {str(i): i for i in range(N)},\
                                        {(str(i), 'A'): i for i in range(N)},\
                    )
                    example_parameters = (\
                        {"t": t * np.ones((N,N)), "mu": mu, "rangeN": [i for i in range(N)], "rangeNN": zip([i for i in range(N-1)], [i for i in range(1,N)])},\
                        {"t": t * np.ones((N,N)), "mu": mu, "rangeN": [str(i) for i in range(N)], "rangeNN": zip([str(i) for i in range(N-1)], [str(i) for i in range(1,N)])},\
                        {"t": t * np.ones((N,N)), "mu": mu, "rangeN": [(str(i),'A') for i in range(N)], "rangeNN": zip([(str(i),'A') for i in range(N-1)], [(str(i),'A') for i in range(1,N)])},\
                    )
                    for (emap, eparam) in zip(example_mapping, example_parameters):
                        generate = mps.Generator(N, ops, map=emap)
                        generate.random_seed(seed=0)
                        
                        H_ref = mpo_XX_model(generate.config, N=N, t=t, mu=mu)
                        H = generate.mpo_from_latex(H_str, eparam)
              
                        psi = generate.random_mps(D_total=8, n=0) + generate.random_mps( D_total=8, n=1)
                        x_ref = mps.measure_mpo(psi, H_ref, psi).item()
                        x = mps.measure_mpo(psi, H, psi).item()
                        assert abs(x_ref - x) < tol

                        psi.canonize_sweep(to='first')
                        psi.canonize_sweep(to='last')
                        x_ref = mps.measure_mpo(psi, H_ref, psi).item()
                        x = mps.measure_mpo(psi, H, psi).item()
                        assert abs(x_ref - x) < tol

def mpo_random_hopping():
    
    # the model is random with handom hopping and on-site energies. sym is symmetry for tensors we will use
    sym, N = 'U1', 3
    
    # generate set of basic ops for the model we want to work with
    ops = yast.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
    
    # generate data for random Hamiltonian
    amplitudes1 = np.random.rand(N, N)
    param1 = amplitudes1 - np.diag(np.diag(amplitudes1))
    param2 = np.diag(amplitudes1)
    
    # use this map which is used for naming the sites in MPO
    # maps between iteractors and MPO
    emap = {i: i for i in range(N)}
    
    # create a generator initialized for emap mapping
    generate = mps.Generator(N, ops, map=emap)
    generate.random_seed(seed=0)
    
    # define parameters for automatic generator and Hamiltonian in a latex-like form
    eparam ={"t": param1, "mu": param2, "rangeN": range(N)}
    h_input = "\sum_{j\in rangeN} \sum_{k\in rangeN} t_{j,k} (cp_{j} c_{k} + cp_{k} c_{j}) + \
            \sum_{j\in rangeN} mu_{j} cp_{j} c_{j}"
    
    # generate MPO from latex-like input
    h_str = generate.mpo_from_latex(h_input, eparam)

    # generate Hamiltonian manually
    man_input = []
    for j, val in enumerate(param2):
        man_input.append(mps.Hterm(val, (j, j,), (ops.cp(), ops.c(),)))
    
    for j, row in enumerate(param1):
        for k, val in enumerate(row):
            man_input.append(mps.Hterm(val, (j, k,), (ops.cp(), ops.c(),)))
            man_input.append(mps.Hterm(val, (k, j,), (ops.cp(), ops.c(),)))
    h_man = mps.generate_mpo(generate.I(), man_input)
    
    # test the result by comparing expectation value for a steady state.
    # use random seed to generate mps
    generate.random_seed(seed=0)

    # generate mps and compare overlaps
    psi = generate.random_mps(D_total=8, n=0) + generate.random_mps( D_total=8, n=1)
    x_man = mps.measure_mpo(psi, h_man, psi).item()
    x_str = mps.measure_mpo(psi, h_str, psi).item()
    
    assert abs(x_man - x_str) < tol


if __name__ == "__main__":
    test_generator_mps()
    #test_generator_mpo()
    mpo_random_hopping()
