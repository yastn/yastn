import pytest
import numpy as np
import yastn
import yastn.tn.mps as mps
try:
    from .configs import config_dense as cfg
    from .configs import config_U1
    from .configs import config_Z2
    # pytest modifies cfg to inject different backends and devices during tests
except ImportError:
    from configs import config_dense as cfg
    from configs import config_U1
    from configs import config_Z2

tol = 1e-12

#
####### MPO for XX model ##########
#
def build_mpo_nn_hopping_manually(N, t, mu, sym='U1', config=None):
    """
    Nearest-neighbor hopping Hamiltonian on N sites
    with hopping amplitude t and chemical potential mu, i.e.,
    H = t * sum_{n=1}^{N-1} (cp_n c_{n+1} + cp_{n+1} c_n)
      + mu * sum_{n=1}^{N} cp_n c_n

    Initialize MPO symmetric tensors by hand with sym = 'dense', 'Z2', or 'U1'.
    Config is used to inject non-default backend and default_device.
    """
    #
    # Build empty MPO for system of N sites
    #
    H = mps.Mpo(N)
    #
    # Depending on the symmetry, define elements of on-site tensor
    #
    # We chose signature convention for indices of the MPO tensor as follows
    #         |
    #         ^(-1)
    #         |
    # (-1)-<-|T|-<-(+1)
    #         |
    #         ^(+1)
    #         |
    #
    # We set fermionic=True for conistency as in tests
    # MPO is compared with operators.SpinlessFermions.
    # backend and device is inherited from config to automitize pytest testing.

    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject backend and device for testing
    config = yastn.make_config(fermionic=True, sym=sym, **opts_config)

    if sym == 'dense':  # no symmetry
        # Basic rank-2 blocks (matrices) of on-site tensors
        cp = np.array([[0, 0], [1, 0]])
        c  = np.array([[0, 1], [0, 0]])
        nn = np.array([[0, 0], [0, 1]])
        ee = np.array([[1, 0], [0, 1]])
        oo = np.array([[0, 0], [0, 0]])

        for n in H.sweep(to='last'):
            # empty tensors
            H[n] = yastn.Tensor(config=config, s=(-1, 1, 1, -1))
            # that gets filled-in
            if n == H.first:
                tmp = np.block([[mu * nn, t * cp, t * c, ee]])
                H[n].set_block(val=tmp, Ds=(1, 2, 4, 2))
            elif n == H.last:
                tmp = np.block([[ee], [c], [cp], [mu * nn]])
                H[n].set_block(val=tmp, Ds=(4, 2, 1, 2))
            else:
                tmp = np.block([[ee, oo, oo, oo],
                                [c, oo, oo, oo],
                                [cp, oo, oo, oo],
                                [mu * nn, t * cp, t * c, ee]])
                H[n].set_block(val=tmp, Ds=(4, 2, 4, 2))

    elif sym == 'Z2':  # Z2 symmetry
        for n in H.sweep(to='last'):
            H[n] = yastn.Tensor(config=config, s=(-1, 1, 1, -1), n=0)
            if n == H.first:
                H[n].set_block(ts=(0, 0, 0, 0),
                               Ds=(1, 1, 2, 1), val=[0, 1])
                H[n].set_block(ts=(0, 1, 0, 1),
                               Ds=(1, 1, 2, 1), val=[mu, 1])
                H[n].set_block(ts=(0, 0, 1, 1),
                               Ds=(1, 1, 2, 1), val=[t, 0])
                H[n].set_block(ts=(0, 1, 1, 0),
                               Ds=(1, 1, 2, 1), val=[0, t])
            elif n == H.last:
                H[n].set_block(ts=(0, 0, 0, 0),
                               Ds=(2, 1, 1, 1), val=[1, 0])
                H[n].set_block(ts=(0, 1, 0, 1),
                               Ds=(2, 1, 1, 1), val=[1, mu])
                H[n].set_block(ts=(1, 1, 0, 0),
                               Ds=(2, 1, 1, 1), val=[1, 0])
                H[n].set_block(ts=(1, 0, 0, 1),
                               Ds=(2, 1, 1, 1), val=[0, 1])
            else:
                H[n].set_block(ts=(0, 0, 0, 0),
                               Ds=(2, 1, 2, 1), val=[[1, 0], [0, 1]])
                H[n].set_block(ts=(0, 1, 0, 1),
                               Ds=(2, 1, 2, 1), val=[[1, 0], [mu, 1]])
                H[n].set_block(ts=(0, 0, 1, 1),
                               Ds=(2, 1, 2, 1), val=[[0, 0], [t, 0]])
                H[n].set_block(ts=(0, 1, 1, 0),
                               Ds=(2, 1, 2, 1), val=[[0, 0], [0, t]])
                H[n].set_block(ts=(1, 1, 0, 0),
                               Ds=(2, 1, 2, 1), val=[[1, 0], [0, 0]])
                H[n].set_block(ts=(1, 0, 0, 1),
                               Ds=(2, 1, 2, 1), val=[[0, 0], [1, 0]])

    elif sym == 'U1':  # U1 symmetry
        for n in H.sweep(to='last'):
            H.A[n] = yastn.Tensor(config=config, s=(-1, 1, 1, -1), n=0)
            if n == H.first:
                H.A[n].set_block(ts=(0, 0, 0, 0),
                                 val=[0, 1], Ds=(1, 1, 2, 1))
                H.A[n].set_block(ts=(0, 1, 0, 1),
                                 val=[mu, 1], Ds=(1, 1, 2, 1))
                H.A[n].set_block(ts=(0, 0, 1, 1),
                                 val=[t], Ds=(1, 1, 1, 1))
                H.A[n].set_block(ts=(0, 1, -1, 0),
                                 val=[t], Ds=(1, 1, 1, 1))
            elif n == H.last:
                H.A[n].set_block(ts=(0, 0, 0, 0),
                                 Ds=(2, 1, 1, 1), val=[1, 0])
                H.A[n].set_block(ts=(0, 1, 0, 1),
                                 Ds=(2, 1, 1, 1), val=[1, mu])
                H.A[n].set_block(ts=(1, 1, 0, 0),
                                 Ds=(1, 1, 1, 1), val=[1])
                H.A[n].set_block(ts=(-1, 0, 0, 1),
                                 Ds=(1, 1, 1, 1), val=[1])
            else:
                H.A[n].set_block(ts=(0, 0, 0, 0),
                                 Ds=(2, 1, 2, 1), val=[[1, 0], [0, 1]])
                H.A[n].set_block(ts=(0, 1, 0, 1),
                                 Ds=(2, 1, 2, 1), val=[[1, 0], [mu, 1]])
                H.A[n].set_block(ts=(0, 0, 1, 1),
                                 Ds=(2, 1, 1, 1), val=[0, t])
                H.A[n].set_block(ts=(0, 1, -1, 0),
                                 Ds=(2, 1, 1, 1), val=[0, t])
                H.A[n].set_block(ts=(1, 1, 0, 0),
                                 Ds=(1, 1, 2, 1), val=[1, 0])
                H.A[n].set_block(ts=(-1, 0, 0, 1),
                                 Ds=(1, 1, 2, 1), val=[1, 0])
    return H


def mpo_hopping_Hterm(J, sym="U1", config=None):
    """
    Fermionic hopping Hamiltonian on N sites with hoppings at arbitrary range.

    The upper triangular part of N x N matrix J defines hopping amplitudes,
    and the diagonal defines on-site chemical potentials.
    """

    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject backend and device for testing
    ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)

    Hterms = []  # list of Hterm(amplitude, positions, operators)
    # Each Hterm corresponds to a single product of local operators.
    # Hamiltonian is a sum of such products.
    #
    N = len(J)
    #
    # chemical potential on site n
    #
    for n in range(N):
        if abs(J[n][n]) > 0:
            Hterms.append(mps.Hterm(amplitude=J[n][n],
                                    positions=[n],
                                    operators=[ops.n()]))

    # hopping term between sites m and n
    for m in range(N):
        for n in range(m + 1, N):
            if abs(J[m][n]) > 0:
                Hterms.append(mps.Hterm(amplitude=J[m][n],
                                        positions=(m, n),
                                        operators=(ops.cp(), ops.c())))
                Hterms.append(mps.Hterm(amplitude=np.conj(J[m][n]),
                                        positions=(n, m),
                                        operators=(ops.cp(), ops.c())))

    # We need an identity MPO operator.
    I = mps.product_mpo(ops.I(), N)

    # Generate MPO for Hterms
    H = mps.generate_mpo(I, Hterms, opts={'tol':1e-14})

    return H


def mpo_nn_hopping_latex(N, t, mu, sym="U1", config=None):
    """
    Nearest-neighbor hopping Hamiltonian on N sites
    with hopping amplitude t and chemical potential mu.

    The upper triangular part of matrix J defines hopping amplitudes,
    and the diagonal defines on-site chemical potentials.
    """

    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject backend and device for testing
    ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)

    Hstr = "\sum_{j,k \in NN} t (cp_{j} c_{k}+cp_{k} c_{j})"
    Hstr += " + \sum_{i \in sites} mu cp_{i} c_{i}"
    parameters = {"t": t,
                  "mu": mu,
                  "sites": list(range(N)),
                  "NN": list((i, i+1) for i in range(N-1))}

    generate = mps.Generator(N, ops)
    H = generate.mpo_from_latex(Hstr, parameters=parameters)
    return H


def mpo_hopping_latex(J=np.array([[0.5, 1], [0, 0.2]]), sym="U1", config=None):
    """
    Nearest-neighbor hopping Hamiltonian on N sites
    with hopping amplitude t and chemical potential mu.

    The upper triangular part of matrix J defines hopping amplitudes,
    and the diagonal defines on-site chemical potentials.
    """

    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject backend and device for testing
    ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)

    N = len(J)

    Hstr = "\sum_{j,k \in NN} J_{j,k} (cp_{j} c_{k}+cp_{k} c_{j})"
    Hstr += " + \sum_{i \in sites} J_{i,i} cp_{i} c_{i}"
    parameters = {"J": J,
                  "sites": list(range(N)),
                  "NN": list((i, j) for i in range(N-1)
                             for j in range(i + 1, N))}

    generate = mps.Generator(N, ops)
    H = generate.mpo_from_latex(Hstr, parameters=parameters)
    return H


def test_generator_mpo():
    # uniform chain with nearest neighbor hopping
    # notation:
    # * in the sum there are all elements which are connected by multiplication, so \sum_{.} -1 ... should be \sum_{.} (-1) ...
    # * 1j is an imaginary number
    # * multiple sums are supported so you can write \sum_{.} \sum_{.} ...
    # * multiplication of the sum is allowed but '*' or bracket is needed.
    #   ---> this is an artifact of allowing space=' ' to be equivalent to multiplication
    #   E.g.1, 2 \sum... can be written as 2 (\sum...) or 2 * \sum... or (2) * \sum...
    #   E.g.2, \sum... \sum.. write as \sum... * \sum... or (\sum...) (\sum...)
    #   E.g.4, -\sum... is supported and equivalent to (-1) * \sum...
    H_str = "\sum_{j,k \in NN} t_{j,k} (cp_{j} c_{k}+cp_{k} c_{j}) + \sum_{i \in sites} mu cp_{i} c_{i}"
    for sym in ['Z2', 'U1']:
        ops = yastn.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        for t in [0, 0.2, -0.3]:
            for mu in [0.2, -0.3]:
                for N in [3, 4]:
                    example_mapping = [{i: i for i in range(N)},
                                       {str(i): i for i in range(N)},
                                       {(str(i), 'A'): i for i in range(N)}]
                    example_parameters = \
                        [{"t": t * np.ones((N,N)), "mu": mu, "sites": list(range(N)), "NN": list((i, i+1) for i in range(N - 1))},
                         {"t": t * np.ones((N,N)), "mu": mu, "sites": [str(i) for i in range(N)], "NN": list((str(i), str(i+1)) for i in range(N - 1))},
                         {"t": t * np.ones((N,N)), "mu": mu, "sites": [(str(i),'A') for i in range(N)], "NN": list(((str(i), 'A'), (str(i+1), 'A')) for i in range(N - 1))}]

                    for (emap, eparam) in zip(example_mapping, example_parameters):
                        generate = mps.Generator(N, ops, map=emap)

                        H1 = generate.mpo_from_latex(H_str, eparam)
                        H2 = build_mpo_nn_hopping_manually(N=N, t=t, mu=mu, sym=sym, config=cfg)
                        H3 = mpo_nn_hopping_latex(N=N, t=t, mu=mu, sym=sym, config=cfg)

                        generate.random_seed(seed=0)
                        psi = generate.random_mps(D_total=8, n=0) + generate.random_mps(D_total=8, n=1)

                        x1 = mps.vdot(psi, H1, psi)
                        x2 = mps.vdot(psi, H2, psi)
                        x3 = mps.vdot(psi, H3, psi)
                        assert abs(x1.item() - x2.item()) < tol
                        assert abs(x1.item() - x3.item()) < tol


def test_mpo_from_latex():
    # the model is random with handom hopping and on-site energies. sym is symmetry for tensors we will use
    sym, N = 'U1', 5

    # generate set of basic ops for the model we want to work with
    ops = yastn.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)

    # generate data for random Hamiltonian
    J = np.random.rand(N, N)
    t = np.triu(J, 1)
    mu = np.diag(J)

    # create a generator initialized for emap mapping
    generate = mps.Generator(N, ops)

    # define parameters for automatic generator and Hamiltonian in a latex-like form
    eparam ={"t": t, "mu": mu, 'sites': list(range(N))}
    h_input = "\sum_{j\in sites} \sum_{k\in sites} t_{j,k} (cp_{j} c_{k} + cp_{k} c_{j}) + \
               \sum_{j\in sites} mu_{j} cp_{j} c_{j}"

    H1 = generate.mpo_from_latex(h_input, eparam)
    H2 = mpo_hopping_Hterm(J, sym=sym, config=cfg)
    H3 = mpo_hopping_latex(J, sym=sym, config=cfg)

    tmp = mps.vdot(H1, H2) / (H1.norm() * H2.norm())
    assert pytest.approx(tmp.item(), rel=tol) == 1
    tmp = mps.vdot(H1, H3) / (H1.norm() * H3.norm())
    assert pytest.approx(tmp.item(), rel=tol) == 1


def test_mpo_from_templete():
    # the model is random with handom hopping and on-site energies. sym is symmetry for tensors we will use
    sym, N = 'U1', 3

    # generate set of basic ops for the model we want to work with
    ops = yastn.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)

    # generate data for random Hamiltonian
    amplitudes1 = np.random.rand(N, N)
    amplitudes1 = 0.5 * (amplitudes1 + amplitudes1.transpose())

    # use this map which is used for naming the sites in MPO
    # maps between iteractors and MPO
    emap = {i: i for i in range(N)}

    # create a generator initialized for emap mapping
    generate = mps.Generator(N, ops, map=emap)
    generate.random_seed(seed=0)

    # define parameters for automatic generator and Hamiltonian in a latex-like form
    eparam ={"A": amplitudes1, "sites": range(N)}
    h_input = "\sum_{j\in sites} \sum_{k\in sites} A_{j,k} cp_{j} c_{k}"

    # generate MPO from latex-like input
    h_str = generate.mpo_from_latex(h_input, eparam)

    # generate Hamiltonian manually
    man_input = []
    for n0 in emap.keys():
        for n1 in emap.keys():
            man_input.append(\
                mps._latex2term.single_term((('A',n0,n1), ('cp',n0), ('c',n1),)))
    h_man = generate.mpo_from_templete(man_input, eparam)

    # test the result by comparing expectation value for a steady state.
    # use random seed to generate mps
    generate.random_seed(seed=0)

    # generate mps and compare overlaps
    psi = generate.random_mps(D_total=8, n=0) + generate.random_mps( D_total=8, n=1)
    x_man = mps.measure_mpo(psi, h_man, psi).item()
    x_str = mps.measure_mpo(psi, h_str, psi).item()
    assert abs(x_man - x_str) < tol


def test_build_mpo_nn_hopping_manually():
    """ test example generating mpo by hand """
    N, t, mu = 10, 1.0, 0.1
    H = {}
    for sym in ['dense', 'Z2', 'U1']:
        H[sym] = build_mpo_nn_hopping_manually(N=N, t=t, mu=mu, sym=sym, config=cfg)

    H_Z2_dense = mps.Mpo(N=N)
    H_U1_dense = mps.Mpo(N=N)
    for n in range(N):
        H_Z2_dense[n] = H['Z2'][n].to_nonsymmetric()
        H_U1_dense[n] = H['U1'][n].to_nonsymmetric()

    assert (H_Z2_dense - H['dense']).norm() < tol
    assert (H_U1_dense - H['dense']).norm() < tol

    # test mpo energy with direct calculation of all terms
    for sym, n in [('Z2', (0,)), ('U1', (N // 2,))]:
        # SpinlessFermions do not support 'dense'
        ops = yastn.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        generate = mps.Generator(N, ops)
        psi = generate.random_mps(D_total=16, n=n).canonize_(to='last').canonize_(to='first')

        cp, c = ops.cp(), ops.c()

        epm = mps.measure_2site(psi, cp, c, psi)
        emp = mps.measure_2site(psi, c, cp, psi)
        en = mps.measure_1site(psi, cp @ c, psi)

        E1 = mps.measure_mpo(psi, H[sym], psi)
        E2 = t * sum(epm[(n, n+1)] - emp[(n, n+1)] for n in range(N - 1)) # minus due to fermionic=True
        E2 += mu * sum(en[n] for n in range(N))

        psi_dense = mps.Mps(N=N)  # test also dense Hamiltonian casting down state psi to dense tensors
        for n in range(N):
            psi_dense[n] = psi[n].to_nonsymmetric()
        E3 = mps.measure_mpo(psi_dense, H['dense'], psi_dense)

        assert pytest.approx(E1.item(), rel=tol) == E2.item()
        assert pytest.approx(E1.item(), rel=tol) == E3.item()


if __name__ == "__main__":
    test_generator_mpo()
    test_mpo_from_latex()
    test_mpo_from_templete()
    test_build_mpo_nn_hopping_manually()
