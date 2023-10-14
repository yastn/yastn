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
def mpo_nn_hopping_manually(N=10, t=1.0, mu=0.0, config=None):
    """
    Nearest-neighbor hopping Hamiltonian on N sites with hopping amplitude t and chemical potential mu.
    i.e. sum_{n=1}^{N-1} t * (sp_n sm_{n+1} + sp_{n+1} sm_n) + sum_{n=1}^{N} mu * sp_n sm_n"

    Initialize MPO tensor by hand with dense, Z2, or U1 symmetric tensors.
    Symmetry is specified in config.
    """
    #
    # Build empty MPO for system of N sites
    #
    H = mps.Mpo(N)
    #
    # Depending on the symmetry, define elements of on-site tensor
    #
    # We chose signature convention for indices of the MPO tensor as follows
    #          |
    #          ^(-1)
    #          |
    # (-1) -<-|T|-<-(+1)
    #          |
    #          ^(+1)
    #          |

    if config is None:
        config = yastn.make_config()  # default is no symmetry, i.e. dense.

    if config.sym.SYM_ID == 'dense':  # no symmetry
        # Basic rank-2 blocks (matrices) of on-site tensors
        cp = np.array([[0, 0], [1, 0]])
        c  = np.array([[0, 1], [0, 0]])
        nn = np.array([[0, 0], [0, 1]])
        ee = np.array([[1, 0], [0, 1]])
        oo = np.array([[0, 0], [0, 0]])

        for n in H.sweep(to='last'):  # empty tensors
            H[n] = yastn.Tensor(config=config, s=(-1, 1, 1, -1))
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

    elif config.sym.SYM_ID == 'Z2':  # Z2 symmetry
        for n in H.sweep(to='last'):
            H[n] = yastn.Tensor(config=config, s=(-1, 1, 1, -1), n=0)
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

    elif config.sym.SYM_ID == 'U(1)':  # U1 symmetry
        for n in H.sweep(to='last'):
            H.A[n] = yastn.Tensor(config=config, s=(-1, 1, 1, -1), n=0)
            if n == H.first:
                H.A[n].set_block(ts=(0, 0, 0, 0), val=[0, 1], Ds=(1, 1, 2, 1))
                H.A[n].set_block(ts=(0, 1, 0, 1), val=[mu, 1], Ds=(1, 1, 2, 1))
                H.A[n].set_block(ts=(0, 0, 1, 1), val=[t], Ds=(1, 1, 1, 1))
                H.A[n].set_block(ts=(0, 1, -1, 0), val=[t], Ds=(1, 1, 1, 1))
            elif n == H.last:
                H.A[n].set_block(ts=(0, 0, 0, 0), val=[1, 0], Ds=(2, 1, 1, 1))
                H.A[n].set_block(ts=(0, 1, 0, 1), val=[1, mu], Ds=(2, 1, 1, 1))
                H.A[n].set_block(ts=(1, 1, 0, 0), val=[1], Ds=(1, 1, 1, 1))
                H.A[n].set_block(ts=(-1, 0, 0, 1), val=[1], Ds=(1, 1, 1, 1))
            else:
                H.A[n].set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 2, 1))
                H.A[n].set_block(ts=(0, 1, 0, 1), val=[[1, 0], [mu, 1]], Ds=(2, 1, 2, 1))
                H.A[n].set_block(ts=(0, 0, 1, 1), val=[0, t], Ds=(2, 1, 1, 1))
                H.A[n].set_block(ts=(0, 1, -1, 0), val=[0, t], Ds=(2, 1, 1, 1))
                H.A[n].set_block(ts=(1, 1, 0, 0), val=[1, 0], Ds=(1, 1, 2, 1))
                H.A[n].set_block(ts=(-1, 0, 0, 1), val=[1, 0], Ds=(1, 1, 2, 1))

    for n in H.sweep():
        H[n].config = H[n].config._replace(fermionic=True)  # change fermionic to True, as those Hamiltonians will be tested again SpinlessFermions
    return H


def mpo_hopping_Hterm(N, J, sym="U1", config=None):
    """
    Fermionic hopping Hamiltonian on N sites with hoppings at arbitrary range.

    The upper triangular part of matrix J defines hopping amplitudes,
    and the diagonal defines on-site chemical potentials.
    """

    if config is None:
        ops = yastn.operators.SpinlessFermions(sym=sym)
    else:  # config is used here by pytest to inject backend and device for testing
        ops = yastn.operators.SpinlessFermions(sym=sym, backend=config.backend, default_device=config.default_device)

    Hterms = []  # list of Hterm(amplitude, positions, operators)
    # Each Hterm corresponds to a single product of local operators.
    # Hamiltonian is a sum of such products.

    # chemical potential on site n
    for n in range(N):
        if abs(J[n][n]) > 0:
            Hterms.append(mps.Hterm(J[n][n], [n], [ops.n()]))

    # hopping term between sites m and n
    for m in range(N):
        for n in range(m + 1, N):
            if abs(J[m][n]) > 0:
                Hterms.append(mps.Hterm(J[m][n], (m, n), (ops.cp(), ops.c())))
                Hterms.append(mps.Hterm(np.conj(J[m][n]), (n, m), (ops.cp(), ops.c())))

    # We need an identity MPO operator. Here it is created manually.
    I = mps.Mpo(N)
    for n in I.sweep():
        I[n] = ops.I().add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    #
    # Identity MPO can be also obtained using mps.Generator class
    #
    # generator = mps.Generator(N, ops)
    # I = generator.I()
    #

    #
    # Generate MPO for Hterms
    #
    H = mps.generate_mpo(I, Hterms, opts={'tol':1e-14})
    return H


def mpo_nn_hopping_latex(N=10, t=1.0, mu=0.0, sym="U1", config=None):
    """
    Nearest-neighbor hopping Hamiltonian on N sites with hopping amplitude t and chemical potential mu.

    The upper triangular part of matrix J defines hopping amplitudes,
    and the diagonal defines on-site chemical potentials.
    """

    if config is None:
        ops = yastn.operators.SpinlessFermions(sym=sym)
    else:  # config is used here by pytest to inject backend and device for testing
        ops = yastn.operators.SpinlessFermions(sym=sym, backend=config.backend, default_device=config.default_device)

    Hstr = "\sum_{j,k \in NN} t (cp_{j} c_{k}+cp_{k} c_{j}) + \sum_{i \in sites} mu cp_{i} c_{i}"
    parameters = {"t": t, "mu": mu, "sites": list(range(N)), "NN": list((i, i+1) for i in range(N-1))}
    generate = mps.Generator(N, ops)
    H = generate.mpo_from_latex(Hstr, parameters=parameters)
    return H


def mpo_hopping_latex(N, J, sym="U1", config=None):
    """
    Nearest-neighbor hopping Hamiltonian on N sites with hopping amplitude t and chemical potential mu.

    The upper triangular part of matrix J defines hopping amplitudes,
    and the diagonal defines on-site chemical potentials.
    """

    if config is None:
        ops = yastn.operators.SpinlessFermions(sym=sym)
    else:  # config is used here by pytest to inject backend and device for testing
        ops = yastn.operators.SpinlessFermions(sym=sym, backend=config.backend, default_device=config.default_device)

    Hstr = "\sum_{j,k \in NN} J_{j,k} (cp_{j} c_{k}+cp_{k} c_{j}) + \sum_{i \in sites} J_{i,i} cp_{i} c_{i}"
    parameters = {"J": J, "sites": list(range(N)), "NN": list((i, j) for i in range(N-1) for j in range(i + 1, N))}

    generate = mps.Generator(N, ops)
    H = generate.mpo_from_latex(Hstr, parameters=parameters)
    return H


def random_mps_spinless_fermions(N=10, D_total=16, sym='Z2', n=1, config=None):
    """
    Generate random MPS of N sites, with bond dimension D_total, tensors with symmetry sym and total charge n.
    """
    if config is None:
        ops = yastn.operators.SpinlessFermions(sym=sym)
    else:  # config is used here by pytest to inject backend and device for testing
        ops = yastn.operators.SpinlessFermions(sym=sym, backend=config.backend, default_device=config.default_device)

    generate = mps.Generator(N, ops)
    psi = generate.random_mps(D_total=D_total, n=n)
    return psi


def random_mpo_spinless_fermions(N=10, D_total=16, sym='Z2', config=None):
    """
    Generate random MPO of N sites, with bond dimension D_total and tensors with symmetry sym.
    """
    if config is None:
        ops = yastn.operators.SpinlessFermions(sym=sym)
    else:  # config is used here by pytest to inject backend and device for testing
        ops = yastn.operators.SpinlessFermions(sym=sym, backend=config.backend, default_device=config.default_device)

    generate = mps.Generator(N, ops)
    H = generate.random_mpo(D_total=D_total)
    return H


def test_generate_random_mps():
    N = 10
    D_total = 16
    bds = (1,) + (D_total,) * (N - 1) + (1,)

    for sym, nn in (('Z2', (0,)), ('Z2', (1,)), ('U1', (N // 2,))):
        ops = yastn.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        generate = mps.Generator(N, ops)
        I = generate.I()
        assert pytest.approx(mps.measure_overlap(I, I).item(), rel=tol) == 2 ** N
        O = I @ I + (-1 * I)
        assert pytest.approx(O.norm().item(), abs=tol) == 0

        n0 = (0,) * len(nn)
        psi = random_mps_spinless_fermions(N, D_total, sym, nn)
        leg = psi[psi.first].get_legs(axes=0)
        assert leg.t == (nn,) and leg.s == -1
        leg = psi[psi.last].get_legs(axes=2)
        assert leg.t == (n0,) and leg.s == 1
        bds = psi.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])

        H = random_mpo_spinless_fermions(N, D_total, sym)
        leg = H[H.first].get_legs(axes=0)
        assert leg.t == (n0,) and leg.s == -1
        leg = H[H.last].get_legs(axes=2)
        assert leg.t == (n0,) and leg.s == 1
        bds = H.get_bond_dimensions()
        assert bds[0] == bds[-1] == 1
        assert all(bd > D_total/2 for bd in bds[2:-2])


def test_generate_product_mps():
    """ test mps.generate_prod_mps"""
    for sym, nl, nr in [('U1', (1,), (0,)), ('Z2', (1,), (0,)), ('dense', (), ())]:
        ops = yastn.operators.Spin12(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        vp1 = ops.vec_z(val=+1)
        vm1 = ops.vec_z(val=-1)

        psi = mps.generate_product_mps(vectors=[vp1, vm1, vp1, vm1, vm1, vp1, vp1])

        assert pytest.approx(mps.vdot(psi, psi).item(), rel=tol) == 1.0
        assert psi.virtual_leg('first').t == (nl,)
        assert psi.virtual_leg('last').t == (nr,)
        assert mps.measure_1site(psi, ops.z(), psi) == {0: +1.0, 1: -1.0, 2: +1.0, 3: -1.0, 4: -1.0, 5: +1.0, 6: +1.0}

    for sym, ntot in [('U1', (4,)), ('Z2', (0,))]:
        ops = yastn.operators.SpinlessFermions(sym=sym, backend=cfg.backend, default_device=cfg.default_device)
        v0 = ops.vec_n(val=0)
        v1 = ops.vec_n(val=1)

        psi = mps.generate_product_mps(vectors=[v1, v0, v1, v0, v0, v1, v1])

        assert pytest.approx(mps.vdot(psi, psi).item(), rel=tol) == 1.0
        assert psi.virtual_leg('first').t == (ntot,)
        assert mps.measure_1site(psi, ops.n(), psi) == {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 1.0}


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
                        H2 = mpo_nn_hopping_manually(N=N, t=t, mu=mu, config=generate.config)
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
    H2 = mpo_hopping_Hterm(N, J, sym=sym, config=cfg)
    H3 = mpo_hopping_latex(N, J, sym=sym, config=cfg)

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


def mps_basis_ex(config):
    plus = yastn.Tensor(config=config, s=[1])
    plus.set_block(val=[0, 1],Ds=(2,))
    minus = yastn.Tensor(config=config, s=[1])
    minus.set_block(val=[1, 0],Ds=(2,))
    return plus, minus


def mpo_basis_ex(config):
    cpc = yastn.Tensor(config=config, s=[1, -1])
    cpc.set_block(val=[[0,0],[0,1]],Ds=(2,2,))
    ccp = yastn.Tensor(config=config, s=[1, -1])
    ccp.set_block(val=[[1,0],[0,0]],Ds=(2,2,))
    I = yastn.Tensor(config=config, s=[1, -1])
    I.set_block(val=[[1,0],[0,1]],Ds=(2,2,))
    return cpc, ccp, I


def test_mpo_nn_example():
    """ test example generating mpo by hand """
    N, t, mu = 10, 1.0, 0.1
    H = {}
    H['dense'] = mpo_nn_hopping_manually(N=N, t=t, mu=mu, config=cfg)
    H['Z2'] = mpo_nn_hopping_manually(N=N, t=t, mu=mu, config=config_Z2)
    H['U1'] = mpo_nn_hopping_manually(N=N, t=t, mu=mu, config=config_U1)

    H_Z2_dense = mps.Mpo(N=N)
    H_U1_dense = mps.Mpo(N=N)
    for n in range(N):
        H_Z2_dense[n] = H['Z2'][n].to_nonsymmetric()
        H_U1_dense[n] = H['U1'][n].to_nonsymmetric()

    assert (H_Z2_dense - H['dense']).norm() < tol
    assert (H_U1_dense - H['dense']).norm() < tol

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
        E2 = t * sum(epm[(n, n+1)] - emp[(n, n+1)] for n in range(N - 1))
        E2 += mu * sum(en[n] for n in range(N))

        psi_dense = mps.Mps(N=N)  # test also dense Hamiltonian casting down state psi to dense tensors
        for n in range(N):
            psi_dense[n] = psi[n].to_nonsymmetric()
        E3 = mps.measure_mpo(psi_dense, H['dense'], psi_dense)

        print(E1, E2, E3)
        assert pytest.approx(E1.item(), rel=tol) == E2.item()
        assert pytest.approx(E1.item(), rel=tol) == E3.item()


if __name__ == "__main__":
    test_mpo_nn_example()
    test_generate_random_mps()
    test_generate_product_mps()
    test_generator_mpo()
    test_mpo_from_latex()
    test_mpo_from_templete()
