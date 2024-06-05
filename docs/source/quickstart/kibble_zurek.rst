Kibble-Zurek quench in 2D transverse-field Ising model
======================================================

This guide provides a quick overview of simulating the real-time evolution of
a Kibble-Zurek quench in 2D Ising model using **YASTN** library.
We show a minimal example of a system defined on a :math:`4{\times}4` lattice with open boundary conditions (OBC).
The Hamiltonian reads

.. math::

 H(s) = f(s) \sum_{\langle i, j \rangle} J_{i,j} \sigma^x_i \sigma^x_j - g(s) \sum_i \sigma^z_i,

where :math:`\sigma^x` and :math:`\sigma^z` are tandard Pauli matrices,
and we assume random couplings :math:`J_{i,j}`.
The amplitude of couplings is gradually turned on as :math:`f(s) = \sin(\pi (s - 0.5))+ 1`,
and the transverse field is gradually turned off as :math:`g(s) = 1 - \sin(\pi (s - 0.5))`.
The system is initialized in the ground state at :math:`s=0`, and the evolution ends upon reaching :math:`s=1`.
The quench rate is controlled by an annealing time :math:`t_a` as :math:`s= t / t_a`.

We compare the results obtained using MPS and PEPS routines.

1. *Initialization of Model Parameters*:
    .. code-block:: python

        import numpy as np
        import yastn
        import yastn.tn.mps as mps
        import yastn.tn.fpeps as peps
        from yastn.tn.fpeps.gates import gate_nn_Ising, gate_local_Ising
        #
        # Employ PEPS lattice geometry for sites and bonds
        Lx, Ly = 4, 4  # lattice size
        geometry = fpeps.SquareLattice(dims=(Lx, Ly), boundary='obc')
        sites = geometry.sites()
        #
        # Draw random couplings
        np.random.seed(seed=0)
        Jij = {k: 2 * np.random.rand() - 1 for k in geometry.bonds()}
        #
        # Define quench protocol
        fXX = lambda s : np.sin((s - 0.5) * np.pi) + 1
        fZ = lambda s :  1 - np.sin((s - 0.5) * np.pi)
        ta = 2.0  # annealing time
        dt = 0.02  # time step
        steps = round(ta / dt)
        dt = ta / steps
        #
        # Load operators. Problem has Z2 symmetry, which we impose.
        ops = yastn.operators.Spin12(sym='Z2')

2. *PEPS simulations; time evolution*:
    .. code-block:: python

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
                gt = gate_local_Ising(fZ(s), dt2, ops.I(), ops.z(), site)
                local.append(gt)
            return fpeps.Gates(nn=nn, local=local)
        #
        # Initialize system in the product ground state at s=0.
        psi = fpeps.product_peps(geometry=geometry, vectors=ops.vec_z(val=1))
        #
        # simulation parameters
        D = 6  # PEPS bond dimension
        opts_svd_ntu = {"D_total": D, "D_block": D // 2}
        env = fpeps.EnvNTU(psi, which='NN+')
        #
        # execute time evolution
        infoss, t = [], 0
        for _ in range(steps):
            t += dt / 2
            gates = gates_Ising(Jij, fXX, fZ, t / ta, dt, sites, ops)
            infos = fpeps.evolution_step_(env, gates, opts_svd=opts_svd_ntu)
            infoss.append(infos)
            t += dt / 2
            print(f"s = {t / ta:0.4f}")
        Delta = fpeps.accumulated_truncation_error(infoss, statistics='mean')
        print(f"Accumulated truncation error {Delta:0.8f}")

3. *PEPS simulations; final correlations*:
    .. code-block:: python

        # We employ boundary MPS to contract the network
        opts_svd_env = {'D_total': 4 * D}
        opts_var_env = {"max_sweeps": 8,
                        "overlap_tol": 1e-5,
                        "Schmidt_tol": 1e-5}
        #
        # setting-up environment
        env = fpeps.EnvBoundaryMps(psi,
                                   opts_svd=opts_svd_env,
                                   opts_var=opts_var_env, setup='lr')
        #
        # Calculating 1-site <Z_i> for all sites
        Ez_peps = env.measure_1site(ops.z())
        #
        # Calculating 2-site <X_i X_j> for all pairs
        Exx_peps = env.measure_2site(ops.x(), ops.x(),
                                    opts_svd=opts_svd_env,
                                    opts_var=opts_var_env)
        #
        # remove diagonal
        Exx_peps = {bd: v for bd, v in Exx_peps.items() if bd[0] != bd[1]}

4. *MPS simulations*:
    .. code-block:: python

        # map between sites and linear MPS ordering.
        i2s = {i: s for i, s in enumerate(sites)}
        s2i = {s: i for i, s in enumerate(sites)}
        b2i = lambda (s1, s2): tuple(sorted([s2i[s1], s2i[s2]]))
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
        # Initial state; product state via dmrg_
        # TDVP is unstable staring in a product state
        # We make bond dimension artificially large
        psi = mps.random_mps(I, D_total=8)
        mps.dmrg_(psi, H(0), method='1site', max_sweeps=8, Schmidt_tol=1e-8)
        #
        # time-evoluion parametters
        Dmax = 128
        opts_expmv = {'hermitian': True, 'tol': 1e-12}
        opts_svd = {'tol': 1e-6, 'D_total': Dmax}
        evol = mps.tdvp_(psi, H, times=(0, ta),
                        method='12site', dt=dt, order='2nd',
                        opts_svd=opts_svd, opts_expmv=opts_expmv):
        #
        # run evolution; # evol is a generaor, with one final snapshot
        next(evol)
        #
        # calculate expectation values
        Ez_mps = mps.measure_1site(psi, ops.z(), psi)
        Exx_mps = mps.measure_2site(psi, ops.x(), ops.x(), psi)

5. *Compare results of PEPS and MPS*:
    .. code-block:: python

        Z1 = np.array([Ez_peps[st].real for st in sites])
        Z2 = np.array([Ez_mps[s2i[st]].real for st in sites])

        XX1 = np.array([Exx_peps.values()])
        XX2 = np.array([Exx_mps[b2i[bond]] for bond in Exx_peps.keys()])
