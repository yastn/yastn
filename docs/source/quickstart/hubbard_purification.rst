2D Fermi-Hubbard model at finite temperature
============================================

This guide provides a quick overview of how to simulate the Fermi-Hubbard model at finite temperature
and on the infinite square lattice using purification and the **YASTN** tensor network library.
We'll focus on initializing the model, setting up the simulation, and calculating expectation values.

The Fermi-Hubbard model describes interacting electrons on a lattice,
where the kinetic energy due to electron hopping competes with on-site Coulomb interaction.
The model's Hamiltonian is expressed as:

.. math::

    H = -t \sum_{\langle i, j \rangle, \sigma} (c_{i, \sigma}^\dagger c_{j, \sigma} + c_{j, \sigma}^\dagger c_{i, \sigma}) + U \sum_i \left( n_{i, \uparrow} - \frac{1}{2} \right) \left(n_{i, \downarrow} - \frac{1}{2} \right) - \mu \sum_{i, \sigma} n_{i, \sigma},

where:
    - :math:`t` is the hopping amplitude,
    - :math:`U` is the on-site interaction strength,
    - :math:`\mu` is the chemical potential,
    - :math:`c_{i, \sigma}^\dagger` and :math:`c_{i, \sigma}` are the creation and annihilation operators at site :math:`i` with spin :math:`\sigma`,
    - :math:`n_{i, \sigma} = c_{i, \sigma}^\dagger c_{i, \sigma}` is the number operator for electrons at site :math:`i` with spin :math:`\sigma`.


1. *Initialization of Model Parameters*:
    We set our model parameters keeping in mind that in the purification there is no clear way to fix particle number
    and must be controlled by changing chemical potential :math:`\mu`.

    .. code-block:: python

        import yastn
        import yastn.tn.fpeps as peps
        from tqdm import tqdm  # progressbar

        t = 1
        mu = 0
        U = 10
        beta = 0.5  # inverse temperature

2. *Local operators*
    .. code-block:: python

        ops = yastn.operators.SpinfulFermions(sym='U1xU1')
        I = ops.I()
        c_up, cdag_up, n_up = ops.c('u'), ops.cp('u'), ops.n('u')
        c_dn, cdag_dn, n_dn = ops.c('d'), ops.cp('d'), ops.n('d')


3. *Lattice Geometry and Initial Product State Creation*:
    .. code-block:: python

        geometry = peps.CheckerboardLattice()
        # for bigger unit cells, set
        # geometry = peps.SquareLattice(dims=(m, n))
        # for finite lattice, set
        # geometry = peps.SquareLattice(dims=(m, n), boundary='finite')
        #
        # purification at infinite temperature (unnormalized)
        psi = peps.product_peps(geometry=geometry, vectors=I)


4. *Hamiltonian Gates Definition*:
    .. code-block:: python

        db = 0.01  # Trotter step size
        # making sure we have integer number of steps to target beta / 2
        steps = round((beta / 2) / db)
        db = (beta / 2) / steps
        #
        g_hop_u = peps.gates.gate_nn_hopping(t, db/2, I, c_up, cdag_up)
        g_hop_d = peps.gates.gate_nn_hopping(t, db/2, I, c_dn, cdag_dn)
        g_loc = peps.gates.gate_local_Coulomb(mu, mu, U, db/2, I, n_up, n_dn)
        gates = peps.gates.distribute(geometry,
                                      gates_nn=[g_hop_u, g_hop_d],
                                      gates_local=g_loc)


5. *Time Evolution*:
    .. code-block:: python

        env = peps.EnvNTU(psi, which='NN')
        # this is set up for neighborhood tensor update optimization
        # as described in https://arxiv.org/pdf/2209.00985.pdf

        D = 12  # bond dimenson

        opts_svd = {'D_total': D, 'tol': 1e-12}
        infoss = []  # for evolution diagnostics
        #
        for step in tqdm(range(1, steps + 1)):
            infos = peps.evolution_step_(env, gates, opts_svd=opts_svd)
            # The state psi is contained in env;
            # evolution_step_ updates psi in place.
            #
            infoss.append(infos)
        #
        Delta = fpeps.accumulated_truncation_error(infoss)
        print(f"Accumulated truncation error {Delta:0.5f}")


5. *CTMRG and Expectation Values*:
    This part sets up CTMRG procedure for calculating corners and
    transfer matrices to be used to calculate any expectation value.
    It can accessed through an instance of peps.EnvCTM class.
    Here, the convergence criterion is based on total energy.

    .. code-block:: python

        env_ctm = peps.EnvCTM(psi, init='eye')
        opts_svd_ctm = {'D_total': 5 * D, 'tol': 1e-10}  # chi = 5 * D

        mean = lambda data: sum(data) / len(data)  # helper function

        energy_old, tol_exp = 0, 1e-7
        for i in range(50):
            #
            env_ctm.update_(opts_svd=opts_svd_ctm)  # single CMTRG sweep
            #
            # calculate energy expectation value
            #
            # calculate for all unique sites; {site: value}
            ev_nn = env_ctm.measure_1site((n_up - I / 2) @ (n_dn - I / 2))
            ev_nn = mean([*ev_nn.values()])  # mean over all sites
            #
            # calculate for all unique bonds; {bond: value}
            ev_cdagc_up = env_ctm.measure_nn(cdag_up, c_up)
            ev_cdagc_dn = env_ctm.measure_nn(cdag_dn, c_dn)
            ev_cdagc_up = mean([*ev_cdagc_up.values()])  # mean over bonds
            ev_cdagc_dn = mean([*ev_cdagc_dn.values()])  # mean over bonds
            #
            energy = -4 * t * (ev_cdagc_up + ev_cdagc_dn) + U * ev_nn
            #
            print(f"Energy per site after iteration {i}: {energy:0.8f}")
            if abs(energy - energy_old) < tol_exp:
                break
            energy_old = energy

        # Energy per site after iteration 0: -2.36130904
        # Energy per site after iteration 1: -2.36554935
        # Energy per site after iteration 2: -2.36557284
        # Energy per site after iteration 3: -2.36557295
        # Energy per site after iteration 4: -2.36557295


6. *Specific Expectation Values*:
    Now we move to calculate expectation values of interest.

    .. code-block:: python

        # average occupation of spin-polarization up and down
        ev_n_up = env_ctm.measure_1site(n_up)
        ev_n_dn = env_ctm.measure_1site(n_dn)
        ev_n_up = mean([*ev_n_up.values()])
        ev_n_dn = mean([*ev_n_dn.values()])
        print(f"Occupation spin up: {ev_n_up:0.8f}")
        print(f"Occupation spin dn: {ev_n_dn:0.8f}")
        # occupation spin up:  0.50000000
        # occupation spin dn:  0.50000000

        print("kinetic energy per bond")
        print(f"spin up electrons: {2 * ev_cdagc_up:0.6f}")
        print(f"spin dn electrons: {2 * ev_cdagc_dn:0.6f}")
        # Kinetic energy per bond
        # spin up electrons: 0.123385
        # spin dn electrons: 0.122360

        ev_double = env_ctm.measure_1site(n_up @ n_dn)
        ev_double = mean([*ev_double.values()])
        print(f"Average double occupancy: {ev_double:0.6f}")
        # Average double occupancy: 0.062592

        Sz = 0.5 * (n_up - n_dn)  # Sz operator
        ev_SzSz = env_ctm.measure_nn(Sz, Sz)
        ev_SzSz = mean([*ev_SzSz.values()])
        print(f"Average NN spin-spin correlator: {ev_SzSz:0.6f}")
        # Average NN spin-spin correlator: -0.006933
