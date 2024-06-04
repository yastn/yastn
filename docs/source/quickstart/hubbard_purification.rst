Purifcation for Fermi-Hubbard Model Quickstart
==============================================

This guide provides a quick overview of how to simulate the Fermi-Hubbard model at finite temperature using the `YASTN`
tensor network library for an infinite square lattice. We'll focus on initializing the model, setting up the simulation,
and calculating expectation values.


Model Overview
--------------

The Fermi-Hubbard model describes interacting electrons on a lattice, where the kinetic energy due to electron hopping competes with on-site Coulomb interaction. The model's Hamiltonian is expressed as:

.. math::

    H = -t \sum_{\langle i, j \rangle, \sigma} (c_{i, \sigma}^\dagger c_{j, \sigma} + h.c.) + U \sum_i \left( n_{i, \uparrow} - \frac{1}{2} \right) \left(n_{i, \downarrow} - \frac{1}{2} \right) - \sum_{i, \sigma} \mu_\sigma n_{i, \sigma},

where:
    - :math:`t` is the hopping amplitude,
    - :math:`U` is the on-site interaction strength,
    - :math:`\mu` is the chemical potential,
    - :math:`c_{i, \sigma}^\dagger` and :math:`c_{i, \sigma}` are the creation and annihilation operators at site :math:`i` with spin :math:`\sigma`,
    - :math:`n_{i, \sigma} = c_{i, \sigma}^\dagger c_{i, \sigma}` is the number operator for electrons at site :math:`i` with spin :math:`\sigma`.


Simulation Setup
----------------

1. *Initialization of Model Parameters*:
    We set our model parameters keeping in mind that in the purification there is no clear way to fix particle number
    and must be controlled by changing chemical potential :math:`\mu`.

    .. code-block:: python

        import yastn
        import yastn.tn.fpeps as fpeps

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

        geometry = fpeps.CheckerboardLattice()

        # for bigger unit cells, set
        # geometry = fpeps.SquareLattice(dims=(m, n))
        # for finite lattice, set
        # geometry = fpeps.SquareLattice(dims=(m, n), boundary='finite')

        # purification at infinite temperature (unnormalized)
        psi = fpeps.product_peps(geometry=geometry, vectors=I)


4. *Hamiltonian Gates Definition*:
    .. code-block:: python

        db = 0.01  # Trotter step size
        steps = round((beta / 2) / db)
        db = (beta / 2) / steps

        g_hop_u = fpeps.gates.gate_nn_hopping(t, db/2, I, c_up, cdag_up)
        g_hop_d = fpeps.gates.gate_nn_hopping(t, db/2, I, c_dn, cdag_dn)
        g_loc = fpeps.gates.gate_local_Coulomb(mu, mu, U, db/2, I, n_up, n_dn)
        gates = fpeps.gates.distribute(geometry,
                                       gates_nn=[g_hop_u, g_hop_d],
                                       gates_local=g_loc)


5. *Time Evolution*:
    .. code-block:: python

        env = fpeps.EnvNTU(psi, which='NN')
        # this is set up for neighborhood tensor update optimization
        # as described in https://arxiv.org/pdf/2209.00985.pdf

        D = 12  # bond dimenson

        opts_svd = {'D_total': D, 'tol': 1e-12}
        infoss = []
        for step in range(1, steps + 1):
            print(f"beta_purification = {step * db:0.3f}" )
            infos = fpeps.evolution_step_(env, gates, opts_svd=opts_svd)
            infoss.append(info)
        Delta = fpeps.accumulated_truncation_error(infoss)
        print(f"Accumulated mean truncation error: {Delta:0.3f}")

5. *CTMRG and Expectation Values*:
    .. code-block:: python

        # This part sets up CTMRG procedure for calculating corners and
        # transfer matrices to be used to calculate any expectation value.
        # It can accessed through an instance of fpeps.EnvCTM class.
        # Here, the convergence criterion is based on total energy.

        env_ctm = fpeps.EnvCTM(psi, init='rand')
        chi = 5 * D
        opts_svd_ctm = {'D_total': chi, 'tol': 1e-10}

        mean = lambda data: sum(data) / len(data)

        energy_old, tol_exp = 0, 1e-7
        for i in range(50):
            #
            env_ctm.update_(opts_svd=opts_svd_ctm)  # single CMTRG sweep
            #
            # calculate energy expectation value
            #
            ev_nn = env_ctm.measure_1site((n_up - I / 2) @ (n_dn - I / 2))
            # calculate for all unique sites; {site: value}
            #
            ev_cdagc_up = env_ctm.measure_nn(cdag_up, c_up)
            ev_cdagc_dn = env_ctm.measure_nn(cdag_dn, c_dn)
            ev_ccdag_up = env_ctm.measure_nn(c_up, cdag_up)
            ev_ccdag_dn = env_ctm.measure_nn(c_dn, cdag_dn)
            # calculate for all unique bonds; {bond: value}
            #
            energy = U * mean([*ev_nn.values()])  # mean over all sites
            energy += -4 * t * mean([*ev_cdagc_up.values()]) # mean over bonds
            energy += -4 * t * mean([*ev_cdagc_dn.values()])
            #
            print(f"Energy per site after iteration {i}: {energy:0.8f}")
            if abs(energy - energy_old) < tol_exp:
                break
            energy_old = energy

6. *Terminal Output Showing Convergence of Energy Calculations*:
    .. code-block:: none

        Energy after iteration 1:  -0.36401150344639244
        Energy after iteration 2:  -0.35722388043232156
        Energy after iteration 3:  -0.3570652371408988
        Energy after iteration 4:  -0.3570627502958944
        Energy after iteration 5:  -0.357062698531201

7. *Specific Expectation Values*:
    Now we move to calculate expectation values of interest.
    We have commands followed by its terminal output.

    .. code-block:: python

        # average occupation of spin-polarization up and down
        ev_n_up = env_ctm.measure_1site(n_up)
        ev_n_dn = env_ctm.measure_1site(n_dn)
        print("occupation spin up: ", mean([*ev_n_up.values()]))
        print("occupation spin dn: ", mean([*ev_n_dn.values()]))

    .. code-block:: none

        occupation spin up:  0.5000000004102714
        occupation spin dn:  0.4999999997308221

    .. code-block:: python

        print("kinetic energy per bond")
        print("spin up electrons: ", 2 * mean([*ev_cdagc_up.values(), *ev_ccdag_up.values()]))
        print("spin dn electrons: ", 2 * mean([*ev_cdagc_dn.values(), *ev_ccdag_dn.values()]))

    .. code-block:: none

        kinetic energy per bond
        spin up electrons:  0.06169239676196566
        spin dn electrons:  0.06118004385332907

    .. code-block:: python

        print("average double occupancy ", np.mean([*ev_nn.values()]))

    .. code-block:: none

        average double occupancy  0.06259168263911569

    .. code-block:: python

        sz = 0.5 * (n_up - n_dn)   # sz operator
        # calculate for all unique bonds
        ev_szsz = env_ctm.measure_nn(sz, sz)

        print("average NN spin-spin correlator ", mean([*ev_szsz.values()]))

    .. code-block:: none

        average NN spin-spin correlator  -0.0069327726073487505





