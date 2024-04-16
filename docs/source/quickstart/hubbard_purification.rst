Purifcation for Fermi-Hubbard Model Quickstart
==============================================

This guide provides a quick overview on how to simulate the Fermi-Hubbard model at finite temperature using the `YASTN`
tensor network library for an infinite square lattice. We'll focus on initializing the model, setting up the simulation, 
and calculating expectation values.


Model Overview
--------------

The Fermi-Hubbard model describes interacting electrons on a lattice, where the kinetic energy due to electron hopping competes with on-site Coulomb interaction. The model's Hamiltonian is expressed as:

.. math::

    H = -t \sum_{\langle i, j \rangle, \sigma} (c_{i, \sigma}^\dagger c_{j, \sigma} + h.c.) + U \sum_i n_{i, \uparrow} n_{i, \downarrow} - \mu \sum_{i, \sigma} n_{i, \sigma}

where:
- :math:`t` is the hopping amplitude,
- :math:`U` is the on-site interaction strength,
- :math:`\mu` is the chemical potential,
- :math:`c_{i, \sigma}^\dagger` and :math:`c_{i, \sigma}` are the creation and annihilation operators at site :math:`i` with spin :math:`\sigma`,
- :math:`n_{i, \sigma} = c_{i, \sigma}^\dagger c_{i, \sigma}` is the number operator for electrons at site :math:`i` with spin :math:`\sigma`.


Simulation Setup
----------------

1. **Initialization of Model Parameters**:

    We set our model parameters keeping in mind that in purification there is no clear way to fix particle number 
    and must be controlled by changing chemical potention :math:`\mu`.

    .. code-block:: python

        mu = 0
        t_up = 1
        t_dn = 1
        mu_up = 0
        mu_dn = 0
        beta = 0.5 # Inverse temperature
        U = 10


Note that we have the freedom to set different tunneling amplitudes of different elctron polarizations: 't_up' and 't_dn'
as 'mu_up' and 'mu_dn' can be set diiferent to get different fillings for up and down polarizations.

2. **Lattice Geometry and Initial State Creation**:

    .. code-block:: python

        import yastn.tn.fpeps as fpeps
        geometry = fpeps.CheckerboardLattice()
    
        # for bigger unit cells, set 'geometry = fpeps.SquareLattice(dims=(m,n))'
        # for finite lattice, set 'geometry = fpeps.SquareLattice(dims=(m,n), boundary='finite')

        ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2')
        psi = fpeps.product_peps(geometry=geometry, vectors = ops.I())

3. **Hamiltonian Gates Definition**:


   .. code-block:: python

       dbeta = 0.01  # Trotter step size
       fid = ops.I()
       fc_up, fc_dn, fcdag_up, fcdag_dn = ops.c('u'), ops.c('d'), ops.cp('u'), ops.cp('d')
       n_up, n_dn = ops.n('u'), ops.n('d')
       n_int = n_up @ n_dn
       g_hop_u = fpeps.gates.gate_nn_hopping(t_up, dbeta / 2, fid, fc_up, fcdag_up)
       g_hop_d = fpeps.gates.gate_nn_hopping(t_dn, dbeta / 2, fid, fc_dn, fcdag_dn)
       g_loc = fpeps.gates.gate_local_Coulomb(mu_up, mu_dn, U, dbeta / 2, fid, n_up, n_dn)
       gates = fpeps.gates.distribute(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_loc)

4. **Optimzation procedure**:

    .. code-block:: python

        env = fpeps.EnvNTU(psi, which='NN')
        # this is set up for neighborhood tensor update optimization as described in https://arxiv.org/pdf/2209.00985.pdf


5. **Time Evolution Setup**:


    .. code-block:: python

        D = 12  # bond dimenson

        opts_svd = {'D_total': D, 'tol_block': 1e-15}
        steps = np.rint((beta / 2) / dbeta).astype(int)
        for step in range(steps):
            print(f"beta = {(step + 1) * dbeta}" )
            evolution_results = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="EAT")
            print(f"Error after optimization for all gates: {evolution_results.truncation_error}")

6. **Calculation of Environmental Tensor for Expectation Value Calculations**:

    .. code-block:: python

        # This part sets up CTMRG procedure for calculating corner and transfer matrices to be used to calulate any expectation value.
        # Here it can accessed through an instance of EnvCTM class fpeps. The convergence criteria is based on total energy.

        chi = 80  # environmental bond dimension
        tol = 1e-10  # truncation of singular values of CTM projectors
        max_sweeps = 50
        tol_exp = 1e-7  # difference of some observable must be lower than tolernace

        energy_old, tol_exp = 0, 1e-7

        opts_svd_ctm = {'D_total': 40, 'tol': 1e-10}

        env_ctm = fpeps.EnvCTM(psi)

        for i in range(50):
            env_ctm.update_(opts_svd=opts_svd_ctm)  # single CMTRG sweep

            # calculate expectation values
            d_oc = env_ctm.measure_1site(n_int)
            cdagc_up = env_ctm.measure_nn(fcdag_up, fc_up)  # calculate for all unique bonds
            cdagc_dn = env_ctm.measure_nn(fcdag_dn, fc_dn)  # -> {bond: value}
            PEn = U * np.mean([*d_oc.values()]) 
            KEn = - 8 * (np.mean([*cdagc_up.values()]) + np.mean([*cdagc_dn.values()]))

            energy = PEn + KEn
            print(f"Energy after iteration {i+1}: ", energy)
            if abs(energy - energy_old) < tol_exp:
                break
            energy_old = energy

    **Terminal Output Showing Convergence of Energy Calculations**:

    .. code-block:: none

        Energy after iteration 1:  -0.36401150344639244
        Energy after iteration 2:  -0.35722388043232156
        Energy after iteration 3:  -0.3570652371408988
        Energy after iteration 4:  -0.3570627502958944
        Energy after iteration 5:  -0.357062698531201

7. **Specific Expectation Values**:
    

    Now we move on to calculate expectation values of interest. We have commands follwed by its terminal output.

    .. code-block:: python

        occupation_up = env_ctm.measure_1site(n_up) # average occupation of spin polarization up
        occupation_dn = env_ctm.measure_1site(n_dn) # average occupation of spin polarization up
        print("average occupation of spin-polarization up: ", np.mean([*occupation_up.values()]))
        print("average occupation of spin-polarization up: ", np.mean([*occupation_dn.values()]))

    .. code-block:: none

        average occupation of spin-polarization up:  0.5000000004102714
        average occupation of spin-polarization up:  0.4999999997308221

    .. code-block:: python

        sz = 0.5*(n_up - n_dn)   # sz operator
        correlation_sz_sz = env_ctm.measure_nn(sz, sz)  # calculate for all unique bonds
        print("kinetic energy per bond - up spin electrons ", np.mean([*cdagc_up.values()]))
        print("kinetic energy per bond - down spin electrons ", np.mean([*cdagc_dn.values()]))

    .. code-block:: none

        kinetic energy per bond - up spin electrons  0.06169239676196566
        kinetic energy per bond - down spin electrons  0.06118004385332907

    .. code-block:: python

        print("average double occupancy ", np.mean([*d_oc.values()]) )

    .. code-block:: none

        average double occupancy  0.06259168263911569

    .. code-block:: python

        print("average spin-spin correlator ", np.mean([*correlation_sz_sz.values()]))


    .. code-block:: none

        average spin-spin correlator  -0.0069327726073487505





