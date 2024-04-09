Purifcation for Fermi-Hubbard Model Quickstart
==============================================

This guide provides a quick overview on how to simulate the Fermi-Hubbard model at finite temperature using the `yastn`
tensor network library. We'll focus on initializing the model, setting up the simulation, and calculating expectation values.


Model Overview
--------------

The Fermi-Hubbard model describes electrons on a lattice, incorporating both their kinetic energy (hopping between sites) and their interaction energy
(Coulomb repulsion at the same site). The Hamiltonian is given by:

.. math::

   H = -t \sum_{\langle i, j \rangle, \sigma} (c_{i, \sigma}^\dagger c_{j, \sigma} + c_{j, \sigma}^\dagger c_{i, \sigma}) + U \sum_i (n_{i, \uparrow} -\frac{1}{2}) (n_{i, \downarrow}-\frac{1}{2}) - \mu \sum_{i, \sigma} n_{i, \sigma}

where :math:`t` is the hopping parameter, :math:`U` is the on-site Coulomb interaction, :math:`\mu` is the chemical potential, :math:`c_{i, \sigma}^\dagger` (:math:`c_{i, \sigma}`) creates
(annihilates) an electron with spin :math:`\sigma` at site :math:`i`, and :math:`n_{i, \sigma} = c_{i, \sigma}^\dagger c_{i, \sigma}` is the number operator.

Simulation Setup
----------------

1. Initialize the model parameters:

.. code-block:: python
    # Chemical potential
    mu = 0
    t_up = 1
    t_dn = 1
    mu_up = 0
    mu_dn = 0
    beta = 0.2
    U = 8


Note that we have the freedom to set different tunneling amplitudes of different elctron polarizations: 't_up' and 't_dn'
as 'mu_up' and 'mu_dn' can be set diiferent to get different fillings for up and down polarizations.

2. Define the lattice geometry and create the initial state:

.. code-block:: python

    import yastn.tn.fpeps as fpeps
    geometry = fpeps.CheckerboardLattice()
  
    # for bigger unit cells, set 'geometry = fpeps.SquareLattice(dims=(m,n))'
    # for finite lattice, set 'geometry = fpeps.SquareLattice(dims=(m,n), boundary='finite')
    ops = yastn.operators.SpinfulFermions(sym='U1xU1xZ2')
    psi = fpeps.product_peps(geometry=geometry, vectors = ops.I())

2. Define the lattice geometry and create the initial state:

.. code-block:: python

    import yastn.tn.fpeps as fpeps
    geometry = fpeps.SquareLattice(lattice='checkerboard')
    opt = yastn.operators.SpinfulFermions(sym='U1xU1xZ2')
    one = opt.I()
    psi = fpeps.product_peps(geometry=geometry, vectors = ops.I())

3. Define the Hubbard NN and local gates:

.. code-block:: python

    dbeta = 0.01 # trotter step size for second-order ST

    fid = ops.I()
    fc_up, fc_dn, fcdag_up, fcdag_dn = ops.c(spin='u'), ops.c(spin='d'), ops.cp(spin='u'), ops.cp(spin='d')
    n_up, n_dn =  ops.n(spin='u'), ops.n(spin='d')

    g_hop_u = fpeps.gates.gate_nn_hopping(t, dbeta / 2, fid, fc_up, fcdag_up)
    g_hop_d = fpeps.gates.gate_nn_hopping(t, dbeta / 2, fid, fc_dn, fcdag_dn)
    g_loc = fpeps.gates.gate_local_Coulomb(mu_up, mu_dn, U, dbeta / 2, fid, n_up, n_dn)
    gates = fpeps.gates.distribute(geometry, gates_nn=[g_hop_u, g_hop_d], gates_local=g_loc)

4. Define the optimzation procedure by setting up the environment:

.. code-block:: python

    env = fpeps.EnvNTU(psi, which='NN')
    # this is set up for neighborhood tensor update optimization as described in https://arxiv.org/pdf/2209.00985.pdf
    # For larger environements you can select from 'NN+', 'NN++', 'NNN', 'NNN+', 'NNN++', look at class: yastn.tn.fpeps.EnvNTU


3. Set up the time evolution:


.. code-block:: python

    D = 16  # bond dimenson

    opts_svd = {'D_total': D, 'tol_block': 1e-15}
    steps = np.rint((beta / 2) / dbeta).astype(int)   # to reach a target inverse temperature \beta
    for step in range(steps):
        print(f"beta = {(step + 1) * dbeta}" )
        evolution_results = fpeps.evolution_step_(env, gates, opts_svd=opts_svd, initialization="EAT")
        print(f"Error after optimization for all gates: {evolution_results.truncation_error}")

4. Evolve the system and calculate expectation values:

.. code-block:: python

    # convergence criteria for CTM based on total energy
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
        KEn = -2 * np.sum([*cdagc_up.values(), *cdagc_dn.values()])

        energy = PEn + KEn
        print(f"Energy after iteration {i+1}: ", energy)
        if abs(energy - energy_old) < tol_exp:
            break
        energy_old = energy

    print("Final Energy:", energy)


5. Output:

.. code-block:: python


