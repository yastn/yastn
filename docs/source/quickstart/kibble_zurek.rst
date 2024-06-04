Kibble-Zurek quench in 2D transverse-field Ising model
======================================================

This guide provides a quick overview of simulating the real time evolution of Kibble-Zurek quench in 2D Ising model using `yastn` tensor network library.
We simulate a small :math:`2{\times}2` lattice with open boundary conditions, comparing the results of PEPS and MPS simulations

    .. code-block:: python

        import numpy as np
        import yastn
        import yastn.tn.mps as mps
        import yastn.tn.fpeps as fpeps
        #
        Lx, Ly = 4, 4
        #
        # Quench protocol
        #
        # H(s) = fXX(s) J_ij X_i X_j - fZ(s) Z_i
        #
        fXX = lambda s : np.sin((s - 0.5) * np.pi) + 1
        fZ = lambda s :  1 - np.sin((s - 0.5) * np.pi)
        ta = 1.0  # annealing time
        dt = 0.02
        steps = round(ta / dt)
        dt = ta / steps
        #
        ops = yastn.operators.Spin12(sym='Z2')
        #
        # random couiplings in 2D square lattice with OBC
        #
        geometry = fpeps.SquareLattice(dims=(Lx, Ly), boundary='obc')
        np.random.seed(seed=0)
        Jij = {k: 2 * np.random.rand() - 1 for k in geometry.bonds()}
        sites = geometry.sites()
        #
        print("PEPS simulations")
        #
        def gates_Ising(Jij, fXX, fZ, s, dt, sites, ops):
            nn = [fpeps.gates.gate_nn_Ising(J * fXX(s), 1j * dt / 2, ops.I(), ops.x(), bd) for bd, J in Jij.items()]
            local = [fpeps.gates.gate_local_Ising(fZ(s), 1j * dt / 2, ops.I(), ops.z(), st) for st in sites]
            return fpeps.Gates(nn=nn, local=local)
        #
        psi = fpeps.product_peps(geometry=geometry, vectors=ops.vec_z(val=1))
        #
        D = 6
        opts_svd_ntu = {"D_total": D, "D_block": D // 2}
        env = fpeps.EnvNTU(psi, which='NN+')
        #
        t = 0
        infoss = []
        for _ in range(steps):
            t += dt / 2
            gates = gates_Ising(Jij, fXX, fZ, t / ta, dt, sites, ops)
            infos = fpeps.evolution_step_(env, gates, opts_svd=opts_svd_ntu)
            infoss.append(infos)
            t += dt / 2
            print(f"s = {t / ta:0.4f}")
        Delta = fpeps.accumulated_truncation_error(infoss, statistics='mean')
        print(f"{Delta=:0.8f}")
        #
        # calculate correlations
        #
        opts_svd_env = {'D_total': 4 * D}
        opts_var_env = {"overlap_tol": 1e-5,
                        "Schmidt_tol": 1e-5,
                        "max_sweeps": 8}
        #
        print(" BoundaryMPS preprocessing ")
        env = fpeps.EnvBoundaryMps(psi, opts_svd=opts_svd_env, opts_var=opts_var_env, setup='lr')
        #
        # Calculating 1-site
        print(" Observables 1-site ")
        Ez_peps = env.measure_1site(ops.z())
        #
        # Calculating 2-site
        print(" Observables 2-site ")
        Exx_peps = env.measure_2site(ops.x(), ops.x(),
                                opts_svd=opts_svd_env,
                                opts_var=opts_var_env)
        #
        print("MPS simulations")
        #
        i2s = {i: s for i, s in enumerate(sites)}
        s2i = {s: i for i, s in enumerate(sites)}
        #
        I = mps.product_mpo(ops.I(), Lx * Ly)  # identity MPO
        termsXX = [mps.Hterm(J, [s2i[s1], s2i[s2]], [ops.x(), ops.x()]) for (s1, s2), J in Jij.items()]
        HXX = mps.generate_mpo(I, termsXX)
        termsZ = [mps.Hterm(-1, [i], [ops.z()]) for i in range(Lx * Ly)]
        HZ = mps.generate_mpo(I, termsZ)
        #
        H = lambda t: [HXX * fXX(t / ta), HZ * fZ(t / ta)]
        #
        psi = mps.random_mps(I, D_total=128)
        t = 0
        mps.dmrg_(psi, H(t), method='1site', max_sweeps=8, Schmidt_tol=1e-8)
        #
        Dmax = 128
        opts_expmv = {'hermitian': True, 'tol': 1e-12}
        opts_svd = {'tol': 1e-6, 'D_total': Dmax}
        # #
        for _ in mps.tdvp_(psi, H, times=(0, ta),
                        method='1site', dt=dt, order='2nd',
                        opts_svd=opts_svd, opts_expmv=opts_expmv):
            pass
        Ez_mps = mps.measure_1site(psi, ops.z(), psi)
        Exx_mps = mps.measure_2site(psi, ops.x(), ops.x(), psi)

        Zerror = sum(abs(Ez_mps[s2i[st]] - Ez_peps[st]) ** 2 for st in sites) ** 0.5
        Zerror /= sum(abs(Ez_peps[st]) ** 2 for st in sites) ** 0.5
        print(f"Normalized error of <Z_i>: {Zerror:0.5f}")

        XX_error = 0.
        for (s1, s2), v in Exx_peps.items():
            if s1 != s2:
                XX_error += abs(v - Exx_mps[tuple(sorted((s2i[s1], s2i[s2])))]) ** 2
        XX_error = XX_error ** 0.5
        XX_error /= sum(abs(v) ** 2 for v in Exx_mps.values()) ** 0.5
        print(f"Normalized error of <X_i X_j>: {XX_error:0.5f}")


