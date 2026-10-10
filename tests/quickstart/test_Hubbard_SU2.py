# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Compare the U1xU1 and SU2xU1 Hubbard quickstart simulations."""
import pytest


def _mean(values):
    return sum(values) / len(values)


def _run_hubbard(config_kwargs, symmetry, D, betas, dbeta=0.02, max_sweeps=4):
    """Run the Hubbard quickstart and return SU(2)-scalar observables."""
    import yastn
    import yastn.tn.fpeps as fpeps

    # Keep these parameters synchronized with test_Hubbard.py.
    t, mu, U = 1, 0, 10
    ops = yastn.operators.SpinfulFermions(sym=symmetry, **config_kwargs)
    I = ops.I()
    n_total = ops.n_total()
    double_occ = ops.d()

    geometry = fpeps.CheckerboardLattice()
    psi = fpeps.product_peps(geometry=geometry, vectors=I)

    beta0, infoss = 0, []
    for beta in betas:
        db = dbeta
        steps = round(((beta - beta0) / 2) / db)
        db = ((beta - beta0) / 2) / steps
        beta0 = beta

        # Strang (second-order Suzuki--Trotter) step.  distribute() appends
        # the reverse gate list, so every gate constructed with db / 2 is
        # applied twice: hopping(db/2) local(db) hopping(db/2), including the
        # corresponding forward/backward checkerboard bond ordering.
        half_step = db / 2
        if symmetry == 'U1xU1':
            c_up, cdag_up = ops.c('u'), ops.cp('u')
            c_dn, cdag_dn = ops.c('d'), ops.cp('d')
            n_up, n_dn = ops.n('u'), ops.n('d')
            # Use one combined hopping gate in both representations.  At a
            # deliberately small D, applying two spin-resolved U1 gates with
            # an intermediate truncation is not numerically equivalent to
            # applying the single SU2 scalar gate before truncation.
            hopping = -t * (
                yastn.fkron(cdag_up, c_up, sites=(0, 1))
                + yastn.fkron(cdag_up, c_up, sites=(1, 0))
                + yastn.fkron(cdag_dn, c_dn, sites=(0, 1))
                + yastn.fkron(cdag_dn, c_dn, sites=(1, 0)))
            gates_nn = [fpeps.gates.gate_nn_exp(half_step, I, hopping)]
            gate_local = fpeps.gates.gate_local_Coulomb(
                mu, mu, U, half_step, I, n_up, n_dn)
        else:
            # The two spin-component hopping terms commute and together form
            # an SU(2) scalar.  The local term is exactly
            # U (n_up - 1/2) (n_dn - 1/2) - mu * n_total.
            gate_builder = getattr(fpeps.gates, 'gate_nn_Hubbard_' + symmetry)
            gates_nn = [gate_builder(t, 0, 0, half_step, I)]
            h_local = U * (double_occ - n_total / 2 + I / 4) - mu * n_total
            gate_local = fpeps.gates.gate_local_exp(half_step, I, h_local)

        gates = fpeps.gates.distribute(
            geometry, gates_nn=gates_nn, gates_local=gate_local)
        env = fpeps.EnvNTU(psi, which='NN')
        opts_svd = {'D_total': D, 'tol': 1e-12}
        for _ in range(steps):
            infoss.append(fpeps.evolution_step_(env, gates, opts_svd=opts_svd))

        env_ctm = fpeps.EnvCTM(psi, init='eye')
        opts_svd_ctm = {'D_total': 5 * D, 'tol': 1e-10}
        density_shift = n_total - I
        old = None
        for info in env_ctm.ctmrg_(opts_svd=opts_svd_ctm,
                                   iterator=True, max_sweeps=max_sweeps):
            current = _mean(list(env_ctm.measure_1site(double_occ).values()))
            if old is not None and abs(current - old) < 1e-7:
                break
            old = current

        result = {
            'density': _mean(list(env_ctm.measure_1site(n_total).values())),
            'double_occ': _mean(list(env_ctm.measure_1site(double_occ).values())),
            'density_nn': _mean(list(env_ctm.measure_nn(
                density_shift, density_shift).values())),
            'truncation_error': fpeps.accumulated_truncation_error(infoss),
            'ctm_sweeps': info.sweeps,
        }
    return result


@pytest.mark.skipif("not config.getoption('quickstart')")
def test_two_site_hubbard_SU2_against_ED(config_kwargs):
    """Exact, untruncated check of gate application and PEPS measurement."""
    import numpy as np
    import scipy.linalg
    import yastn
    import yastn.tn.fpeps as fpeps

    t, U, mu, step = 0.7, 3.2, 1.6, 0.02
    # Independent four-mode Fock-space Hamiltonian, with modes (0u,0d,1u,1d).
    annihilation = []
    for mode in range(4):
        c = np.zeros((16, 16))
        for state in range(16):
            if state & (1 << mode):
                parity = (state & ((1 << mode) - 1)).bit_count()
                c[state ^ (1 << mode), state] = (-1) ** parity
        annihilation.append(c)
    number = [c.T @ c for c in annihilation]
    H = U * (number[0] @ number[1] + number[2] @ number[3]) - mu * sum(number)
    for left, right in ((0, 2), (1, 3)):
        hopping = annihilation[left].T @ annihilation[right]
        H -= t * (hopping + hopping.T)
    rho = scipy.linalg.expm(-2 * step * H)
    rho /= np.trace(rho)
    expected = np.trace(rho @ number[0] @ number[1])

    ops = yastn.operators.SpinfulFermions(sym='SU2xU1', **config_kwargs)
    geometry = fpeps.SquareLattice((1, 2), boundary='obc')
    psi = fpeps.product_peps(geometry, ops.I())
    gate = fpeps.gates.gate_nn_Hubbard_SU2xU1(t, U, mu, step, ops.I())
    gates = fpeps.gates.distribute(geometry, gates_nn=gate, symmetrize=False)
    infos = fpeps.evolution_step_(fpeps.EnvNTU(psi, which='NN'), gates,
                                  opts_svd={'D_total': 32}, max_iter=1)
    assert all(info.truncation_error < 1e-12 for info in infos)
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 32}, max_sweeps=2)
    for value in env.measure_1site(ops.d()).values():
        assert value == pytest.approx(expected, abs=1e-10)


@pytest.mark.skipif("not config.getoption('quickstart')")
@pytest.mark.parametrize('D, betas, dbeta', [(8, [0.04], 0.02)])
def test_quickstart_hubbard_SU2_matches_U1xU1(config_kwargs, D, betas, dbeta):
    """SU2xU1 and U1xU1 give the same scalar Hubbard observables."""
    reference = _run_hubbard(config_kwargs, 'U1xU1', D, betas, dbeta=dbeta)
    su2 = _run_hubbard(config_kwargs, 'SU2xU1', D, betas, dbeta=dbeta)

    # The two runs use the same physical gates and numerical cutoffs, but
    # organize the retained states into different symmetry multiplets.
    assert su2['density'] == pytest.approx(reference['density'], abs=2e-3)
    assert su2['double_occ'] == pytest.approx(reference['double_occ'], abs=3e-3)
    assert su2['density_nn'] == pytest.approx(reference['density_nn'], abs=3e-3)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--quickstart"])
