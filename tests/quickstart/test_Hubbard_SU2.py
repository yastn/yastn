# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Benchmark SU2 Hubbard purification against the METTS data in test_Hubbard."""
import numpy as np
import pytest
from time import perf_counter

from ..peps._multiplet_models import hubbard_hamiltonian


def _mean(values):
    return sum(values) / len(values)


def _measure_hubbard(env, ops, t, U):
    """Measure scalar densities and invariant spinor/vector contractions."""
    cp, c, S = ops.cp_spinor(), ops.c_spinor(), ops.S()
    forward = env.measure_nn(cp, c)
    reverse = env.measure_nn(c, cp)
    # c_i c_j^dagger = -c_j^dagger c_i for distinct sites.
    kinetic = -t * _mean([forward[bond] - reverse[bond] for bond in forward])
    spin = _mean(env.measure_nn(S, S).values())
    density = _mean(env.measure_1site(ops.n_total()).values())
    double_occ = _mean(env.measure_1site(ops.d()).values())
    # Square lattice: two bonds per site; particle-hole-symmetric convention.
    energy = 2 * kinetic + U * (double_occ - density / 2 + 0.25)
    return {name: float(np.real_if_close(value)) for name, value in
            {'energy': energy, 'density': density, 'double_occ': double_occ,
             'SzSz_NN': spin / 3}.items()}


def _run_hubbard(config_kwargs, symmetry, D, betas, dbeta=0.05, max_sweeps=50):
    """Return measurements at every target beta, using the METTS Hamiltonian."""
    import yastn
    import yastn.tn.fpeps as fpeps

    t, mu, U = 1, 0, 10  # synchronized with test_Hubbard.py and its METTS data
    ops = yastn.operators.SpinfulFermions(sym=symmetry, **config_kwargs)
    I = ops.I()
    h_local = U * (ops.d() - ops.n_total() / 2 + I / 4) - mu * ops.n_total()
    geometry = fpeps.CheckerboardLattice()
    psi = fpeps.product_peps(geometry=geometry, vectors=I)
    env = fpeps.EnvNTU(psi, which='NN')
    beta0, infoss, results = 0, [], {}
    for beta in betas:
        if beta <= beta0 or dbeta <= 0:
            raise ValueError('Target betas must increase and dbeta must be positive.')
        steps = max(1, round(((beta - beta0) / 2) / dbeta))
        db = ((beta - beta0) / 2) / steps
        # Purification evolves through beta/2. distribute() appends the
        # reverse list, giving a Strang step with gates of duration db/2.
        gates = fpeps.gates.distribute(
            geometry, gates_nn=[fpeps.gates.gate_nn_hopping(t, db / 2, I, ops.c_spinor(), ops.cp_spinor())],
            gates_local=fpeps.gates.gate_local_exp(db / 2, I, h_local))
        for step in range(steps):
            infoss.append(fpeps.evolution_step_(
                env, gates, opts_svd={'D_total': D, 'tol': 1e-12}))
            print(f'{symmetry} D={D} beta={beta0 + 2 * db * (step + 1):.6f}', flush=True)
        beta0 = beta

        env_ctm = fpeps.EnvCTM(psi, init='eye')
        old_energy, converged = None, False
        started = perf_counter()
        for info in env_ctm.ctmrg_(opts_svd={'D_total': 5 * D, 'tol': 1e-10},
                                   iterator=True, max_sweeps=max_sweeps):
            updated = perf_counter()
            print(f'beta={beta:g} CTM sweep={info.sweeps} update={updated - started:.2f}s; measuring multiplets', flush=True)
            result = _measure_hubbard(env_ctm, ops, t, U)
            started = perf_counter()
            print(f"beta={beta:g} CTM sweep={info.sweeps} energy={result['energy']:.8f} "
                  f"measurement={started - updated:.2f}s", flush=True)
            if old_energy is not None and abs(result['energy'] - old_energy) < 1e-7:
                converged = True
                break
            old_energy = result['energy']
        results[beta] = {**result,
                         'truncation_error': fpeps.accumulated_truncation_error(infoss),
                         'ctm_sweeps': info.sweeps, 'ctm_converged': converged}
    return results


@pytest.mark.skipif("not config.getoption('quickstart')")
@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('dims', [(1, 2), (2, 1)])
@pytest.mark.parametrize('step', [0.02, 1.0])
def test_two_site_hubbard_SU2_against_ED(config_kwargs, sym, dims, step):
    """Exact, untruncated check of gate application and PEPS measurement."""
    import numpy as np
    import scipy.linalg
    import yastn
    import yastn.tn.fpeps as fpeps

    t, U, mu = 0.7, 3.2, 1.6
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
    hopping = np.zeros((16, 16))
    for left, right in ((0, 2), (1, 3)):
        hop = annihilation[left].T @ annihilation[right]
        hopping -= t * (hop + hop.T)
    H = hopping + U * (number[0] @ number[1] + number[2] @ number[3]) - mu * sum(number)
    rho = scipy.linalg.expm(-2 * step * H)
    rho /= np.trace(rho)
    expected = np.trace(rho @ number[0] @ number[1])

    ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
    geometry = fpeps.SquareLattice(dims, boundary='obc')
    psi = fpeps.product_peps(geometry, ops.I())
    gate = fpeps.gates.gate_nn_exp(step, ops.I(), hubbard_hamiltonian(ops, t, U, mu))
    gates = fpeps.gates.distribute(geometry, gates_nn=gate, symmetrize=False)
    infos = fpeps.evolution_step_(fpeps.EnvNTU(psi, which='NN'), gates,
                                  opts_svd={'D_total': 32}, max_iter=1)
    assert all(info.truncation_error < 1e-12 for info in infos)
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 32}, max_sweeps=2)
    for value in env.measure_1site(ops.d()).values():
        assert value == pytest.approx(expected, abs=1e-10)
    measured = _measure_hubbard(env, ops, t, U)
    kinetic = np.trace(rho @ hopping)
    szsz = (number[0] - number[1]) @ (number[2] - number[3]) / 4
    density = np.trace(rho @ (number[0] + number[1]))
    assert measured['SzSz_NN'] == pytest.approx(np.trace(rho @ szsz), abs=1e-10)
    # The benchmark reports square-lattice energy (two bonds per site).
    assert measured['energy'] == pytest.approx(2 * kinetic + U * (expected - density / 2 + 0.25), abs=1e-10)


@pytest.mark.skipif("not config.getoption('quickstart')")
@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_hubbard_SU2_temperature_schedule(config_kwargs, sym):
    """Keep all checkpoints, including intervals smaller than the requested step."""
    results = _run_hubbard(config_kwargs, sym, D=1, betas=[0.1, 0.2],
                           dbeta=0.1, max_sweeps=2)
    assert list(results) == [0.1, 0.2]
    for result in results.values():
        assert result['density'] == pytest.approx(1, abs=1e-10)
        assert all(np.isfinite(result[name]) for name in ('energy', 'double_occ', 'SzSz_NN'))


@pytest.mark.skipif("not config.getoption('quickstart')")
@pytest.mark.parametrize('D, betas, dbeta', [(12, [2.0], 0.05)])
@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_quickstart_hubbard_SU2_against_METTS(config_kwargs, D, betas, dbeta, sym):
    """Compare with the 4 x 16 cylinder METTS reference, including its error bars.

    Extend betas to [2.0, 4.0, 6.0] for the full reference temperature range.
    METTS sampling errors do not include PEPS truncation, Trotter, CTM, or
    geometry errors, so they are reported rather than used as pass thresholds.
    """
    from .test_Hubbard import metts_values

    results = _run_hubbard(config_kwargs, sym, D, betas, dbeta=dbeta)
    for beta, result in results.items():
        print(f"\n{sym}: beta={beta:g}, D={D}, dbeta={dbeta:g}, "
              f"CTM converged={result['ctm_converged']}, "
              f"truncation error={result['truncation_error']:.6g}")
        for name in ('energy', 'double_occ', 'SzSz_NN'):
            value = result[name]
            reference, error = metts_values[name, beta]
            print(f'{name:12s} PEPS={value:.8f} METTS={reference:.8f} +/- {error:.4g} '
                  f'difference={value - reference:+.8f}')
            assert np.isfinite(value)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--quickstart"])
