# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Compare U1 and SU2 Heisenberg evolution on a checkerboard lattice."""
import pytest


def _mean(values):
    return sum(values) / len(values)


def _run_heisenberg(config_kwargs, symmetry, D=4, beta=0.5,
                    dbeta=0.05, max_sweeps=50):
    import yastn
    import yastn.tn.fpeps as fpeps

    J = 1
    ops = yastn.operators.Spin12(sym=symmetry, **config_kwargs)
    I = ops.I()
    geometry = fpeps.CheckerboardLattice()
    psi = fpeps.product_peps(geometry, I)
    steps = max(1, round((beta / 2) / dbeta))
    dbeta = (beta / 2) / steps

    if symmetry == 'U1':
        gate = fpeps.gates.gate_nn_Heisenberg(
            J, dbeta / 2, I, ops.sz(), ops.sp(), ops.sm())
    else:
        gate = fpeps.gates.gate_nn_Heisenberg(J, dbeta / 2, I, ops.S())
    gates = fpeps.gates.distribute(geometry, gates_nn=[gate])
    env_ntu = fpeps.EnvNTU(psi, which='NN')
    infos = []
    for _ in range(steps):
        infos.append(fpeps.evolution_step_(
            env_ntu, gates, opts_svd={'D_total': D, 'tol': 1e-12}))

    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 4 * D, 'tol': 1e-10},
               max_sweeps=max_sweeps, corner_tol=1e-8)
    if symmetry == 'U1':
        energy = J * (_mean(env.measure_nn(ops.sz(), ops.sz()).values())
                      + 0.5 * _mean(env.measure_nn(ops.sp(), ops.sm()).values())
                      + 0.5 * _mean(env.measure_nn(ops.sm(), ops.sp()).values()))
    else:
        energy = J * _mean(env.measure_nn(ops.S(), ops.S()).values())
    return {'energy': energy,
            'truncation_error': fpeps.accumulated_truncation_error(infos)}


@pytest.mark.skipif("not config.getoption('quickstart')")
@pytest.mark.parametrize('D, beta, dbeta, energy_tol', [(4, 0.5, 0.05, 1e-4)])
def test_checkerboard_heisenberg_SU2_matches_U1(config_kwargs, D, beta, dbeta, energy_tol):
    reference = _run_heisenberg(config_kwargs, 'U1', D=D, beta=beta, dbeta=dbeta)
    su2 = _run_heisenberg(config_kwargs, 'SU2', D=D, beta=beta, dbeta=dbeta)
    print(f'beta={beta:g}, D={D}, dbeta={dbeta:g}: U1={reference}, SU2={su2}')
    # Finite-D evolutions truncate differently in the two symmetry bases.
    # Exact two-site ED tests separately enforce numerical precision.
    assert su2['energy'] == pytest.approx(reference['energy'], abs=energy_tol, rel=0)



if __name__ == '__main__':
    pytest.main([__file__, '-vs', '--durations=0', '--quickstart'])
