# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Compare U1 and SU2 Heisenberg evolution on a checkerboard lattice."""
import numpy as np
import pytest


def _mean(values):
    return sum(values) / len(values)


def _su2_bond_hamiltonian(yastn, ops, J):
    """Return J S.S in the coupled singlet/triplet representation."""
    leg = ops.I().get_legs(0)
    coupled = yastn.leg_product(leg, leg)
    fused = yastn.zeros(config=ops.config, legs=(coupled, coupled.conj()))
    for charge, dim in coupled.tD.items():
        spin = charge[0] / 2
        energy = 0.5 * J * (spin * (spin + 1) - 1.5)
        fused.set_block(ts=charge + charge, Ds=(dim, dim),
                        val=energy * np.eye(dim))
    return fused.unfuse_legs((0, 1)).transpose((0, 2, 1, 3))


def _run_heisenberg(config_kwargs, symmetry, D=4, beta=0.5,
                    dbeta=0.05, max_sweeps=50):
    import yastn
    import yastn.tn.fpeps as fpeps
    from yastn.tn.fpeps.envs.rdm import rdm1x2, rdm2x1

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
        gate = fpeps.gates.gate_nn_Heisenberg_SU2(J, dbeta / 2, I)
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
        energies = []
        for direction, builder in (('h', rdm1x2), ('v', rdm2x1)):
            for site, _ in psi.bonds(dirn=direction):
                density, _ = builder(site, psi, env)
                density = density.fuse_legs(((0, 2), (1, 3)), mode='hard')
                weights, weighted_energy = 0, 0
                for charge in density.get_legs(0).t:
                    block = density[charge + charge]
                    multiplicity = np.trace(density.config.backend.to_numpy(block))
                    qdim = charge[0] + 1
                    spin = charge[0] / 2
                    sector_energy = 0.5 * J * (spin * (spin + 1) - 1.5)
                    # Each reduced eigenvalue represents qdim magnetic states.
                    sector_weight = qdim * multiplicity
                    weights += sector_weight
                    weighted_energy += sector_energy * sector_weight
                energies.append(weighted_energy / weights)
        energy = _mean(energies)
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
