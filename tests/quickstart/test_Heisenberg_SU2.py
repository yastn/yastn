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


def _run_heisenberg(config_kwargs, symmetry, D=4, beta=0.04,
                    dbeta=0.02, max_sweeps=4):
    import yastn
    import yastn.tn.fpeps as fpeps
    from yastn.tn.fpeps.envs.rdm import rdm1x2

    J = 1
    ops = yastn.operators.Spin12(sym=symmetry, **config_kwargs)
    I = ops.I()
    geometry = fpeps.CheckerboardLattice()
    psi = fpeps.product_peps(geometry, I)
    steps = round((beta / 2) / dbeta)
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
               max_sweeps=max_sweeps)
    if symmetry == 'U1':
        energy = J * (_mean(env.measure_nn(ops.sz(), ops.sz()).values())
                      + 0.5 * _mean(env.measure_nn(ops.sp(), ops.sm()).values())
                      + 0.5 * _mean(env.measure_nn(ops.sm(), ops.sp()).values()))
    else:
        site = next(iter(psi.sites()))
        density, _ = rdm1x2(site, psi, env)
        density = density.fuse_legs(((0, 2), (1, 3)), mode='hard')
        weights, weighted_energy = 0, 0
        for charge in density.get_legs(0).t:
            block = density[charge + charge]
            multiplicity = np.trace(block)
            qdim = charge[0] + 1
            spin = charge[0] / 2
            sector_energy = 0.5 * J * (spin * (spin + 1) - 1.5)
            # Hard fusion uses normalized CG intertwiners.  Recover the
            # magnetic trace of the two spin-1/2 legs from the reduced block.
            sector_weight = np.sqrt(qdim) * multiplicity
            weights += sector_weight
            weighted_energy += sector_energy * sector_weight
        energy = weighted_energy / weights
    return {'energy': energy,
            'truncation_error': fpeps.accumulated_truncation_error(infos)}


@pytest.mark.skipif("not config.getoption('quickstart')")
def test_checkerboard_heisenberg_SU2_matches_U1(config_kwargs):
    reference = _run_heisenberg(config_kwargs, 'U1')
    su2 = _run_heisenberg(config_kwargs, 'SU2')
    # At D=4 the SU2 update discards about 2e-2 in norm; at this deliberately
    # tiny CI cutoff the leading high-temperature energy is only O(1e-2).
    assert su2['energy'] == pytest.approx(reference['energy'], abs=1e-2)


if __name__ == '__main__':
    pytest.main([__file__, '-vs', '--durations=0', '--quickstart'])
