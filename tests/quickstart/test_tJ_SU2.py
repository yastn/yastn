"""Finite-temperature t-J checkerboard comparison with explicit spin components."""
import pytest
import yastn
from yastn.tn import fpeps


def _run_tj(config_kwargs, symmetry, D=8, beta=0.5, dbeta=0.05, max_sweeps=50):
    ops = yastn.operators.SpinfulFermions_tJ(sym=symmetry, **config_kwargs)
    I = ops.I()
    J, t, mu = 0.3, 1., 0.2
    geometry = fpeps.CheckerboardLattice()
    psi = fpeps.product_peps(geometry, I)
    steps = max(1, round((beta / 2) / dbeta))
    dbeta = (beta / 2) / steps
    step = dbeta / 2
    if symmetry == 'U1xU1xZ2':
        gate = fpeps.gates.gate_nn_tJ(J, t, t, mu / 4, mu / 4, mu / 4, mu / 4,
                                     step, I, ops.c('u'), ops.cp('u'), ops.c('d'), ops.cp('d'))
    else:
        gate = getattr(fpeps.gates, 'gate_nn_tJ_' + symmetry)(J, t, mu / 4, mu / 4, step, I)
    gates = fpeps.gates.distribute(geometry, gates_nn=gate)
    env_ntu = fpeps.EnvNTU(psi, which='NN')
    infos = []
    for _ in range(steps):
        infos.append(fpeps.evolution_step_(env_ntu, gates,
                                          opts_svd={'D_total': D, 'tol': 1e-12}))
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 5 * D, 'tol': 1e-10}, max_sweeps=max_sweeps, corner_tol=1e-8)
    mean = lambda values: sum(values) / len(values)
    return {'density': mean(env.measure_1site(ops.n_total()).values()),
            'density_nn': mean(env.measure_nn(ops.n_total(), ops.n_total()).values()),
            'truncation_error': fpeps.accumulated_truncation_error(infos)}


@pytest.mark.skipif("not config.getoption('quickstart')")
@pytest.mark.parametrize('D, beta, dbeta, density_tol, correlation_tol',
                         [(8, 0.5, 0.05, 1e-4, 2e-3)])
@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_checkerboard_tJ_SU2_matches_abelian(config_kwargs, sym, D, beta, dbeta,
                                             density_tol, correlation_tol):
    reference = _run_tj(config_kwargs, 'U1xU1xZ2', D=D, beta=beta, dbeta=dbeta)
    result = _run_tj(config_kwargs, sym, D=D, beta=beta, dbeta=dbeta)
    print(f'beta={beta:g}, D={D}, dbeta={dbeta:g}: U1xU1xZ2={reference}, {sym}={result}')
    # At beta=0.5, D=8 the accumulated truncation error is about 1e-2.
    # Multiplet-preserving and Abelian truncations need not give identical
    # correlations; exact ED/RDM tests separately enforce numerical precision.
    assert result['density'] == pytest.approx(reference['density'], abs=density_tol, rel=0)
    assert result['density_nn'] == pytest.approx(reference['density_nn'], abs=correlation_tol, rel=0)
