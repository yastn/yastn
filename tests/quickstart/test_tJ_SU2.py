"""High-temperature t-J checkerboard comparison with explicit spin components."""
import pytest
import yastn
from yastn.tn import fpeps


def _run_tj(config_kwargs, symmetry, D=8, beta=0.04):
    ops = yastn.operators.SpinfulFermions_tJ(sym=symmetry, **config_kwargs)
    I = ops.I()
    J, t, mu = 0.3, 1., 0.2
    geometry = fpeps.CheckerboardLattice()
    psi = fpeps.product_peps(geometry, I)
    step = beta / 4
    if symmetry == 'U1xU1xZ2':
        gate = fpeps.gates.gate_nn_tJ(J, t, t, mu / 4, mu / 4, mu / 4, mu / 4,
                                     step, I, ops.c('u'), ops.cp('u'), ops.c('d'), ops.cp('d'))
    else:
        gate = getattr(fpeps.gates, 'gate_nn_tJ_' + symmetry)(J, t, mu / 4, mu / 4, step, I)
    gates = fpeps.gates.distribute(geometry, gates_nn=gate)
    infos = fpeps.evolution_step_(fpeps.EnvNTU(psi, which='NN'), gates,
                                 opts_svd={'D_total': D, 'tol': 1e-12})
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 5 * D, 'tol': 1e-10}, max_sweeps=4)
    mean = lambda values: sum(values) / len(values)
    return {'density': mean(env.measure_1site(ops.n_total()).values()),
            'density_nn': mean(env.measure_nn(ops.n_total(), ops.n_total()).values()),
            'truncation_error': fpeps.accumulated_truncation_error([infos])}


@pytest.mark.skipif("not config.getoption('quickstart')")
@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_checkerboard_tJ_SU2_matches_abelian(config_kwargs, sym):
    reference = _run_tj(config_kwargs, 'U1xU1xZ2')
    result = _run_tj(config_kwargs, sym)
    assert result['density'] == pytest.approx(reference['density'], abs=1e-4)
    assert result['density_nn'] == pytest.approx(reference['density_nn'], abs=1e-4)
