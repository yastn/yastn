"""Non-Abelian ladder metrics against exact finite PEPS contractions."""
import numpy as np
import pytest
import yastn
from yastn.tn import fpeps
from .test_ntu_metric_streaming import _state, _qr


def _finite_ladder(config_kwargs, sym, layers, vertical):
    fermionic = (False, True) if sym == 'SU2xU1' else False
    config = yastn.make_config(sym=sym, fermionic=fermionic, **config_kwargs)
    config.backend.random_seed(17)
    dims = (2, 2 * layers + 1) if vertical else (2 * layers + 1, 2)
    geometry = fpeps.SquareLattice(dims, boundary='obc')
    charges = (0, 1) if sym == 'SU2' else ((0, 0), (1, 1), (1, -1))
    bond = yastn.Leg(config, s=1, t=charges, D=(1,) * len(charges))
    trivial = yastn.Leg(config, s=1, t=(config.sym.zero(),), D=(1,))
    physical = yastn.Leg(config, s=1, t=(config.sym.zero(),), D=(2,))
    s0 = (0, layers) if vertical else (layers, 0)
    s1 = (1, layers) if vertical else (layers, 1)
    tensors = {}
    for site in geometry.sites():
        legs = [bond if geometry.nn_site(site, direction) is not None else trivial
                for direction in ('t', 'l', 'b', 'r')]
        legs[0], legs[3] = legs[0].conj(), legs[3].conj()
        legs.append(physical if site in (s0, s1) else trivial)
        a = yastn.rand(config, legs=legs, dtype='complex128')
        tensors[site] = a / a.norm()
    return fpeps.Peps(geometry, tensors=tensors), s0, s1


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('vertical', [False, True])
@pytest.mark.parametrize('layers', [1, 2, 3])
def test_ladder_exact_finite_norm(config_kwargs, sym, vertical, layers):
    psi, s0, s1 = _finite_ladder(config_kwargs, sym, layers, vertical)
    if vertical:
        q0, r0 = psi[s0].qr(axes=((0, 1, 3, 4), 2), sQ=1, Qaxis=2)
        q1, r1 = psi[s1].qr(axes=((1, 2, 3, 4), 0), sQ=-1, Qaxis=0, Raxis=-1)
    else:
        q0, r0 = psi[s0].qr(axes=((0, 1, 2, 4), 3), sQ=-1, Qaxis=3)
        q1, r1 = psi[s1].qr(axes=((0, 2, 3, 4), 1), sQ=1, Qaxis=1, Raxis=-1)
    direction = 'tb' if vertical else 'lr'
    qr_leg = q0.get_legs(2 if vertical else 3)
    assert any(q0.config.sym.irrep_dimension(t) > 1 for t in qr_leg.t)
    g = fpeps.EnvNTU(psi, which=f'Ladder{layers}').bond_metric(q0, q1, s0, s1, direction).g
    x = (r0 @ r1).fuse_legs([(0, 1)])
    assert g.size > 0 and x.size > 1
    exact = psi.to_tensor().norm().item() ** 2
    assert exact > 1e-20
    assert yastn.vdot(x, g @ x).item() == pytest.approx(exact, rel=1e-10, abs=1e-20)
    scale = g.norm().item()
    assert scale > 0 and (g - g.H).norm().item() < 1e-11 * scale
    assert np.linalg.eigvalsh(g.to_numpy()).min() > -1e-11 * scale
    if layers == 1:
        nn = fpeps.EnvNTU(psi, which='NN').bond_metric(q0, q1, s0, s1, direction).g
        assert (g - nn).norm().item() < 1e-11 * scale


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('vertical', [False, True])
def test_ladder_streamed_metric(config_kwargs, sym, vertical):
    psi, _ = _state(config_kwargs, sym)
    q0, q1, r0, r1, s0, s1, direction = _qr(psi, vertical)
    full = fpeps.EnvNTU(psi, which='Ladder2').bond_metric(q0, q1, s0, s1, direction).g
    metric = fpeps.EnvNTU(psi, which='Ladder', metric_chunk_size=1).bond_metric(
        q0, q1, s0, s1, direction).g
    assert len(metric.projectors) > 1
    x = (r0 @ r1).fuse_legs([(0, 1)])
    expected = ((full + full.H) / 2) @ x
    assert expected.norm().item() > 0
    assert (metric @ x - expected).norm().item() < 1e-10 * expected.norm().item()


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
def test_ladder_gate_evolution(config_kwargs, sym):
    psi, gate = _state(config_kwargs, sym, prepare=True)
    gates = fpeps.gates.distribute(psi, gates_nn=gate, symmetrize=False)
    infos = fpeps.evolution_step_(fpeps.EnvNTU(psi, which='Ladder2'), gates,
                                 opts_svd={'D_total': 2}, max_iter=2)
    assert len(infos) == len(gates) == 4
    assert all(np.isfinite(info.truncation_error) and info.normalization_factor > 0 for info in infos)
    assert psi.to_tensor().norm().item() > 0


def test_ladder_requires_environment_layer(config_kwargs):
    psi, _ = _state(config_kwargs, 'SU2')
    q0, q1, _, _, s0, s1, direction = _qr(psi, False)
    with pytest.raises(yastn.YastnError, match='at least one environment layer'):
        fpeps.EnvNTU(psi, which='Ladder0').bond_metric(q0, q1, s0, s1, direction)
