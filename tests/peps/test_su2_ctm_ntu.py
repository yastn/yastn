import pytest
import yastn
from yastn.tn import fpeps
from yastn.tn.fpeps.envs.rdm import _fermionic_exchange


def _scalar_nonabelian_peps(config):
    leg = yastn.Leg(config, s=1, t=(config.sym.zero(),), D=(1,))
    site = yastn.ones(config, legs=(leg, leg, leg.conj(), leg.conj()))
    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    return fpeps.Peps(geometry, tensors=site)


def _run_ctm(config_kwargs, sym, fermionic=False):
    config = yastn.make_config(sym=sym, fermionic=fermionic, **config_kwargs)
    psi = _scalar_nonabelian_peps(config)
    env = fpeps.EnvCTM(psi, init='eye')
    info = env.ctmrg_(opts_svd={'D_total': 2}, max_sweeps=1)
    assert info.sweeps == 1
    assert all(env[site].tl.config.sym.SYM_ID == sym for site in psi.sites())
    assert all(env[site].tl.config.fermionic == fermionic for site in psi.sites())


def test_ctm_SU2(config_kwargs):
    _run_ctm(config_kwargs, 'SU2')


def test_ctm_SU2xU1(config_kwargs):
    _run_ctm(config_kwargs, 'SU2xU1')


def test_ctm_fermionic_SU2(config_kwargs):
    _run_ctm(config_kwargs, 'SU2', True)


def test_ctm_fermionic_SU2xU1(config_kwargs):
    _run_ctm(config_kwargs, 'SU2xU1', (False, True))


def _run_model_ctm(ops, gate):
    """Run CTMRG and an identity measurement on a nontrivial gated PEPS."""
    geometry = fpeps.SquareLattice(dims=(2, 2), boundary='obc')
    psi = fpeps.product_peps(geometry, ops.I())
    gates = fpeps.gates.distribute(geometry, gates_nn=[gate], symmetrize=False)
    psi.apply_gate_(gates[0])
    env = fpeps.EnvCTM(psi, init='eye')
    info = env.ctmrg_(opts_svd={'D_total': 12}, max_sweeps=1)
    assert info.sweeps == 1
    assert all(abs(value - 1) < 1e-12
               for value in env.measure_1site(ops.I()).values())


def test_ctm_Heisenberg_SU2_nontrivial(config_kwargs):
    opts = {**config_kwargs, 'lazy_threshold': 0}
    ops = yastn.operators.Spin12(sym='SU2', **opts)
    _run_model_ctm(ops, fpeps.gates.gate_nn_Heisenberg_SU2(1.0, 0.02, ops.I()))


def test_ctm_Hubbard_SU2xU1_nontrivial(config_kwargs):
    opts = {**config_kwargs, 'lazy_threshold': 0}
    ops = yastn.operators.SpinfulFermions(sym='SU2xU1', **opts)
    gate = fpeps.gates.gate_nn_Hubbard_SU2xU1(0.7, 3.2, -0.15, 0.02, ops.I())
    _run_model_ctm(ops, gate)


def test_Hubbard_SU2xU1_infinite_temperature_weights(config_kwargs):
    """Purification counts a spin doublet twice and agrees with U1xU1."""
    results = {}
    for sym in ('U1xU1', 'SU2xU1'):
        ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
        psi = fpeps.product_peps(fpeps.CheckerboardLattice(), ops.I())
        env = fpeps.EnvCTM(psi, init='eye')
        env.ctmrg_(opts_svd={'D_total': 20}, max_sweeps=1)
        mean = lambda values: sum(values.values()) / len(values)
        results[sym] = (mean(env.measure_1site(ops.n_total())),
                        mean(env.measure_1site(ops.d())))

    assert results['SU2xU1'] == pytest.approx(results['U1xU1'], abs=1e-12)
    assert results['SU2xU1'] == pytest.approx((1, 0.25), abs=1e-12)


def test_ctm_tJ_SU2xU1_nontrivial(config_kwargs):
    opts = {**config_kwargs, 'lazy_threshold': 0}
    ops = yastn.operators.SpinfulFermions_tJ(sym='SU2xU1', **opts)
    gate = fpeps.gates.gate_nn_tJ_SU2xU1(0.3, 0.7, -0.1, 0.2, 0.02, ops.I())
    _run_model_ctm(ops, gate)


def _run_ntu(config_kwargs, sym, fermionic=False):
    config = yastn.make_config(sym=sym, fermionic=fermionic, **config_kwargs)
    env = fpeps.EnvNTU(_scalar_nonabelian_peps(config), which='NN')
    assert env.psi.config.sym.SYM_ID == sym
    assert env.psi.config.fermionic == fermionic


def test_ntu_SU2(config_kwargs):
    _run_ntu(config_kwargs, 'SU2')


def test_ntu_SU2xU1(config_kwargs):
    _run_ntu(config_kwargs, 'SU2xU1')


def test_ntu_fermionic_SU2(config_kwargs):
    _run_ntu(config_kwargs, 'SU2', True)


def test_ntu_fermionic_SU2xU1(config_kwargs):
    _run_ntu(config_kwargs, 'SU2xU1', (False, True))


def _run_model_ntu(ops, gate):
    """Exercise gate application, fusion trees, bond metric and NTU SVD."""
    geometry = fpeps.SquareLattice(dims=(2, 1), boundary='obc')
    # I is the infinite-temperature purification tensor; unlike a selected
    # spinor it is an invariant local tensor for both SU2 and SU2xU1.
    psi = fpeps.product_peps(geometry, ops.I())
    gates = fpeps.gates.distribute(geometry, gates_nn=[gate], symmetrize=False)
    env = fpeps.EnvNTU(psi, which='NN')
    infos = fpeps.evolution_step_(env, gates, opts_svd={'D_total': 20},
                                  max_iter=1)
    assert infos
    assert all(info.truncation_error < 1e-10 for info in infos)
    assert all(psi[site].norm() > 0 for site in psi.sites())


def test_ntu_Heisenberg_SU2(config_kwargs):
    opts = {**config_kwargs, 'lazy_threshold': 0}
    ops = yastn.operators.Spin12(sym='SU2', **opts)
    gate = fpeps.gates.gate_nn_Heisenberg_SU2(1.0, 0.04, ops.I())
    _run_model_ntu(ops, gate)


def test_ntu_Hubbard_SU2xU1(config_kwargs):
    opts = {**config_kwargs, 'lazy_threshold': 0}
    ops = yastn.operators.SpinfulFermions(sym='SU2xU1', **opts)
    gate = fpeps.gates.gate_nn_Hubbard_SU2xU1(0.7, 3.2, -0.15, 0.04, ops.I())
    _run_model_ntu(ops, gate)


def test_ntu_tJ_SU2xU1(config_kwargs):
    opts = {**config_kwargs, 'lazy_threshold': 0}
    ops = yastn.operators.SpinfulFermions_tJ(sym='SU2xU1', **opts)
    gate = fpeps.gates.gate_nn_tJ_SU2xU1(0.3, 0.7, -0.1, 0.2, 0.04, ops.I())
    _run_model_ntu(ops, gate)


def test_full_multibond_ntu_then_ctm(config_kwargs):
    """All four 2x2 bonds: NTU update followed by CTMRG and measurement."""
    opts = {**config_kwargs, 'lazy_threshold': 0}
    cases = []
    ops = yastn.operators.Spin12(sym='SU2', **opts)
    cases.append((ops, fpeps.gates.gate_nn_Heisenberg_SU2(1.0, 0.02, ops.I())))
    ops = yastn.operators.SpinfulFermions(sym='SU2xU1', **opts)
    cases.append((ops, fpeps.gates.gate_nn_Hubbard_SU2xU1(0.7, 3.2, -0.15, 0.02, ops.I())))
    ops = yastn.operators.SpinfulFermions_tJ(sym='SU2xU1', **opts)
    cases.append((ops, fpeps.gates.gate_nn_tJ_SU2xU1(0.3, 0.7, -0.1, 0.2, 0.02, ops.I())))

    for ops, gate in cases:
        geometry = fpeps.SquareLattice(dims=(2, 2), boundary='obc')
        psi = fpeps.product_peps(geometry, ops.I())
        gates = fpeps.gates.distribute(geometry, gates_nn=[gate], symmetrize=False)
        infos = fpeps.evolution_step_(fpeps.EnvNTU(psi, which='NN'), gates,
                                      opts_svd={'D_total': 12}, max_iter=1)
        assert len(infos) == 4
        env = fpeps.EnvCTM(psi, init='eye')
        assert env.ctmrg_(opts_svd={'D_total': 12}, max_sweeps=1).sweeps == 1
        assert all(abs(value - 1) < 1e-12
                   for value in env.measure_1site(ops.I()).values())


def test_SU2xU1_operator_parity_uses_particle_number(config_kwargs):
    """PEPS measurements must not infer fermionic parity from the SU(2) label."""
    config = yastn.make_config(sym='SU2xU1', fermionic=(False, True), **config_kwargs)

    def operator(charge):
        leg = yastn.Leg(config, t=(charge,), D=(1,))
        return yastn.ones(config=config, legs=(leg,), n=charge)

    odd = operator((1, 1))
    spin_only = operator((1, 0))
    even_particle = operator((0, 2))
    assert _fermionic_exchange(odd, odd)
    assert not _fermionic_exchange(spin_only, spin_only)
    assert not _fermionic_exchange(even_particle, odd)
