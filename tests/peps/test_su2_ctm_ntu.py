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
