import yastn
from yastn.tn import fpeps


def _scalar_su2_peps(config):
    leg = yastn.Leg(config, s=1, t=(0,), D=(1,))
    site = yastn.ones(config, legs=(leg, leg, leg.conj(), leg.conj()))
    geometry = fpeps.SquareLattice(dims=(1, 1), boundary='infinite')
    return fpeps.Peps(geometry, tensors=site)


def test_su2_ctm_uses_common_tensor_pipeline(config_kwargs):
    config = yastn.make_config(sym='SU2', **config_kwargs)
    psi = _scalar_su2_peps(config)
    env = fpeps.EnvCTM(psi, init='eye')
    info = env.ctmrg_(opts_svd={'D_total': 2}, max_sweeps=1)
    assert info.sweeps == 1
    assert all(env[site].tl.config.sym.SYM_ID == 'SU2' for site in psi.sites())


def test_su2_ntu_accepts_same_peps_without_special_environment(config_kwargs):
    config = yastn.make_config(sym='SU2', **config_kwargs)
    env = fpeps.EnvNTU(_scalar_su2_peps(config), which='NN')
    assert env.psi.config.sym.SYM_ID == 'SU2'
