"""Each ordinary CTMRG half consumes its corners without retaining four of them."""
import weakref
import pytest
import yastn
from yastn.tn import fpeps
from yastn.tn.fpeps.envs import _env_ctm


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('move', ['h', 'v', 'l', 'r', 't', 'b'])
@pytest.mark.parametrize('use_qr', [False, True])
def test_corner_pairs_are_released(config_kwargs, sym, move, use_qr, monkeypatch):
    ops = yastn.operators.SpinfulFermions_tJ(sym=sym, **config_kwargs)
    charges = (0, 1) if sym == 'SU2' else ((0, 0), (1, 1))
    leg = yastn.Leg(ops.config, s=1, t=charges, D=(1, 1))
    a = yastn.rand(ops.config, legs=(leg, leg, leg.conj(), leg.conj(), ops.I().get_legs(0)))
    psi = fpeps.Peps(fpeps.CheckerboardLattice(), tensors=a)
    env = fpeps.EnvCTM(psi, init='eye')
    original = _env_ctm.corner2x2
    references, names, live = [], [], []

    def record(name, *args, **kwargs):
        live.append(sum(ref() is not None for ref in references))
        result = original(name, *args, **kwargs)
        references.append(weakref.ref(result))
        names.append(name)
        return result

    monkeypatch.setattr(_env_ctm, 'corner2x2', record)
    env.update_(opts_svd={'D_total': 4}, moves=move, use_qr=use_qr)
    assert len(names) == 4 * len(env.sites())
    assert max(live) == 1 and all(ref() is None for ref in references)
    for offset in range(0, len(names), 4):
        assert set(names[offset:offset + 4]) == {'tl', 'tr', 'bl', 'br'}
    assert all(getattr(env[site], move if move in 'lrtb' else 'l' if move == 'h' else 't').size > 0
               for site in env.sites())
