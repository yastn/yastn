"""NNN append contractions must release inputs between hard unfusions."""

import pytest
import yastn
from yastn.tensor import _merging as merging
from yastn.tn.fpeps.envs import _env_contractions as contractions, rdm

torch = pytest.importorskip('torch')
from yastn.backend import backend_torch


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('corner', ['tl', 'br', 'tr', 'bl'])
@pytest.mark.parametrize('open_physical', [False, True])
def test_append_combines_unfusions(config_kwargs, sym, corner, open_physical, monkeypatch):
    monkeypatch.setattr(merging, '_USE_STREAMED_NONABELIAN_FUSION', True)
    monkeypatch.setattr(merging, '_NONABELIAN_FUSION_MIN_SIZE', 0)
    config = yastn.make_config(**{**config_kwargs, 'backend': 'torch'}, sym=sym)
    charges = (0, 1, 2) if sym == 'SU2' else ((0, 0), (1, 1), (2, 2))
    # D=12 including magnetic states; external boundary legs are trivial.
    leg = yastn.Leg(config, s=1, t=charges, D=(3, 3, 1))
    trivial = yastn.Leg(config, s=1, t=(config.sym.zero(),), D=(1,))
    a = yastn.rand(config, legs=(leg.conj(), leg, leg, leg.conj(), trivial),
                   dtype='complex128')
    first, second = {'tl': (1, 0), 'br': (3, 2), 'tr': (0, 3), 'bl': (2, 1)}[corner]
    left, right = a.get_legs((first, second))
    vector = yastn.rand(config, legs=(trivial, left.conj(), left,
                                     right.conj(), right, trivial.conj()),
                        dtype='complex128')
    vector = vector.fuse_legs((0, (1, 2), (3, 4), 5), mode='hard')
    vector.requires_grad_()

    operation = (getattr(rdm, '_append_vec_' + corner + '_open') if open_physical
                 else getattr(contractions, 'append_vec_' + corner))
    with monkeypatch.context() as patch:
        patch.setattr(backend_torch, 'recouple_and_unfuse', None)
        ref = operation(a, a, vector)

    original = backend_torch.recouple_and_unfuse
    calls = []

    def combined(data, steps, outputs, size):
        assert len(steps) > 2 and outputs and size > 0
        calls.append(size)
        result = original(data, steps, outputs, size)
        assert result is not None
        return result

    monkeypatch.setattr(backend_torch, 'recouple_and_unfuse', combined)
    found = operation(a, a, vector)
    assert len(calls) == 1
    assert found.size > 0 and found.nblocks > 1
    assert found.struct == ref.struct and found.hfs == ref.hfs and found.trans == ref.trans
    torch.testing.assert_close(found.data, ref.data, atol=1e-10, rtol=1e-12)
    gradient = torch.randn_like(found.data)
    actual_grad, = torch.autograd.grad(found.data, vector.data, gradient)
    expected_grad, = torch.autograd.grad(ref.data, vector.data, gradient)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-12)
