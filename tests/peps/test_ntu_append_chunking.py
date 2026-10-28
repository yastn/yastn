"""Close internal NNN boundary selections without full append intermediates."""
import pytest
import weakref
import yastn
from yastn.tn.fpeps.envs import _env_ntu as ntu, _env_contractions as ec

torch = pytest.importorskip('torch')


@pytest.mark.parametrize('value', ['', '0', '-1', '1GiB', '1.5'])
def test_append_invalid_chunk_bytes(value, monkeypatch):
    monkeypatch.setenv('YASTN_NNN_APPEND_CHUNK_BYTES', value)
    with pytest.raises(yastn.YastnError, match='YASTN_NNN_APPEND_CHUNK_BYTES.*positive integer'):
        ntu._append_vec_nnn(None, None, None, None, None, False)


def half_environment(a, corner, vertical):
    if corner == 'tl':
        vector = ec.edge_l(a) @ (ec.cor_tl(a) @ ec.edge_t(a))
        closing = ec.cor_tr(a) @ ec.edge_r(a) if vertical else ec.edge_b(a) @ ec.cor_bl(a)
    else:
        vector = ec.edge_r(a) @ (ec.cor_br(a) @ ec.edge_b(a))
        closing = ec.cor_bl(a) @ ec.edge_l(a) if vertical else ec.edge_t(a) @ ec.cor_tr(a)
    return vector, closing


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('corner', ['tl', 'br'])
@pytest.mark.parametrize('vertical', [False, True])
@pytest.mark.parametrize('masked', [False, True])
def test_append_close_selections(config_kwargs, sym, corner, vertical, masked, monkeypatch):
    backend = config_kwargs['backend'] if 'torch' in config_kwargs['backend'] else 'torch'
    config = yastn.make_config(**{**config_kwargs, 'backend': backend}, sym=sym,
                               fermionic=False if sym == 'SU2' else (False, True))
    charges = (0, 1, 2) if sym == 'SU2' else ((0, 0), (1, 1), (2, 2))
    leg = yastn.Leg(config, t=charges, D=(1, 1, 1))
    physical_charges = (0, 2) if sym == 'SU2' else ((0, 0), (2, 0))
    physical = yastn.Leg(config, t=physical_charges, D=(1, 1))
    a = yastn.rand(config, legs=(leg.conj(), leg, leg, leg.conj(), physical), dtype='complex128')
    ac = yastn.rand_like(a)
    vector, closing = half_environment(a, corner, vertical)
    if masked:
        axis = 0 if vertical else 2
        closing = closing.unfuse_legs(axis)
        mask = yastn.eye(config, legs=closing.get_legs(axis), device=closing.device)
        mask._data[0] = 0
        closing = mask.apply_mask(closing, axes=axis)
        groups = tuple(range(axis)) + ((axis, axis + 1),) + tuple(range(axis + 2, closing.ndim))
        closing = closing.fuse_legs(groups, mode='hard')
    for tensor in (vector, closing, a, ac):
        tensor.requires_grad_()
    append = getattr(ec, 'append_vec_' + corner)
    with monkeypatch.context() as patch:
        patch.setenv('YASTN_NNN_APPEND_CHUNK_BYTES', str(1 << 60))
        expected = ntu._append_vec_nnn(ac, a, vector, closing, append, vertical)
    calls = []

    def tracked(ac, a, selected):
        calls.append(selected.size)
        return append(ac, a, selected)

    monkeypatch.setenv('YASTN_NNN_APPEND_CHUNK_BYTES', str(vector.size * vector.data.element_size() // 3))
    found = ntu._append_vec_nnn(ac, a, vector, closing, tracked, vertical)
    assert len(calls) > 1 and 0 < max(calls) < vector.size
    if not masked:
        assert sum(calls) == vector.size
    assert found.size > 0 and expected.norm().item() > 0
    assert found.hfs == expected.hfs and found.trans == expected.trans
    assert (found - expected).norm().item() < 1e-10 * expected.norm().item()
    gradient = torch.randn_like(expected.data)
    inputs = tuple(tensor.data for tensor in (vector, closing, a, ac))
    reference_grad = torch.autograd.grad(expected.data, inputs, gradient)
    actual_grad = torch.autograd.grad(found.data, inputs, gradient)
    for actual, reference in zip(actual_grad, reference_grad):
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)


def test_append_close_buffer_lifetimes(monkeypatch):
    ops = yastn.operators.Spin12(sym='SU2', backend='torch')
    leg = yastn.Leg(ops.config, t=(0, 1, 2), D=(3, 3, 1))  # D=12.
    physical = ops.I().fuse_legs(((0, 1),)).get_legs(0)
    a = yastn.rand(ops.config, legs=(leg.conj(), leg, leg, leg.conj(), physical))
    vector, closing = half_environment(a, 'br', False)
    refs, calls = [], []

    def append(ac, a, selected):
        assert all(ref() is None for ref in refs), 'Previous append output is still live'
        calls.append(selected.size)
        result = ec.append_vec_br(ac, a, selected)
        refs.append(weakref.ref(result.data))
        return result

    with torch.no_grad():
        monkeypatch.delenv('YASTN_NNN_APPEND_CHUNK_BYTES', raising=False)
        expected = ntu._append_vec_nnn(a, a, vector, closing, ec.append_vec_br, False)
        monkeypatch.setenv('YASTN_NNN_APPEND_CHUNK_BYTES', str(16 << 20))
        found = ntu._append_vec_nnn(a, a, vector, closing, append, False)
    assert len(calls) > 1 and sum(calls) == vector.size
    assert max(calls) * vector.data.element_size() <= 16 << 20
    assert refs and all(ref() is None for ref in refs)
    assert expected.norm().item() > 0
    torch.testing.assert_close(found.data, expected.data)

    refs.clear()
    calls.clear()

    def owned_vector():
        result = vector.copy()
        refs.append(weakref.ref(result.data))
        return result

    def consumes_input(ac, a, selected):
        selected = selected * 1
        assert refs[0]() is None, 'Unchunked wrapper retained the original input'
        calls.append(selected.size)
        return ec.append_vec_br(ac, a, selected)

    monkeypatch.setenv('YASTN_NNN_APPEND_CHUNK_BYTES', str(1 << 60))
    with torch.no_grad():
        found = ntu._append_vec_nnn(a, a, owned_vector(), closing, consumes_input, False)
    assert calls == [vector.size]
    torch.testing.assert_close(found.data, expected.data)
