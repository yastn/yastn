
# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" yastn.truncation_mask() """
import pytest
import numpy as np
import yastn


def _minimum_spectrum(config_kwargs, sym='SU2'):
    config = yastn.make_config(sym=sym, **config_kwargs)
    charges = ((0, 0), (1, 1), (2, 0)) if sym == 'SU2xU1' else ((0,), (1,), (2,))
    leg = yastn.Leg(config, t=charges, D=(3, 2, 2))
    spectrum = yastn.zeros(config, legs=(leg, leg.conj()), isdiag=True)
    for charge, values in zip(charges, ([10., 9., 8.], [1e-8, 2e-8], [0., 0.])):
        spectrum.set_block(ts=charge + charge, Ds=len(values), val=values)
    return spectrum, charges


@pytest.mark.parametrize('sym', ['U1', 'SU2', 'SU2xU1'])
@pytest.mark.parametrize('which', ['LR', 'LM', 'SR', 'SM'])
def test_truncation_sector_minimum(config_kwargs, sym, which):
    spectrum, charges = _minimum_spectrum(config_kwargs, sym)
    # Reserve a weak sector first, then fill remaining capacity by spectral order.
    cap = 2 + (2 if sym != 'U1' else 1)
    opts = dict(which=which, D_total=cap, D_block_min={charges[1]: 1},
                D_block={charges[0]: 3, charges[1]: 1, charges[2]: 2})
    if which in ('LR', 'LM'):
        opts.update(tol=0.1, tol_block={charges[0]: 0.1, charges[1]: 2.})
    mask = spectrum.truncation_mask(**opts)
    backend = spectrum.config.backend
    weak = backend.to_numpy(mask[charges[1] + charges[1]])
    np.testing.assert_array_equal(weak, [which in ('SR', 'SM'), which in ('LR', 'LM')])
    kept = mask.apply_mask(spectrum)
    assert kept.size > 0
    cost = sum(spectrum.config.sym.irrep_dimension(t) * d
               for t, d in zip(kept.struct.legs[0].t, kept.struct.legs[0].D)) if sym != 'U1' else kept.size
    assert cost <= cap
    if which in ('LR', 'LM'):
        np.testing.assert_array_equal(backend.to_numpy(mask[charges[0] + charges[0]]), [True, True, False])


@pytest.mark.parametrize('sym', ['U1', 'SU2', 'SU2xU1'])
@pytest.mark.parametrize('count', [1, 2])
def test_truncation_minimum_all_sectors_and_zeros(config_kwargs, sym, count):
    spectrum, charges = _minimum_spectrum(config_kwargs, sym)
    mask = spectrum.truncation_mask(D_block_min=count, tol=2., tol_block=2., D_total=6 * count)
    for charge in charges:
        assert spectrum.config.backend.sum_elements(mask[charge + charge]).item() == count
    assert mask.apply_mask(spectrum).size == 3 * count
    # A zero minimum preserves the complete legacy mask, including multiplets.
    for setting in (0, {}, {charges[1]: 0}):
        found = spectrum.truncation_mask(D_total=3, D_block_min=setting, eps_multiplet=0.1)
        reference = spectrum.truncation_mask(D_total=3, eps_multiplet=0.1)
        np.testing.assert_array_equal(found.to_numpy(), reference.to_numpy())


@pytest.mark.parametrize('opts,match', [
    ({'D_block_min': {(2,): 1}, 'D_total': 2}, 'dense dimension 3'),
    ({'D_block_min': {(1,): 1, (2,): 1}, 'D_total': 4}, 'dense dimension 5'),
    ({'D_block_min': {(1,): 1}, 'D_total': 0}, 'exceeding D_total'),
    ({'D_block_min': {(1,): 3}, 'D_total': 3}, 'dense dimension 4'),
    ({'D_block_min': {(1,): 3}, 'D_block': 1}, 'D_block=1'),
    ({'D_block_min': {(1,): 1}, 'D_block': 0}, 'D_block=0'),
    ({'D_block_min': {(1,): 1}, 'D_block': {(0,): 2}}, 'D_block=0'),
    ({'D_block_min': -1}, 'nonnegative integers'),
    ({'D_block_min': 1.5}, 'nonnegative integers'),
    ({'D_block_min': True}, 'nonnegative integers'),
    ({'D_block_min': {(1,): -1}}, 'nonnegative integers'),
    ({'D_block_min': {(1, 2): 1}}, 'charge tuples'),
    ({'D_block_min': {'1': 1}}, 'charge tuples'),
    ({'D_block_min': {1: 1, (1,): 2}}, 'conflicting counts'),
])
def test_truncation_minimum_errors(config_kwargs, opts, match):
    spectrum, _ = _minimum_spectrum(config_kwargs)
    with pytest.raises(yastn.YastnError, match=match):
        spectrum.truncation_mask(**opts)


@pytest.mark.parametrize('sym', ['U1', 'SU2', 'SU2xU1'])
@pytest.mark.parametrize('all_sectors', [False, True])
def test_truncation_minimum_capped_at_available(config_kwargs, sym, all_sectors):
    spectrum, charges = _minimum_spectrum(config_kwargs, sym)
    minimum = 100 if all_sectors else {charges[1]: 100}
    mask = spectrum.truncation_mask(D_block_min=minimum, tol=2., tol_block=2.,
                                    D_block=3 if all_sectors else 2,
                                    D_total=13 if all_sectors else 4)
    expected = [True] * 7 if all_sectors else [False] * 3 + [True] * 2 + [False] * 2
    np.testing.assert_array_equal(spectrum.config.backend.to_numpy(mask.data), expected)
    assert mask.apply_mask(spectrum).size == (7 if all_sectors else 2)


@pytest.mark.parametrize('sym', ['U1', 'SU2', 'SU2xU1'])
@pytest.mark.parametrize('protect_existing', [False, True])
def test_truncation_minimum_ignores_absent_sectors(config_kwargs, sym, protect_existing):
    spectrum, charges = _minimum_spectrum(config_kwargs, sym)
    absent = (4, 0) if sym == 'SU2xU1' else (4,)
    minimum = {charges[1]: 1} if protect_existing else {}
    options = dict(D_total=4, tol=0.1)
    if not protect_existing:
        # An entirely absent request must retain legacy multiplet handling.
        options['eps_multiplet'] = 0.1
    reference = spectrum.truncation_mask(D_block_min=minimum, **options)
    found = spectrum.truncation_mask(D_block_min={**minimum, absent: 100}, **options)
    assert found.apply_mask(spectrum).size > 0
    np.testing.assert_array_equal(found.to_numpy(), reference.to_numpy())
    if protect_existing:
        assert spectrum.config.backend.sum_elements(found[charges[1] + charges[1]]).item() == 1


def test_truncation_minimum_hermitian_pairs(config_kwargs):
    config = yastn.make_config(sym='U1', **config_kwargs)
    spectrum = yastn.Tensor(config, isdiag=True)
    for charge, values in ((-1, [2e-8, 1e-8]), (0, [10., 9.]), (1, [3e-8, 4e-8])):
        spectrum.set_block(ts=(charge, charge), Ds=2, val=values)
    mask = spectrum.truncation_mask(D_total=3, tol=0.1, D_block_min={1: 1}, hermitian=True)
    backend = config.backend
    np.testing.assert_array_equal(backend.to_numpy(mask[(-1, -1)]), [True, False])
    np.testing.assert_array_equal(backend.to_numpy(mask[(1, 1)]), [False, True])
    np.testing.assert_array_equal(backend.to_numpy(mask[(0, 0)]), [True, False])
    with pytest.raises(yastn.YastnError, match='exceeding D_total'):
        spectrum.truncation_mask(D_total=1, D_block_min={1: 1}, hermitian=True)
    with pytest.raises(yastn.YastnError, match='D_block=0'):
        spectrum.truncation_mask(D_block={1: 1}, D_block_min={1: 1}, hermitian=True)
    missing = yastn.Tensor(config, isdiag=True)
    missing.set_block(ts=(1, 1), Ds=1, val=[1.])
    with pytest.raises(yastn.YastnError, match='conjugate sector'):
        missing.truncation_mask(D_block_min={1: 1}, hermitian=True)
    # Unequal conjugate sectors retain as many complete pairs as available.
    unequal = yastn.Tensor(config, isdiag=True)
    unequal.set_block(ts=(-1, -1), Ds=2, val=[2e-8, 1e-8])
    unequal.set_block(ts=(1, 1), Ds=1, val=[3e-8])
    for setting in (100, {1: 100}, {-1: 100}):
        mask = unequal.truncation_mask(D_block_min=setting, hermitian=True,
                                        tol=2., D_total=2, D_block=1)
        np.testing.assert_array_equal(backend.to_numpy(mask.data), [True, False, True])


@pytest.mark.parametrize('sym', ['U1', 'SU2'])
def test_truncation_minimum_multiplets_and_custom_mask(config_kwargs, sym):
    spectrum, _ = _minimum_spectrum(config_kwargs, sym)
    for opts in ({'largest_gap': True}, {'eps_multiplet': 0}, {'eps_multiplet': 0.1}):
        if sym == 'U1':
            with pytest.raises(yastn.YastnError, match='D_block_min cannot be combined'):
                spectrum.truncation_mask(D_block_min=1, **opts)
        else:
            found = spectrum.truncation_mask(D_block_min=1, D_total=6, **opts)
            expected = spectrum.truncation_mask(D_block_min=1, D_total=6)
            np.testing.assert_array_equal(found.to_numpy(), expected.to_numpy())
    custom = spectrum > 1.
    assert spectrum.truncation_mask(D_block_min=-1, mask_f=lambda _: custom) is custom


@pytest.mark.parametrize('sym', ['U1', 'SU2', 'SU2xU1'])
@pytest.mark.parametrize('method', ['svd', 'eigh'])
@pytest.mark.parametrize('minimum', [1, 100])
def test_minimum_decomposition_wrappers(config_kwargs, sym, method, minimum):
    spectrum, charges = _minimum_spectrum(config_kwargs, sym)
    tensor = spectrum.diag()
    options = dict(D_total=4, D_block_min={charges[1]: minimum}, tol=0.1)
    if method == 'svd':
        u, s, v = tensor.svd_with_truncation(sU=-1, **options)
        reconstructed = u @ s @ v
    else:
        s, u = tensor.eigh_with_truncation(axes=(0, 1), sU=-1, **options)
        reconstructed = u @ s @ u.conj().transpose((1, 0))
    assert charges[1] in s.struct.legs[0].t
    mask = spectrum.truncation_mask(**options)
    expected = spectrum._replace(data=spectrum.data * mask.data).diag()
    assert reconstructed.size > 0 and (reconstructed - expected).norm().item() < 1e-12


def test_minimum_dense_and_empty(config_kwargs):
    config = yastn.make_config(sym='dense', **config_kwargs)
    spectrum = yastn.Tensor(config, isdiag=True)
    spectrum.set_block(Ds=3, val=[0., 2., 1.])
    mask = spectrum.truncation_mask(tol=2., D_block_min={(): 1}, D_total=1)
    np.testing.assert_array_equal(config.backend.to_numpy(mask.data), [False, True, False])
    empty = yastn.Tensor(yastn.make_config(sym='SU2', **config_kwargs), isdiag=True)
    assert not empty.truncation_mask(D_block_min=1, tol=0).struct.legs[0].t
    assert not empty.truncation_mask(D_block_min={0: 1}, tol=0).struct.legs[0].t


@pytest.mark.skipif("'torch' not in config.getoption('--backend')", reason='Requires Torch autograd.')
def test_minimum_mask_preserves_gradient(config_kwargs):
    spectrum, _ = _minimum_spectrum(config_kwargs)
    spectrum.requires_grad_()
    mask = spectrum.truncation_mask(D_block_min={1: 1}, tol=0.1, D_total=4)
    retained = mask.apply_mask(spectrum)
    retained.data.sum().backward()
    np.testing.assert_array_equal(spectrum.config.backend.to_numpy(spectrum.data.grad),
                                  [1., 1., 0., 0., 1., 0., 0.])


def test_truncation_mask_SU2_counts_complete_irreps(config_kwargs):
    config = yastn.make_config(sym='SU2', **config_kwargs)
    leg = yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 1, 1))
    spectrum = yastn.zeros(config, legs=(leg, leg.conj()), isdiag=True)
    spectrum.set_block(ts=(0, 0), Ds=1, val=[0.8])
    spectrum.set_block(ts=(1, 1), Ds=1, val=[1.0])
    spectrum.set_block(ts=(2, 2), Ds=1, val=[0.9])
    mask = spectrum.truncation_mask(D_total=3)
    assert mask[(1, 1)].item() and mask[(0, 0)].item()
    assert not mask[(2, 2)].item()
    assert sum(config.sym.irrep_dimension((j,)) * mask[(j, j)].item()
               for j in (0, 1, 2)) == 3


def test_truncation_mask_SU2xU1_counts_complete_irreps(config_kwargs):
    config = yastn.make_config(sym='SU2xU1', **config_kwargs)
    charges = ((0, 0), (1, 0), (2, 0))
    leg = yastn.Leg(config, s=1, t=charges, D=(1, 1, 1))
    spectrum = yastn.zeros(config, legs=(leg, leg.conj()), isdiag=True)
    for charge, value in zip(charges, (0.8, 1.0, 0.9)):
        spectrum.set_block(ts=charge + charge, Ds=1, val=[value])
    mask = spectrum.truncation_mask(D_total=3)
    assert mask[charges[1] + charges[1]].item()
    assert mask[charges[0] + charges[0]].item()
    assert not mask[charges[2] + charges[2]].item()

@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('which', ['LR', 'LM', 'SR', 'SM'])
@pytest.mark.parametrize('multiplet_opts', [
    {'largest_gap': True},
    {'eps_multiplet': 0.1},
    {'eps_multiplet': 0},
    {'largest_gap': True, 'eps_multiplet': 0.1},
])
def test_nonabelian_truncation_ignores_multiplet_options(config_kwargs, sym, which, multiplet_opts):
    config = yastn.make_config(sym=sym, **config_kwargs)
    charges = ((0,), (1,), (2,)) if sym == 'SU2' else ((0, 0), (1, 1), (2, 2))
    leg = yastn.Leg(config, s=1, t=charges, D=(2, 2, 2))
    spectrum = yastn.zeros(config, legs=(leg, leg.conj()), isdiag=True)
    for charge, values in zip(charges, ([0.8, -0.2], [1.0, 0.4], [-0.9, 0.3])):
        spectrum.set_block(ts=charge + charge, Ds=2, val=values)

    block_opts = {'D_block': 1}
    if which in ('LR', 'LM'):
        block_opts['tol_block'] = 0.5
    for opts in ({'D_total': 5}, {'D_total': 5, **block_opts}, block_opts):
        expected = spectrum.truncation_mask(which=which, **opts)
        actual = spectrum.truncation_mask(which=which, **opts, **multiplet_opts)
        np.testing.assert_array_equal(config.backend.to_numpy(actual.data),
                                      config.backend.to_numpy(expected.data))


tol = 1e-12


def test_svd_multiplets(config_kwargs):
    #
    # fixing tensor for testing
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    S0 = yastn.Tensor(config_U1, isdiag=True)
    S1 = yastn.Tensor(config_U1, isdiag=True)
    v0  = [1, -1, 0.1001, 0.1000, 0.1000, 0.0999, 0.001001, 0.001000, 0, 0, 0]
    vp1 = [1,  1, 0.1000, 0.1000, 0.0010, 0]
    vm1 = [1, -0.98, 0.1000, -0.0999, 0]
    vm2 = [-0.1000, -0.0998, 0.0010]
    S0.set_block(ts=(0, 0), Ds=11, val=v0)
    S0.set_block(ts=(1, 1), Ds=6, val=vp1)
    S0.set_block(ts=(-1, -1), Ds=5, val=vm1)
    S0.set_block(ts=(-2, -2), Ds=3, val=vm2)
    #
    v0  = [-1, 0.001001, 1, 0.001000, 0, 0, 0, 0.1001, 0.1000, 0.1000, 0.0999]
    vp1 = [1, -0.0999, 0, -0.98, 0.1000]
    vm1 = [1, 0.0010, 1, 0.1000, 0.1000, 0]
    vm2 = [-0.0998, 0.0010, -0.1000]
    S1.set_block(ts=(0, 0), Ds=11, val=v0)
    S1.set_block(ts=(1, 1), Ds=5, val=vp1)
    S1.set_block(ts=(-1, -1), Ds=6, val=vm1)
    S1.set_block(ts=(-2, -2), Ds=3, val=vm2)
    #
    Db0 = {(0,): 3, (1,): 4,  (-2,): 5}
    tb0 = {(0,): 0.01, (-1,): 0.5, (-2,): 0.1}
    Db1 = {(0,): 3, (-1,): 4, (-2,): 5}
    tb1 = {(0,): 0.01, (1,): 0.5,  (-2,): 0.1}

    for S, Db, tb in [(S0, Db0, tb0), (S1, Db1, tb1)]:
        #
        #  D_total
        Smask = yastn.truncation_mask(S, which='LM', D_total=12)
        assert sum(Smask.data) == 12 and abs(sum(Smask.data * S.data).item() - 2.4201) < tol
        Smask = yastn.truncation_mask(S, which='LR', D_total=12)
        assert sum(Smask.data) == 12 and abs(sum(Smask.data * S.data).item() - 4.701001) < tol
        Smask = yastn.truncation_mask(S, which='SM', D_total=13)
        assert sum(Smask.data) == 13 and abs(sum(Smask.data * S.data).item() - 0.004201) < tol
        Smask = yastn.truncation_mask(S, which='SR', D_total=11)
        assert sum(Smask.data) == 11 and abs(sum(Smask.data * S.data).item() + 2.2787) < tol
        #
        #  tol
        Smask = yastn.truncation_mask(S, which='LM', tol=0.09999999)
        assert sum(Smask.data) == 13 and abs(sum(Smask.data * S.data).item() - 2.5201) < tol
        Smask = yastn.truncation_mask(S, which='LR', tol=0.09999999)
        assert sum(Smask.data) == 10 and abs(sum(Smask.data * S.data).item() - 4.6001) < tol
        #
        #  D_block
        Smask = yastn.truncation_mask(S, which='LM', D_block=5)
        assert sum(Smask.data) == 18 and abs(sum(Smask.data * S.data).item() - 2.3224) < tol
        Smask = yastn.truncation_mask(S, which='SR', D_block=Db)
        assert sum(Smask.data) == 10 and abs(sum(Smask.data * S.data).item() + 0.9978) < tol
        #
        #  tol_block
        Smask = yastn.truncation_mask(S, which='LM', tol_block=0.09999999)
        assert sum(Smask.data) == 14 and abs(sum(Smask.data * S.data).item() - 2.4203) < tol
        Smask = yastn.truncation_mask(S, which='LR', tol_block=tb)
        assert sum(Smask.data) == 6 and abs(sum(Smask.data * S.data).item() - 2.4000) < tol
        #
        #  hermitian
        Smask = yastn.truncation_mask(S, which='LR', tol=0.09999999, hermitian=True)
        assert sum(Smask.data) == 8 and abs(sum(Smask.data * S.data).item() - 4.4001) < tol
        #
        #  largest_gap
        Smask = yastn.truncation_mask(S, which='LM', D_total=12, largest_gap=True)
        assert sum(Smask.data) == 16 and abs(sum(Smask.data * S.data).item() - 2.4203) < tol
        Smask = yastn.truncation_mask(S, which='LR', D_total=12, largest_gap=True)
        assert sum(Smask.data) == 15 and abs(sum(Smask.data * S.data).item() - 4.704001) < tol
        Smask = yastn.truncation_mask(S, which='LR', D_total=15, largest_gap=True)
        assert sum(Smask.data) == 15 and abs(sum(Smask.data * S.data).item() - 4.704001) < tol
        #
        #  eps_multiplet
        Smask = yastn.truncation_mask(S, which='LM', D_total=12, eps_multiplet=1e-2)
        assert sum(Smask.data) == 6 and abs(sum(Smask.data * S.data).item() - 2.0200) < tol
        Smask = yastn.truncation_mask(S, which='LM', D_total=6, eps_multiplet=1e-2)
        assert sum(Smask.data) == 6 and abs(sum(Smask.data * S.data).item() - 2.0200) < tol
        Smask = yastn.truncation_mask(S, which='LR', D_total=17, eps_multiplet=1e-6)
        assert sum(Smask.data) == 15 and abs(sum(Smask.data * S.data).item() - 4.704001) < tol
        #
        with pytest.raises(yastn.YastnError,
                        match="Truncation by tolerance with which='SR' or 'SM' is not supported."):
            yastn.truncation_mask(S, which="SR", tol=1e-2)
        #
        with pytest.raises(yastn.YastnError,
                        match="Truncation by tolerance with which='SR' or 'SM' is not supported."):
            yastn.truncation_mask(S, which="SM", tol=1e-2)
        #
        with pytest.raises(yastn.YastnError,
                        match="Truncation by block cannot be used when multiplet-related schmes are invoked."):
            yastn.truncation_mask(S, tol_block=1e-2, largest_gap=True)
        #
        with pytest.raises(yastn.YastnError,
                        match="Only which = 'LM' or 'LR' are supported when multiplet-related schmes are invoked."):
            yastn.truncation_mask(S, which='SR', eps_multiplet=1e-2)
        #
        with pytest.raises(yastn.YastnError,
                        match="Truncation multiplets cannot perform both schemes largest_gap and eps_multiplets simultaneously."):
            yastn.truncation_mask(S, eps_multiplet=1e-2, largest_gap=True)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
