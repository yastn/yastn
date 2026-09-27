# Copyright 2024 The YASTN Authors. All Rights Reserved.
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
""" Structured options for the CTM environment routines. """
import json
import pickle
from dataclasses import FrozenInstanceError

import pytest

from yastn import YastnError
from yastn.tn.fpeps.envs._ctm_opts import (
    SIOpts, CTMOpts, FixedPointOpts,
    make_si_opts, make_ctm_opts, make_fixed_point_opts,
    to_dict, from_dict, override, argspec,
    DEFAULT_SVD_TOL, SI_TRUNCATION_KEYS,
)


# ----------------------------------------------------------------------
# defaults
# ----------------------------------------------------------------------

def test_ctm_opts_defaults():
    """ Defaults live in one place and match what the routines used to hardcode. """
    o = make_ctm_opts()
    assert o.moves == 'hv'
    assert o.method == '2x2 corner'
    assert o.max_sweeps == 1
    assert o.corner_tol is None and o.conv_check is None
    assert o.iterator_step == 0
    assert o.use_qr is True
    assert o.cutoff == 0.0
    assert o.opts_si is None and o.si_enabled is False
    assert o.checkpoint_move is False
    assert o.devices is None
    assert o.profiling_mode is None
    assert o.verbosity == 0


def test_si_opts_defaults():
    """ The twelve SI keys, with the values the SI module actually uses. """
    s = make_si_opts()
    assert (s.enabled, s.oversampling, s.niter, s.tol, s.warmup) == (False, 5, 5, 1e-3, 5)
    assert (s.redistribute_sectors, s.redistribute_frequency) == (False, 0)
    assert s.refinement == 'per_sector_oversampling'
    assert s.adaptive_spectrum_iterations == 5
    assert (s.skip_SI_update, s.rebase, s.recycle_grad) == (False, True, False)


def test_svd_tol_fallback():
    """ tol defaults to 1e-14 unless the caller pinned tol or tol_block. """
    assert make_ctm_opts().opts_svd['tol'] == DEFAULT_SVD_TOL
    assert make_ctm_opts(opts_svd={'D_total': 8}).opts_svd['tol'] == DEFAULT_SVD_TOL
    assert make_ctm_opts(opts_svd={'tol': 1e-8}).opts_svd['tol'] == 1e-8
    # tol_block alone suppresses the fallback, as in the original code
    assert 'tol' not in make_ctm_opts(opts_svd={'tol_block': 1e-9}).opts_svd


# ----------------------------------------------------------------------
# unknown keys are rejected, not swallowed
# ----------------------------------------------------------------------

@pytest.mark.parametrize('kwargs', [
    {'max_sweep': 10},          # typo for max_sweeps
    {'opts_sii': {}},           # typo for opts_si
    {'ctm_jobs_vh': []},        # a real stale kwarg found in the test suite
    {'fix_signs': True},        # belongs inside opts_svd
])
def test_unknown_ctm_key_raises(kwargs):
    with pytest.raises(YastnError, match='not recognized'):
        make_ctm_opts(**kwargs)


def test_unknown_key_error_lists_accepted_names():
    with pytest.raises(YastnError, match='max_sweeps'):
        make_ctm_opts(max_sweep=10)


@pytest.mark.parametrize('kwargs', [{'nitr': 3}, {'oversample': 2}])
def test_unknown_si_key_raises(kwargs):
    with pytest.raises(YastnError, match='not recognized'):
        make_si_opts(**kwargs)


def test_renames_without_a_shim_are_hard_errors():
    """ 'correct' and 'correction_frequency' were renamed with no alias,
        so a stored config using them must fail loudly rather than drift. """
    for stale in ('correct', 'correction_frequency'):
        with pytest.raises(YastnError, match='not recognized'):
            make_si_opts(**{stale: True})


# ----------------------------------------------------------------------
# legacy spellings that DO still work
# ----------------------------------------------------------------------

def test_si_alias_asvr_iterations():
    assert make_si_opts(asvr_iterations=7).adaptive_spectrum_iterations == 7
    with pytest.raises(YastnError, match='duplicates'):
        make_si_opts(asvr_iterations=7, adaptive_spectrum_iterations=9)


@pytest.mark.parametrize('old,new', [
    ('cwo', 'per_sector_oversampling'),
    ('asvr', 'adaptive_spectrum'),
    ('rds', 'sector_dimensions'),
])
def test_refinement_acronyms_normalize(old, new):
    assert make_si_opts(refinement=old).refinement == new
    assert make_si_opts(refinement=new).refinement == new


def test_bad_refinement_raises():
    with pytest.raises(YastnError, match='refinement'):
        make_si_opts(refinement='nonsense')


def test_ctm_aliases():
    assert make_ctm_opts(iterator=True).iterator_step == 1
    assert make_ctm_opts(iterator=False).iterator_step == 0
    assert make_ctm_opts(iterator_step=5).iterator_step == 5
    assert make_ctm_opts(opts_svd_ctm={'D_total': 4}).opts_svd['D_total'] == 4


def test_callable_corner_tol_routes_to_conv_check():
    """ corner_tol used to accept a callable; keep that working while making
        the numeric field expressible in a config file. """
    check = lambda env, history: (True, history)
    o = make_ctm_opts(corner_tol=check)
    assert o.conv_check is check and o.corner_tol is None
    o = make_ctm_opts(corner_tol=1e-8)
    assert o.corner_tol == 1e-8 and o.conv_check is None
    with pytest.raises(YastnError, match='not both'):
        make_ctm_opts(corner_tol=check, conv_check=check)


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------

@pytest.mark.parametrize('method', ['2x2', '2x2 corner', '1x2', '1x2 corner',
                                    '2x1 svd', '2x1 qr', '1site', '2site'])
def test_accepted_methods(method):
    """ Every method string the dispatch in _update_projectors_ accepts. """
    assert make_ctm_opts(method=method).method == method


def test_bad_method_raises():
    with pytest.raises(YastnError, match="not recognized"):
        make_ctm_opts(method='something')


@pytest.mark.parametrize('moves', ['hv', 'lrtb', 'h', 'v', 'd'])
def test_accepted_moves(moves):
    assert make_ctm_opts(moves=moves).moves == moves


@pytest.mark.parametrize('moves', ['hx', '', 'xy'])
def test_bad_moves_raises(moves):
    with pytest.raises(YastnError, match='moves'):
        make_ctm_opts(moves=moves)


@pytest.mark.parametrize('value', [False, 'reentrant', 'nonreentrant'])
def test_accepted_checkpoint_move(value):
    assert make_ctm_opts(checkpoint_move=value).checkpoint_move == value


@pytest.mark.parametrize('value', [True, 'yes', 1, 'Reentrant'])
def test_bad_checkpoint_move_raises(value):
    """ checkpoint_move=True was the value that left use_reentrant unbound. """
    with pytest.raises(YastnError, match='checkpoint_move'):
        make_ctm_opts(checkpoint_move=value)


@pytest.mark.parametrize('value', [0, None, ''])
def test_falsy_checkpoint_move_normalizes_to_False(value):
    assert make_ctm_opts(checkpoint_move=value).checkpoint_move is False


def test_negative_numbers_rejected():
    with pytest.raises(YastnError):
        make_ctm_opts(max_sweeps=-1)
    with pytest.raises(YastnError):
        make_ctm_opts(corner_tol=-1e-8)
    with pytest.raises(YastnError):
        make_si_opts(niter=-1)


# ----------------------------------------------------------------------
# immutability / copy-on-construct
# ----------------------------------------------------------------------

def test_frozen():
    o = make_ctm_opts()
    with pytest.raises(FrozenInstanceError):
        o.max_sweeps = 5


def test_opts_svd_copied_on_construct():
    """ A callee must not be able to write through into the caller's dict --
        this is the bug class that let k_block leak across sweeps. """
    caller = {'D_total': 8}
    o = make_ctm_opts(opts_svd=caller)
    assert 'tol' not in caller, "constructor mutated the caller's dict"
    o.opts_svd['sneaky'] = 1          # writing into the stored dict ...
    assert 'sneaky' not in caller     # ... still cannot reach the caller


def test_svd_kwargs_is_fresh_each_call():
    o = make_ctm_opts(opts_svd={'D_total': 8})
    a = o.svd_kwargs(k_block=4)
    b = o.svd_kwargs()
    assert a['k_block'] == 4
    assert 'k_block' not in b, "per-call override leaked into the stored options"
    assert 'k_block' not in o.opts_svd


def test_svd_kwargs_defaults_and_strips():
    o = make_ctm_opts(opts_svd={'D_total': 8, 'verbosity': 3,
                                'profiling_mode': 'NVTX'})
    kw = o.svd_kwargs()
    assert kw['fix_signs'] is True          # CTM's default, not linalg's
    assert kw['D_total'] == 8
    # linalg.svd reads verbosity out of its own **kwargs to log spectra, so it
    # must survive; profiling_mode is read nowhere in linalg, so it is dropped.
    assert kw['verbosity'] == 3
    assert 'profiling_mode' not in kw
    assert o.svd_verbosity() == 3
    # an explicit fix_signs wins over the CTM default
    assert make_ctm_opts(opts_svd={'fix_signs': False}).svd_kwargs()['fix_signs'] is False


def test_si_trunc_kwargs_allowlist():
    o = make_ctm_opts(opts_svd={'D_total': 8, 'eps_multiplet': 1e-8,
                                'policy': 'fullrank', 'svds_thresh': 0.1,
                                'k_block': 4})
    kw = o.si_trunc_kwargs()
    assert set(kw) <= set(SI_TRUNCATION_KEYS)
    assert kw['D_total'] == 8 and kw['eps_multiplet'] == 1e-8
    for dropped in ('policy', 'svds_thresh', 'k_block'):
        assert dropped not in kw


# ----------------------------------------------------------------------
# deriving / merging
# ----------------------------------------------------------------------

def test_make_from_base():
    base = make_ctm_opts(opts_svd={'D_total': 8}, max_sweeps=50, use_qr=False)
    tighter = make_ctm_opts(base, corner_tol=1e-10)
    assert tighter.corner_tol == 1e-10
    assert tighter.max_sweeps == 50 and tighter.use_qr is False
    assert tighter.opts_svd['D_total'] == 8
    assert base.corner_tol is None, "deriving mutated the base"


def test_nested_si_merges_field_wise():
    base = make_ctm_opts(opts_si={'enabled': True, 'oversampling': 2, 'niter': 4})
    derived = make_ctm_opts(base, opts_si={'niter': 1})
    assert derived.opts_si.niter == 1
    assert derived.opts_si.oversampling == 2, "unspecified SI fields were lost"
    assert derived.opts_si.enabled is True
    assert base.opts_si.niter == 4


def test_opts_si_accepts_dict_or_object():
    a = make_ctm_opts(opts_si={'enabled': True, 'niter': 3})
    b = make_ctm_opts(opts_si=SIOpts(enabled=True, niter=3))
    assert a.opts_si == b.opts_si and a.si_enabled
    with pytest.raises(YastnError, match='opts_si'):
        make_ctm_opts(opts_si=7)


def test_devices_normalized_to_tuple():
    assert make_ctm_opts(devices=['cuda:0', 'cuda:1']).devices == ('cuda:0', 'cuda:1')
    assert make_ctm_opts().devices is None
    with pytest.raises(YastnError, match='devices'):
        make_ctm_opts(devices=[])


# ----------------------------------------------------------------------
# config-file / CLI helpers
# ----------------------------------------------------------------------

def test_to_dict_from_dict_round_trip():
    o = make_ctm_opts(opts_svd={'D_total': 64, 'eps_multiplet': 1e-8},
                      max_sweeps=200, corner_tol=1e-8, use_qr=False,
                      checkpoint_move='nonreentrant', devices=['cuda:0'],
                      opts_si={'enabled': True, 'niter': 3, 'refinement': 'adaptive_spectrum'})
    d = to_dict(o)
    assert isinstance(d, dict) and isinstance(d['opts_si'], dict)
    assert d['devices'] == ['cuda:0']            # tuple -> list, for YAML
    assert d['opts_si']['refinement'] == 'adaptive_spectrum'
    assert from_dict(CTMOpts, d) == o


def test_round_trip_through_yaml_like_text():
    """ The nested dict must survive a real serializer, not just dict equality. """
    o = make_ctm_opts(opts_svd={'D_total': 32}, max_sweeps=10,
                      opts_si={'enabled': True, 'oversampling': 2})
    text = json.dumps(to_dict(o))
    assert from_dict(CTMOpts, json.loads(text)) == o


def test_from_dict_rejects_unknown_key():
    d = to_dict(make_ctm_opts())
    d['max_sweep'] = 10
    with pytest.raises(YastnError, match='not recognized'):
        from_dict(CTMOpts, d)


def test_override_dotted_paths():
    o = make_ctm_opts(opts_svd={'D_total': 8}, max_sweeps=10,
                      opts_si={'enabled': True, 'niter': 4, 'oversampling': 2})
    out = override(o, {'max_sweeps': 200,
                       'opts_svd.D_total': 64,
                       'opts_si.niter': 1})
    assert out.max_sweeps == 200
    assert out.opts_svd['D_total'] == 64
    assert out.opts_svd['tol'] == DEFAULT_SVD_TOL, 'other opts_svd keys were dropped'
    assert out.opts_si.niter == 1 and out.opts_si.oversampling == 2
    # the base is untouched
    assert o.max_sweeps == 10 and o.opts_svd['D_total'] == 8 and o.opts_si.niter == 4


def test_override_validates():
    o = make_ctm_opts()
    with pytest.raises(YastnError, match='not recognized'):
        override(o, {'max_sweep': 5})
    with pytest.raises(YastnError, match='not recognized'):
        override(o, {'opts_si.nitr': 5})
    with pytest.raises(YastnError, match='not recognized'):
        override(o, {'method': 'something'})


def test_override_empty_is_identity():
    o = make_ctm_opts(max_sweeps=3)
    assert override(o, {}) is o


def test_layering_defaults_then_file_then_cli():
    """ The intended downstream flow. """
    from_file = from_dict(CTMOpts, {'opts_svd': {'D_total': 16}, 'max_sweeps': 50})
    from_cli = override(from_file, {'opts_svd.D_total': 64, 'corner_tol': 1e-9})
    assert from_cli.opts_svd['D_total'] == 64
    assert from_cli.max_sweeps == 50
    assert from_cli.corner_tol == 1e-9


def test_argspec_covers_schema_with_dotted_nested_names():
    spec = {name: (default, help_, choices)
            for name, _, default, help_, choices in argspec(CTMOpts)}
    assert 'max_sweeps' in spec and 'opts_svd' in spec
    assert 'opts_si.niter' in spec, 'nested dataclass fields must be dotted'
    assert spec['opts_si.niter'][0] == 5
    # choices come from the Literal annotations, not a hand-kept list
    assert spec['checkpoint_move'][2] == (False, 'reentrant', 'nonreentrant')
    assert spec['opts_si.refinement'][2] == (
        'per_sector_oversampling', 'adaptive_spectrum', 'sector_dimensions')
    assert spec['max_sweeps'][2] is None
    assert all(help_ for _, help_, _ in spec.values()), 'every field needs help text'


def test_argspec_help_matches_metadata():
    spec = dict((name, help_) for name, _, _, help_, _ in argspec(CTMOpts))
    assert 'sweep' in spec['max_sweeps'].lower()


# ----------------------------------------------------------------------
# pickling -- the distributed paths ship these to worker processes
# ----------------------------------------------------------------------

def test_pickle_round_trip():
    o = make_ctm_opts(opts_svd={'D_total': 8}, devices=['cuda:0', 'cuda:1'],
                      opts_si={'enabled': True, 'niter': 2},
                      checkpoint_move='reentrant')
    assert pickle.loads(pickle.dumps(o)) == o


def test_pickle_fixed_point_opts():
    fp = make_fixed_point_opts(fwd={'opts_svd': {'D_total': 8}, 'max_sweeps': 50},
                               fp={'opts_svd': {'policy': 'fullrank'}},
                               devices=['cuda:0'])
    assert pickle.loads(pickle.dumps(fp)) == fp


# ----------------------------------------------------------------------
# FixedPointOpts
# ----------------------------------------------------------------------

def test_fixed_point_opts_carries_forward_settings():
    """ FixedPoint reverses the CTM by hand, so forward settings carry over to
        the backward path. The Neumann budget IS the FP step's sweep budget. """
    fp = make_fixed_point_opts(
        fwd={'opts_svd': {'D_total': 64}, 'max_sweeps': 100, 'corner_tol': 1e-8},
        fp={'opts_svd': {'policy': 'fullrank'}},
        devices=['cuda:0', 'cuda:1'])
    assert fp.fwd.max_sweeps == 100 and fp.fwd.corner_tol == 1e-8
    assert fp.fp.opts_svd['policy'] == 'fullrank'
    assert fp.devices == ('cuda:0', 'cuda:1')
    # the Neumann loop reads its budget/tolerance from the FP step, by design
    assert fp.neumann_max_iter == fp.fp.max_sweeps
    assert fp.neumann_tol == fp.fp.corner_tol


def test_from_legacy_dicts_reproduces_the_historical_merge():
    """ fp = deepcopy(fwd), then opts_svd merged key-by-key and the rest
        overridden -- the derivation the backward pass depends on. """
    fwd_d = {'method': '2x1', 'max_sweeps': 7, 'use_qr': False,
             'corner_tol': 1e-9, 'opts_svd': {'D_total': 8}}
    fp_d = {'opts_svd': {'policy': 'fullrank'}}
    o = FixedPointOpts.from_legacy_dicts(fwd_d, fp_d)

    # carried over from fwd
    assert o.fp.method == '2x1' and o.fp.use_qr is False
    assert o.fp.max_sweeps == 7 and o.fp.corner_tol == 1e-9
    # opts_svd merged, not replaced
    assert o.fp.opts_svd['D_total'] == 8 and o.fp.opts_svd['policy'] == 'fullrank'
    # and therefore the Neumann loop inherits the forward budget
    assert o.neumann_max_iter == 7 and o.neumann_tol == 1e-9
    # caller dicts untouched
    assert fwd_d['opts_svd'] == {'D_total': 8} and fp_d['opts_svd'] == {'policy': 'fullrank'}


def test_from_legacy_dicts_fp_overrides_win():
    o = FixedPointOpts.from_legacy_dicts(
        {'max_sweeps': 7, 'corner_tol': 1e-9, 'opts_svd': {'D_total': 8}},
        {'max_sweeps': 3, 'corner_tol': 1e-12})
    assert o.fp.max_sweeps == 3 and o.fp.corner_tol == 1e-12
    assert o.neumann_max_iter == 3 and o.neumann_tol == 1e-12
    assert o.fwd.max_sweeps == 7, "overriding fp must not touch fwd"


def test_from_legacy_dicts_lifts_non_ctm_keys():
    """ neumann_patience and fp_devices are not CTM options; before they were
        lifted out, make_ctm_opts rejected them (a Phase-2 regression). """
    o = FixedPointOpts.from_legacy_dicts(
        {'opts_svd': {'D_total': 8}, 'max_sweeps': 5},
        {'neumann_patience': 3, 'fp_devices': ['cuda:0', 'cuda:1']})
    assert o.neumann_patience == 3
    assert o.devices == ('cuda:0', 'cuda:1')


def test_from_legacy_dicts_backward_verbosity_does_not_inherit():
    """ ctx.verbosity came from ctm_opts_fp alone, defaulting to 0 -- distinct
        from fp.verbosity, which is the merged CTM-level setting. """
    o = FixedPointOpts.from_legacy_dicts({'verbosity': 3, 'opts_svd': {}}, {})
    assert o.verbosity == 0, "backward verbosity must not inherit fwd's"
    assert o.fp.verbosity == 3, "the CTM-level setting still carries over"
    o2 = FixedPointOpts.from_legacy_dicts({'verbosity': 3, 'opts_svd': {}},
                                          {'verbosity': 1})
    assert o2.verbosity == 1 and o2.fp.verbosity == 1


def test_from_legacy_dicts_defaults():
    o = FixedPointOpts.from_legacy_dicts()
    assert o.fwd == CTMOpts() and o.fp == CTMOpts()
    assert o.neumann_patience == 10 and o.devices is None and o.verbosity == 0


def test_fixed_point_opts_round_trip_and_override():
    fp = make_fixed_point_opts(fwd={'opts_svd': {'D_total': 8}})
    assert from_dict(FixedPointOpts, to_dict(fp)) == fp
    out = override(fp, {'fwd.opts_svd.D_total': 32, 'neumann_patience': 4})
    assert out.fwd.opts_svd['D_total'] == 32 and out.neumann_patience == 4
    assert fp.fwd.opts_svd['D_total'] == 8


def test_fixed_point_opts_rejects_unknown():
    with pytest.raises(YastnError, match='not recognized'):
        make_fixed_point_opts(fp_devices=['cuda:0'])   # the old smuggled key


def test_override_can_clear_a_field():
    """ An override is an explicit assignment, so null in a config file must
        clear the field rather than be ignored (unlike a signature default). """
    o = make_ctm_opts(corner_tol=1e-8, profiling_mode='NVTX')
    out = override(o, {'corner_tol': None, 'profiling_mode': None})
    assert out.corner_tol is None and out.profiling_mode is None
    # ... whereas make_ctm_opts treats None as "not supplied", so that a public
    # signature can forward all of its defaults without clobbering a base.
    assert make_ctm_opts(o, corner_tol=None).corner_tol == 1e-8


def test_override_still_validates_after_the_replace():
    o = make_ctm_opts()
    with pytest.raises(YastnError, match='checkpoint_move'):
        override(o, {'checkpoint_move': True})
    with pytest.raises(YastnError, match='not recognized'):
        override(o, {'opts_si.refinement': 'nonsense'})


def test_override_accepts_legacy_names():
    o = make_ctm_opts(opts_si={'enabled': True})
    assert override(o, {'opts_si.asvr_iterations': 7}).opts_si.adaptive_spectrum_iterations == 7
    assert override(o, {'opts_si.refinement': 'cwo'}).opts_si.refinement == 'per_sector_oversampling'


def test_refinement_constant_matches_the_literal():
    """ REFINEMENTS is validated against; the Literal drives argspec choices.
        They must not drift apart. """
    from typing import get_args, get_type_hints
    from yastn.tn.fpeps.envs._ctm_opts import REFINEMENTS
    assert get_args(get_type_hints(SIOpts)['refinement']) == REFINEMENTS


def test_none_uniformly_means_not_supplied():
    base = make_ctm_opts(opts_si={'enabled': True, 'niter': 3}, corner_tol=1e-8)
    assert make_ctm_opts(base, opts_si=None, corner_tol=None) == base


def test_fix_signs_default_is_single_sourced():
    """ CTM wants a deterministic SVD gauge, unlike linalg's own fix_signs=False.
        proj_corners re-applies it for callers that still pass a raw dict, so the
        value must be named once rather than written out at both sites. """
    from yastn.tn.fpeps.envs import _env_ctm
    from yastn.tn.fpeps.envs._ctm_opts import DEFAULT_FIX_SIGNS
    import inspect
    assert make_ctm_opts().svd_kwargs()['fix_signs'] is DEFAULT_FIX_SIGNS
    src = inspect.getsource(_env_ctm.proj_corners)
    assert 'DEFAULT_FIX_SIGNS' in src and "'fix_signs', True" not in src


def test_nested_bundles_merge_into_the_base():
    """ opts_svd and opts_si follow the same rule: deriving from a base merges
        into it rather than replacing it. The fixed-point layer's
        fp-derived-from-fwd relationship depends on this for opts_svd. """
    base = make_ctm_opts(opts_svd={'D_total': 8, 'tol': 1e-10},
                         opts_si={'enabled': True, 'oversampling': 2, 'niter': 4})
    derived = make_ctm_opts(base, opts_svd={'policy': 'fullrank'},
                            opts_si={'niter': 1})
    assert derived.opts_svd == {'D_total': 8, 'tol': 1e-10, 'policy': 'fullrank'}
    assert derived.opts_si.oversampling == 2 and derived.opts_si.niter == 1
    # base untouched
    assert 'policy' not in base.opts_svd and base.opts_si.niter == 4
    # with no base there is nothing to merge into
    assert make_ctm_opts(opts_svd={'policy': 'fullrank'}).opts_svd == {
        'policy': 'fullrank', 'tol': DEFAULT_SVD_TOL}


def test_fixed_point_opts_pickles():
    """ The FP step ships these to worker processes on the distributed path. """
    o = FixedPointOpts.from_legacy_dicts(
        {'opts_svd': {'D_total': 8}, 'max_sweeps': 5},
        {'opts_svd': {'policy': 'fullrank'}, 'neumann_patience': 4},
        devices=['cpu', 'cpu'])
    assert pickle.loads(pickle.dumps(o)) == o


def test_fixed_point_opts_frozen_and_derivable():
    o = FixedPointOpts.from_legacy_dicts({'max_sweeps': 5, 'opts_svd': {}})
    with pytest.raises(FrozenInstanceError):
        o.neumann_patience = 1
    tighter = make_fixed_point_opts(o, neumann_patience=2)
    assert tighter.neumann_patience == 2 and o.neumann_patience == 10
    assert tighter.fwd == o.fwd and tighter.fp == o.fp


def test_iterator_default_does_not_clobber_a_supplied_base():
    """ iterator=False is a signature default, not a caller's choice -- it must
        not override iterator_step coming from `opts=`. A non-None default here
        silently turned a generator call back into an eager one. """
    base = make_ctm_opts(opts_svd={'D_total': 4}, iterator_step=1)
    assert make_ctm_opts(base, iterator=None).iterator_step == 1
    # an explicit choice still wins
    assert make_ctm_opts(base, iterator=False).iterator_step == 0
    assert make_ctm_opts(base, iterator=True).iterator_step == 1
