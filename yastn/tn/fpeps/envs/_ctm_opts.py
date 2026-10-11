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
r"""
Structured options for the PEPS CTM environment routines.

* :class:`CTMOpts` -- main CTM algorithm options.
* :class:`SIOpts` -- subspace-iteration options (optionally used to accelerate CTM).
* :class:`FixedPointOpts` -- the CTM with fixed-point gradient algorithm.

All three are plain stdlib dataclasses carrying only config-representable
types, so a downstream application can build them from a YAML/TOML/JSON file
with :func:`from_dict`, layer command-line overrides on top with
:func:`override`, and generate its own CLI flags from :func:`argspec` -- without
YASTN depending on any configuration framework.

Single point of reference for default values. :func:`make_ctm_opts` is the single
boundary that accepts loose keyword arguments; it rejects unknown names.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass, replace
from typing import Any, Callable, Literal, get_args, get_origin, get_type_hints

from ....tensor import YastnError


__all__ = ['SIOpts', 'CTMOpts', 'FixedPointOpts', 'make_si_opts', 'make_ctm_opts',
           'DEFAULT_SVD_TOL', 'DEFAULT_FIX_SIGNS',
           'make_fixed_point_opts',
           'to_dict', 'from_dict', 'override', 'argspec']


# Default ``tol`` handed to svd_with_truncation when the caller pinned neither
# 'tol' nor 'tol_block'.
DEFAULT_SVD_TOL = 1e-14

# Force deterministic (CTM) projectors SVD gauge.
DEFAULT_FIX_SIGNS = True

# Truncation keys the SI projector path forwards to ``truncation_mask``.
# Narrower than the full opts_svd since SI path always uses full-rank SVD.
SI_TRUNCATION_KEYS = ('tol', 'tol_block', 'D_block', 'D_total', 'largest_gap',
                      'eps_multiplet', 'hermitian', 'mask_f')

# Keys that are meaningful to CTM but are not svd_with_truncation arguments,
# and so must be stripped before opts_svd is splatted into it.
# NOTE 'verbosity' is deliberately NOT here: yastn.linalg reads it out of its
# own **kwargs (linalg.py:145, 247, 625, 864) to log spectra, and the
# fixed-point tests set it inside opts_svd. Stripping it would silently
# disable that. 'profiling_mode' by contrast is read nowhere in linalg.
_CTM_ONLY_SVD_KEYS = ('profiling_mode',)

# Renamed options still accepted on input. The canonical name is the value.
# These exist purely as a deprecation shim; delete a row once no stored config
# can carry the old name.
_SI_ALIASES = {'asvr_iterations': 'adaptive_spectrum_iterations'}
_REFINEMENT_ALIASES = {'cwo': 'per_sector_oversampling',
                       'asvr': 'adaptive_spectrum',
                       'rds': 'sector_dimensions'}

# Must stay in step with the Literal on SIOpts.refinement; a test asserts it.
REFINEMENTS = ('per_sector_oversampling', 'adaptive_spectrum', 'sector_dimensions')
_CTM_ALIASES = {'opts_svd_ctm': 'opts_svd'}

_MOVES = 'hvlrtbd'


def _unknown(name, allowed, what):
    return YastnError(
        f"{what} {name!r} not recognized. Accepted: {', '.join(sorted(allowed))}.")


@dataclass(frozen=True, kw_only=True)
class SIOpts:
    r"""
    Options of the recycled subspace-iteration (SI) projectors.

    Parameters
    ----------
    enabled: bool
        Whether SI projectors are used at all. The default is ``False``.
    oversampling: int
        Extra directions carried beyond the requested bond dimension.
    niter: int
        Number of subspace iterations when adjusting the range-finders.
    tol: float
        Target subspace error of the range-finders, weighted by the singular
        value of each direction.
    warmup: int
        Number of projector updates before the redistribution schedule starts.
    redistribute_sectors: bool
        Reallocate the SI rank between charge sectors, on the warmup /
        ``redistribute_frequency`` schedule.
    redistribute_frequency: int
        Redistribute every that many updates once past ``warmup``.
        ``0`` disables the recurring redistribution.
    refinement: str
        Sector-refinement algorithm: ``'per_sector_oversampling'``,
        ``'adaptive_spectrum'`` or ``'sector_dimensions'``.
    adaptive_spectrum_iterations: int
        Refinement passes used by ``refinement='adaptive_spectrum'``.
        Ignored by the other refinements.
    skip_SI_update: bool
        Skip the subspace iteration entirely on an update that is past
        ``warmup``, outside the redistribution schedule, and whose bases
        already report an error below ``tol``. Trades accuracy for speed.
    rebase: bool
        When changed corner legs invalidate the recycled bases, carry them onto
        the new row space instead of drawing fresh ones.
    recycle_grad: bool
        Keep the recycled bases attached to the autograd graph.
    """
    enabled: bool = field(default=False, metadata={
        'help': 'use recycled subspace-iteration projectors'})
    oversampling: int = field(default=5, metadata={
        'help': 'extra directions beyond the requested bond dimension'})
    niter: int = field(default=5, metadata={
        'help': 'subspace iterations when adjusting range-finders'})
    tol: float = field(default=1e-3, metadata={
        'help': 'target weighted subspace error of the range-finders'})
    warmup: int = field(default=5, metadata={
        'help': 'projector updates before the redistribution schedule starts'})
    redistribute_sectors: bool = field(default=False, metadata={
        'help': 'reallocate SI rank between charge sectors'})
    redistribute_frequency: int = field(default=0, metadata={
        'help': 'redistribute every N updates past warmup; 0 disables'})
    refinement: Literal['per_sector_oversampling',
                        'adaptive_spectrum',
                        'sector_dimensions'] = field(
        default='per_sector_oversampling',
        metadata={'help': 'sector-refinement algorithm'})
    adaptive_spectrum_iterations: int = field(default=5, metadata={
        'help': "refinement passes for refinement='adaptive_spectrum'"})
    skip_SI_update: bool = field(default=False, metadata={
        'help': 'skip subspace iteration once the bases are converged'})
    rebase: bool = field(default=True, metadata={
        'help': 'carry invalidated bases onto changed corner legs'})
    recycle_grad: bool = field(default=False, metadata={
        'help': 'keep recycled bases attached to the autograd graph'})

    def __post_init__(self):
        refinement = _REFINEMENT_ALIASES.get(self.refinement, self.refinement)
        if refinement != self.refinement:
            object.__setattr__(self, 'refinement', refinement)
        if refinement not in REFINEMENTS:
            raise YastnError(
                f"SI {refinement=} not recognized. Accepted: "
                f"{', '.join(repr(a) for a in REFINEMENTS)} "
                f"(or the former acronyms {', '.join(repr(a) for a in _REFINEMENT_ALIASES)}).")
        for name in ('oversampling', 'niter', 'warmup',
                     'redistribute_frequency', 'adaptive_spectrum_iterations'):
            if getattr(self, name) < 0:
                raise YastnError(f"SI {name}={getattr(self, name)} must be non-negative.")


@dataclass(frozen=True, kw_only=True)
class CTMOpts:
    r"""
    Options of a CTM environment sweep.

    Construct with :func:`make_ctm_opts`, which accepts the legacy keyword
    spellings and reports unknown names. Direct construction is fine too; all
    normalization and validation happens in ``__post_init__``, so
    ``dataclasses.replace`` and :func:`from_dict` are equally safe.

    Parameters
    ----------
    moves: str
        Sequence of moves forming a single sweep, drawn from ``'l'``, ``'r'``,
        ``'t'``, ``'b'``, ``'h'``, ``'v'`` (and ``'d'`` for the c4v variant).
        Sensible choices are ``'hv'`` and ``'lrtb'``.
    method: str
        Projector construction. Must contain ``'2x2'``, ``'1x2'`` or ``'2x1'``,
        or be exactly ``'1site'`` / ``'2site'``.
    max_sweeps: int
        Maximal number of sweeps.
    corner_tol: float | None
        Convergence tolerance on the change of corner singular values.
        ``None`` disables the built-in check.
    conv_check: Callable | None
        Custom convergence check ``f(env, history) -> (converged, history)``,
        used instead of the ``corner_tol`` comparison. Previously this was
        passed as a callable ``corner_tol``; splitting the two keeps
        ``corner_tol`` expressible in a configuration file.
    iterator_step: int
        Yield an intermediate result every that many sweeps. ``0`` runs all
        sweeps eagerly and returns a single result.
    opts_svd: dict
        Options forwarded to :meth:`yastn.linalg.svd_with_truncation`, setting
        the environment bond dimension. Copied on construction; treat the
        stored dict as read-only and go through :meth:`svd_kwargs`.
    use_qr: bool
        Whether to include an intermediate QR while calculating projectors.
    cutoff: float
        Absolute pseudo-inverse cutoff in projector construction.
    opts_si: SIOpts | None
        Subspace-iteration options; see :class:`SIOpts`. A plain dict is
        accepted and converted.
    checkpoint_move: False | str
        ``'reentrant'`` or ``'nonreentrant'`` to checkpoint each move on the
        PyTorch backend; ``False`` disables checkpointing.
    devices: tuple[str, ...] | None
        Devices for the distributed CTM paths. A list is accepted and converted.
    profiling_mode: str | None
        Profiling instrumentation, e.g. ``'NVTX'``.
    verbosity: int
        Diagnostic verbosity.
    """
    # --- sweep control ---
    moves: str = field(default='hv', metadata={
        'help': "moves forming one sweep, e.g. 'hv' or 'lrtb'"})
    method: str = field(default='2x2 corner', metadata={
        'help': "projector construction; contains '2x2', '1x2' or '2x1'"})
    max_sweeps: int = field(default=1, metadata={'help': 'maximal number of sweeps'})
    corner_tol: float | None = field(default=None, metadata={
        'help': 'tolerance on the change of corner singular values'})
    conv_check: Callable | None = field(default=None, metadata={
        'help': 'custom convergence check (not expressible in a config file)'})
    iterator_step: int = field(default=0, metadata={
        'help': 'yield an intermediate result every N sweeps; 0 runs eagerly'})
    # --- projector construction ---
    opts_svd: dict[str, Any] = field(default_factory=dict, metadata={
        'help': 'options forwarded to svd_with_truncation'})
    use_qr: bool = field(default=True, metadata={
        'help': 'intermediate QR while calculating projectors'})
    cutoff: float = field(default=0.0, metadata={
        'help': 'absolute pseudo-inverse cutoff in projector construction'})
    opts_si: SIOpts | None = field(default=None, metadata={
        'help': 'subspace-iteration options; see SIOpts'})
    # --- execution ---
    checkpoint_move: Literal[False, 'reentrant', 'nonreentrant'] = field(
        default=False, metadata={'help': 'checkpoint each move (PyTorch backend)'})
    devices: tuple[str, ...] | None = field(default=None, metadata={
        'help': 'devices for the distributed CTM paths'})
    profiling_mode: str | None = field(default=None, metadata={
        'help': "profiling instrumentation, e.g. 'NVTX'"})
    verbosity: int = field(default=0, metadata={'help': 'diagnostic verbosity'})

    def __post_init__(self):
        st = lambda name, value: object.__setattr__(self, name, value)

        # A callable corner_tol used to be the way to plug in a custom check.
        if callable(self.corner_tol):
            if self.conv_check is not None:
                raise YastnError("Pass either a callable corner_tol or conv_check, not both.")
            st('conv_check', self.corner_tol)
            st('corner_tol', None)
        if self.corner_tol is not None and self.corner_tol < 0:
            raise YastnError(f"corner_tol={self.corner_tol} must be non-negative or None.")

        # opts_svd is caller-owned; copy it so no callee can write through.
        opts_svd = dict(self.opts_svd)
        if 'tol' not in opts_svd and 'tol_block' not in opts_svd:
            opts_svd['tol'] = DEFAULT_SVD_TOL
        st('opts_svd', opts_svd)

        if isinstance(self.opts_si, dict):
            st('opts_si', make_si_opts(**self.opts_si))
        elif self.opts_si is not None and not isinstance(self.opts_si, SIOpts):
            raise YastnError(f"opts_si must be a dict or SIOpts, got {type(self.opts_si).__name__}.")

        if self.devices is not None:
            st('devices', tuple(self.devices))
            if not self.devices:
                raise YastnError("devices must be a non-empty sequence, or None.")

        if not self.checkpoint_move:  # normalize 0/None/'' to False
            st('checkpoint_move', False)
        elif self.checkpoint_move not in ('reentrant', 'nonreentrant'):
            raise YastnError(
                f"checkpoint_move={self.checkpoint_move!r} not recognized. "
                "Should be 'reentrant', 'nonreentrant', or False.")

        # Mirror the dispatch in EnvCTM._update_projectors_ exactly, so that no
        # method string accepted by the code is rejected here.
        m = self.method
        if not ('1x2' in m or '2x1' in m or '2x2' in m or m in ('1site', '2site')):
            raise YastnError(
                f"CTM update method={m!r} not recognized. Should contain '1x2' or '2x2'.")

        bad = sorted(set(self.moves) - set(_MOVES))
        if bad or not self.moves:
            raise YastnError(
                f"CTM moves={self.moves!r} not recognized. "
                f"Each move should be one of {', '.join(repr(c) for c in _MOVES)}.")

        if self.max_sweeps < 0 or self.iterator_step < 0:
            raise YastnError("max_sweeps and iterator_step must be non-negative.")

    # ------------------------------------------------------------------
    @property
    def si_enabled(self) -> bool:
        """Whether SI projectors are active for this sweep."""
        return self.opts_si is not None and self.opts_si.enabled

    def svd_kwargs(self, **overrides) -> dict:
        r"""
        The keyword arguments for :meth:`yastn.linalg.svd_with_truncation`.

        This is the single boundary at which options become SVD arguments.
        Returns a fresh dict every call, so a per-pair prediction such as
        ``svd_kwargs(k_block=...)`` never leaks into the next projector or the
        next sweep.
        """
        kwargs = {k: v for k, v in self.opts_svd.items()
                  if k not in _CTM_ONLY_SVD_KEYS}
        kwargs.setdefault('fix_signs', DEFAULT_FIX_SIGNS)
        # 'verbosity' stays in: linalg.svd reads it to log spectra.
        kwargs.update(overrides)
        return kwargs

    def si_trunc_kwargs(self) -> dict:
        r"""
        The subset of ``opts_svd`` the SI path forwards to ``truncation_mask``.

        Narrower than :meth:`svd_kwargs` by design; see
        ``SI_TRUNCATION_KEYS``.
        """
        return {k: self.opts_svd[k] for k in SI_TRUNCATION_KEYS if k in self.opts_svd}

    def svd_verbosity(self) -> int:
        """``verbosity`` as carried inside ``opts_svd``, falling back to the sweep-level one."""
        return self.opts_svd.get('verbosity', self.verbosity)


@dataclass(frozen=True, kw_only=True)
class FixedPointOpts:
    r"""
    Options of the fixed-point CTM, see :func:`yastn.tn.fpeps.envs.fixed_pt.fp_ctmrg`.

    ``FixedPoint`` reverses the CTM, so the forward settings carry over
    to the backward path by design: ``fp`` is ``fwd`` with selective overrides,
    and the Neumann series in the backward pass takes its iteration budget from
    ``fp.max_sweeps`` and its gradient tolerance from ``fp.corner_tol``.
    The additional configuration specific to fixed point approach is defined here.
    
    Parameters
    ----------
    fwd: CTMOpts
        Options of the forward CTMRG convergence.
    fp: CTMOpts
        Options of the single gauge-fixing CTM step, and -- through
        ``max_sweeps`` and ``corner_tol`` -- of the Neumann backward loop.
        Built from ``fwd`` with overrides applied.
    neumann_patience: int
        Iterations without improvement before the Neumann series gives up.
        Unlike the budget and tolerance, this has no forward counterpart.
    devices: tuple[str, ...] | None
        Devices for the distributed fixed-point path. Was ``fp_devices``.
    verbosity: int
        Diagnostic verbosity.
    """
    fwd: CTMOpts = field(default_factory=CTMOpts, metadata={
        'help': 'options of the forward CTMRG convergence'})
    fp: CTMOpts = field(default_factory=CTMOpts, metadata={
        'help': 'options of the gauge-fixing CTM step; inherits from fwd'})
    neumann_patience: int = field(default=10, metadata={
        'help': 'Neumann iterations without improvement before giving up'})
    devices: tuple[str, ...] | None = field(default=None, metadata={
        'help': 'devices for the distributed fixed-point path'})
    verbosity: int = field(default=0, metadata={'help': 'diagnostic verbosity'})

    def __post_init__(self):
        for name in ('fwd', 'fp'):
            value = getattr(self, name)
            if isinstance(value, dict):
                object.__setattr__(self, name, make_ctm_opts(**value))
            elif not isinstance(value, CTMOpts):
                raise YastnError(
                    f"FixedPointOpts.{name} must be a dict or CTMOpts, "
                    f"got {type(value).__name__}.")
        if self.devices is not None:
            object.__setattr__(self, 'devices', tuple(self.devices))
        if self.neumann_patience < 0:
            raise YastnError("neumann_patience must be non-negative.")

    # ------------------------------------------------------------------
    @property
    def neumann_max_iter(self) -> int:
        """Neumann iteration budget, carried over from the FP step's sweep budget."""
        return self.fp.max_sweeps

    @property
    def neumann_tol(self) -> float:
        """Neumann gradient tolerance, carried over from the FP step's corner tolerance."""
        return self.fp.corner_tol

    @classmethod
    def from_legacy_dicts(cls, ctm_opts_fwd=None, ctm_opts_fp=None, devices=None):
        r"""
        Build from the ``ctm_opts_fwd`` / ``ctm_opts_fp`` dicts of ``fp_ctmrg``.

        Reproduces the historical derivation exactly: ``fp`` starts as a copy of
        ``fwd`` and is then overridden selectively, with ``opts_svd`` merged key
        by key rather than replaced.

        Keys that are not CTM options (``neumann_patience``, and the legacy
        ``fp_devices``) are lifted out before the rest is handed to
        :func:`make_ctm_opts`, which would otherwise reject them.
        """
        fwd_kwargs = dict(ctm_opts_fwd) if ctm_opts_fwd else {}
        fp_kwargs = dict(ctm_opts_fp) if ctm_opts_fp else {}

        # Not CTM options: make_ctm_opts would reject them.
        patience = fp_kwargs.pop('neumann_patience',
                                 fwd_kwargs.pop('neumann_patience', 10))
        legacy_devices = fp_kwargs.pop('fp_devices', None)

        # The diagnostic verbosity of the backward pass came from ctm_opts_fp
        # alone, defaulting to 0 -- it did NOT inherit fwd's. Keep that: it is
        # distinct from fp.verbosity, which is the merged CTM-level setting.
        verbosity = fp_kwargs.get('verbosity', 0)

        fwd = make_ctm_opts(**fwd_kwargs)
        fp = make_ctm_opts(fwd, **fp_kwargs)  # inherit everything, override selectively
        return cls(fwd=fwd, fp=fp, devices=devices or legacy_devices,
                   neumann_patience=patience, verbosity=verbosity)


# ----------------------------------------------------------------------
# factories
# ----------------------------------------------------------------------

def _resolve(kwargs, aliases, allowed, what):
    """Apply the alias table, then reject anything not in ``allowed``."""
    out = {}
    for name, value in kwargs.items():
        canonical = aliases.get(name, name)
        if canonical not in allowed:
            raise _unknown(name, allowed, what)
        if canonical in out:
            raise YastnError(
                f"{what} {name!r} duplicates {canonical!r}; pass only one.")
        out[canonical] = value
    return out


def make_si_opts(base: SIOpts | None = None, **kwargs) -> SIOpts:
    r"""
    Build :class:`SIOpts`, accepting the legacy key spellings.

    Unknown keys raise :class:`YastnError` rather than being silently dropped.
    """
    allowed = {f.name for f in fields(SIOpts)}
    resolved = _resolve(kwargs, _SI_ALIASES, allowed, 'SI option')
    if base is None:
        return SIOpts(**resolved)
    if not isinstance(base, SIOpts):
        raise YastnError(f"base must be SIOpts or None, got {type(base).__name__}.")
    return replace(base, **resolved) if resolved else base


def make_ctm_opts(base: CTMOpts | None = None, **kwargs) -> CTMOpts:
    r"""
    Build :class:`CTMOpts` from loose keyword arguments.

    This is the single entry point through which every calling convention
    funnels -- a direct keyword call, :func:`from_dict` for a configuration
    file, and :func:`override` for command-line layering -- so the unknown-key
    error cannot be bypassed.

    ``base`` supplies the starting values; anything passed as a keyword
    overrides it. ``iterator=True`` is accepted as the historical spelling of
    ``iterator_step=1``, and ``opts_svd_ctm`` as that of ``opts_svd``.

    Parameters
    ----------
    base: CTMOpts | None
        Options to derive from. ``None`` starts from the defaults.

    Example
    -------

    ::

        opts = make_ctm_opts(opts_svd={'D_total': 64}, max_sweeps=200,
                             corner_tol=1e-8, use_qr=False)
        tighter = make_ctm_opts(opts, corner_tol=1e-10)
    """
    if base is not None and not isinstance(base, CTMOpts):
        raise YastnError(f"base must be CTMOpts or None, got {type(base).__name__}.")

    # None uniformly means "not supplied", so a public signature can forward all
    # of its defaults without clobbering `base`. To clear a field, use override().
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # 'iterator' is a bool spelling of iterator_step.
    if 'iterator' in kwargs:
        iterator = kwargs.pop('iterator')
        if 'iterator_step' not in kwargs:
            kwargs['iterator_step'] = int(iterator)

    allowed = {f.name for f in fields(CTMOpts)}
    resolved = _resolve(kwargs, _CTM_ALIASES, allowed, 'CTM option')

    # Nested bundles are merged into the base's rather than replacing them, so
    # make_ctm_opts(base, opts_si={'niter': 3}) keeps the rest of base's SIOpts
    # and make_ctm_opts(base, opts_svd={'policy': ...}) keeps base's truncation.
    # This is what the fixed-point layer's fp-derived-from-fwd merge needs, and
    # it is the same rule for both bundles.
    if isinstance(resolved.get('opts_si'), dict):
        current = base.opts_si if base is not None else None
        resolved['opts_si'] = make_si_opts(current, **resolved['opts_si'])

    if base is not None and isinstance(resolved.get('opts_svd'), dict):
        resolved['opts_svd'] = {**base.opts_svd, **resolved['opts_svd']}

    if base is None:
        return CTMOpts(**resolved)
    return replace(base, **resolved) if resolved else base


def make_fixed_point_opts(base: FixedPointOpts | None = None, **kwargs) -> FixedPointOpts:
    r"""
    Build :class:`FixedPointOpts`. ``fwd`` and ``fp`` accept plain dicts.
    """
    if base is not None and not isinstance(base, FixedPointOpts):
        raise YastnError(f"base must be FixedPointOpts or None, got {type(base).__name__}.")
    allowed = {f.name for f in fields(FixedPointOpts)}
    resolved = _resolve(kwargs, {}, allowed, 'fixed-point option')
    for name in ('fwd', 'fp'):
        if isinstance(resolved.get(name), dict):
            current = getattr(base, name) if base is not None else None
            resolved[name] = make_ctm_opts(current, **resolved[name])
    if base is None:
        return FixedPointOpts(**resolved)
    return replace(base, **resolved) if resolved else base


# ----------------------------------------------------------------------
# configuration-file and CLI helpers
# ----------------------------------------------------------------------

_FACTORIES = {SIOpts: make_si_opts, CTMOpts: make_ctm_opts,
              FixedPointOpts: make_fixed_point_opts}


def to_dict(opts) -> dict:
    r"""
    Nested plain-dict representation, ready for YAML/JSON.

    Note that a callable option has no serializable form: ``conv_check``, and
    ``mask_f`` inside ``opts_svd``, come out as the objects themselves. They
    round-trip in memory but not through a file, and :func:`from_dict` cannot
    set them -- code must.
    """
    if not is_dataclass(opts):
        raise YastnError(f"to_dict expects an options dataclass, got {type(opts).__name__}.")
    out = {}
    for f in fields(opts):
        value = getattr(opts, f.name)
        if is_dataclass(value):
            out[f.name] = to_dict(value)
        elif isinstance(value, dict):
            out[f.name] = dict(value)
        elif isinstance(value, tuple):
            out[f.name] = list(value)
        else:
            out[f.name] = value
    return out


def from_dict(cls, d: dict):
    r"""
    Build an options object of type ``cls`` from a nested plain dict.

    The inverse of :func:`to_dict`, and the entry point for a configuration
    file. Validates through the same factory as every other path, so an
    unknown key in the file is an error rather than a silent no-op.
    """
    factory = _FACTORIES.get(cls)
    if factory is None:
        raise YastnError(
            f"from_dict expects one of {', '.join(c.__name__ for c in _FACTORIES)}, "
            f"got {getattr(cls, '__name__', cls)!r}.")
    if not isinstance(d, dict):
        raise YastnError(f"from_dict expects a dict, got {type(d).__name__}.")
    return factory(**d)


def override(opts, overrides: dict):
    r"""
    Layer dotted-path overrides on top of ``opts``, returning a new object.

    The command-line half of the configuration story: a config file builds the
    base with :func:`from_dict`, then flags layered on top override individual
    leaves.

    ::

        opts = override(opts, {'max_sweeps': 200, 'opts_svd.D_total': 64,
                               'opts_si.niter': 3})

    A path may address a dataclass field, a nested dataclass field, or a key
    inside ``opts_svd``. Each layer is validated.
    """
    if not overrides:
        return opts
    tree = {}
    for dotted, value in overrides.items():
        parts = str(dotted).split('.')
        if not all(parts):
            raise YastnError(f"Malformed override path {dotted!r}.")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise YastnError(f"Override path {dotted!r} conflicts with another override.")
        if parts[-1] in node and isinstance(node[parts[-1]], dict):
            raise YastnError(f"Override path {dotted!r} conflicts with another override.")
        node[parts[-1]] = value
    return _apply_overrides(opts, tree)


def _apply_overrides(obj, tree):
    if is_dataclass(obj):
        aliases = _SI_ALIASES if isinstance(obj, SIOpts) else _CTM_ALIASES
        allowed = {f.name for f in fields(obj)}
        changes = {}
        for name, value in tree.items():
            canonical = aliases.get(name, name)
            if canonical not in allowed:
                raise _unknown(name, allowed, f'{type(obj).__name__} option')
            current = getattr(obj, canonical)
            if isinstance(value, dict) and (is_dataclass(current) or isinstance(current, dict)):
                changes[canonical] = _apply_overrides(current, value)
            elif isinstance(value, dict) and current is None:
                changes[canonical] = value  # e.g. opts_si.* on a bare CTMOpts
            else:
                changes[canonical] = value
        # replace() re-runs __post_init__, so every override layer is validated
        # and normalized exactly like a fresh construction. Unlike the factory it
        # does not drop None, so an override can deliberately clear a field.
        return replace(obj, **changes)
    if isinstance(obj, dict):
        out = dict(obj)
        for name, value in tree.items():
            current = out.get(name)
            out[name] = (_apply_overrides(current, value)
                         if isinstance(value, dict) and isinstance(current, (dict,))
                         else value)
        return out
    raise YastnError(f"Cannot apply overrides to a {type(obj).__name__}.")


def argspec(cls, prefix: str = ''):
    r"""
    Yield ``(dotted_name, type, default, help, choices)`` for every option.

    Lets a downstream application generate its command-line flags from the
    schema, so adding a CTM option never means editing that CLI. ``choices``
    is derived from the ``Literal`` annotations, and ``help`` from each
    field's ``metadata['help']``.

    ::

        for name, typ, default, help_, choices in argspec(CTMOpts):
            parser.add_argument(f'--{name}', default=default, help=help_,
                                **({'choices': choices} if choices else {}))
    """
    if not (isinstance(cls, type) and is_dataclass(cls)):
        raise YastnError(f"argspec expects an options dataclass, got {cls!r}.")
    hints = get_type_hints(cls)
    for f in fields(cls):
        hint = hints[f.name]
        name = f'{prefix}{f.name}'
        nested = next((a for a in (get_args(hint) or (hint,))
                       if isinstance(a, type) and is_dataclass(a)), None)
        if nested is not None:
            yield from argspec(nested, prefix=f'{name}.')
            continue
        choices = None
        for candidate in (hint, *get_args(hint)):
            if get_origin(candidate) is Literal:
                choices = get_args(candidate)
                break
        if f.default is not MISSING:
            default = f.default
        elif f.default_factory is not MISSING:
            default = f.default_factory()
        else:
            default = None
        yield name, hint, default, f.metadata.get('help', ''), choices
