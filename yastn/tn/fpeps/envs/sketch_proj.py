"""Sketched (randomized range-finding) projectors for the cuts of a measurement window.

Fits the projector pair of every interior cut to the window maps themselves,
sketched with :meth:`EnvCTM.measure_nsite_cut_map_oe`, instead of the CTM pairs
in ``env.proj``, which serve only as the probe's warm start.  One pooled SVD
per cut serves the norm window and every numerator window of the patch; the
norm block is scaled by ``|norm|`` and every numerator block by the same
``|sum of numerators|``, the quantities whose relative errors they control.
``opts['mode'] = 'two_sided'`` also sketches the adjoint maps.  The method,
error and gradient bounds are in "Cut maps and randomized range finding" of
``docs/source/fpeps/measurement_oe.rst``.

Nothing is written to ``env.proj``; the pairs are returned in the
``{site: {slot: tensor}}`` form ``projectors=`` accepts, computed under
``torch.no_grad()``.
"""

import logging
import math

import torch

from ....initialize import rand, zeros, block
from ....tensor import Leg, vdot, svd_with_truncation, leg_product
from ._env_ctm_oe_measure_network import _compress_bond_side

log = logging.getLogger(__name__)


_DEFAULT_OPTS = {
    'oversample': 32,   # additive probe budget over the kept rank (columns)
    'floor': 4,         # per-charge-sector probe floor even when n_c = 0
    'mode': 'pooled',   # 'pooled' or 'two_sided' (see module docstring)
    'rank': None,       # explicit rank override (diagnostics); None = use D_total
    'D_total': None,    # observable truncation; None = match the stored rank
    'tol': 0,           # relative singular-value cutoff (projector_svd_reltol)
    'seed': 0,          # probe RNG seed; None draws a fresh probe each call
}


def _opts(opts):
    o = {**_DEFAULT_OPTS, **(opts or {})}
    if o['mode'] not in ('pooled', 'two_sided'):
        raise ValueError(
            f"sketch mode must be 'pooled' or 'two_sided', got {o['mode']!r}")
    return o


def _as_projectors(pairs):
    """``{(site, slot): tensor}`` -> the ``{site: {slot: tensor}}`` that
    ``projectors=`` accepts."""
    out = {}
    for (site, slot), tensor in pairs.items():
        out.setdefault(site, {})[slot] = tensor
    return out


def _probe_for_slot(config, p, oversample, floor, full=False):
    """Build the probe for one half-projector slot.

    The probe is ``[ p | random ]`` blocked along the thin leg, where ``p`` is
    the CTM half-projector at this slot.  Its columns serve as the warm
    start and as the readout handle for :func:`_readout_scalar`.  The budget
    per charge sector ``c`` admissible on the fat cut leg is

        k_c = n_c + max(2, ceil(0.15 * n_c))

    plus a share of ``oversample``, with every admissible sector given at
    least ``floor`` columns even when ``n_c = 0`` (the sector-misalignment
    detector), all clipped at the sector's fat dimension.

    With ``full=True`` the probe spans the whole fat cut space, so the SVD
    rank is the only truncation (diagnostic rank sweeps / convergence tests).

    ``p`` is passed explicitly rather than read from ``env.proj`` so the warm
    start does not depend on what earlier patches wrote to the slot.

    Returns ``(probe, rand_leg)``; ``rand_leg`` is the random block's leg, so
    the caller can build the matching zero padding for the partner (``None``
    when the warm start already saturates every sector).
    """
    l0, l1 = p.get_legs(0), p.get_legs(1)
    thin = p.get_legs(2)
    fat = leg_product(l0, l1)
    fat_tD, thin_tD = fat.tD, thin.tD

    k_t, k_D = [], []
    for c in fat.t:
        k_t.append(c)
        if full:
            k_D.append(fat_tD[c])
            continue
        nc = thin_tD.get(c, 0)
        kc = nc + max(2, math.ceil(0.15 * nc))
        k_D.append(min(kc, fat_tD[c]))

    if not full:
        # distribute the oversample headroom to sectors with free capacity
        headroom = [fat_tD[c] - kD for c, kD in zip(k_t, k_D)]
        budget = oversample
        while budget > 0:
            free = [(i, h) for i, h in enumerate(headroom) if h > 0]
            if not free:
                break
            i, _ = max(free, key=lambda q: (q[1], -q[0]))
            k_D[i] += 1
            headroom[i] -= 1
            budget -= 1
        for i, c in enumerate(k_t):
            k_D[i] = min(max(floor, k_D[i]), fat_tD[c])

    # the random block supplies the budget BEYOND the warm-start columns
    rand_t, rand_D = [], []
    for c, kD in zip(k_t, k_D):
        extra = max(kD - thin_tD.get(c, 0), 0)
        if extra > 0:
            rand_t.append(c)
            rand_D.append(extra)

    if not rand_t:  # warm start already saturates every sector
        return p, None

    rand_leg = Leg(config, s=thin.s, t=tuple(rand_t), D=tuple(rand_D))
    G = rand(config, n=p.n, legs=[l0, l1, rand_leg])
    return block({(0,): p, (1,): G}, common_legs=(0, 1)), rand_leg


def _pad_partner(p_top, rand_leg):
    """Zero-pad the partner half so its thin leg matches the blocked probe's.

    Blocking ``[ p_top | 0 ]`` in the same layout as the probe makes the
    all-leg contraction with Y pick out exactly the warm-start block,
    independently of the order the sectors ended up in.
    """
    if rand_leg is None:
        return p_top
    cfg = p_top.config
    pad = zeros(cfg, n=p_top.n,
                legs=[p_top.get_legs(0), p_top.get_legs(1),
                      Leg(cfg, s=p_top.get_legs(2).s, t=rand_leg.t, D=rand_leg.D)])
    return block({(0,): p_top, (1,): pad}, common_legs=(0, 1))


def _readout_scalar(Y, p_top_padded):
    """``Tr(P_2x2 X)`` read off the sketch -- no extra window contraction.

    Contracting the open-leg cut map against a top-slot half reproduces the
    scalar path with that pair inserted.  ``p_top_padded`` is the oblique
    ``proj_corners`` half zero-padded onto the probe's thin leg, so only the
    warm-start columns contribute and the result is the value the current
    compressed path reports for this map.
    """
    return vdot(Y, p_top_padded.unfuse_legs(axes=(1,)), conj=(0, 0))


def _cuts_from_proj(proj, patch):
    """Split a ``{site: slot(s)}`` projector dict into its cuts.

    Slots are grouped by the bond they sever (``_compress_bond_side``).  For
    each cut the slot on the ``'t'``/``'r'`` side is the *top* slot (where Q
    is stored) and the slot on the ``'b'``/``'l'`` side is the *bottom* slot
    (where the probe is inserted and Q.conj() is stored).

    Returns a list of ``{top, bottom, slots}`` dicts.
    """
    minx = min(s[0] for s in patch)
    miny = min(s[1] for s in patch)
    pairs = {}
    for site, slots in proj.items():
        for slot in slots:
            i, j = site[0] - minx, site[1] - miny
            env_bond, D2_bond, side = _compress_bond_side(i, j, slot)
            key = (env_bond, D2_bond)
            pairs.setdefault(key, []).append((site, slot, side))

    cuts = []
    for _key, halves in pairs.items():
        top = [h for h in halves if h[2] in ('t', 'r')]
        bottom = [h for h in halves if h[2] in ('b', 'l')]
        if len(top) != 1 or len(bottom) != 1:
            raise ValueError(f"malformed projector cut: {halves!r}")
        t_site, t_slot, _ = top[0]
        b_site, b_slot, _ = bottom[0]
        cuts.append({
            'top': (t_site, t_slot),
            'bottom': (b_site, b_slot),
            'slots': [(t_site, t_slot), (b_site, b_slot)],
        })
    return cuts


def _other_cut_projectors(cuts, idx, done):
    """``projectors=`` dict closing every cut except ``idx``.

    Slots of a cut already in ``done`` carry its freshly sketched tensor; the
    rest map to ``None``, which reads the CTM pair from ``env.proj``.
    """
    other = {}
    for j, cut in enumerate(cuts):
        if j == idx:
            continue
        for site, slot in cut['slots']:
            other.setdefault(site, {})[slot] = done.get((site, slot))
    return other


def _sketch_cut(env, patch, cuts, idx, op_sites, opts, done,
                devices=None, mp_workers_per_device=0):
    """Sketch one interior cut; return its projector pairs without writing.

    Contracts the cut map once per quantity sharing the cut (norm + one per
    operator tuple; twice in ``'two_sided'`` mode) and truncates the pooled,
    normalized ``W`` once.  Returns ``{(site, slot): tensor}`` for the cut's
    two slots.  The rank is matched to ``opts['D_total']`` (or the current 2x2
    pair's rank when unset) unless ``opts['rank']`` overrides it.

    The warm start is read straight from ``env.proj``, which still holds the
    converged CTM pairs -- nothing here writes to it.  ``done`` maps the slots
    of already-sketched cuts to their fresh tensors, so those cuts are closed
    by the sketched pair while the rest fall back to the CTM one.
    """
    o = _opts(opts)
    mode = o['mode']

    cut = cuts[idx]
    (top_site, top_slot), (bottom_site, bottom_slot) = cut['top'], cut['bottom']
    other_proj = _other_cut_projectors(cuts, idx, done) or None

    p_top = getattr(env.proj[top_site], top_slot)
    if p_top is None:
        raise ValueError(
            f"env carries no projector at {top_slot}@{top_site}; the sketch "
            f"warm-starts from the converged CTM projectors")
    thin_top = p_top.get_legs(2)
    # The stored pair comes from CTM at rank ~chi, so the observable rank has
    # to come from D_total, not from the stored thin leg.
    n = o['rank'] if o['rank'] is not None else o['D_total']
    n = int(n) if n is not None else sum(thin_top.D)

    probe, rand_leg = _probe_for_slot(
        env.config, getattr(env.proj[bottom_site], bottom_slot),
        o['oversample'], o['floor'], full=o['rank'] is not None)

    # Each entry carries its OWN sites: terms sharing a measurement window are
    # sketched together, and they need not act on the same sites -- only span
    # the same bounding box. The leading () is the norm map over the window.
    maps = [((), patch), *op_sites]

    def _sketch(pr_site, pr_slot, pr):
        return [env.measure_nsite_cut_map_oe(
                    *ops, sites=sts,
                    probe_site=pr_site, probe_slot=pr_slot, probe=pr,
                    projectors=other_proj, optimizer="dp",
                    devices=devices, mp_workers_per_device=mp_workers_per_device)
                for ops, sts in maps]

    Ys = _sketch(bottom_site, bottom_slot, probe)

    Zs = None
    if mode == 'two_sided':
        # Build the top probe AFTER the forward one, so the bottom probe is
        # drawn from the same RNG position as in pooled mode.
        probe_top, _ = _probe_for_slot(
            env.config, p_top, o['oversample'], o['floor'],
            full=o['rank'] is not None)
        # the slot-swapped contraction returns Z = X^T . Omega'; conj() makes it
        # a sketch of X^dagger in Y's space (flip_signature() would not)
        Zs = [Z.conj() for Z in _sketch(top_site, top_slot, probe_top)]

    def _truncate(W):
        U, _S, _V = svd_with_truncation(
            W, axes=((0, 1, 2), (3,)), sU=thin_top.s, nU=True,
            D_total=n, tol=o['tol'])
        # The kept rank IS the compressed bond dimension of the cut.  Log which
        # of the three limits bound it: the probe width (sketch under-resolves,
        # raise oversample/floor), D_total (raise chi_obs), or tol.
        kept, probe = sum(U.get_legs(3).D), sum(W.get_legs(3).D)
        bound = ("probe" if kept >= probe else
                 "D_total" if kept >= n else "tol")
        log.info("[sketch] cut %s/%s -> %s: D=%d (probe %d, D_total %d, "
                 "tol %.1e) bound-by=%s sectors=%s",
                 top_site, top_slot, bottom_site, kept, probe, n, o['tol'],
                 bound, {t[0]: d for t, d in zip(U.get_legs(3).t,
                                                 U.get_legs(3).D)})
        # U.conj(), not U.flip_signature(): the kept subspace must be span(U) for
        # complex data too ("Which half carries the conjugate matters" in the docs)
        Q = U.conj().fuse_legs(axes=((0,), (1, 2), (3,)))  # (env, ket-bra, thin)
        return {(top_site, top_slot): Q, (bottom_site, bottom_slot): Q.conj()}

    # norm block by |norm|, EVERY numerator block by the same |num_total|
    p_top_padded = _pad_partner(p_top, rand_leg)
    vals = [_readout_scalar(Y, p_top_padded) for Y in Ys]
    s_rho = abs(vals[0])
    # guard a num_total sitting near a zero crossing
    s_num = max(abs(sum(vals[1:])), s_rho * 1e-8)
    scales = [s_rho] + [s_num] * (len(Ys) - 1)

    # one truncation over the union, used by both contractions
    blocks = [Y * (1.0 / s) for Y, s in zip(Ys, scales)]
    if Zs is not None:
        # sigma(X^T) = sigma(X), so the adjoint blocks use the SAME scales as
        # their forward partners; rescaling them independently would reweight
        # the truncation.
        blocks += [Z * (1.0 / s) for Z, s in zip(Zs, scales)]
    return _truncate(block({(i,): B for i, B in enumerate(blocks)}, common_legs=(0, 1, 2)))


def sketch_projectors_(env, patch, proj, op_sites, opts,
                       devices=None, mp_workers_per_device=0):
    """Sketch every cut of the layout ``proj`` and return the new pairs.

    ``proj`` is the ``{site: slot(s)}`` layout of the cuts to compress in the
    window spanned by ``patch``; ``op_sites`` is the ``(operator tuple, sites)``
    pair of every task sharing the window -- tasks in one window may act on
    different sites, so each carries its own.  The warm start is read from ``env.proj``,
    which is never written.

    Returns ``{site: {slot: tensor}}`` covering every slot of ``proj``, in the
    shape ``projectors=`` accepts.  Cuts are sketched in order, each with the
    cuts already done closed by their fresh pairs and the rest by their CTM pairs.
    Runs entirely under ``torch.no_grad()``, so the returned tensors carry no
    grad.
    """
    with torch.no_grad():
        if not proj:
            raise ValueError(f"patch {patch} has no projector layout")

        # Seed the probe so it -- and hence the energy -- is a reproducible
        # function of the state; an unseeded redraw makes E(x) stochastic,
        # which a line search must not see.  One fixed seed is enough: the
        # patches are independent range-finding problems, so reusing the same
        # draw across them is harmless.  The previous RNG state is restored so
        # the seeding stays local to the sketch.
        seed, rng_state = _opts(opts)['seed'], None
        if seed is not None:
            rng_state = torch.random.get_rng_state()
            env.config.backend.random_seed(seed)
        try:
            cuts = _cuts_from_proj(proj, patch)
            done = {}
            for idx in range(len(cuts)):
                # the next cut's sketch closes this one with the fresh pair
                done.update(_sketch_cut(
                    env, patch, cuts, idx, op_sites, opts, done,
                    devices=devices,
                    mp_workers_per_device=mp_workers_per_device))
        finally:
            if rng_state is not None:
                torch.random.set_rng_state(rng_state)
    return _as_projectors(done)
