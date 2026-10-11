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
""" Network construction for the opt_einsum-based n-site CTM measurement.

Builds the interleaved ``(tensor, labels, ...)`` argument list that
``contract_with_unroll`` consumes for one rectangular window of the double
layer (corners, edges, site tensors, optional projectors and operators), the
unroll-label translation, and the fermionic-sign bookkeeping: the swap gates of
the bonds of an MPO (``_mpo_path_swaps``), which also carry the charge of plain
operators.  ``_env_ctm_measure`` orchestrates the contraction itself.  Bond
labels and fermionic signs are described in ``docs/source/fpeps/measurement_oe.rst``.
"""
import itertools
from .._geometry import Site
from .._gates_auxiliary import BOND_FUSION, ordering_swaps, ordering_swap_axes
from ....tensor import YastnError
from ....tensor._auxiliary import get_blocks

# In the double layer the ket's left leg crosses the bra's top leg, and the ket's bottom leg
# the bra's right leg; a bond fused into one of those ket legs crosses the same bra leg.
_KET_BRA = {1: 'b0', 2: 'b3'}


def _translate_unroll(unroll, Nx, Ny):
    """Map user-facing fused bond labels to unfused ket/bra sub-labels.

    Interior PEPS bonds (``('v', i, j)`` with ``0 <= j < Ny`` and
    ``('h', i, j)`` with ``0 <= i < Nx``) are split into ket/bra
    sub-labels. Explicit layer-qualified labels ``(..., 'k')`` and
    ``(..., 'b')`` are kept as-is. Boundary (chi) bonds are kept as-is.
    """
    def _is_interior_peps_bond(label):
        return (label[0] == 'v' and 0 <= label[2] < Ny) or \
               (label[0] == 'h' and 0 <= label[1] < Nx)

    if unroll is None:
        return None
    translated = {}
    for label, val in unroll.items():
        if len(label) == 4:
            if label[-1] not in ('k', 'b'):
                raise YastnError(f"Invalid layer-qualified unroll label {label}; expected trailing 'k' or 'b'.")
            if not _is_interior_peps_bond(label[:-1]):
                raise YastnError(f"Layer-qualified unroll label {label} is only valid for PEPS ket/bra bonds.")
            translated[label] = val
        elif _is_interior_peps_bond(label):
            translated[label + ('k',)] = val
            translated[label + ('b',)] = val
        else:
            translated[label] = val
    return translated


def _pad_unfused_edge(edge_uf, peps_ket_leg, peps_bra_leg, ket_ax=1, bra_ax=2):
    r"""
    Pad an unfused edge tensor with zero blocks so that its ket/bra
    sub-legs match the PEPS ket/bra legs.

    After CTM expansion with OBC boundary projectors, unfused edge
    sub-legs can have fewer charge sectors than the PEPS legs.
    Adding zero blocks for the missing sectors restores compatibility
    without affecting the contraction result (zero blocks contribute
    nothing).
    """
    ket_sub = edge_uf.get_legs(axes=ket_ax)
    bra_sub = edge_uf.get_legs(axes=bra_ax)

    existing_ket_t = set(ket_sub.t)
    missing_ket_t = set(peps_ket_leg.t) - existing_ket_t

    existing_bra_t = set(bra_sub.t)
    missing_bra_t = set(peps_bra_leg.t) - existing_bra_t

    if not missing_ket_t and not missing_bra_t:
        return edge_uf

    legs = edge_uf.get_legs()
    sigs = edge_uf.s
    n_total = edge_uf.n
    ndim = edge_uf.ndim_n
    other_axes = [ax for ax in range(ndim) if ax != ket_ax and ax != bra_ax]

    peps_ket_tD = dict(zip(peps_ket_leg.t, peps_ket_leg.D))
    peps_bra_tD = dict(zip(peps_bra_leg.t, peps_bra_leg.D))

    # Collect existing chi charge pairs (leg_first: block charges are derived
    # from the legs via get_blocks; existing_blocks kept as flat charge tuples
    # to match ts_flat below — set_block ravels ts, so flat form is correct for
    # any NSYM).
    nsym = edge_uf.config.sym.NSYM
    chi_pairs = set()
    existing_blocks = set()
    for blk in get_blocks(edge_uf.config.sym, edge_uf.struct).t.tolist():
        charges = tuple(tuple(c) for c in blk)   # per-native-leg charge tuples
        chi_pairs.add(tuple(charges[ax] for ax in other_axes))
        existing_blocks.add(tuple(x for c in charges for x in c))

    def _infer_missing_charge(chi_combo, known_charge, known_ax, unknown_ax):
        """Infer the charge on unknown_ax from the symmetry constraint."""
        mb_list = []
        for s in range(nsym):
            partial = sum(sigs[ax] * chi_combo[idx][s]
                          for idx, ax in enumerate(other_axes))
            partial += sigs[known_ax] * known_charge[s]
            mb_list.append((n_total[s] - partial) // sigs[unknown_ax])
        return tuple(mb_list)

    def _add_zero_block(chi_combo, mk, mb):
        """Add a zero block for (chi_combo, mk, mb) if it doesn't exist."""
        D_k = peps_ket_tD[mk]
        D_b = peps_bra_tD[mb]

        ts_list = [None] * ndim
        Ds_list = [None] * ndim
        ts_list[ket_ax] = mk
        ts_list[bra_ax] = mb
        Ds_list[ket_ax] = D_k
        Ds_list[bra_ax] = D_b
        for idx, ax in enumerate(other_axes):
            ts_list[ax] = chi_combo[idx]
            Ds_list[ax] = legs[ax].D[list(legs[ax].t).index(chi_combo[idx])]

        ts_flat = sum(ts_list, ())
        if ts_flat not in existing_blocks:
            edge_uf.set_block(ts=ts_flat, Ds=tuple(Ds_list), val='zeros')
            existing_blocks.add(ts_flat)

    for chi_combo in chi_pairs:
        for mk in missing_ket_t:
            mb = _infer_missing_charge(chi_combo, mk, ket_ax, bra_ax)
            if mb in peps_bra_tD:
                _add_zero_block(chi_combo, mk, mb)

        for mb in missing_bra_t:
            mk = _infer_missing_charge(chi_combo, mb, bra_ax, ket_ax)
            if mk in peps_ket_tD:
                _add_zero_block(chi_combo, mk, mb)

    return edge_uf


def _uf_middle_padded(edge_tensor, peps_ket_leg, peps_bra_leg):
    """Unfuse edge middle leg and pad to match PEPS ket/bra legs."""
    uf = edge_tensor.unfuse_legs(axes=(1,))
    uf = uf.drop_leg_history(axes=(1, 2))
    return _pad_unfused_edge(uf, peps_ket_leg, peps_bra_leg)


def _bond_labels(i, j):
    """Bond labels of window site ``(i, j)`` in direction order t, l, b, r."""
    return [('v', i, j), ('h', i, j - 1), ('v', i + 1, j), ('h', i, j)]


def _boundary_args(env, peps_legs, Nx, Ny, minx, miny, maxx, maxy, tl, tr, bl, br,
                   tag=lambda label, i, j: label):
    """Corners and edges of the window, edge middle legs unfused into ket/bra
    and padded to ``peps_legs[i, j] = (ket legs, bra legs)`` of the adjacent
    site, in direction order t, l, b, r.  ``tag(label, i, j)`` renames the end
    of a bond at the tensor sitting at ``(i, j)``, used to insert projectors.
    Corners sit at TL = (-1, -1), TR = (-1, Ny), BL = (Nx, -1), BR = (Nx, Ny)."""
    args = [env[tl].tl, [tag(('v', 0, -1), -1, -1), tag(('h', -1, -1), -1, -1)],
            env[bl].bl, [tag(('h', Nx, -1), Nx, -1), tag(('v', Nx, -1), Nx, -1)],
            env[tr].tr, [tag(('h', -1, Ny - 1), -1, Ny), tag(('v', 0, Ny), -1, Ny)],
            env[br].br, [tag(('v', Nx, Ny), Nx, Ny), tag(('h', Nx, Ny - 1), Nx, Ny)]]

    def edge(ten, site, d, i, j, labels):
        k_legs, b_legs = peps_legs[site]
        args.extend([_uf_middle_padded(ten, k_legs[d], b_legs[d]), [tag(lb, i, j) for lb in labels]])

    for i in range(Nx):  # left edge i sits at (i, -1)
        edge(env[Site(minx + i, miny)].l, (i, 0), 1, i, -1,
             [('v', i + 1, -1), ('h', i, -1, 'k'), ('h', i, -1, 'b'), ('v', i, -1)])
    for i in range(Nx):  # right edge i sits at (i, Ny)
        edge(env[Site(minx + i, maxy)].r, (i, Ny - 1), 3, i, Ny,
             [('v', i, Ny), ('h', i, Ny - 1, 'k'), ('h', i, Ny - 1, 'b'), ('v', i + 1, Ny)])
    for j in range(Ny):  # top edge j sits at (-1, j)
        edge(env[Site(minx, miny + j)].t, (0, j), 0, -1, j,
             [('h', -1, j - 1), ('v', 0, j, 'k'), ('v', 0, j, 'b'), ('h', -1, j)])
    for j in range(Ny):  # bottom edge j sits at (Nx, j)
        edge(env[Site(maxx, miny + j)].b, (Nx - 1, j), 2, Nx, j,
             [('h', Nx, j), ('v', Nx, j, 'k'), ('v', Nx, j, 'b'), ('h', Nx, j - 1)])
    return args


_CUT = {'lt': ('v', 0, -1, 0, 0, 'b'), 'rt': ('v', 0, 1, 0, 0, 'b'),
        'lb': ('v', 1, -1, 1, 0, 't'), 'rb': ('v', 1, 1, 1, 0, 't'),
        'tl': ('h', -1, -1, 0, -1, 'r'), 'bl': ('h', 1, -1, 0, -1, 'r'),
        'tr': ('h', -1, 0, 0, 0, 'l'), 'br': ('h', 1, 0, 0, 0, 'l')}
_PARTNER_FACE = {'t': 'b', 'b': 't', 'l': 'r', 'r': 'l'}


def _compress_bond_side(i, j, proj_name):
    """Return (bond1, bond2, side) for the half-projector at (i, j),
    where bond1 (dim=chi) and bond2 (dim=D) are the two bonds that
    get compressed.
    """
    kind, di1, dj1, di2, dj2, side = _CUT[proj_name[-2:]]
    return (kind, i + di1, j + dj1), (kind, i + di2, j + dj2), side


def _bond_endpoint(bond, side):
    """(i, j) of the bond's endpoint on the given side."""
    if bond[0] == 'v':
        return (bond[1] - 1, bond[2]) if side == 't' else (bond[1], bond[2])
    return (bond[1], bond[2]) if side == 'l' else (bond[1], bond[2] + 1)


def _norm_slots(slots):
    """A slot name, a sequence of names, or a ``{slot: tensor}`` dict -> ``{slot: tensor-or-None}``."""
    if isinstance(slots, str):
        return {slots: None}
    return dict(slots) if isinstance(slots, dict) else dict.fromkeys(slots)


def _build_ketbra_separate(env, tens, Nx, Ny, minx, miny, maxx, maxy, tl, tr, bl, br,
                           projectors=None, op_bonds=None, bond_crossings=(), bond_pairs=(), probe=None):
    r"""
    Window network with separate ket and bra tensors (5 legs each) per site,
    joined by a shared physical-leg label, and the operator, if any, as a
    tensor of its own between them.  Crossings inside the bra are applied as
    ``swap_gate``; ket x bra crossings are returned as ncon swap pairs.

    ``op_bonds`` maps a site to the network labels of its MPO tensor's bond
    legs, left then right, the chain's ends having one; an MPO tensor enters in
    the leg order of a gate, ``(phys_out, phys_in, left, right)``, absent bonds
    dropped.  ``bond_crossings`` lists ``(bond_label, (site, axis))`` pairs from
    :func:`_mpo_path_swaps`; each becomes a swap pair between the bond and the network
    leg of that axis.  ``bond_pairs`` are further swap pairs of network labels.

    ``projectors`` maps a site to one slot name, a sequence of slot names, or a
    ``{slot: tensor}`` dict, the slots of :class:`EnvCTM_projectors` (``"hlt"``,
    ``"hlb"``, ``"hrt"``, ``"hrb"``, ``"vtl"``, ``"vtr"``, ``"vbl"``,
    ``"vbr"``), e.g. ``{site: ("hrt", "hrb")}``; a name (or a ``None`` tensor)
    reads the half-projector from ``env.proj[site]``, a tensor is inserted as
    given.  Every half needs its partner.  A pair compresses the two parallel
    bonds of its cut into one thin bond, as a CTM move does, which makes the
    measurement approximate but cheaper.  Requires a ``DoublePepsTensor`` PEPS.

    ``probe=(site, slot, tensor)`` inserts one half-projector, in the stored
    form ``(env chi, fused ket-D x bra-D, thin)``, without its partner.  The
    partner's side of the cut stays open: its severed-bond labels
    ``[env_bond, D2_bond+('k',), D2_bond+('b',)]`` and the probe's thin label
    ``('proj',)+env_bond`` form the network's output spec.  A half and its
    partner sever the same two bonds, so these follow from the probe alone.

    Returns ``(tn_args, swap_pairs)``.
    """
    projectors = {site: _norm_slots(slots) for site, slots in (projectors or {}).items()}
    op_bonds = op_bonds or {}
    # Every half must have its partner: the slot with the last letter flipped
    # ('t' <-> 'b', 'l' <-> 'r') on the neighbouring site in that letter's direction.
    for site, slots in projectors.items():
        for slot in slots:
            partner_slot = slot[:-1] + _PARTNER_FACE[slot[-1]]
            partner_site = env.nn_site(site, slot[-1])
            if partner_slot not in projectors.get(partner_site, ()):
                raise YastnError(
                    f"projector half {slot}@{site} is missing its partner "
                    f"{partner_slot}@{partner_site}.")

    # Each inserted half renames the endpoint, on its own side, of the two bonds
    # it severs; rename is keyed by (label, i, j) of that endpoint.  The other
    # endpoint keeps its label and meets the partner half, or stays open for the probe.
    rename, inserts = {}, []

    def register(key, label, slot, site):
        if rename.setdefault(key, label) != label:
            raise YastnError(
                f"projector {slot}@{site} conflicts with another projector "
                f"trying to rename the same bond endpoint {key}.")

    halves = [(site, slot, t) for site, slots in projectors.items() for slot, t in slots.items()]
    for site, slot, proj in halves + ([probe] if probe else []):
        env_bond, D2_bond, side = _compress_bond_side(site[0] - minx, site[1] - miny, slot)
        new_env, new_ket, new_bra = env_bond + (side,), D2_bond + ('k', side), D2_bond + ('b', side)
        register((env_bond, *_bond_endpoint(env_bond, side)), new_env, slot, site)
        register((D2_bond + ('k',), *_bond_endpoint(D2_bond, side)), new_ket, slot, site)
        register((D2_bond + ('b',), *_bond_endpoint(D2_bond, side)), new_bra, slot, site)
        if proj is None:
            proj = getattr(env.proj[site], slot)
        inserts += [proj.unfuse_legs(axes=(1,)), [new_env, new_ket, new_bra, ('proj',) + env_bond]]

    def tag(label, i, j):
        return rename.get((label, i, j), label)

    site_args, swap_pairs, peps_legs = [], [], {}
    for i in range(Nx):
        for j in range(Ny):
            s = Site(minx + i, miny + j)
            dpt = tens[s]
            Ab, Ak = dpt.bra, dpt.ket
            Ab = Ab.conj().swap_gate(axes=(1, 0, 2, 3))  # inside the bra: l_b x t_b, b_b x r_b
            # history is dropped on the virtual legs, to match the padded edges, but kept on the
            # physical leg: for a purification it is a fusion of system and ancilla, and the
            # operator, matched to it by `match_ancilla`, lines up only through that history.
            Ak = Ak.transpose(axes=dpt.trans + (4,)).drop_leg_history(axes=(0, 1, 2, 3))
            Ab = Ab.transpose(axes=dpt.trans + (4,)).drop_leg_history(axes=(0, 1, 2, 3))
            peps_legs[i, j] = (Ak.get_legs(axes=(0, 1, 2, 3)), Ab.get_legs(axes=(0, 1, 2, 3)))

            # The operator, if any, stays a separate network tensor so the path
            # search can place it: ket --('pin',i,j)--> op --('p',i,j)--> bra.
            lbls = _bond_labels(i, j)
            ket_p = ('pin', i, j) if dpt.op is not None else ('p', i, j)
            site_args += [Ak, [tag(lb + ('k',), i, j) for lb in lbls] + [ket_p],
                          Ab, [tag(lb + ('b',), i, j) for lb in lbls] + [('p', i, j)]]
            if dpt.op is not None:  # Gate leg order (phys_out, phys_in, left, right), ends trimmed
                op = dpt.op.drop_leg_history(axes=tuple(range(2, dpt.op.ndim)))  # bond legs only
                site_args += [op, [('p', i, j), ket_p, *op_bonds.get(s, ())]]

            inv = [dpt.trans.index(d) for d in range(4)]  # canonical direction -> label index

            def leg(axis):
                """Network label of an axis of the site's own (pre-transpose) ket/bra."""
                if axis == 'k4':
                    return ket_p  # the ket's bare physical leg, before any operator
                if axis == 'p':
                    return ('p', i, j)  # the physical leg after the operator
                return tag(lbls[inv[int(axis[1])]] + (axis[0],), i, j)

            # ket x bra crossings in canonical order: l_k x t_b and b_k x r_b
            swap_pairs += [(leg(f'k{k}'), leg(b)) for k, b in _KET_BRA.items()]
            # an MPO bond swapped against the legs of this site it crosses
            swap_pairs += [(b, leg(axis)) for b, (site_x, axis) in bond_crossings if site_x == s]
    swap_pairs += list(bond_pairs)

    args = _boundary_args(env, peps_legs, Nx, Ny, minx, miny, maxx, maxy, tl, tr, bl, br, tag=tag)
    out = ()  # scalar output, unless the probe leaves the partner side open
    if probe:
        env_bond, D2_bond, _ = _compress_bond_side(probe[0][0] - minx, probe[0][1] - miny, probe[1])
        out = (env_bond, D2_bond + ('k',), D2_bond + ('b',), ('proj',) + env_bond)
    return tuple(args + site_args + inserts + [out]), swap_pairs


def _window_bounds(env, sites):
    """Bounding window ``(minx, miny, maxx, maxy)`` of ``sites``; on a finite
    lattice a single row or column is widened to two."""
    if not sites:
        raise YastnError("`sites` must be non-empty.")
    minx, miny = min(s[0] for s in sites), min(s[1] for s in sites)
    maxx, maxy = max(s[0] for s in sites), max(s[1] for s in sites)
    if minx == maxx and env.nn_site((minx, miny), 'b') is None:
        minx -= 1
    if miny == maxy and env.nn_site((minx, miny), 'r') is None:
        miny -= 1
    return minx, miny, maxx, maxy


# The layer a line running along a cut lattice bond moves outside of, to pass its
# half-projectors on the side away from their environment leg, by that side: in the
# projected double layer the bra line lies left of and below the ket line.
_JUMP_LAYER = {'l': 'k', 'r': 'b', 't': 'b', 'b': 'k'}


def _cut_bonds(projectors, probe, minx, miny):
    """Lattice bonds, as sets of their two sites, that the half-projectors of
    ``projectors`` and ``probe`` cut, mapped to the side of the window, ``'l'``,
    ``'r'``, ``'t'`` or ``'b'``, whose environment leg the half-projectors take in."""
    halves = [(site, slot) for site, slots in (projectors or {}).items() for slot in _norm_slots(slots)]
    halves += [probe[:2]] if probe else []
    cut = {}
    for site, slot in halves:
        _, (kind, i, j), _ = _compress_bond_side(site[0] - minx, site[1] - miny, slot)
        a, b = ((i - 1, j), (i, j)) if kind == 'v' else ((i, j), (i, j + 1))
        cut[frozenset((Site(a[0] + minx, a[1] + miny), Site(b[0] + minx, b[1] + miny)))] = slot[1]
    return cut


def _route(a, b):
    """The nearest-neighbour route from site ``a`` to site ``b``, both included: along the
    column of ``a`` first, then along the row of ``b``."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = (1 if x1 >= x0 else -1), (1 if y1 >= y0 else -1)
    column = [Site(x, y0) for x in range(x0, x1 + dx, dx)]
    return column + [Site(x1, y) for y in range(y0 + dy, y1 + dy, dy)]


# The direction of a step between neighbouring window sites, as :meth:`SquareLattice.nn_bond_dirn`
# names it, read from the window coordinates: a step never crosses a periodic boundary there.
_STEP = {(0, 1): 'lr', (1, 0): 'tb', (0, -1): 'rl', (-1, 0): 'bt'}


def _mpo_path_swaps(tens, ops, chain, f_ordered, cut):
    """
    Attach the MPO tensors of ``chain`` to their sites and return the swap gates their
    bonds need, ``(crossings, pairs)``: ``crossings`` lists ``(bond_label, (site, axis))``,
    a bond against a leg of a site, axis ``'p'`` being the output leg of the site's
    operator, and ``pairs`` lists pairs of bond labels.  ``f_ordered`` is the lattice's
    fermionic order.

    Each bond joins its two MPO tensors directly, and its swap gates are those that
    applying the MPO to the ket along a path, as :meth:`Peps.apply_gate_` does, would
    produce -- replayed instead of performed.  The walk from one chain site to the next
    follows :func:`_route`, so the chain may list its sites in any order and they need not
    be adjacent.  At every step :func:`ordering_swaps` adapts the tensors to the step's
    direction and fermionic order.  At every site the bond is fused into a ket leg as in
    :func:`apply_gate_onsite`, crossing the leg of :data:`BOND_FUSION` and every bond
    already fused into it, and inheriting the ket x bra crossing of that leg.  When a
    lattice bond carries several MPO bonds, fused in different orders at its two ends,
    they cross.  A site the walk only passes carries an identity, as in
    :meth:`Peps.apply_gate_`, whose two bond legs are the same bond; where the replay
    swaps them, the swap is that bond's parity.  The bond carries one charge sector per
    block along its whole length, so its parity is applied to the tensor it leaves.
    See "Building the MPO" in ``docs/source/fpeps/measurement_oe.rst``.

    ``cut`` maps the lattice bonds cut by half-projectors to the side of the window
    their environment leg comes from (:func:`_cut_bonds`).  Half-projectors compress
    the environment leg and the ket and bra legs of a lattice bond together, so a bond
    running along a cut lattice bond, between its ket and bra lines, would pass through
    them.  Where the walk runs along a cut lattice bond, the bond of the MPO passes the
    half-projectors on the side away from their environment leg instead: it crosses the
    ket or the bra leg of the cut lattice bond at both of its ends, which moves it
    outside that line.  This gives the same contraction as a walk around the cut, so
    the walk need not avoid cut lattice bonds, even where half-projectors cut every
    lattice bond between two rows or columns of the window.  In the projected double
    layer the bra lies left of and below the ket, so the leg crossed follows from the
    environment side (:data:`_JUMP_LAYER`)::

        half-projectors   environment side   leg crossed
        hlt, hlb          left               ket
        hrt, hrb          right              bra
        vtl, vtr          top                bra
        vbl, vbr          bottom             ket
    """
    G = [ops[s] for s in chain]
    walk, walk_bond = [(chain[0], 0)], []  # (site, chain index or None where only passed)
    for k in range(len(chain) - 1):
        route = _route(chain[k], chain[k + 1])
        walk += [(u, None) for u in route[1:-1]] + [(chain[k + 1], k + 1)]
        walk_bond += [k + 1] * (len(route) - 1)
    own = {u: i for i, (u, c) in enumerate(walk) if c is not None}
    dirns = [_STEP[(v[0] - u[0], v[1] - u[1])] for (u, _), (v, _) in zip(walk, walk[1:])]

    toggled, parity = set(), set()  # a swap gate applied twice is none

    def toggle(m, target):
        toggled.symmetric_difference_update({(m, target)})

    def cross_bonds(m, x):
        if m == x:  # the two bond legs of an identity at a passed site: the bond's parity
            parity.symmetric_difference_update({m})
        else:
            toggle(min(m, x), ('opb', max(m, x)))

    def cross_physical(m, u, i):  # below u's operator if the walk reaches u first, above it if later
        toggle(m, ('leg', u, 'p' if own.get(u, i) < i else 'k4'))

    for i, (u, c) in enumerate(walk):  # the bond x physical swap of an identity passed through
        if c is None:
            cross_physical(walk_bond[i - 1], u, i)
    for i, dirn in enumerate(dirns):
        m = walk_bond[i]
        for tensor, leg in ordering_swaps(dirn, f_ordered(walk[i][0], walk[i + 1][0])):
            u, c = walk[i + tensor]
            if c is not None:
                G[c] = G[c].swap_gate(axes=ordering_swap_axes(tensor, leg))
            elif leg == 'bond':
                parity.symmetric_difference_update({m})
            else:  # the in and out legs of an identity are both the physical line
                cross_physical(m, u, i + tensor)

    for i, dirn in enumerate(dirns):  # a step along a cut lattice bond
        side = cut.get(frozenset((walk[i][0], walk[i + 1][0])))
        if side is not None:  # pass the half-projectors outside the layer facing away from their environment
            layer = _JUMP_LAYER[side]
            toggle(walk_bond[i], ('leg', walk[i][0], f'{layer}{BOND_FUSION[dirn[0]][0]}'))
            toggle(walk_bond[i], ('leg', walk[i + 1][0], f'{layer}{BOND_FUSION[dirn[1]][0]}'))

    fused = {}  # (site, ket leg) -> the bonds fused into it, in order
    for i, (u, c) in enumerate(walk):
        steps = [(dirns[i][0], walk_bond[i])] if i < len(dirns) else []
        steps += [(dirns[i - 1][1], walk_bond[i - 1])] if i > 0 else []
        for role, m in steps:  # the outgoing bond first, as in apply_gate_onsite
            leg, crossed = BOND_FUSION[role]
            if crossed is not None:
                toggle(m, ('leg', u, f'k{crossed}'))
                for x in fused.get((u, crossed), []):
                    cross_bonds(m, x)
            fused.setdefault((u, leg), []).append(m)
            if leg in _KET_BRA:
                toggle(m, ('leg', u, _KET_BRA[leg]))
    for (u, leg), ms in fused.items():  # the two ends of a lattice bond, in fusion order
        if leg in (2, 3):
            v = Site(u[0] + 1, u[1]) if leg == 2 else Site(u[0], u[1] + 1)
            other = fused.get((v, leg - 2), [])
            for a, b in itertools.combinations([m for m in ms if m in other], 2):
                if (ms.index(a) < ms.index(b)) != (other.index(a) < other.index(b)):
                    cross_bonds(a, b)

    for m in parity:
        ax = G[m - 1].ndim - 1
        G[m - 1] = G[m - 1].swap_gate(axes=(ax, ax))
    for k, s in enumerate(chain):
        i = own[s]
        roles = (dirns[i - 1][1] if i > 0 else '') + (dirns[i][0] if i < len(dirns) else '')
        tens[s].set_operator_(G[k], dirn=roles)

    crossings = sorted((('opb', m), target[1:]) for m, target in toggled if target[0] == 'leg')
    pairs = sorted((('opb', m), target) for m, target in toggled if target[0] == 'opb')
    return crossings, pairs


def _build_fused(env, tens, Nx, Ny, minx, miny, maxx, maxy, tl, tr, bl, br):
    """Interleaved network of a single-layer (fused) PEPS window; ``tens``
    maps window sites to 4-leg tensors.  Same bond labels as the unfused
    builders, without the ket/bra split."""
    def _d(t):
        return t.drop_leg_history() if hasattr(t, 'drop_leg_history') else t
    args = []
    args += [_d(env[tl].tl), [('v', 0, -1), ('h', -1, -1)]]
    args += [_d(env[bl].bl), [('h', Nx, -1), ('v', Nx, -1)]]
    args += [_d(env[tr].tr), [('h', -1, Ny - 1), ('v', 0, Ny)]]
    args += [_d(env[br].br), [('v', Nx, Ny), ('h', Nx, Ny - 1)]]
    for i in range(Nx):
        args += [_d(env[Site(minx + i, miny)].l), [('v', i + 1, -1), ('h', i, -1), ('v', i, -1)]]
    for i in range(Nx):
        args += [_d(env[Site(minx + i, maxy)].r), [('v', i, Ny), ('h', i, Ny - 1), ('v', i + 1, Ny)]]
    for j in range(Ny):
        args += [_d(env[Site(minx, miny + j)].t), [('h', -1, j - 1), ('v', 0, j), ('h', -1, j)]]
    for j in range(Ny):
        args += [_d(env[Site(maxx, miny + j)].b), [('h', Nx, j), ('v', Nx, j), ('h', Nx, j - 1)]]
    for i in range(Nx):
        for j in range(Ny):
            args += [_d(tens[Site(minx + i, miny + j)]),
                     [('v', i, j), ('h', i, j - 1), ('v', i + 1, j), ('h', i, j)]]
    args.append(())
    return tuple(args)
