"""Magnetic-basis normalization at reduced tensor interfaces."""
from array import array
from collections import namedtuple
from functools import lru_cache
import numpy as np
import opt_einsum
from .._cache_registry import register_cache


@register_cache(key='nonabelian_cg_vertex')
@lru_cache(maxsize=4096)
def _cg_vertex(sym, left, right, target):
    vertex = sym.fusion_isometry(left, right, target).reshape(
        sym.irrep_dimension(target), sym.irrep_dimension(left), sym.irrep_dimension(right))
    vertex.setflags(write=False)
    return vertex


@register_cache(key='nonabelian_dual_metric')
@lru_cache(maxsize=256)
def _dual_metric(j):
    dual = np.zeros((j + 1, j + 1))
    for m in range(j + 1):
        dual[j - m, m] = (-1.) ** (j - m)
    dual.setflags(write=False)
    return dual


def tree_network(sym, charges, signs, total, channel, labels, fresh, root=None):
    """Small CG vertices, leaving only external magnetic indices open.

    Keeping the tree factorized avoids constructing arrays whose sizes grow
    as the product of every external irrep dimension.
    """
    from ._contractions import _split_channel_flat
    if not charges:
        return []
    oriented = tuple(t if s == 1 else sym.conj_charge(t) for t, s in zip(charges, signs))
    path = _split_channel_flat(channel, sym.NSYM) + (total,)
    factors = []
    leaves = []
    for charge, sign, label in zip(charges, signs, labels):
        if sign == 1:
            leaves.append(label)
        else:
            internal = next(fresh)
            factors.extend((_dual_metric(charge[0]), [internal, label]))
            leaves.append(internal)
    if len(charges) == 1:
        dimension = sym.irrep_dimension(charges[0])
        factors.extend((np.ones(dimension), [leaves[0]]) if root is None
                       else (np.eye(dimension), [root, leaves[0]]))
    else:
        left, left_label = oriented[0], leaves[0]
        for i in range(1, len(charges)):
            target = path[i - 1]
            right = oriented[i]
            vertex = _cg_vertex(sym, left, right, target)
            if i == len(charges) - 1:
                factors.extend((vertex[0], [left_label, leaves[i]]) if root is None
                               else (vertex, [root, left_label, leaves[i]]))
            else:
                label = next(fresh)
                factors.extend((vertex, [label, left_label, leaves[i]]))
                left_label = label
            left = target
    return factors


@register_cache(key='nonabelian_contraction_coefficient')
@lru_cache(maxsize=65536)
def contraction_coefficient(sym, charges_a, signs_a, total_a, channel_a,
                            charges_b, signs_b, total_b, channel_b,
                            nin_a, nin_b, charges_c, signs_c, total_c, channel_c):
    from itertools import count
    fresh = count()
    la = [next(fresh) for _ in charges_a]
    lb = [None] * len(charges_b)
    for ia, ib in zip(nin_a, nin_b):
        lb[ib] = la[ia]
    for i in range(len(lb)):
        if lb[i] is None:
            lb[i] = next(fresh)
    lc = [l for i, l in enumerate(la) if i not in nin_a] + [l for i, l in enumerate(lb) if i not in nin_b]
    factors = tree_network(sym, charges_a, signs_a, total_a, channel_a, la, fresh)
    factors += tree_network(sym, charges_b, signs_b, total_b, channel_b, lb, fresh)
    factors += tree_network(sym, charges_c, signs_c, total_c, channel_c, lc, fresh)
    value = _contract_scalar(factors)
    return value * scale(sym, charges_a, signs_a, total_a) * scale(sym, charges_b, signs_b, total_b) / scale(sym, charges_c, signs_c, total_c)


def _contract_scalar(factors):
    if not factors:
        return 1.
    # Singlet magnetic axes have dimension one. Remove them before planning:
    # otherwise equivalent CG networks generate many distinct large paths.
    scalar, operands, topology, relabel = 1., [], [], {}
    for array, labels in zip(factors[::2], factors[1::2]):
        axes = tuple(i for i, size in enumerate(array.shape) if size != 1)
        if not axes:
            scalar *= array.reshape(()).item()
            continue
        shape = tuple(array.shape[i] for i in axes)
        labels = tuple(relabel.setdefault(labels[i], len(relabel)) for i in axes)
        operands.append(array.reshape(shape))
        topology.append(labels)
    if not operands or scalar == 0:
        return float(scalar)
    return float(scalar * _network_contraction(tuple(topology))(*operands, backend='numpy'))


@register_cache(key='nonabelian_network_contraction')
@lru_cache(maxsize=4096)
def _network_contraction(topology):
    # Cache the executable contraction, not only its path: contract(...,
    # optimize=path) still rebuilds contraction metadata on every invocation.
    # ContractExpression supports varying dimensions at fixed ranks. These
    # small CG networks share connectivity across many spin sectors, so plan
    # with uniform nontrivial dimensions instead of caching every spin size.
    expression = ','.join(''.join(opt_einsum.get_symbol(i) for i in labels)
                          for labels in topology) + '->'
    shapes = [(2,) * len(labels) for labels in topology]
    planned = opt_einsum.contract_expression(expression, *shapes, optimize='greedy')
    program = []
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for positions, removed, equation, _, blas in planned.contraction_list:
        inputs, output = equation.split('->')
        labels = dict.fromkeys(inputs.replace(',', '') + output)
        if len(labels) > len(alphabet):
            return planned
        renamed = dict(zip(labels, alphabet))
        local_equation = ''.join(renamed.get(c, c) for c in equation)
        axes, transpose = None, None
        if blas and 'EINSUM' not in blas:
            left, right = inputs.split(',')
            contracted = tuple(c for c in left if c in removed)
            axes = (tuple(left.index(c) for c in contracted),
                    tuple(right.index(c) for c in contracted))
            remaining = ''.join(c for c in left + right if c not in removed)
            if remaining != output:
                transpose = tuple(remaining.index(c) for c in output)
        program.append((positions, local_equation, axes, transpose))

    def evaluate(*arrays, backend='numpy'):
        # CG vertices are tiny arrays. Avoid re-parsing backend dispatch,
        # contraction axes and permutations for every coefficient evaluation.
        operands = list(arrays)
        for positions, equation, axes, transpose in program:
            values = [operands.pop(i) for i in positions]
            if axes is not None and max(a.size for a in values) > 256:
                value = np.tensordot(*values, axes=axes)
                if transpose is not None:
                    value = value.transpose(transpose)
            else:
                value = np.einsum(equation, *values, optimize=False)
            operands.append(value)
        return operands[0]

    return evaluate


def scale(sym, charges, signs, total):
    """Convert a normalized invariant CG tree to an ordinary rank-two map.

    For signatures (+,-) this gives an identity in magnetic space; for
    (-,+) the dual-basis phase cancels the Frobenius--Schur sign.
    """
    if len(charges) == 2 and total == sym.zero():
        return np.sqrt(sym.irrep_dimension(charges[0])) * (-1.) ** (charges[0][0] if signs[0] == -1 else 0)
    return 1.


@lru_cache(maxsize=4096)
def intertwiner(sym, charges, signs, total, channel):
    from ._contractions import _fusion_intertwiner
    return scale(sym, charges, signs, total) * _fusion_intertwiner(sym, charges, signs, total, channel)


def fusion_coefficient(sym, charges, signs, total, channel, grouped, out_charges, out_signs, out_channel, paths):
    # Keep full-charge lookups in a separate LRU: U1-labelled aliases must
    # never evict the shared spin coefficients.
    if sym.SYM_ID == 'SU2xU1':
        return _fusion_coefficient_u1_cached(charges, signs, total, channel, grouped,
                                            out_charges, out_signs, out_channel, paths)
    return _fusion_coefficient_cached(sym, charges, signs, total, channel, grouped,
                                      out_charges, out_signs, out_channel, paths)


@register_cache(key='nonabelian_fusion_u1')
@lru_cache(maxsize=4096)
def _fusion_coefficient_u1_cached(charges, signs, total, channel, grouped,
                                  out_charges, out_signs, out_channel, paths):
    # Charge conservation is enforced by fusion metadata. The numerical
    # isometry is SU(2); a miss here can still hit the independent spin cache.
    from ..sym import sym_SU2
    return _fusion_coefficient_cached(
        sym_SU2, tuple((t[0],) for t in charges), signs, (total[0],), channel[::2], grouped,
        tuple((t[0],) for t in out_charges), out_signs, out_channel[::2],
        tuple(path[::2] for path in paths))


@register_cache(key='nonabelian_fusion_coefficient')
@lru_cache(maxsize=65536)
def _fusion_coefficient_cached(sym, charges, signs, total, channel, grouped,
                               out_charges, out_signs, out_channel, paths):
    coefficient = _prefix_fusion_coefficient(sym, charges, signs, total, channel, grouped,
                                             out_charges, out_signs, out_channel, paths)
    if coefficient is not None:
        return coefficient
    return _fusion_coefficient_network(sym, charges, signs, total, channel, grouped,
                                       out_charges, out_signs, out_channel, paths)


def _prefix_fusion_coefficient(sym, charges, signs, total, channel, grouped,
                               out_charges, out_signs, out_channel, paths):
    """Orthogonality for a prefix already present in the canonical SU2 tree.

    Reversing every orientation in a subtree transports the dual metric to
    its root. Each originally positive leaf then contributes D_j**2=(-1)**2j.
    All remaining vertices cancel against their identical partners.
    """
    if sym.SYM_ID != 'SU2' or not grouped:
        return None
    k, n = len(grouped[0]), len(charges)
    if not 1 < k < n or grouped != (tuple(range(k)),) + tuple((i,) for i in range(k, n)):
        return None
    if (len(channel) != n - 2 or out_channel != channel[k - 1:]
            or out_charges != ((channel[k - 2],),) + charges[k:]
            or out_signs != (signs[0],) + signs[k:]
            or paths != (channel[:k - 2],) + ((),) * (n - k)):
        return None
    phase = (-1.) ** sum(charges[i][0] for i in range(k) if signs[i] == 1) if signs[0] == -1 else 1.
    return phase * scale(sym, charges, signs, total) / scale(sym, out_charges, out_signs, total)


def _fusion_coefficient_network(sym, charges, signs, total, channel, grouped,
                                out_charges, out_signs, out_channel, paths):
    """General CG-network reference for groupings not covered by the fast path."""
    from itertools import count
    fresh = count()
    old_labels = [next(fresh) for _ in charges]
    new_labels = [next(fresh) for _ in out_charges]
    factors = tree_network(sym, charges, signs, total, channel, old_labels, fresh)
    factors += tree_network(sym, out_charges, out_signs, total, out_channel, new_labels, fresh)
    for i, group in enumerate(grouped):
        if not group:
            factors.extend((np.ones((1,)), [new_labels[i]]))
        else:
            relative = tuple(signs[j] * out_signs[i] for j in group)
            factors += tree_network(sym, tuple(charges[j] for j in group), relative,
                                    out_charges[i], paths[i], [old_labels[j] for j in group],
                                    fresh, root=new_labels[i])
    coefficient = _contract_scalar(factors) * scale(sym, charges, signs, total) / scale(sym, out_charges, out_signs, total)
    if abs(coefficient) < 1e-13:
        raise ValueError('Inconsistent non-Abelian fusion basis')
    return coefficient


def fusion_scale(config, data, struct, fused, axes, meta, inverse=False):
    if getattr(config.sym, 'IS_ABELIAN', True):
        return data
    weights, lengths = fusion_weights(config.sym, struct, fused, axes, meta)
    return config.backend.scale_blocks(data, tuple(weights), tuple(lengths), inverse)


def fusion_transpose_and_merge(config, data, struct, fused, axes, order, meta, size):
    """Apply fusion weights while writing merged blocks when the backend supports it."""
    if not getattr(config.sym, 'IS_ABELIAN', True):
        merge_scaled = getattr(config.backend, 'transpose_and_merge_scaled', None)
        if merge_scaled is not None:
            weights, lengths = fusion_weights(config.sym, struct, fused, axes, meta)
            return merge_scaled(data, order, meta, size, tuple(weights), tuple(lengths))
        data = fusion_scale(config, data, struct, fused, axes, meta)
    return config.backend.transpose_and_merge(data, order, meta, size)


#: Merging metas are nested tuples of slices, integer tuples and tuples of slices. They are
#: encoded below as fixed-width int64 fields, each container prefixed by its length and each
#: element by a type tag, which makes the encoding injective: distinct metas can never share a
#: byte string, so the bytes serve as an exact dict key (the cache compares keys by value,
#: never by hash alone).
_TAG_INT, _TAG_SEQ, _TAG_NONE, _TAG_SLICE = 1, 2, 3, 4


def _freeze_pair(x):
    """ ``(start, stop)`` of a slice; containers are frozen element-wise.

    Reproduces the old ``freeze`` applied to the meta field being looked up, where a
    subslices entry is either one slice or a (possibly empty) tuple of them.
    """
    if isinstance(x, slice):
        return (x.start, x.stop)
    if isinstance(x, (tuple, list)):
        return tuple(_freeze_pair(y) for y in x)
    return x


def _encode_meta(flat, meta, fields=None) -> bool:
    """ Append ``meta`` to the int64 buffer; ``False`` if it has no lossless encoding.

    ``fields`` selects which positions of every record are encoded; the default encodes all of
    them.  Encoding only the positions a consumer reads makes the key canonical for the *result*
    rather than for the raw meta, so inputs that differ solely in ignored positions share one
    cached value instead of splitting it into several -- fewer misses, less encoding, smaller keys.

    Written flat (no recursion, no intermediate tuples) because it runs on every
    ``fusion_weights`` call, including cache hits: the specialized encoding must stay
    cheaper than the nested tuple it replaces.
    """
    append, extend = flat.append, flat.extend
    if not isinstance(meta, (tuple, list)):
        return False
    extend((_TAG_SEQ, len(meta)))  # self-delimiting: the record count is part of the key
    for record in meta:
        if not isinstance(record, (tuple, list)):
            return False
        # A generator instead of a list: the canonical key is rebuilt on every call, hits
        # included, so do not throw away one list per record.
        if fields is None:
            selected, count = record, len(record)
        else:
            selected, count = (record[i] for i in fields), len(fields)
        extend((_TAG_SEQ, count))
        for field in selected:
            if isinstance(field, (tuple, list)):
                extend((_TAG_SEQ, len(field)))
                for value in field:
                    if type(value) is int:
                        append(_TAG_INT)
                        append(value)
                    elif type(value) is slice:
                        extend((_TAG_SLICE, value.start, value.stop))
                    elif isinstance(value, np.integer):
                        append(_TAG_INT)
                        append(int(value))
                    else:
                        return False
            elif type(field) is slice:
                extend((_TAG_SLICE, field.start, field.stop))
            elif type(field) is int:
                append(_TAG_INT)
                append(field)
            elif isinstance(field, np.integer):
                append(_TAG_INT)
                append(int(field))
            elif field is None:
                append(_TAG_NONE)
            else:
                return False
    return True


def meta_key(meta, fields=None):
    """ Compact lossless key of a merging meta: 8 bytes per scalar instead of a nested tuple.

    The meta is needed to compute the fusion weights but must not be retained with them, so
    the cache keys on this encoding and drops the meta itself. Distinct metas always encode to
    distinct byte strings; should a value not fit int64 (or have an unexpected type), the frozen
    form of the meta is returned instead -- equally exact and hashable on every supported Python,
    but larger.

    ``fields`` restricts the encoding to those record positions; two metas agreeing on them then
    share a key.  That is exact whenever the consumer ignores the remaining positions, and it keeps
    one value from being recomputed and stored once per irrelevant variation.
    """
    flat = array('q')
    try:
        if _encode_meta(flat, meta, fields):
            return flat.tobytes()
    except (OverflowError, TypeError, ValueError):
        pass
    try:
        if fields is None:
            return _freeze_pair(meta)
        return _freeze_pair([[record[i] for i in fields] for record in meta])
    except (TypeError, IndexError):
        return _freeze_pair(meta)          # finer key than asked for, never a wrong one


#: Same shape as ``functools.lru_cache().cache_info()``, so ``yastn.get_cache_info()``
#: reports every cache uniformly.
CacheInfo = namedtuple('CacheInfo', ['hits', 'misses', 'maxsize', 'currsize'])


class _FusionWeightsCache:
    """ Bounded LRU cache for :func:`_compute_fusion_weights`, keyed by :func:`meta_key`.

    ``functools.lru_cache`` cannot express this cache: the meta is an argument but must not be
    part of the retained key, since the frozen nested tuple it used to key on costs about 36
    bytes per scalar (one Python object per integer) as against 8 bytes encoded.
    """
    __slots__ = ('_entries', '_maxsize', 'hits', 'misses')

    def __init__(self, maxsize=1024):
        self._entries: dict = {}
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def lookup(self, key):
        """ Cached value, or ``None`` on a miss -- entries are never ``None``. """
        try:
            value = self._entries.pop(key)
        except KeyError:
            self.misses += 1
            return None
        self._entries[key] = value  # reinserted last: most recently used
        self.hits += 1
        return value

    def store(self, key, value):
        if self._maxsize == 0:
            return
        if self._maxsize is not None:
            while len(self._entries) >= self._maxsize:
                self._entries.pop(next(iter(self._entries)))
        self._entries[key] = value

    def cache_clear(self):
        self._entries.clear()
        self.hits = self.misses = 0

    def set_maxsize(self, maxsize):
        self._maxsize = maxsize
        if maxsize is not None:
            while len(self._entries) > maxsize:
                self._entries.pop(next(iter(self._entries)))

    def cache_info(self):
        return CacheInfo(self.hits, self.misses, self._maxsize, len(self._entries))


_fusion_weights_cache = _FusionWeightsCache(maxsize=1024)


#: Record positions of a merging meta that :func:`_compute_fusion_weights` reads: ``sln``,
#: ``slo`` and ``subslices``.  The other three slots carry decompositions and channel labels
#: that never enter the weights, so encoding them would split a single cached value across
#: several keys -- more misses, more encoding, and larger keys for no gain.
_WEIGHTS_META_FIELDS = (0, 2, 4)


def fusion_weights(sym, struct, fused, axes, meta):
    """ Fusion weights of one hard-fusion scale step, cached under the encoded meta. """
    key = (sym, struct, fused, axes, meta_key(meta, _WEIGHTS_META_FIELDS))
    value = _fusion_weights_cache.lookup(key)
    if value is None:
        value = _compute_fusion_weights(sym, struct, fused, axes, meta)
        _fusion_weights_cache.store(key, value)
    return value


def _compute_fusion_weights(sym, struct, fused, axes, meta):
    from ._auxiliary import get_blocks
    from ._merging import _leg_structure_combine_charges_prod
    old, new = get_blocks(sym, struct), get_blocks(sym, fused)
    old_lookup = {tuple(s): i for i, s in enumerate(old.slc)}
    new_lookup = {tuple(s): i for i, s in enumerate(new.slc)}
    signs = tuple(l.s for l in struct.legs)
    out_signs = tuple(l.s for l in fused.legs)
    decompositions = []
    for group, leg in zip(axes, fused.legs):
        if len(group) > 1:
            dec = _leg_structure_combine_charges_prod(sym, tuple(struct.legs[j] for j in group), leg.t, leg.s)
            lookup = {}
            for charge, entries in zip(dec.t, dec.dec):
                for entry in entries:
                    # Preserve the first-match behavior of the reference search.
                    lookup.setdefault((charge, entry.t, entry.Dslc), entry.channel)
            decompositions.append(lookup)
        else:
            decompositions.append(None)
    # Cache one scalar per reduced block, not per multiplicity entry.
    weights = np.ones(old.nblocks)
    # Only these three slots are read; the cache key encodes exactly them
    # (see _WEIGHTS_META_FIELDS -- keep the two in sync if this loop changes).
    for sln, _, slo, _, subslices, _ in meta:
        ia, ic = old_lookup[_freeze_pair(slo)], new_lookup[_freeze_pair(sln)]
        charges, out_charges = tuple(map(tuple, old.t[ia])), tuple(map(tuple, new.t[ic]))
        paths = []
        for i, (group, dec) in enumerate(zip(axes, decompositions)):
            if dec is None:
                paths.append(())
            else:
                flat = sum((charges[j] for j in group), ())
                slcs = _freeze_pair(subslices[i])
                paths.append(dec[(out_charges[i], flat, slcs)])
        coef = fusion_coefficient(sym, charges, signs, struct.n, tuple(old.channels[ia]), axes,
                                  out_charges, out_signs, tuple(new.channels[ic]), tuple(paths))
        weights[ia] = coef
    return weights, old.slc[:, 1] - old.slc[:, 0]
