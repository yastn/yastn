"""Magnetic-basis normalization at reduced tensor interfaces."""
from functools import lru_cache
import numpy as np
import opt_einsum


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
            j = charge[0]
            dual = np.zeros((j + 1, j + 1))
            for m in range(j + 1):
                dual[j - m, m] = (-1.) ** (j - m)
            factors.extend((dual, [internal, label]))
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
            vertex = sym.fusion_isometry(left, right, target).reshape(
                sym.irrep_dimension(target), sym.irrep_dimension(left), sym.irrep_dimension(right))
            if i == len(charges) - 1:
                factors.extend((vertex[0], [left_label, leaves[i]]) if root is None
                               else (vertex, [root, left_label, leaves[i]]))
            else:
                label = next(fresh)
                factors.extend((vertex, [label, left_label, leaves[i]]))
                left_label = label
            left = target
    return factors


@lru_cache(maxsize=16384)
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
    topology = tuple((tuple(a.shape), tuple(labels)) for a, labels in zip(factors[::2], factors[1::2]))
    return float(opt_einsum.contract(*factors, [], optimize=_network_path(topology)))


@lru_cache(maxsize=1024)
def _network_path(topology):
    expression = ','.join(''.join(opt_einsum.get_symbol(i) for i in labels)
                          for _, labels in topology) + '->'
    shapes = [shape for shape, _ in topology]
    return opt_einsum.contract_path(expression, *shapes, shapes=True, optimize='greedy')[0]


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


@lru_cache(maxsize=4096)
def fusion_coefficient(sym, charges, signs, total, channel, grouped, out_charges, out_signs, out_channel, paths):
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
    def freeze(x):
        if isinstance(x, slice):
            return (x.start, x.stop)
        return tuple(freeze(y) for y in x) if isinstance(x, (list, tuple)) else x
    weights, lengths = fusion_weights(config.sym, struct, fused, axes, freeze(meta))
    if inverse:
        weights = 1 / weights
    weights = np.repeat(weights, lengths)
    return data * config.backend.to_tensor(weights, dtype=config.backend.get_yastn_dtype(data),
                                         device=config.backend.get_device(data))


@lru_cache(maxsize=1024)
def fusion_weights(sym, struct, fused, axes, meta):
    from ._auxiliary import get_blocks
    from ._merging import _leg_structure_combine_charges_prod
    old, new = get_blocks(sym, struct), get_blocks(sym, fused)
    old_lookup = {tuple(s): i for i, s in enumerate(old.slc)}
    new_lookup = {tuple(s): i for i, s in enumerate(new.slc)}
    signs = tuple(l.s for l in struct.legs)
    out_signs = tuple(l.s for l in fused.legs)
    decompositions = []
    for group, leg in zip(axes, fused.legs):
        decompositions.append(_leg_structure_combine_charges_prod(sym, tuple(struct.legs[j] for j in group), leg.t, leg.s) if len(group) > 1 else None)
    # Cache one scalar per reduced block, not per multiplicity entry.
    weights = np.ones(old.nblocks)
    for sln, _, slo, _, subslices, _ in meta:
        ia, ic = old_lookup[tuple(slo)], new_lookup[tuple(sln)]
        charges, out_charges = tuple(map(tuple, old.t[ia])), tuple(map(tuple, new.t[ic]))
        paths = []
        for i, (group, dec) in enumerate(zip(axes, decompositions)):
            if dec is None:
                paths.append(())
            else:
                candidates = dec.dec[dec.t.index(out_charges[i])]
                flat = sum((charges[j] for j in group), ())
                paths.append(next(d.channel for d in candidates if d.t == flat and d.Dslc == tuple(subslices[i])))
        coef = fusion_coefficient(sym, charges, signs, struct.n, tuple(old.channels[ia]), axes,
                                  out_charges, out_signs, tuple(new.channels[ic]), tuple(paths))
        weights[ia] = coef
    return weights, old.slc[:, 1] - old.slc[:, 0]
