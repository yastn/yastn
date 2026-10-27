"""Bounded structural composition for a sequence of fusion-tree basis changes."""
from functools import lru_cache
from math import prod

from .._cache_registry import register_cache


def adjoint_recoupling_steps(steps):
    """Transpose a sequence of real reduced-basis maps, including axis orders."""
    return tuple((tuple(order.index(i) for i in range(len(order))),
                  tuple((source, destination, tuple(shape[i] for i in order), coefficient)
                        for destination, source, shape, coefficient in meta))
                 for order, meta in reversed(steps))


@register_cache(key='recoupling_sequence')
@lru_cache(maxsize=16)
def recoupling_sequence_plan(steps):
    """Index raw sparse steps without constructing their global product."""
    order = tuple(range(len(steps[0][0])))
    indexed = []
    shapes = {}
    for i, (permutation, meta) in enumerate(steps):
        incoming = {}
        for destination, source, shape, coefficient in meta:
            incoming.setdefault(destination, []).append((source, coefficient))
            if i == 0:
                shapes[source] = shape
        indexed.append({key: tuple(value) for key, value in incoming.items()})
        order = tuple(order[j] for j in permutation)
    return order, tuple(indexed), shapes


def composed_recoupling_blocks(plan):
    """Yield one destination block's coefficients, discarding them before the next.

    The live coefficient map is bounded by the channels connected to one
    external-charge block, rather than the size of the global composed map.
    No tensor values or element-sized index maps are constructed here.
    """
    _, indexed, _ = plan
    for destination in indexed[-1]:
        yield destination, composed_recoupling_terms(plan, destination)


def composed_recoupling_terms(plan, destination):
    """Compose only the paths reaching one requested destination block."""
    _, indexed, shapes = plan
    weights = {destination: 1.0}
    for incoming in reversed(indexed):
        previous = {}
        for middle, right in weights.items():
            for source, left in incoming.get(middle, ()):
                previous[source] = previous.get(source, 0.0) + left * right
        weights = {source: value for source, value in previous.items() if value != 0.0}
    return tuple((source, shapes[source], coefficient)
                 for source, coefficient in weights.items())


_UNFUSE_COMPOSITION_MAX_WORK_RATIO = 4


def unfuse_sequence_plan(steps, outputs):
    """Index operations, declining compositions whose path count explodes.

    This conservative bound ignores cancellations and identical source views.
    It is computed from block metadata before any tensor allocation.
    """
    indexed = []
    counts, sequential_work = None, 0
    for kind, parameter, records in steps:
        incoming, next_counts = {}, {}
        for destination, *record in records:
            incoming.setdefault(destination, []).append(tuple(record))
            source = record[0]
            count = 1 if counts is None else counts.get(source, 0)
            next_counts[destination] = next_counts.get(destination, 0) + count
            shape = record[1] if kind == 'recouple' else record[2]
            sequential_work += prod(shape)
        indexed.append((kind, parameter, incoming))
        counts = next_counts
    composed_work = (sum(prod(shape) * counts.get(destination, 0) for destination, shape in outputs)
                     if counts is not None else 0)
    if composed_work > _UNFUSE_COMPOSITION_MAX_WORK_RATIO * sequential_work:
        return None
    return tuple(indexed)


def unfuse_sequence_blocks(plan, outputs):
    """Compose source views for one final block at a time.

    Unmerge only slices and splits axes; recoupling only permutes them.
    Every path therefore remains an affine strided view of the original
    input. Coefficients for identical views are summed before execution.
    The memo is local to one destination, never the global composed map.
    """
    def sources(level, destination, shape, memo):
        key = (level, destination)
        if key in memo:
            return memo[key]
        if level < 0:
            strides = tuple(prod(shape[i + 1:]) if dimension > 1 else 0
                            for i, dimension in enumerate(shape))
            return {(destination, strides): 1.}
        kind, parameter, incoming = plan[level]
        result = {}
        for record in incoming.get(destination, ()):
            source, old_shape = record[:2]
            previous = sources(level - 1, source, old_shape, memo)
            for (offset, strides), coefficient in previous.items():
                if kind == 'recouple':
                    new_strides = tuple(strides[i] for i in parameter)
                    coefficient *= record[2]
                else:
                    _, _, new_shape, starts, weight = record
                    offset += sum(start * stride for start, stride in zip(starts, strides))
                    new_strides = tuple(strides[0] * prod(new_shape[i + 1:parameter])
                                        for i in range(parameter)) + strides[1:]
                    coefficient *= weight
                # Strides on singleton axes do not change the source view.
                new_strides = tuple(stride if dimension > 1 else 0
                                    for stride, dimension in zip(new_strides, shape))
                target = (offset, new_strides)
                result[target] = result.get(target, 0.) + coefficient
        result = {key: value for key, value in result.items() if value != 0.}
        memo[key] = result
        return result

    for destination, shape in outputs:
        terms = sources(len(plan) - 1, destination, shape, {})
        yield destination, shape, tuple((int(offset), tuple(map(int, strides)), float(weight))
                                        for (offset, strides), weight in terms.items())
