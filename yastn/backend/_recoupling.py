"""Bounded structural composition for a sequence of fusion-tree basis changes."""
from functools import lru_cache

from .._cache_registry import register_cache


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
    _, indexed, shapes = plan
    for destination in indexed[-1]:
        weights = {destination: 1.0}
        for incoming in reversed(indexed):
            previous = {}
            for middle, right in weights.items():
                for source, left in incoming.get(middle, ()):
                    previous[source] = previous.get(source, 0.0) + left * right
            weights = {source: value for source, value in previous.items() if value != 0.0}
        yield destination, tuple((source, shapes[source], coefficient)
                                 for source, coefficient in weights.items())
