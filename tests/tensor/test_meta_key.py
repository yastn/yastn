"""The fusion-weights cache key depends only on the meta fields the consumer reads.

A merging meta record has six slots, but :func:`_compute_fusion_weights` reads only ``sln``,
``slo`` and ``subslices``; the other three (decompositions and channel labels) never enter the
weights.  Encoding them into the key splits one cached value over several entries, so the
canonical key selects the fields that matter.
"""
from yastn.tensor._nonabelian import _WEIGHTS_META_FIELDS, meta_key


def _records(used, junk):
    """One meta record: three used slots interleaved with three ignored ones."""
    return (used[0], junk[0], used[1], junk[1], used[2], junk[2])


def test_ignored_slots_do_not_change_the_weights_key():
    used = (slice(0, 2), slice(0, 3), (slice(1, 2), slice(0, 1)))
    a = (_records(used, ((1, 2, 1, 1), (10, 20), (7,))),)
    b = (_records(used, ((), (), ())),)
    assert meta_key(a, _WEIGHTS_META_FIELDS) == meta_key(b, _WEIGHTS_META_FIELDS)
    # the generic encoder still tells the two raw metas apart
    assert meta_key(a) != meta_key(b)


def test_used_slots_do_change_the_weights_key():
    a = (_records((slice(0, 2), slice(0, 3), (slice(1, 2),)), ((), (), ())),)
    b = (_records((slice(0, 2), slice(1, 4), (slice(1, 2),)), ((), (), ())),)
    assert meta_key(a, _WEIGHTS_META_FIELDS) != meta_key(b, _WEIGHTS_META_FIELDS)


def test_field_selection_is_still_injective_on_a_few_records():
    a = (_records((slice(0, 2), slice(0, 3), (slice(1, 2),)), ((), (), ())),) * 2
    b = (_records((slice(0, 2), slice(0, 3), (slice(1, 2),)), ((), (), ())),
         _records((slice(0, 2), slice(1, 4), (slice(1, 2),)), ((), (), ())))
    assert meta_key(a, _WEIGHTS_META_FIELDS) != meta_key(b, _WEIGHTS_META_FIELDS)
