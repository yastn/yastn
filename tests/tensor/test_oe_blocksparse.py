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
"""Tests for yastn.get_contraction_path and yastn.contract_with_unroll."""
import pytest
import yastn
from opt_einsum.contract import PathInfo

tol = 1e-10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _u1_chain(cfg, *bond_dims):
    """Create a chain of U1-symmetric matrices A(i,j), B(j,k), ...

    bond_dims = (D_i, D_j, D_k, ...) — each is used for charges t=(0,1) with
    D_sector=bond_dims[n].  Returns (tensors, legs).
    """
    def _leg(D):
        return yastn.Leg(cfg, s=1, t=(0, 1), D=(D, D))

    legs = [_leg(d) for d in bond_dims]
    tensors = [
        yastn.rand(config=cfg, legs=[legs[n], legs[n + 1].conj()], n=0)
        for n in range(len(legs) - 1)
    ]
    return tensors, legs


def _split_leg_intra(leg):
    """Split sector (0,) of leg in two halves; keep sector (1,) whole."""
    D0 = leg[(0,)]
    half = D0 // 2
    return [
        yastn.SlicedLeg(t=[(0,)], D=[half],      slices={(0,): slice(0, half)}),
        yastn.SlicedLeg(t=[(0,)], D=[D0 - half], slices={(0,): slice(half, D0)}),
        yastn.SlicedLeg(t=[(1,)], D=[leg[(1,)]]),
    ]


# ---------------------------------------------------------------------------
# 1. Path finding
# ---------------------------------------------------------------------------

def test_get_contraction_path(config_kwargs):
    """get_contraction_path returns a valid (path, PathInfo) for 2- and 3-tensor networks."""
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    (A, B), legs = _u1_chain(cfg, 2, 3, 2)

    path, info = yastn.get_contraction_path(A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'))
    assert path == [(0, 1)]
    assert isinstance(info, PathInfo)

    (A, B, C), legs = _u1_chain(cfg, 2, 3, 4, 2)
    path3, info3 = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l')
    )
    assert len(path3) == 2
    assert all(len(p) == 2 for p in path3)
    assert isinstance(info3, PathInfo)


# ---------------------------------------------------------------------------
# 2. contract_with_unroll without sliced unrolling (plain delegation to ncon)
# ---------------------------------------------------------------------------

def test_contract_with_unroll_no_unroll(config_kwargs):
    """contract_with_unroll without unroll= matches ncon; works for U1 and dense."""
    # U1 two-tensor
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    (A, B), _ = _u1_chain(cfg, 2, 3, 2)
    path, _ = yastn.get_contraction_path(A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'))
    result = yastn.contract_with_unroll(A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'), optimize=path)
    assert yastn.norm(result - yastn.ncon([A, B], [[-1, 1], [1, -2]])) < tol

    # U1 three-tensor chain
    (A, B, C), _ = _u1_chain(cfg, 2, 3, 2, 2)
    path3, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l')
    )
    result3 = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'), optimize=path3
    )
    assert yastn.norm(result3 - yastn.ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]])) < tol

    # Dense tensors
    cfg_d = yastn.make_config(sym='none', **config_kwargs)
    Ad = yastn.rand(config=cfg_d, s=(1, -1), D=(4, 6))
    Bd = yastn.rand(config=cfg_d, s=(1, -1), D=(6, 5))
    path_d, _ = yastn.get_contraction_path(Ad, ('i', 'j'), Bd, ('j', 'k'), ('i', 'k'))
    result_d = yastn.contract_with_unroll(
        Ad, ('i', 'j'), Bd, ('j', 'k'), ('i', 'k'), optimize=path_d
    )
    assert yastn.norm(result_d - yastn.ncon([Ad, Bd], [[-1, 1], [1, -2]])) < tol


# ---------------------------------------------------------------------------
# 3. SlicedLeg API
# ---------------------------------------------------------------------------

def test_sliced_leg_api(config_kwargs):
    """SlicedLeg construction and make_sliced_legs behave as documented."""
    # Construction: plain ints normalised to 1-tuples
    sl = yastn.SlicedLeg(t=[0, 1], D=[3, 4])
    assert sl.t == ((0,), (1,))
    assert sl.D == (3, 4)
    assert sl.tD == {(0,): 3, (1,): 4}

    # Custom slices
    sl2 = yastn.SlicedLeg(t=[(0,)], D=[2], slices={(0,): slice(0, 2)})
    assert sl2.slices[(0,)] == slice(0, 2)

    # make_sliced_legs: one SlicedLeg per charge sector
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg = yastn.Leg(cfg, s=1, t=(0, 1, 2), D=(2, 3, 4))
    parts = yastn.make_sliced_legs(leg)
    assert len(parts) == 3
    for part, ti, Di in zip(parts, leg.t, leg.D):
        assert part.t == (ti,)
        assert part.D == (Di,)
        assert part.slices[ti] == slice(None)


# ---------------------------------------------------------------------------
# 4. Sliced unrolling of contracted indices
# ---------------------------------------------------------------------------

def test_sliced_unroll_contracted_index(config_kwargs):
    """
    Unrolling a contracted index by charge sector and by intra-sector slice
    both recover the full contraction result.
    """
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    # Use D=4 sectors so we can split them in half
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(4, 4))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    path, _ = yastn.get_contraction_path(A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'))
    expected = yastn.ncon([A, B], [[-1, 1], [1, -2]])

    # Charge-sector unrolling: one SlicedLeg per sector
    result_cs = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'),
        unroll={'j': yastn.make_sliced_legs(leg_j)},
        optimize=path,
    )
    assert yastn.norm(result_cs - expected) < tol

    # Intra-sector slicing: sector (0,) split in two halves
    result_is = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'),
        unroll={'j': _split_leg_intra(leg_j)},
        optimize=path,
    )
    assert yastn.norm(result_is - expected) < tol


# ---------------------------------------------------------------------------
# 5. Sliced unrolling of multiple contracted indices across several tensors
# ---------------------------------------------------------------------------

def test_sliced_unroll_multi_index(config_kwargs):
    """
    Simultaneously unrolling two contracted indices (with intra-sector slicing
    on both) in a 3-tensor chain recovers the full contraction.
    """
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(4, 4))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(4, 4))
    leg_l = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    C = yastn.rand(config=cfg, legs=[leg_k, leg_l.conj()], n=0)

    path, _ = yastn.get_contraction_path(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l')
    )
    expected = yastn.ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]])

    result = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), C, ('k', 'l'), ('i', 'l'),
        unroll={'j': _split_leg_intra(leg_j), 'k': _split_leg_intra(leg_k)},
        optimize=path,
    )
    assert yastn.norm(result - expected) < tol


# ---------------------------------------------------------------------------
# 6. Sliced unrolling of output (uncontracted) indices
# ---------------------------------------------------------------------------

def test_sliced_unroll_output_index(config_kwargs):
    """
    Unrolling an OUTPUT index works for both charge-sector and intra-sector
    slicing.

    Network: A(i,j) x B(j,k) -> result(i,k).

    'i' is an output index of A; unrolling it restricts the rows of A that
    enter the contraction.  Summing over slices must recover the full result.
    """
    cfg = yastn.make_config(sym='U1', **config_kwargs)
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(4, 4))   # D=4 so we can split
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)

    path, _ = yastn.get_contraction_path(A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'))
    expected = yastn.ncon([A, B], [[-1, 1], [1, -2]])

    # Charge-sector output unrolling: each sector in its own SlicedLeg
    result_cs = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'),
        unroll={'i': yastn.make_sliced_legs(leg_i)},
        optimize=path,
    )
    assert yastn.norm(result_cs - expected) < tol

    # Intra-sector output unrolling: sector (0,) split in two halves
    result_is = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'),
        unroll={'i': _split_leg_intra(leg_i)},
        optimize=path,
    )
    assert yastn.norm(result_is - expected) < tol

    # Simultaneously unroll output index 'i' and contracted index 'j'
    result_both = yastn.contract_with_unroll(
        A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'),
        unroll={'i': _split_leg_intra(leg_i), 'j': yastn.make_sliced_legs(leg_j)},
        optimize=path,
    )
    assert yastn.norm(result_both - expected) < tol


# ---------------------------------------------------------------------------
# 7. Block-structure diagnostics
# ---------------------------------------------------------------------------

def test_partial_block_structure(config_kwargs):
    """
    Verify the three partial-block scenarios that arise with sliced unrolling:

    (a) Charge-sector unrolling of a contracted index: each partial covers a
        DIFFERENT output block charge set (disjoint), their union = full result.

    (b) Intra-sector slicing of a contracted index: all partials have the
        SAME output block charges but accumulate correctly via + (element-wise
        sum of same-sized blocks).

    (c) Incompatible charge combos (B with n=0 requires j==k): the partial is
        an empty tensor; adding it is a no-op.
    """
    cfg = yastn.make_config(sym='U1', **config_kwargs)

    # --- (a) disjoint blocks ---
    leg_i = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 3))
    leg_j = yastn.Leg(cfg, s=1, t=(0, 1), D=(4, 5))
    leg_k = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 2))
    A = yastn.rand(config=cfg, legs=[leg_i, leg_j.conj()], n=0)
    B = yastn.rand(config=cfg, legs=[leg_j, leg_k.conj()], n=0)
    path, _ = yastn.get_contraction_path(A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'))
    expected = yastn.ncon([A, B], [[-1, 1], [1, -2]])

    partials = [
        yastn.contract_with_unroll(
            A, ('i', 'j'), B, ('j', 'k'), ('i', 'k'),
            unroll={'j': [sl]}, optimize=path,
        )
        for sl in yastn.make_sliced_legs(leg_j)
    ]
    blocks = [set(p.get_blocks_charge()) for p in partials]
    assert blocks[0].isdisjoint(blocks[1]), "Charge-sector partials must be disjoint"
    total = partials[0] + partials[1]
    assert yastn.norm(total - expected) < tol

    # --- (b) overlapping blocks (intra-sector, single-sector leg) ---
    leg_ik = yastn.Leg(cfg, s=1, t=(0,), D=(3,))
    leg_j2 = yastn.Leg(cfg, s=1, t=(0,), D=(6,))
    A2 = yastn.rand(config=cfg, legs=[leg_ik, leg_j2.conj()], n=0)
    B2 = yastn.rand(config=cfg, legs=[leg_j2, leg_ik.conj()], n=0)
    path2, _ = yastn.get_contraction_path(A2, ('i', 'j'), B2, ('j', 'k'), ('i', 'k'))
    expected2 = yastn.ncon([A2, B2], [[-1, 1], [1, -2]])

    slices_j = [
        yastn.SlicedLeg(t=[(0,)], D=[2], slices={(0,): slice(0, 2)}),
        yastn.SlicedLeg(t=[(0,)], D=[2], slices={(0,): slice(2, 4)}),
        yastn.SlicedLeg(t=[(0,)], D=[2], slices={(0,): slice(4, 6)}),
    ]
    partials2 = [
        yastn.contract_with_unroll(
            A2, ('i', 'j'), B2, ('j', 'k'), ('i', 'k'),
            unroll={'j': [sl]}, optimize=path2,
        )
        for sl in slices_j
    ]
    # All three cover the same single block
    assert all(len(p.get_blocks_charge()) == 1 for p in partials2)
    assert partials2[0].get_blocks_charge() == partials2[1].get_blocks_charge()
    result2 = yastn.contract_with_unroll(
        A2, ('i', 'j'), B2, ('j', 'k'), ('i', 'k'),
        unroll={'j': slices_j}, optimize=path2,
    )
    assert yastn.norm(result2 - expected2) < tol

    # --- (c) empty partial from incompatible (j,k) charges ---
    leg_j3 = yastn.Leg(cfg, s=1, t=(0, 1), D=(4, 4))
    leg_k3 = yastn.Leg(cfg, s=1, t=(0, 1), D=(3, 3))
    leg_l3 = yastn.Leg(cfg, s=1, t=(0, 1), D=(2, 2))
    A3 = yastn.rand(config=cfg, legs=[leg_i, leg_j3.conj()], n=0)
    B3 = yastn.rand(config=cfg, legs=[leg_j3, leg_k3.conj()], n=0)
    C3 = yastn.rand(config=cfg, legs=[leg_k3, leg_l3.conj()], n=0)
    path3, _ = yastn.get_contraction_path(
        A3, ('i', 'j'), B3, ('j', 'k'), C3, ('k', 'l'), ('i', 'l')
    )
    expected3 = yastn.ncon([A3, B3, C3], [[-1, 1], [1, 2], [2, -2]])
    # (j=sector0, k=sector1) is incompatible with B3 (n=0 requires j==k)
    sl_j0 = yastn.SlicedLeg(t=[(0,)], D=[4])
    sl_k1 = yastn.SlicedLeg(t=[(1,)], D=[3])
    p_incompatible = yastn.contract_with_unroll(
        A3, ('i', 'j'), B3, ('j', 'k'), C3, ('k', 'l'), ('i', 'l'),
        unroll={'j': [sl_j0], 'k': [sl_k1]}, optimize=path3,
    )
    assert len(p_incompatible.get_blocks_charge()) == 0, "Incompatible charges → empty partial"
    result3 = yastn.contract_with_unroll(
        A3, ('i', 'j'), B3, ('j', 'k'), C3, ('k', 'l'), ('i', 'l'),
        unroll={'j': yastn.make_sliced_legs(leg_j3), 'k': yastn.make_sliced_legs(leg_k3)},
        optimize=path3,
    )
    assert yastn.norm(result3 - expected3) < tol
