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
"""Tests for the experimental charged-sector DoublePepsTensor path."""

import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps._doublePepsTensor_charged import DoublePepsTensorCharged

tol = 1e-12  # pylint: disable=invalid-name


def create_charged_double_peps_tensor(config_kwargs, charge=1, dtype="complex128"):
    config = yastn.make_config(sym="U1", fermionic=True, **config_kwargs)
    leg0 = yastn.Leg(config, s=-1, t=(-1, 0, 1), D=(1, 2, 1))
    leg1 = yastn.Leg(config, s=1, t=(-1, 0, 1), D=(2, 1, 2))
    leg2 = yastn.Leg(config, s=1, t=(-1, 0, 1), D=(3, 1, 2))
    leg3 = yastn.Leg(config, s=-1, t=(-1, 0, 1), D=(2, 2, 3))
    leg4 = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
    A = yastn.rand(config, legs=[leg0, leg1, leg2, leg3, leg4], dtype=dtype, n=charge)
    return DoublePepsTensorCharged(bra=A, ket=A)


def test_original_double_peps_tensor_rejects_charged_tensors(config_kwargs):
    T0 = create_charged_double_peps_tensor(config_kwargs, charge=1)
    original = fpeps.DoublePepsTensor(bra=T0.bra, ket=T0.ket)
    fused = original.fuse_layers()
    l0 = yastn.Leg(fused.config, s=1, t=(-1, 0, 1), D=(1, 2, 1))
    lfs = original.get_legs()
    probe = yastn.rand(fused.config, legs=[l0, lfs[0].conj(), lfs[1].conj(), l0], n=0)

    with pytest.raises(AssertionError, match="nor carry charge"):
        original.tensordot(probe, axes=((0, 1), (1, 2)))


def test_charged_double_peps_tensor_tl_br_matches_fused_layers(config_kwargs):
    T0 = create_charged_double_peps_tensor(config_kwargs, charge=1)
    ops = yastn.operators.SpinlessFermions(sym="U1", **config_kwargs)
    T0.set_operator_(ops.c())
    T0.add_charge_swaps_((1,), ["k1"])
    fused = T0.fuse_layers()
    lfs = T0.get_legs()
    l0 = yastn.Leg(fused.config, s=1, t=(-1, 0, 1), D=(1, 2, 1))
    l3 = yastn.Leg(fused.config, s=-1, t=(-3, 0, 2), D=(1, 1, 2))

    t01 = yastn.rand(fused.config, legs=[l0, lfs[0].conj(), lfs[1].conj(), l3], n=1)
    t32 = yastn.rand(fused.config, legs=[l0, lfs[3].conj(), lfs[2].conj()], n=1)

    a01 = fused.tensordot(t01, axes=((0, 1), (1, 2)))
    b01 = T0.tensordot(t01, axes=((0, 1), (1, 2)))
    assert (a01 - b01).norm() < tol

    a32 = fused.tensordot(t32, axes=((3, 2), (1, 2)))
    b32 = T0.tensordot(t32, axes=((3, 2), (1, 2)))
    assert (a32 - b32).norm() < tol


def test_charged_double_peps_tensor_full_tensordot_matches_fused_layers(config_kwargs):
    T0 = create_charged_double_peps_tensor(config_kwargs, charge=1)
    ops = yastn.operators.SpinlessFermions(sym="U1", **config_kwargs)
    T0.set_operator_(ops.c())
    T0.add_charge_swaps_((1,), ["k1"])
    fused = T0.fuse_layers()

    allowed_transpose = (
        (0, 1, 2, 3),
        (1, 2, 3, 0),
        (2, 3, 0, 1),
        (3, 0, 1, 2),
        (0, 3, 2, 1),
        (1, 0, 3, 2),
        (2, 1, 0, 3),
        (3, 2, 1, 0),
    )

    for axes1 in allowed_transpose:
        T1 = T0.transpose(axes=axes1)
        r1 = fused.transpose(axes=axes1)
        lfs = T1.get_legs()
        l0 = yastn.Leg(fused.config, s=1, t=(-1, 0, 1), D=(1, 2, 1))
        l3 = yastn.Leg(fused.config, s=-1, t=(-3, 0, 2), D=(1, 1, 2))

        t01 = yastn.rand(fused.config, legs=[l0, lfs[0].conj(), lfs[1].conj(), l3], n=1)
        t12 = yastn.rand(fused.config, legs=[lfs[1].conj(), lfs[2].conj(), l3], n=1)
        t32 = yastn.rand(fused.config, legs=[l0, lfs[3].conj(), lfs[2].conj()], n=1)
        t30 = yastn.rand(fused.config, legs=[l0, lfs[3].conj(), lfs[0].conj(), l3, l3], n=1)

        a01 = r1.tensordot(t01, axes=((0, 1), (1, 2)))
        b01 = T1.tensordot(t01, axes=((0, 1), (1, 2)))
        assert (a01 - b01).norm() < tol

        a12 = yastn.tensordot(r1, t12, axes=((1, 2), (0, 1)))
        b12 = yastn.tensordot(T1, t12, axes=((1, 2), (0, 1)))
        assert (a12 - b12).norm() < tol

        a32 = r1.tensordot(t32, axes=((3, 2), (1, 2)))
        b32 = T1.tensordot(t32, axes=((3, 2), (1, 2)))
        assert (a32 - b32).norm() < tol

        a30 = yastn.tensordot(t30, r1, axes=((2, 1), (0, 3)))
        b30 = yastn.tensordot(t30, T1, axes=((2, 1), (0, 3)))
        assert (a30 - b30).norm() < tol


if __name__ == "__main__":
    pytest.main([__file__, "-vs", "--durations=0"])
