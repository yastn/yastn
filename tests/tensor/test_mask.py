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
""" Test yastn.mask() """
import pytest
import yastn
try:
    from .configs import config_U1, config_Z2xU1
except ImportError:
    from configs import config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def test_mask_basic():
    """ series of tests for apply_mask """
    config_U1.backend.random_seed(seed=0)  # fix for tests

    # start with U1
    leg1 =  yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9))
    leg2 =  yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(5, 6, 7))
    a = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1, leg2.conj()])

    legd =  yastn.Leg(config_U1, s=1, t=(-1, 1), D=(7, 8))
    b0 = yastn.rand(config=config_U1, isdiag=True, legs=legd)

    b = b0.copy()  # create a mask by hand
    b[(-1, -1)] = b[(-1, -1)] < -2  # all false
    b[(1, 1)] = b[(1, 1)] > 0  # some true
    tr_b = b.trace(axes=(0, 1)).item()

    c = b.apply_mask(a, axes=2)

    # application of the mask should leave a single charge (1,) on this leg
    l = c.get_legs(axes=2)
    assert l.t == ((1,),) and l[(1,)] == tr_b  # in second checks the bond dimension

    d0 = b.apply_mask(b0, axes=0)
    d1 = b.apply_mask(b0, axes=-1)
    assert yastn.norm(d0 - d1) < tol
    l = d1.get_legs(axes=1)
    assert l.t == ((1,),) and l[(1,)] == tr_b

    # apply the same mask on 2 tensors
    d2, c2 = b.apply_mask(b0, a, axes=(-1, 2))
    assert (d2 - d0).norm() < tol
    assert (c2 - c).norm() < tol

    # here using Z2xU1 symmetry
    legs = [yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(6, 3, 9, 6)),
            yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2)), D=(3, 2)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, 1), (1, 0), (0, 0), (1, 1)), D=(4, 5, 6, 3)),
            yastn.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2)), D=(2, 3))]
    a = yastn.rand(config=config_Z2xU1, legs=legs)

    # diagonal tensor exactly matching first leg of a
    b = yastn.rand(config=config_Z2xU1, isdiag=True, legs=legs[0])

    b[(0, 0, 0, 0)] *= 0  # put block (0, 0, 0, 0) in diagonal b to 0.

    bgt = b > 0
    blt = b < 0
    bge = b >= 0
    ble = b <= 0
    assert bgt.trace().item() + ble.trace().item() == blt.trace().item() + bge.trace().item() == b.get_shape(axes=0)

    assert all(bgt.bitwise_not().data == ble.data)

    for bb in [bgt, blt, bge, ble]:
        bnd_dim = bb.trace(axes=(0, 1)).item()
        c = bb.apply_mask(a, axes=0)
        l = c.get_legs(axes=0)
        assert sum(l.D) == bnd_dim

    assert blt.apply_mask(bge, axes=0).trace() < tol  # == 0.
    assert ble.apply_mask(bgt, axes=1).trace() < tol  # == 0.


def test_mask_exceptions():
    """ trigger exceptions for apply_mask """
    legd =  yastn.Leg(config_U1, s=1, t=(-1, 1), D=(8, 8))
    a = yastn.rand(config=config_U1, isdiag=True, legs=legd)
    a_nondiag = a.diag()

    leg1 =  yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9))
    leg2 =  yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(5, 6, 7))
    b = yastn.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1, leg2.conj()])

    with pytest.raises(yastn.YastnError):
        _ = a_nondiag.apply_mask(b, axes=2)
        # First tensor should be diagonal.
    with pytest.raises(yastn.YastnError):
        bmf = b.fuse_legs(axes=(0, (1, 2), 3), mode='meta')
        _ = a.apply_mask(bmf, axes=1)
        # Second tensor`s leg specified by axes cannot be fused.
    with pytest.raises(yastn.YastnError):
        bhf = b.fuse_legs(axes=(0, (1, 2), 3), mode='hard')
        _ = a.apply_mask(bhf, axes=1)
        # Second tensor`s leg specified by axes cannot be fused.
    with pytest.raises(yastn.YastnError):
        _ = a.apply_mask(b, axes=1)  # Bond dimensions do not match.
    with pytest.raises(yastn.YastnError):
        _, _ = a.apply_mask(b, b, axes=[2, 2, 1])
        # There should be exactly one axis for each tensor to be projected.


if __name__ == '__main__':
    test_mask_basic()
    test_mask_exceptions()
