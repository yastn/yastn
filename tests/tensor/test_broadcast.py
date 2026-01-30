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
""" yastn.broadcast() """
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_broadcast_dense(config_kwargs):
    """ test broadcast on dense tensors """
    # a is a diagonal tensor to be broadcasted

    config = yastn.make_config(sym='dense', **config_kwargs)
    a = yastn.rand(config=config, s=(1, -1), isdiag=True, D=5)
    a1 = a.diag()

    # broadcast on tensor b
    b = yastn.rand(config=config, s=(-1, 1, 1, -1), D=(2, 5, 2, 5))

    r1 = a.broadcast(b, axes=1)
    r2 = a1.tensordot(b, axes=(1, 1)).transpose((1, 0, 2, 3))
    r3 = a.tensordot(b, axes=(1, 1)).transpose((1, 0, 2, 3))
    r4 = b.tensordot(a, axes=(1, 1)).transpose((0, 3, 1, 2))
    assert all(yastn.norm(r1 - x) < tol for x in (r2, r3, r4))

    # broadcast with conj
    a = yastn.randC(config=config, s=(-1, 1), D=(5, 5))
    a = a.diag()  # 5x5 isdiag == True
    a1 = a.diag()  # 5x5 isdiag == False
    b = yastn.randC(config=config, s=(1, -1, 1, -1), D=(2, 5, 2, 5))

    r1 = a.conj().broadcast(b, axes=1)
    r2 = a.broadcast(b.conj(), axes=1).conj()
    r3 = a1.tensordot(b, axes=(0, 1), conj=(1, 0)).transpose((1, 0, 2, 3))
    r5 = a.tensordot(b, axes=(0, 1), conj=(1, 0)).transpose((1, 0, 2, 3))
    r4 = b.tensordot(a, axes=(1, 0), conj=(0, 1)).transpose((0, 3, 1, 2))
    assert all(yastn.norm(r1 - x) < tol for x in [r2, r3, r4, r5])
    assert all(x.is_consistent() for x in [r1, r2, r3, r4, r5])

    # broadcast and trace
    a = yastn.rand(config=config, isdiag=True, D=5)
    a1 = a.diag()  # 5x5 isdiag=False
    b = yastn.rand(config=config, s=(-1, 1, 1, -1), D=(2, 5, 3, 5))
    r1 = a1.tensordot(b, axes=((1, 0), (1, 3)))
    r2 = a.tensordot(b, axes=((0, 1), (3, 1)))
    r3 = b.tensordot(a, axes=((1, 3), (1, 0)))
    assert all(yastn.norm(r1 - x) < tol for x in [r2, r3])


def test_broadcast_U1(config_kwargs):
    """ test broadcast on U1 tensors """

    config = yastn.make_config(sym='U1', **config_kwargs)
    leg0 = yastn.Leg(config, s=1, t=(-1, 1), D=(7, 8))
    leg1 = yastn.Leg(config, s=1, t=(-1, 1, 2), D=(1, 2, 3))
    leg2 = yastn.Leg(config, s=1, t=(-1, 1, 2), D=(4, 5, 6))
    leg3 = yastn.Leg(config, s=1, t=(-1, 1, 2), D=(7, 8, 9))
    leg4 = yastn.Leg(config, s=1, t=(-1, 1, 2), D=(10, 11, 12))

    a = yastn.rand(config=config, isdiag=True, legs=(leg0, leg0.conj()))
    a1 = a.diag()
    assert a.get_shape() == a1.get_shape() == (15, 15)

    b = yastn.rand(config=config, legs=[leg1.conj(), leg2, leg3, leg4.conj()])
    assert b.get_shape() == (6, 15, 24, 33)

    # broadcast via tensordot
    r1 = a.broadcast(b, axes=2)
    r2 = a.tensordot(b, axes=(1, 2)).transpose((1, 2, 0, 3))
    r3 = b.tensordot(a1, axes=(2, 1)).transpose((0, 1, 3, 2))
    r4 = b.tensordot(a, axes=(2, 1)).transpose((0, 1, 3, 2))
    assert all(x.is_consistent() for x in [r1, r2, r3, r4])
    assert all(yastn.norm(r1 - x) < tol for x in [r2, r3, r4])

    c = yastn.rand(config=config, legs=[leg1.conj(), leg2, leg3, leg3.conj()])

    # broadcast with trace
    r1 = a1.tensordot(c, axes=((0, 1), (3, 2)))
    r2 = a.tensordot(c, axes=((0, 1), (3, 2)))
    r3 = c.tensordot(a, axes=((2, 3), (1, 0)))
    assert all(x.is_consistent() for x in [r1, r2, r3])
    assert all(yastn.norm(r1 - x) < tol for x in [r2, r3])

    # broadcast with transpose and meta
    bf = b.fuse_legs(axes=(3, (1, 0), 2), mode='meta')
    assert bf.trans == (3, 1, 0, 2)
    rfb = a.broadcast(bf, axes=2)
    assert rfb.trans == (3, 1, 0, 2)
    rb = a.broadcast(b, axes=2)
    rbf = rb.fuse_legs(axes=(3, (1, 0), 2), mode='meta')
    assert (rbf - rfb).norm() < tol

    # broadcast with transposed diag -- transpose is ignored
    at = a.T
    assert at.trans == (1, 0)
    r1 = a.broadcast(b, axes=2)
    r2 = at.broadcast(b, axes=2)
    assert (r1 - r2).norm() < tol


def test_broadcast_Z2xU1(config_kwargs):
    """ test broadcast on Z2xU1 tensors """
    config = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    leg0a = yastn.Leg(config, s=1, t=[(0, 0), (0, 2), (1, 0), (1, 2)], D=[6, 3, 9, 6])
    leg0b = yastn.Leg(config, s=1, t=[(0, 0), (0, 2), (1, 0), (1, 2), (1, 3)], D=[6, 3, 9, 6, 5])
    leg0c = yastn.Leg(config, s=1, t=[(0, 2), (0, 3)], D=[3, 4])
    leg1 = yastn.Leg(config, s=1, t=[(0, 0), (0, 2)], D=[2, 3])
    leg2 = yastn.Leg(config, s=1, t=[(0, 1), (1, 0), (0, 0), (1, 1)], D=[4, 5, 6, 3])
    leg3 = yastn.Leg(config, s=1, t=[(0, 0), (0, 2)], D=[3, 2])

    a = yastn.rand(config=config, legs=[leg0a.conj(), leg1.conj(), leg2, leg3])
    b = yastn.eye(config=config, legs=[leg0b, leg0b.conj()])
    c = yastn.eye(config=config, legs=[leg0c, leg0c.conj()])

    # broadcast
    r1 = b.broadcast(a, axes=0)
    r2 = b.tensordot(a, axes=(0, 0))
    r3 = a.tensordot(b, axes=(0, 0)).transpose((3, 0, 1, 2))
    assert all(x.is_consistent() for x in [r1, r2, r3])
    assert all(yastn.norm(a - x) < tol for x in [r1, r2, r3])
    assert not any((r1.isdiag, r2.isdiag, r3.isdiag))

    # broadcast on diagonal
    r4 = c.broadcast(b, axes=1)
    r5 = b.broadcast(c, axes=1)
    assert r4.is_consistent()
    assert r4.isdiag
    nr4 = r4.to_numpy()
    assert nr4.shape == (3, 3)
    assert np.trace(nr4) == 3.
    assert (r4 - r5).norm() < tol

    # broadcast tensor b over multiple tensors in single call
    r1p, r5p = b.broadcast(a, c, axes=(0, 1))
    assert (r1 - r1p).norm() < tol
    assert (r5 - r5p).norm() < tol


def test_broadcast_exceptions(config_kwargs):
    """ test broadcast raising errors """
    config = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.rand(config=config, isdiag=True, t=(-1, 1), D=(7, 8))
    a_nondiag = a.diag()
    b = yastn.rand(config=config, s=(-1, 1, 1, -1),
                    t=((-1, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                    D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    with pytest.raises(yastn.YastnError):
        _ = a_nondiag.broadcast(b, axes=2)
        # First tensor should be diagonal.
    with pytest.raises(yastn.YastnError):
        bmf = b.fuse_legs(axes=(0, (1, 2), 3), mode='meta')
        _ = a.broadcast(bmf, axes=1)
        # Second tensor`s leg specified by axes cannot be fused.
    with pytest.raises(yastn.YastnError):
        bhf = b.fuse_legs(axes=(0, (1, 2), 3), mode='hard')
        _ = a.broadcast(bhf, axes=1)
        # Second tensor`s leg specified by axes cannot be fused.
    with pytest.raises(yastn.YastnError):
        a.broadcast(b, axes=1)  # Bond dimensions do not match.
    with pytest.raises(yastn.YastnError):
        _, _ = a.broadcast(b, b, axes=(1, 1, 1))
        # There should be exactly one axes for each tensor to be projected.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
