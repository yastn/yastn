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
""" Test elements of fuse_legs(... mode='meta') """
import numpy as np
import pytest
import yastn

tol = 1e-10  #pylint: disable=invalid-name


def test_fuse(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1,),
                  t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))
    b = a.fuse_legs(axes=(0, 1, (2, 3, 4)), mode='meta')
    c = b.fuse_legs(axes=(1, (0, 2)), mode='meta')
    c = c.unfuse_legs(axes=1)
    c = c.unfuse_legs(axes=2)
    d = c.move_leg(source=1, destination=0)
    assert yastn.norm(a - d) < tol  # == 0.0


def test_fuse_split(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1,),
                  t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))

    af = a.fuse_legs(axes=(0, (2, 1), (3, 4)), mode='meta')
    af = af.fuse_legs(axes=((0, 1), 2), mode='meta')
    Uf, Sf, Vf = yastn.linalg.svd(af, axes=(0, 1))

    U, S, V = yastn.linalg.svd(a, axes=((0, 1, 2), (3, 4)))
    U = U.fuse_legs(axes=(0, (2, 1), 3), mode='meta')
    U = U.fuse_legs(axes=((0, 1), 2), mode='meta')
    V = V.fuse_legs(axes=(0, (1, 2)), mode='meta')

    US = yastn.tensordot(U, S, axes=(1, 0))
    a2 = yastn.tensordot(US, V, axes=(1, 0))
    assert yastn.norm(af - a2) < tol  # == 0.0
    USf = yastn.tensordot(Uf, Sf, axes=(1, 0))
    a3 = yastn.tensordot(USf, Vf, axes=(1, 0))
    assert yastn.norm(af - a3) < tol  # == 0.0
    a3 = a3.unfuse_legs(axes=0)
    a3 = a3.unfuse_legs(axes=(1, 2)).move_leg(source=2, destination=1)
    assert yastn.norm(a - a3) < tol  # == 0.0

    Qf, Rf = yastn.linalg.qr(af, axes=(0, 1))
    Q, R = yastn.linalg.qr(a, axes=((0, 1, 2), (3, 4)))
    Q = Q.fuse_legs(axes=(0, (2, 1), 3), mode='meta')
    Q = Q.fuse_legs(axes=((0, 1), 2), mode='meta')
    assert yastn.norm(Q - Qf) < tol  # == 0.0
    Rf = Rf.unfuse_legs(axes=1)
    assert yastn.norm(R - Rf) < tol  # == 0.0

    aH = yastn.tensordot(af, af, axes=(1, 1), conj=(0, 1))
    Vf, Uf = yastn.linalg.eigh(aH, axes=(0, 1))
    Uf = Uf.unfuse_legs(axes=0)
    UVf = yastn.tensordot(Uf, Vf, axes=(2, 0))
    aH2 = yastn.tensordot(UVf, Uf, axes=(2, 2), conj=(0, 1))
    aH = aH.unfuse_legs(axes=(0, 1))
    assert yastn.norm(aH2 - aH) < tol  # == 0.0


def test_fuse_transpose(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    assert a.get_shape() == (3, 5, 7, 9, 11, 13)

    fa = a.fuse_legs(axes=((0, 1), 2, (3, 4), 5), mode='meta')
    assert fa.get_shape() == (15, 7, 99, 13)

    fc = np.transpose(fa, axes=(3, 2, 1, 0))
    assert fc.get_shape() == (13, 99, 7, 15)

    fc = fc.unfuse_legs(axes=(1, 3))
    assert fc.get_shape() == (13, 9, 11, 7, 3, 5)

    fc = fa.move_leg(source=1, destination=2)
    assert fc.get_shape() == (15, 99, 7, 13)

    c = fc.unfuse_legs(axes=(1, 0))
    assert c.get_shape() == (3, 5, 9, 11, 7, 13)


def test_get_shapes(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])

    assert a.get_shape() == a.to_numpy().shape == (3, 5, 7, 9, 11, 13)
    assert a.get_signature() == (-1, -1, -1, 1, 1, 1)

    b = a.to_nonsymmetric()
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)

    a = a.fuse_legs(axes=[0, 1, (2, 3), (4, 5)], mode='meta')
    assert a.get_shape() == a.to_numpy().shape == (3, 5, 63, 143)
    assert a.get_signature() == (-1, -1, -1, 1)

    b = a.to_nonsymmetric()
    assert b.get_shape() == (3, 5, 63, 143)
    b = a.to_nonsymmetric(native=True)
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)

    a = a.fuse_legs(axes=[0, (1, 2, 3)], mode='meta')
    assert a.get_shape() == a.to_numpy().shape == (3, 28389)
    assert a.get_signature() == (-1, -1)

    b = a.to_nonsymmetric()
    assert b.get_shape() == (3, 28389)
    b = a.to_nonsymmetric(native=True)
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)

    a = a.fuse_legs(axes=[(0, 1)], mode='meta')
    assert a.get_shape() == a.to_numpy().shape == (a.size,)
    assert a.get_signature() == (-1,)

    b = a.to_nonsymmetric()
    assert b.get_shape() == (a.size,)
    b = a.to_nonsymmetric(native=True)
    assert b.get_shape() == (3, 5, 7, 9, 11, 13)


def test_fuse_get_legs(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.ones(config=config_U1, s=(-1, -1, -1, 1, 1, 1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    b = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1,),
                  t=((-1, 0, 1), (1,), (-1, 1), (0, 1), (0, 1, 2)),
                  D=((2, 1, 2), (4,), (4, 6), (7, 8), (9, 10, 11)))
    af = a.fuse_legs(axes=((0, 1), (2, 3, 4), 5), mode='meta')
    bf = b.fuse_legs(axes=(0, (1, 2), 3, 4), mode='meta')
    bff = bf.fuse_legs(axes=(0, (1, 2), 3), mode='meta')

    legs1 = [a.get_legs(2).conj(), a.get_legs(3).conj(), a.get_legs(4).conj(), b.get_legs(1), b.get_legs(2), b.get_legs(3)]
    c1 = yastn.ones(config=a.config, legs=legs1)
    r1 = yastn.ncon([a, b, c1], [[-1, -2, 1, 2, 3, -3], [-4, 4, 5, 6, -5], [1, 2, 3, 4, 5, 6]], [0, 1, 0])

    legs2 = [af.get_legs(1).conj(), bff.get_legs(1)]
    c2 = yastn.ones(config=af.config, legs=legs2)
    r2 = yastn.ncon([af, bff, c2], [[-1, 1, -2], [-3, 2, -4], [1, 2]], [0, 1, 0])

    c3 = c2.unfuse_legs(axes=1)  # partial unfuse
    r3 = yastn.ncon([af, bf, c3], [[-1, 1, -2], [-3, 2, 3, -4], [1, 2, 3]], [0, 1, 0])
    assert yastn.norm(r3 - r2) < tol  # == 0.0

    r2 = r2.unfuse_legs(axes=0)
    assert yastn.norm(r1 - r2) < tol  # == 0.0


def test_fuse_legs_exceptions(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.rand(config=config_U1, s=(-1, 1, 1, -1, 1,),
                  t=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
                  D=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)))
    b = yastn.rand(config=config_U1, isdiag=True, t=(0, 1), D=(1, 2))
    with pytest.raises(yastn.YastnError):
        b.fuse_legs(axes=((0, 1),), mode='meta')
        # Cannot fuse legs of a diagonal tensor.
    with pytest.raises(yastn.YastnError):
        b.unfuse_legs(axes=0)
        # Cannot unfuse legs of a diagonal tensor.
    with pytest.raises(yastn.YastnError):
        a.fuse_legs(axes=((0, 1, 2, 3, 4),), mode='wrong')
    # mode not in (`meta`, `hard`). Mode can be specified in config file.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
