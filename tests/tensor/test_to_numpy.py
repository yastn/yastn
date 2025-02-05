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
""" yastn.to_nonsymmetric() yastn.to_dense() yastn.to_numpy() """
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def test_dense_basic(config_kwargs):
    """ a.to_numpy() is equivalent to np.array(a.to_dense())"""
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    norms = []
    tens = [yastn.rand(config=config_U1, s=(-1, -1, 1),
                      t=((-1, 1, 2), (-1, 1, 2), (1, 2)),
                      D=((1, 3, 4), (4, 5, 6), (3, 4))),
            yastn.rand(config=config_U1, s=(-1, -1, 1),
                      t=((-1, 0), (1, 2), (0, 1)),
                      D=((1, 2), (5, 6), (2, 3))),
            yastn.rand(config=config_U1, s=(-1, -1, 1),
                      t=((0, 1), (0, 1), (1, 2)),
                      D=((2, 3), (5, 5), (3, 4)))]
    shapes = [(8, 15, 7), (3, 11, 5), (5, 10, 7)]
    common_shape = (10, 20, 9)
    norms.append(_test_dense_v1(tens, shapes, common_shape))

    # with meta-fusion
    mtens = [a.fuse_legs(axes=((0, 1), 2), mode='meta') for a in tens]
    mtens = [ma.fuse_legs(axes=[(0, 1)], mode='meta') for ma in mtens]
    mshapes = [(126,), (58,), (135,)]
    mcommon_shape = (211,)
    norms.append(_test_dense_v1(mtens, mshapes, mcommon_shape))

    htens = [a.fuse_legs(axes=((0, 1), 2), mode='hard') for a in tens]
    htens = [ha.fuse_legs(axes=[(0, 1)], mode='hard') for ha in htens]
    hshapes = [(126,), (58,), (135,)]
    hcommon_shape = (383,)
    norms.append(_test_dense_v1(htens, hshapes, hcommon_shape))

    assert all(pytest.approx(n, rel=tol) == norms[0] for n in norms)

    # provided individual legs
    d = yastn.rand(config=config_U1, s=(-1, 1, 1),
                  t=((-1, 0), (-2, -1), (0, 1, 2)),
                  D=((1, 2), (3, 4), (2, 3, 4)))
    a = tens[0]
    lsa = {1: d.get_legs(1).conj(), 0: d.get_legs(2).conj()}
    lsd = {1: a.get_legs(1).conj(), 2: a.get_legs(0).conj()}
    na = a.to_numpy(legs=lsa)
    nd = d.to_numpy(legs=lsd)
    nad = np.tensordot(na, nd, axes=((1, 0), (1, 2)))

    ad = yastn.tensordot(a, d, axes=((1, 0), (1, 2)))
    lsad = {0: a.get_legs(2), 1: d.get_legs(0)}
    assert np.allclose(ad.to_numpy(legs=lsad), nad)

    # the same with leg fusion
    for mode in ('meta', 'hard'):
        fa = a.fuse_legs(axes=(2, (1, 0)), mode=mode)
        fd = d.fuse_legs(axes=((1, 2), 0), mode=mode)
        fad = fa @ fd
        na = fa.to_numpy(legs={1: fd.get_legs(0).conj()})
        nd = fd.to_numpy(legs={0: fa.get_legs(1).conj()})
        assert np.allclose(na @ nd, nad)
        assert np.allclose(ad.to_numpy(legs=lsad), nad)
        assert np.allclose(fad.to_numpy(legs=lsad), nad)


def test_to_raw_tensor(config_kwargs):
    """ test to_raw_tensor and getting single block """
    # leg with a single charge sector
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    leg = yastn.Leg(config_Z2xU1, s=1, t=[(0, 0)], D=[2])
    a = yastn.ones(config=config_Z2xU1, legs=[leg, leg, leg])
    assert pytest.approx(a.norm().item() ** 2) == 8.

    raw = a.to_raw_tensor()  # if there is only one single block in tensor
    assert raw.shape == (2, 2, 2)

    raw = a[((0, 0), (0, 0), (0, 0))]  # accesing specific block
    assert raw.shape == (2, 2, 2)

    # add 1 to all elements of the block
    a[((0, 0), (0, 0), (0, 0))] += 1  # (broadcasted by underlaying backend tensors)
    assert pytest.approx(a.norm().item() ** 2) == 32.

    # add 2nd block to the tensor
    a.set_block(ts=((1, 0), (1, 0), (0, 0)), Ds=(2, 2, 2), val='ones')
    with pytest.raises(yastn.YastnError):
        _ = a.to_raw_tensor()
        # Only tensor with a single block can be converted to raw tensor.


def test_to_nonsymmetric_diag(config_kwargs):
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    a = yastn.rand(config=config_U1, t=(-1, 0, 1), D=(2, 3, 4), isdiag=True)
    an = a.to_nonsymmetric()
    assert an.isdiag == True

    na = a.to_numpy()
    da = np.diag(np.diag(na))
    assert np.allclose(na, da)
    assert pytest.approx(a.trace().item(), rel=tol) == np.trace(da)
    assert pytest.approx(an.trace().item(), rel=tol) == np.trace(da)


def test_to_nonsymmetric_basic(config_kwargs):
    """ test to_nonsymmetric() """
    # dense to dense (trivial)
    config_dense = yastn.make_config(sym='dense', **config_kwargs)
    a = yastn.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    an = a.to_nonsymmetric()
    # for dense, to_nonsymetric() should result in the same config
    assert yastn.norm(a - an) < tol  # == 0.0
    assert an.are_independent(a)
    assert an.is_consistent()

    # U1 to dense
    config_U1 = yastn.make_config(sym='U1', **config_kwargs)
    legs = [yastn.Leg(config_U1, s=-1, t=(-1, 1, 0), D=(1, 2, 3)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yastn.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9)),
            yastn.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))]
    a = yastn.rand(config=config_U1, legs=legs)

    legs[0] = yastn.Leg(config_U1, s=-1, t=(-2, 1, 2), D=(1, 2, 3))
    legs[2] = yastn.Leg(config_U1, s=1, t=(1, 2, 3), D=(8, 9, 10))
    b = yastn.rand(config=config_U1, legs=legs)

    an = a.to_nonsymmetric(legs=dict(enumerate(b.get_legs())))
    bn = b.to_nonsymmetric(legs=dict(enumerate(a.get_legs())))
    assert pytest.approx(yastn.vdot(an, bn).item(), rel=tol) == yastn.vdot(a, b).item()
    with pytest.raises(yastn.YastnError):
        a.vdot(bn)
        # Two tensors have different symmetry rules.
    assert an.are_independent(a)
    assert bn.are_independent(b)
    assert an.is_consistent()
    assert bn.is_consistent()

    with pytest.raises(yastn.YastnError):
        _ = a.to_nonsymmetric(legs={5: legs[0]})
        # Specified leg out of ndim


def _test_dense_v1(tens, shapes, common_shape):
    assert all(a.to_numpy().shape == sh for a, sh in zip(tens, shapes))

    legs = [a.get_legs() for a in tens]
    ndim = tens[0].ndim

    # all dense tensors will have matching shapes
    lss = {ii: yastn.legs_union(*(a_legs[ii] for a_legs in legs)) for ii in range(ndim)}
    ntens = [a.to_numpy(legs=lss) for a in tens]
    assert all(na.shape == common_shape for na in ntens)
    sum_tens = tens[0]
    sum_ntens = ntens[0]
    for n in range(1, len(tens)):
        sum_tens = sum_tens + tens[n]
        sum_ntens = sum_ntens + ntens[n]
    assert np.allclose(sum_tens.to_numpy(), sum_ntens)
    return np.linalg.norm(sum_ntens)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
