"""
to_nonsymmetric()  to_dense()  to_numpy()
"""
import numpy as np
import pytest
import yast
try:
    from .configs import config_dense, config_U1
except ImportError:
    from configs import config_dense, config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_dense_basic():
    """ a.to_numpy() is equivalent to np.array(a.to_dense())"""
    norms = []
    tens = [yast.rand(config=config_U1, s=(-1, -1, 1),
                      t=((-1, 1, 2), (-1, 1, 2), (1, 2)),
                      D=((1, 3, 4), (4, 5, 6), (3, 4))),
            yast.rand(config=config_U1, s=(-1, -1, 1),
                      t=((-1, 0), (1, 2), (0, 1)),
                      D=((1, 2), (5, 6), (2, 3))),
            yast.rand(config=config_U1, s=(-1, -1, 1),
                      t=((0, 1), (0, 1), (1, 2)),
                      D=((2, 3), (5, 5), (3, 4)))]
    shapes = [(8, 15, 7), (3, 11, 5), (5, 10, 7)]
    common_shape = (10, 20, 9)
    norms.append(_tens_dense_v1(tens, shapes, common_shape))

    # with meta-fusion
    mtens = [a.fuse_legs(axes=((0, 1), 2), mode='meta') for a in tens]
    mtens = [ma.fuse_legs(axes=[(0, 1)], mode='meta') for ma in mtens]
    mshapes = [(126,), (58,), (135,)]
    mcommon_shape = (211,)
    norms.append(_tens_dense_v1(mtens, mshapes, mcommon_shape))

    htens = [a.fuse_legs(axes=((0, 1), 2), mode='hard') for a in tens]
    htens = [ha.fuse_legs(axes=[(0, 1)], mode='hard') for ha in htens]
    hshapes = [(126,), (58,), (135,)]
    hcommon_shape = (383,)
    norms.append(_tens_dense_v1(htens, hshapes, hcommon_shape))

    assert all(pytest.approx(n, rel=tol) == norms[0] for n in norms)

    # provided individual legs
    d = yast.rand(config=config_U1, s=(-1, 1, 1),
                  t=((-1, 0), (-2, -1), (0, 1, 2)),
                  D=((1, 2), (3, 4), (2, 3, 4)))
    a = tens[0]
    lsa = {1: d.get_legs(1).conj(), 0: d.get_legs(2).conj()}
    lsd = {1: a.get_legs(1).conj(), 2: a.get_legs(0).conj()}
    na = a.to_numpy(legs=lsa)
    nd = d.to_numpy(legs=lsd)
    nad = np.tensordot(na, nd, axes=((1, 0), (1, 2)))
    
    ad = yast.tensordot(a, d, axes=((1, 0), (1, 2)))
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


def test_dense_diag():
    a = yast.rand(config=config_U1, t=(-1, 0, 1), D=(2, 3, 4), isdiag=True)
    an = a.to_nonsymmetric()
    assert an.isdiag == True

    na = a.to_numpy()
    da = np.diag(np.diag(na))
    assert np.allclose(na, da)
    assert pytest.approx(a.trace().item(), rel=tol) == np.trace(da)
    assert pytest.approx(an.trace().item(), rel=tol) == np.trace(da)


def test_to_nonsymmetric_basic():
    """ test to_nonsymmetric() """
    # dense to dense (trivial)
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    an = a.to_nonsymmetric()
    # for dense, to_nonsymetric() should result in the same config
    assert yast.norm(a - an) < tol  # == 0.0
    assert an.are_independent(a)
    assert an.is_consistent()

    # U1 to dense
    legs = [yast.Leg(config_U1, s=-1, t=(-1, 1, 0), D=(1, 2, 3)),
            yast.Leg(config_U1, s=1, t=(-1, 1, 2), D=(4, 5, 6)),
            yast.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9)),
            yast.Leg(config_U1, s=-1, t=(-1, 1, 2), D=(10, 11, 12))]
    a = yast.rand(config=config_U1, legs=legs)

    legs[0] = yast.Leg(config_U1, s=-1, t=(-2, 1, 2), D=(1, 2, 3))
    legs[2] = yast.Leg(config_U1, s=1, t=(1, 2, 3), D=(8, 9, 10))
    b = yast.rand(config=config_U1, legs=legs)

    an = a.to_nonsymmetric(legs=dict(enumerate(b.get_legs())))
    bn = b.to_nonsymmetric(legs=dict(enumerate(a.get_legs())))
    assert pytest.approx(yast.vdot(an, bn).item(), rel=tol) == yast.vdot(a, b).item()
    with pytest.raises(yast.YastError):
        a.vdot(bn)
        # Two tensors have different symmetry rules.
    assert an.are_independent(a)
    assert bn.are_independent(b)
    assert an.is_consistent()
    assert bn.is_consistent()

    with pytest.raises(yast.YastError):
        _ = a.to_nonsymmetric(legs={5: legs[0]})
        # Specified leg out of ndim


def _tens_dense_v1(tens, shapes, common_shape):
    assert all(a.to_numpy().shape == sh for a, sh in zip(tens, shapes))

    legs = [a.get_legs() for a in tens]
    ndim = tens[0].ndim

    # all dense tensors will have matching shapes
    lss = {ii: yast.leg_union(*(a_legs[ii] for a_legs in legs)) for ii in range(ndim)}
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
    test_dense_basic()
    test_dense_diag()
    test_to_nonsymmetric_basic()
