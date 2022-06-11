import numpy as np
import pytest
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


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


def _test_dense_v2(a, d):
    """ a.s == (-1, -1, 1), d.s == (-1, 1, 1) """

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


def test_dense_basic():
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

    # mixed inputs
    d = yast.rand(config=config_U1, s=(-1, 1, 1),
                  t=((-1, 0), (-2, -1), (0, 1, 2)),
                  D=((1, 2), (3, 4), (2, 3, 4)))
    _test_dense_v2(tens[0], d)


def test_dense_diag():
    a = yast.rand(config=config_U1, t=(-1, 0, 1), D=(2, 3, 4), isdiag=True)
    b = a.to_nonsymmetric()
    na = a.to_numpy()
    da = np.diag(np.diag(na))
    assert np.allclose(na, da)
    assert pytest.approx(a.trace().item(), rel=tol) == np.trace(da)


if __name__ == '__main__':
    test_dense_basic()
    test_dense_diag()
