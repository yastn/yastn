import numpy as np
import pytest
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_dense_1():
    a = yast.rand(config=config_U1, s=(-1, -1, 1),
                  t=((-1, 1, 2), (-1, 1, 2), (1, 2)),
                  D=((1, 3, 4), (4, 5, 6), (3, 4)))
    b = yast.rand(config=config_U1, s=(-1, -1, 1),
                  t=((-1, 0), (1, 2), (0, 1)),
                  D=((1, 2), (5, 6), (2, 3)))
    c = yast.rand(config=config_U1, s=(-1, -1, 1),
                  t=((0, 1), (0, 1), (1, 2)),
                  D=((2, 3), (5, 5), (3, 4)))

    assert a.to_numpy().shape == (8, 15, 7)
    assert b.to_numpy().shape == (3, 11, 5)
    assert c.to_numpy().shape == (5, 10, 7)

    legs_a = a.get_leg(range(a.ndim))
    legs_b = b.get_leg(range(b.ndim))
    legs_c = c.get_leg(range(c.ndim))
    
    lss = {ii: yast.leg_union(legs_a[ii], legs_b[ii], legs_c[ii]) for ii in range(a.ndim)}
    # all dense tensors will have matching shapes

    na = a.to_numpy(legs=lss)
    nb = b.to_numpy(legs=lss)
    nc = c.to_numpy(legs=lss)
    assert na.shape == (10, 20, 9)
    assert nb.shape == (10, 20, 9)
    assert nc.shape == (10, 20, 9)
    assert np.allclose((a + b + c).to_numpy(), na + nb + nc)


    # mixed inputs
    d = yast.rand(config=config_U1, s=(-1, 1, 1),
                  t=((-1, 0), (-2, -1), (0, 1, 2)),
                  D=((1, 2), (3, 4), (2, 3, 4)))
    assert d.to_numpy().shape == (3, 7, 9)

    ad = yast.tensordot(a, d, axes=((1, 0), (1, 2)))
    lssa = {1: d.get_leg(1).conj(), 0: d.get_leg(2).conj()}
    lssd = {1: a.get_leg(1).conj(), 2: a.get_leg(0).conj()}
    na = a.to_numpy(legs=lssa)
    nd = d.to_numpy(legs=lssd)

    nad = np.tensordot(na, nd, axes=((1, 0), (1, 2)))
    lssad = {0: a.get_leg(2), 1: d.get_leg(0)}
    assert np.allclose(ad.to_numpy(legs=lssad), nad)

    # the same with leg fusion
    a = a.fuse_legs(axes=(2, (1, 0)), mode='meta')
    d = d.fuse_legs(axes=((1, 2), 0), mode='meta')
    fad = yast.tensordot(a, d, axes=(1, 0))
    na = a.to_numpy(legs={1: d.get_leg(0).conj()})   # TODO: support of leg_union for hard fusion
    nd = d.to_numpy(legs={0: a.get_leg(1).conj()})
    nad = np.tensordot(na, nd, axes=(1, 0))
    assert np.allclose(ad.to_numpy(legs=lssad), nad)
    assert np.allclose(fad.to_numpy(legs=lssad), nad)

def test_dense_diag():
    a = yast.rand(config=config_U1, t=(-1, 0, 1), D=(2, 3, 4), isdiag=True)
    b = a.to_nonsymmetric()
    na = a.to_numpy()
    da = np.diag(np.diag(na))
    assert np.allclose(na, da)
    assert pytest.approx(a.trace().item(), rel=tol) == np.trace(da)


if __name__ == '__main__':
    test_dense_1()
    test_dense_diag()
