import numpy as np
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

    # all dense tensors will have matching shapes
    lss = yast.leg_structures_for_dense(tensors=[a, b, c])
    # the same as yast.leg_structures_for_dense(tensors=[a, {0: 0, 1: 1, 2: 2}, b, {0: 0, 1: 1, 2: 2}, c, {0: 0, 1: 1, 2: 2}])
    assert lss == yast.leg_structures_for_dense(tensors=[a, {0: 0, 1: 1, 2: 2}, b, c, {0: 0, 1: 1, 2: 2}])
    
    na = a.to_numpy(leg_structures=lss)
    nb = b.to_numpy(leg_structures=lss)
    nc = c.to_numpy(leg_structures=lss)
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
    lssa = yast.leg_structures_for_dense(tensors=[d, {1: 1, 2: 0}])
    lssd = yast.leg_structures_for_dense(tensors=[a, {1: 1, 0: 2}])
    na = a.to_numpy(leg_structures=lssa)
    nd = d.to_numpy(leg_structures=lssd)
    nad = np.tensordot(na, nd, axes=((1, 0), (1, 2)))
    lssad = yast.leg_structures_for_dense(tensors=[a, {2: 0}, d, {0: 1}])
    assert np.allclose(ad.to_numpy(leg_structures=lssad), nad)

    # the same with leg fusion
    a = a.fuse_legs(axes=(2, (1, 0)))
    d = d.fuse_legs(axes=((1, 2), 0))
    lssa = yast.leg_structures_for_dense(tensors=[d, {0: 1}])
    lssd = yast.leg_structures_for_dense(tensors=[a, {1: 0}])
    na = a.to_numpy(leg_structures=lssa)
    nd = d.to_numpy(leg_structures=lssd)
    nad = np.tensordot(na, nd, axes=(1, 0))
    assert np.allclose(ad.to_numpy(leg_structures=lssad), nad)


if __name__ == '__main__':
    test_dense_1()

