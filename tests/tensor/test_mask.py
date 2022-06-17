""" Test yaps.mask """
import pytest
import yast
try:
    from .configs import config_U1, config_Z2xU1
except ImportError:
    from configs import config_U1, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def test_mask_basic():
    """ series of tests for apply_mask """
    config_U1.backend.random_seed(seed=0)  # fix for tests

    # start with U1
    leg1 =  yast.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9))
    leg2 =  yast.Leg(config_U1, s=1, t=(-1, 1, 2), D=(5, 6, 7))
    a = yast.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1, leg2.conj()])
 
    legd =  yast.Leg(config_U1, s=1, t=(-1, 1), D=(7, 8))
    b0 = yast.rand(config=config_U1, isdiag=True, legs=legd)

    b = b0.copy()  # create a mask by hand
    b[(-1, -1)] = b[(-1, -1)] < -2  # all false
    b[(1, 1)] = b[(1, 1)] > 0  # some true
    tr_b = b.trace(axes=(0, 1)).item()

    c = b.apply_mask(a, axis=2)

    # application of the mask should leave a single charge (1,) on this leg
    l = c.get_legs(axis=2)
    assert l.t == ((1,),) and l[(1,)] == tr_b  # in second checks the bond dimension

    d0 = b.apply_mask(b0, axis=0)
    d1 = b.apply_mask(b0, axis=-1)
    assert yast.norm(d0 - d1) < tol
    l = d1.get_legs(axis=1)
    assert l.t == ((1,),) and l[(1,)] == tr_b

    # apply the same mask on 2 tensors
    d2, c2 = b.apply_mask(b0, a, axis=(-1, 2))
    assert (d2 - d0).norm() < tol
    assert (c2 - c).norm() < tol

    # here using Z2xU1 symmetry
    legs = [yast.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2), (1, 0), (1, 2)), D=(6, 3, 9, 6)),
            yast.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, 2)), D=(3, 2)),
            yast.Leg(config_Z2xU1, s=1, t=((0, 1), (1, 0), (0, 0), (1, 1)), D=(4, 5, 6, 3)),
            yast.Leg(config_Z2xU1, s=1, t=((0, 0), (0, 2)), D=(2, 3))]
    a = yast.rand(config=config_Z2xU1, legs=legs)

    # diagonal tensor exactly matching first leg of a
    b = yast.rand(config=config_Z2xU1, isdiag=True, legs=legs[0])

    b[(0, 0, 0, 0)] *= 0  # put block (0, 0, 0, 0) in diagonal b to 0.

    bgt = b > 0
    blt = b < 0
    bge = b >= 0
    ble = b <= 0
    assert bgt.trace().item() + ble.trace().item() == blt.trace().item() + bge.trace().item() == b.get_shape(axis=0)

    for bb in [bgt, blt, bge, ble]:
        bnd_dim = bb.trace(axes=(0, 1)).item()
        c = bb.apply_mask(a, axis=0)
        l = c.get_legs(axis=0)
        assert sum(l.D) == bnd_dim

    assert blt.apply_mask(bge, axis=0).trace() < tol  # == 0.
    assert ble.apply_mask(bgt, axis=1).trace() < tol  # == 0.


def test_mask_exceptions():
    """ trigger exceptions for apply_mask """
    legd =  yast.Leg(config_U1, s=1, t=(-1, 1), D=(8, 8))
    a = yast.rand(config=config_U1, isdiag=True, legs=legd)
    a_nondiag = a.diag()

    leg1 =  yast.Leg(config_U1, s=1, t=(-1, 1, 2), D=(7, 8, 9))
    leg2 =  yast.Leg(config_U1, s=1, t=(-1, 1, 2), D=(5, 6, 7))
    b = yast.rand(config=config_U1, legs=[leg1.conj(), leg2, leg1, leg2.conj()])

    with pytest.raises(yast.YastError):
        _ = a_nondiag.apply_mask(b, axis=2)
        # First tensor should be diagonal.
    with pytest.raises(yast.YastError):
        bmf = b.fuse_legs(axes=(0, (1, 2), 3), mode='meta')
        _ = a.apply_mask(bmf, axis=1)
        # Second tensor`s leg specified by axis cannot be fused.
    with pytest.raises(yast.YastError):
        bhf = b.fuse_legs(axes=(0, (1, 2), 3), mode='hard')
        _ = a.apply_mask(bhf, axis=1)
        # Second tensor`s leg specified by axis cannot be fused.
    with pytest.raises(yast.YastError):
        _ = a.apply_mask(b, axis=1)  # Bond dimensions do not match.
    with pytest.raises(yast.YastError):
        _, _ = a.apply_mask(b, b, axis=[2, 2, 1])
        # There should be exactly one axis for each tensor to be projected.


if __name__ == '__main__':
    test_mask_basic()
    test_mask_exceptions()
