""" yast.diag() """
import pytest
import numpy as np
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_diag_basic():
    a1 = yast.rand(config=config_U1, s=(-1, 1),
                   t=((-1, 1, 2), (-1, 1, 2)),
                   D=((4, 5, 6), (4, 5, 6)))

    a2 = a1.diag()
    a3 = a2.diag()
    a4 = a3.diag()
    a5 = a4.diag()

    assert a1.are_independent(a2)
    assert a2.are_independent(a3)
    assert a3.are_independent(a4)
    assert a4.are_independent(a5)

    na1 = a1.to_numpy()
    na5 = a5.to_numpy()
    assert np.allclose(np.diag(np.diag(na1)), na5)
    assert yast.norm(a2 - a4) < tol  # == 0.0
    assert yast.norm(a3 - a5) < tol  # == 0.0
    assert yast.norm(a1 - a5) > tol  # are not identical


def test_diag_exceptions():
    t1, D1, D2 = (-1, 0, 1), (2, 3, 4), (3, 4, 5)
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(-1, 1, 1), t=(t1, t1, t1), D=(D1, D1, D1))
        _ = a.diag()  # Diagonal tensor requires 2 legs with opposite signatures.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(1, 1), t=(t1, t1), D=(D1, D1))
        _ = a.diag()  # Diagonal tensor requires 2 legs with opposite signatures.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(-1, 1), t=(t1, t1), D=(D1, D1), n=1)
        a.diag()  # Diagonal tensor requires zero tensor charge.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(-1, 1, 1), t=(t1, t1, t1), D=(D1, D1, D1))
        a.fuse_legs(axes=((0, 1), 2), mode='hard', inplace=True)
        a.diag()  # Diagonal tensor cannot have fused legs.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(-1, 1), t=(t1, t1), D=(D1, D1))
        a.fuse_legs(axes=((0, 1),), mode='meta', inplace=True)
        a.diag()  # Diagonal tensor cannot have fused legs.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(-1, 1), t=(t1, t1), D=(D1, D2))
        a.diag()  # yast.diag() allowed only for square blocks.


if __name__ == '__main__':
    test_diag_basic()
    test_diag_exceptions()
