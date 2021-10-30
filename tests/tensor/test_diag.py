import numpy as np
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_diag_1():
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
    assert a2.norm_diff(a4) < tol  # == 0.0
    assert a3.norm_diff(a5) < tol  # == 0.0
    assert a1.norm_diff(a5) > tol  # are not identical
    


if __name__ == '__main__':
    test_diag_1()
