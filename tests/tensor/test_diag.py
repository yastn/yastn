import numpy as np
import yast
from .configs import config_U1

tol = 1e-12


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

    na1 = a1.to_dense()
    na2 = a2.to_dense()
    na3 = a3.to_dense()
    na4 = a4.to_dense()
    na5 = a5.to_dense()
    assert np.allclose(np.diag(np.diag(na1)), na5)
    assert a2.norm_diff(a4) < tol  # == 0.0
    assert a3.norm_diff(a5) < tol  # == 0.0


if __name__ == '__main__':
    test_diag_1()
