import yamps.yast as yast
import config_U1_R
from math import isclose
import numpy as np

tol = 1e-12

def test_diag_1():
    a1 = yast.rand(config=config_U1_R, s=(-1, 1),
                  t=((-1, 1, 2), (-1, 1, 2)),
                  D=((4, 5, 6), (4, 5, 6)))

    a2 = a1.diag()
    a3 = a2.diag()
    a4 = a3.diag()
    a5 = a4.diag()

    assert a1.is_independent(a2)
    assert a2.is_independent(a3)
    assert a3.is_independent(a4)
    assert a4.is_independent(a5)

    na1 = a1.to_dense()
    na2 = a2.to_dense()
    na3 = a3.to_dense()
    na4 = a4.to_dense()
    na5 = a5.to_dense()
    assert np.allclose(np.diag(np.diag(na1)), na5)
    assert isclose(a2.norm_diff(a4), 0, rel_tol=tol, abs_tol=tol)
    assert isclose(a3.norm_diff(a5), 0, rel_tol=tol, abs_tol=tol)

if __name__ == '__main__':
    test_diag_1()
