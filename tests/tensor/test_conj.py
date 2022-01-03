""" yast.conj() """
import numpy as np
import yast
try:
    from .configs import config_Z2, config_Z2xU1
except ImportError:
    from configs import config_Z2, config_Z2xU1

tol = 1e-12  #pylint: disable=invalid-name


def test_conj_1():
    a = yast.rand(config=config_Z2, s=(1, 1, 1, -1, -1, -1), n=1,
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    b = a.conj()
    assert a.struct.n == b.struct.n
    na = a.to_numpy()
    nb = b.to_numpy()
    assert np.linalg.norm(na - nb) < tol


def test_conj_2():
    a = yast.rand(config=config_Z2xU1, s=(1, -1), n=(1, 1),
                  t=[[(0, 0), (1, 1), (0, 2)], [(0, 1), (0, 0), (1, 1)]],
                  D=[[1, 2, 3], [4, 5, 6]])
    b = a.conj()
    assert b.get_tensor_charge() == (1, -1)
    assert b.get_signature() == (-1, 1)
    c = yast.tensordot(a, b, axes=((0, 1), (0, 1)))
    assert c.get_tensor_charge() == (0, 0)
    assert c.get_signature() == ()


if __name__ == '__main__':
    test_conj_1()
    test_conj_2()
