from yamps.tensor import Tensor
import settings_full_torch as settings_full
import settings_U1_torch as settings_U1
import settings_Z2_U1_torch as settings_Z2_U1
import settings_U1_U1_torch as settings_U1_U1
import numpy as np
from math import isclose

rel_tol=1.0e-14

def test_block1():
    a = Tensor(settings=settings_U1, s=(-1, 1, 1))
    a.set_block(ts=(1, -1, 2), Ds=(1, 1, 1), val='randR')
    a.set_block(ts=(2, 0, 2), Ds=(1, 1, 1), val='randR')
    a.show_properties()

    b = Tensor(settings=settings_U1, s=(-1, 1, 1))
    b.set_block(ts=(1, 0, 1), Ds=(1, 1, 1), val='randR')
    b.set_block(ts=(2, 0, 2), Ds=(1, 1, 1), val='randR')

    c1 = a.dot(a, axes=((0, 1, 2), (0, 1, 2)), conj=(0, 1))
    c2 = b.dot(b, axes=((1, 2), (1, 2)), conj=(0, 1))
    c3 = a.dot(b, axes=(0, 2), conj=(1, 1))

    a1 = a.to_numpy()
    b1 = b.to_numpy()
    cc1 = np.tensordot(a1, a1.conj(), axes=((0, 1, 2), (0, 1, 2)))
    cc2 = np.tensordot(b1, b1.conj(), axes=((1, 2), (1, 2)))
    cc3 = np.tensordot(a1.conj(), b1.conj(), axes=(0, 2))

    assert isclose(c1.norm(), np.linalg.norm(cc1), rel_tol=rel_tol)
    assert isclose(c2.norm(), np.linalg.norm(cc2), rel_tol=rel_tol)
    assert isclose(c3.norm(), np.linalg.norm(cc3), rel_tol=rel_tol)


if __name__ == '__main__':
    test_block1()
