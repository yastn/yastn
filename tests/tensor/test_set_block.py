from yamps.tensor import Tensor
import settings_U1_R
import pytest
import numpy as np


def test_block1():
    a = Tensor(settings=settings_U1_R, s=(-1, 1, 1))
    a.set_block(ts=(1, -1, 2), Ds=(1, 1, 1), val='randR')
    a.set_block(ts=(2, 0, 2), Ds=(1, 1, 1), val='randR')
    a.show_properties()

    b = Tensor(settings=settings_U1_R, s=(-1, 1, 1))
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

    assert 0 == pytest.approx(c1.norm() - np.linalg.norm(cc1))
    assert 0 == pytest.approx(c2.norm() - np.linalg.norm(cc2))
    assert 0 == pytest.approx(c3.norm() - np.linalg.norm(cc3))


if __name__ == '__main__':
    test_block1()
