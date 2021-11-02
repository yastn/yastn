""" yast.linalg.entropy """
import numpy as np
import pytest
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_entropy():
    a = yast.rand(config=config_U1, s=(1, 1, -1, -1), n=1,
                  t=[(0, 1), (-1, 0), (-1, 0, 1), (-1, 0, 1)],
                  D=[(5, 6), (5, 6), (2, 3, 4), (2, 3, 4)])
    U, S, V = yast.linalg.svd(a, axes=((0, 1), (2, 3)), sU=-1)
    S.set_block(ts=-2, Ds=4, val='ones')
    S.set_block(ts=-1, Ds=12, val='ones')
    S.set_block(ts=0, Ds=25, val='ones')
    US = yast.tensordot(U, S, axes=(2, 0))
    a = yast.tensordot(US, V, axes=(2, 0))

    entropy, Smin, normalization = yast.entropy(a, axes=((0, 1), (2, 3)))
    assert pytest.approx(entropy.item(), rel=tol) == np.log2(41)
    assert pytest.approx(Smin.item(), rel=tol) == 1
    assert pytest.approx(normalization.item(), rel=tol) == np.sqrt(41)

    entropy2, Smin, normalization = yast.entropy(a, axes=((0, 1), (2, 3)), alpha=2)
    assert pytest.approx(entropy2.item(), rel=tol) == np.log2(41)
    assert pytest.approx(Smin.item(), rel=tol) == 1
    assert pytest.approx(normalization.item(), rel=tol) == np.sqrt(41)

    _, _, _, uS = yast.svd(a, axes=((0, 1), (2, 3)), sU=-1, untruncated_S=True)
    assert pytest.approx(sum(uS[(-2,)]).item(), rel=tol) == 4.
    assert pytest.approx(sum(uS[(-1,)]).item(), rel=tol) == 12.
    assert pytest.approx(sum(uS[(0,)]).item(), rel=tol) == 25.


if __name__ == '__main__':
    test_entropy()
