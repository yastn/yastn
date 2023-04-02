""" yastn.linalg.entropy """
import numpy as np
import pytest
import yastn
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_entropy():
    legs = [yastn.Leg(config_U1, s=1, t=(0, 1), D=(5, 6)),
            yastn.Leg(config_U1, s=1, t=(-1, 0), D=(5, 6)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4)),
            yastn.Leg(config_U1, s=-1, t=(-1, 0, 1), D=(2, 3, 4))]
    a = yastn.rand(config=config_U1, n=1, legs=legs)

    U, S, V = yastn.linalg.svd(a, axes=((0, 1), (2, 3)), sU=-1)
    S2 = yastn.eye(config=config_U1, legs=yastn.Leg(config_U1, s=1, t=(-2, -1, 0), D=(4, 12, 25)))
    a = U @ S2 @ V

    entropy, Smin, normalization = yastn.entropy(a, axes=((0, 1), (2, 3)))
    assert pytest.approx(entropy.item(), rel=tol) == np.log2(41)
    assert pytest.approx(Smin.item(), rel=tol) == 1
    assert pytest.approx(normalization.item(), rel=tol) == np.sqrt(41)

    entropy2, Smin, normalization = yastn.entropy(a, axes=((0, 1), (2, 3)), alpha=2)
    assert pytest.approx(entropy2.item(), rel=tol) == np.log2(41)
    assert pytest.approx(Smin.item(), rel=tol) == 1
    assert pytest.approx(normalization.item(), rel=tol) == np.sqrt(41)

    entropy, Smin, normalization = yastn.entropy(a * 0, axes=((0, 1), (2, 3))) # zero tensor
    assert (entropy, Smin, normalization) == (0, 0, 0)
    b = yastn.Tensor(config=config_U1, s=(1, 1, -1, -1))  #  empty tensor
    entropy, Smin, normalization = yastn.entropy(b, axes=((0, 1), (2, 3)))
    assert (entropy, Smin, normalization) == (0, 0, 0)


if __name__ == '__main__':
    test_entropy()
