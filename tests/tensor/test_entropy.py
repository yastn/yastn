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

    P = yastn.eye(config=config_U1, legs=yastn.Leg(config_U1, s=1, t=(-2, -1, 0), D=(4, 12, 25)))

    # here P does not sum to 1.
    # This gets normalized during calculation of entropy
    entropy = yastn.entropy(P)
    assert pytest.approx(entropy.item(), rel=tol) == np.log2(41)

    entropy = yastn.entropy(P, alpha=2)
    assert pytest.approx(entropy.item(), rel=tol) == np.log2(41)

    # zero tensor
    assert yastn.entropy(P * 0) == 0

    #  empty tensor
    b = yastn.Tensor(config=config_U1, s=(1, -1), isdiag=True)
    assert yastn.entropy(b) == 0


    with pytest.raises(yastn.YastnError):
        Pnondiag = P.diag()
        entropy = yastn.entropy(Pnondiag)
        # yastn.linalg.entropy requires diagonal tensor.

    with pytest.raises(yastn.YastnError):
        entropy = yastn.entropy(P, alpha=-2)
        # yastn.linalg.entropy requires positive order alpha.


if __name__ == '__main__':
    test_entropy()
