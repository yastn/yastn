import pytest
import yast
from yast.backend import backend_np
try:
    from .configs import config_U1, config_Z2, config_Z2_fermionic
except ImportError:
    from configs import config_U1, config_Z2, config_Z2_fermionic


def test_config_exceptions():
    """ handling mismatches of tensor configs when combining two tensors"""
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
        b = yast.rand(config=config_Z2, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
        _ = a + b  # Two tensors have different symmetry rules.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_Z2, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
        b = yast.rand(config=config_Z2_fermionic, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
        _ = a + b  # Two tensors have different assigment of fermionic statistics.


@pytest.mark.skipif(config_U1.backend.BACKEND_ID=="numpy", reason="requires different backends or devices")
def test_config_exceptions_2():
    """ mismatches requiring different backends or devices"""
    a = yast.rand(config=config_U1, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
    with pytest.raises(yast.YastError):
        wrong_config = a.config._replace(backend=backend_np)
        b = yast.rand(config=wrong_config, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
        _ = a + b
    if config_U1.backend.torch.cuda.is_available():
        with pytest.raises(yast.YastError):
            a = a.to(device='cpu')
            b = yast.rand(config=config_U1, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
            b = b.to(device='cuda')
            _ = a + b


if __name__ == '__main__':
    test_config_exceptions()
