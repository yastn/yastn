import pytest
import yast
from yast.backend import backend_np
try:
    from .configs import config_U1, config_Z2, config_Z2_fermionic
except ImportError:
    from configs import config_U1, config_Z2, config_Z2_fermionic


def test_config_exceptions():
    """ handling mismatches of tensor configs when combining two tensors"""
    leg_U1 = yast.Leg(config_U1, s=1, t=(0, 1), D=(1, 2))
    leg_Z2 = yast.Leg(config_Z2, s=1, t=(0, 1), D=(1, 2))
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_U1, legs=[leg_U1, leg_U1, leg_U1.conj()])
        b = yast.rand(config=config_Z2, legs=[leg_Z2, leg_Z2, leg_Z2.conj()])
        _ = a + b
        # Two tensors have different symmetry rules.
    with pytest.raises(yast.YastError):
        a = yast.rand(config=config_Z2, legs=[leg_Z2, leg_Z2, leg_Z2.conj()])
        # leg do not depend on fermionic statistics so the next line is fine
        b = yast.rand(config=config_Z2_fermionic, legs=[leg_Z2, leg_Z2, leg_Z2.conj()])
        _ = a + b
        # Two tensors have different assigment of fermionic statistics.


@pytest.mark.skipif(config_U1.backend.BACKEND_ID=="numpy", reason="requires different backends or devices")
def test_config_exceptions_2():
    """ mismatches requiring different backends or devices"""
    a = yast.rand(config=config_U1, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
    with pytest.raises(yast.YastError):
        wrong_config = a.config._replace(backend=backend_np)
        b = yast.rand(config=wrong_config, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
        _ = a + b
        # Two tensors have different backends.
    if config_U1.backend.torch.cuda.is_available():
        with pytest.raises(yast.YastError):
            a = a.to(device='cpu')
            b = yast.rand(config=config_U1, s=(1, -1, 1), t=((0, 1), (0, 1), (0, 1)), D=((1, 2), (1, 2), (1, 2)))
            b = b.to(device='cuda')
            _ = a + b
            # Devices of the two tensors do not match.


if __name__ == '__main__':
    test_config_exceptions()
    test_config_exceptions_2()
