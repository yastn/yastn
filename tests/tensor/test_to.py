import pytest
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import  config_U1

tol = 1e-12  #pylint: disable=invalid-name


@pytest.mark.skipif(config_U1.backend.BACKEND_ID=="numpy", reason="numpy works on single device and does not have problems with promoting types")
def test_to_1():
    ta = yast.rand(config=config_U1, s=(1, 1, -1),
                    t=((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)),
                    D=((1, 2, 3), (1, 2, 3), (1, 2, 3)),
                    dtype='float64', device='cpu')

    tb = ta.to(dtype='complex128')
    assert tb.config.dtype == 'complex128'
    assert tb[(0, 0, 0)].dtype == config_U1.backend.DTYPE['complex128']
    assert tb[(0, 0, 0)].device.type == 'cpu'
    assert tb.is_consistent()

    if config_U1.backend.torch.cuda.is_available():
        tc = ta.to(device='cuda')
        assert tc.config.device == 'cuda'
        assert tc.config.dtype == 'float64'
        assert tc[(0, 0, 0)].dtype == config_U1.backend.DTYPE['float64']
        assert tc[(0, 0, 0)].device.type == 'cuda'
        assert tc.is_consistent()

        td = ta.to(device='cuda', dtype='complex128')
        assert td.config.device == 'cuda'
        assert td.config.dtype == 'complex128'
        assert td[(0, 0, 0)].dtype == config_U1.backend.DTYPE['complex128']
        assert td[(0, 0, 0)].device.type == 'cuda'
        assert td.is_consistent()


if __name__ == '__main__':
    test_to_1()
