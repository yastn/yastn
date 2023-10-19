""" Test tensor operations on an empty tensor """
import yastn
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_empty_tensor():
    """ Test some tensor operations on an empty tensor. """
    a = yastn.Tensor(config=config_U1, s=(1, 1, -1, -1))

    assert a.norm() < tol

    b = a.fuse_legs(axes=((0, 1), (2, 3)), mode='hard')
    c = b.unfuse_legs(axes=(0, 1))
    assert (a - c).norm() < tol

    d = yastn.tensordot(a, a, axes=((2, 3), (0, 1)))
    assert (a - d).norm() < tol

    assert a.item() == 0.


if __name__ == '__main__':
    test_empty_tensor()
