""" Predefined dense qdit operator """
import numpy as np
import yastn
try:
    from .configs import config_dense
except ImportError:
    from configs import config_dense


tol = 1e-12  #pylint: disable=invalid-name


def test_qdit(d=5):
    """ Standard operators and some vectors in two-dimensional Hilbert space for various symmetries. """
    # pytest switches backends and default_device in config files for testing
    backend = config_dense.backend
    default_device = config_dense.default_device

    ops_dense = yastn.operators.Qdit(d=d, backend=backend, default_device=default_device)

    I = ops_dense.I()
    leg = ops_dense.space()

    assert leg == I.get_legs(axes=0)
    assert np.allclose(I.to_numpy(), np.eye(d))
    assert I.device[:len(default_device)] == default_device  # for cuda, accept cuda:0 == cudav

    # used in mps Generator
    d = ops_dense.to_dict()
    (d["I"](3) - I).norm() < tol  # here 3 is a posible position in the mps
    assert all(k in d for k in ('I',))


if __name__ == '__main__':
    test_qdit(d=5)
