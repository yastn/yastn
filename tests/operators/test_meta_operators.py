""" Predefined dense qdit operator """
import yastn
try:
    from .configs import config_dense
except ImportError:
    from configs import config_dense


def test_meta_operators():
    """ Parent class for defining operator classes """
    # pytest switches backends and default_device in config files for testing
    backend = config_dense.backend
    default_device = config_dense.default_device

    # allows initializing config
    ops = yastn.operators.meta_operators(backend=backend, default_device=default_device)

    # desired signature of matrix operators
    assert ops.s == (1, -1)

    # provides a short-cut to set the seed of random number generator in the backend
    ops.random_seed(seed=0)


if __name__ == '__main__':
    test_meta_operators()
