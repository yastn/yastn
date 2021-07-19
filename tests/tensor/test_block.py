""" test yast.block """
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


def test_block_U1():
    w = 0.6
    mu = -0.4

    A = yast.Tensor(config=config_U1, s=(1, 1, -1, -1))
    A.set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 1, 2))
    A.set_block(ts=(0, 1, 1, 0), val=[[1, 0], [mu, 1]], Ds=(2, 1, 1, 2))
    A.set_block(ts=(0, 0, 1, -1), val=[0, w], Ds=(2, 1, 1, 1))
    A.set_block(ts=(0, 1, 0, 1), val=[0, w], Ds=(2, 1, 1, 1))
    A.set_block(ts=(-1, 1, 0, 0), val=[1, 0], Ds=(1, 1, 1, 2))
    A.set_block(ts=(1, 0, 1, 0), val=[1, 0], Ds=(1, 1, 1, 2))

    II = yast.ones(config=config_U1, t=(0, (0, 1), (0, 1), 0), D=(1, (1, 1), (1, 1), 1), s=(1, 1, -1, -1))
    nn = yast.ones(config=config_U1, t=(0, 1, 1, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    c01 = yast.ones(config=config_U1, t=(0, 0, 1, -1), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    cp01 = yast.ones(config=config_U1, t=(0, 1, 0, 1), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    cp10 = yast.ones(config=config_U1, t=(1, 0, 1, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    c10 = yast.ones(config=config_U1, t=(-1, 1, 0, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))

    B1 = yast.block({(0, 0): II, (1, 0): cp10, (2, 0): c10, (3, 0): mu * nn, (3, 1): w * c01, (3, 2): w * cp01, (3, 3): II}, common_legs=(1, 2))
    B2 = yast.block({(1, 1): II, (3, 1): cp10, (5, 1): c10, (7, 1): mu * nn, (7, 3): w * c01, (7, 5): w * cp01, (7, 9): II}, common_legs=(1, 2))

    assert A.norm_diff(B1) < tol  # == 0.0
    assert A.norm_diff(B2) < tol  # == 0.0
    assert B1.are_independent(B2)


if __name__ == "__main__":
    test_block_U1()
