import yamps.tensor.yast as yast
import config_U1_R
import pytest


def test_block_U1():

    w = 0.6
    mu = -0.4

    A = yast.Tensor(config=config_U1_R, s=(1, 1, -1, -1))
    A.set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 1, 2))
    A.set_block(ts=(0, 1, 1, 0), val=[[1, 0], [mu, 1]], Ds=(2, 1, 1, 2))
    A.set_block(ts=(0, 0, 1, -1), val=[0, w], Ds=(2, 1, 1, 1))
    A.set_block(ts=(0, 1, 0, 1), val=[0, w], Ds=(2, 1, 1, 1))
    A.set_block(ts=(-1, 1, 0, 0), val=[1, 0], Ds=(1, 1, 1, 2))
    A.set_block(ts=(1, 0, 1, 0), val=[1, 0], Ds=(1, 1, 1, 2))

    II = yast.ones(config=config_U1_R, t=(0, (0, 1), (0, 1), 0), D=(1, (1, 1), (1, 1), 1), s=(1, 1, -1, -1))
    nn = yast.ones(config=config_U1_R, t=(0, 1, 1, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    c01 = yast.ones(config=config_U1_R, t=(0, 0, 1, -1), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    cp01 = yast.ones(config=config_U1_R, t=(0, 1, 0, 1), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    cp10 = yast.ones(config=config_U1_R, t=(1, 0, 1, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    c10 = yast.ones(config=config_U1_R, t=(-1, 1, 0, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))

    B = yast.block({(0, 0): II, (1, 0): cp10, (2, 0): c10, (3, 0): mu * nn, (3, 1): w * c01, (3, 2): w * cp01, (3, 3): II}, common_legs=(1, 2))
    # B.show_properties()

    C = yast.block({(1, 1): II, (3, 1): cp10, (5, 1): c10, (7, 1): mu * nn, (7, 3): w * c01, (7, 5): w * cp01, (7, 9): II}, common_legs=(1, 2))
    # C.show_properties()

    assert pytest.approx(A.norm_diff(B)) == 0
    assert pytest.approx(A.norm_diff(C)) == 0


if __name__ == "__main__":
    test_block_U1()
