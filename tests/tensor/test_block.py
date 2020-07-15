import settings_U1 as settings
import yamps.tensor as tensor
import pytest


def test_block_U1():

    w = 0.6
    mu = -0.4

    II = tensor.ones(settings=settings, t=(0, (0, 1), (0, 1), 0), D=(1, (1, 1), (1, 1), 1), s=(1, 1, -1, -1))
    nn = tensor.ones(settings=settings, t=(0, 1, 1, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    c01 = tensor.ones(settings=settings, t=(0, 0, 1, -1), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    cp01 = tensor.ones(settings=settings, t=(0, 1, 0, 1), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    cp10 = tensor.ones(settings=settings, t=(1, 0, 1, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    c10 = tensor.ones(settings=settings, t=(-1, 1, 0, 0), D=(1, 1, 1, 1), s=(1, 1, -1, -1))
    C = tensor.block({(0, 0): II, (1, 0): cp10, (2, 0): c10, (3, 0): mu * nn, (3, 1): w * c01, (3, 2): w * cp01, (3, 3): II}, common_legs=(1, 2))
    C.show_properties()

    A = tensor.Tensor(settings=settings, s=(1, 1, -1, -1))
    A.set_block(ts=(0, 0, 0, 0), val=[[1, 0], [0, 1]], Ds=(2, 1, 1, 2))
    A.set_block(ts=(0, 1, 1, 0), val=[[1, 0], [mu, 1]], Ds=(2, 1, 1, 2))
    A.set_block(ts=(0, 0, 1, -1), val=[0, w], Ds=(2, 1, 1, 1))
    A.set_block(ts=(0, 1, 0, 1), val=[0, w], Ds=(2, 1, 1, 1))
    A.set_block(ts=(-1, 1, 0, 0), val=[1, 0], Ds=(1, 1, 1, 2))
    A.set_block(ts=(1, 0, 1, 0), val=[1, 0], Ds=(1, 1, 1, 2))

    assert pytest.approx(A.norm_diff(C)) == 0
