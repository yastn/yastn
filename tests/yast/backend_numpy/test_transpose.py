import yamps.tensor.yast as yast
import config_dense_R
import config_U1_R
import config_Z2_U1_R


def test_transpose_0():
    a = yast.ones(config=config_dense_R, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    assert a.get_total_shape() == (2, 3, 4, 5)
    b = a.transpose(axes=(1, 3, 2, 0)).to_numpy()
    assert b.shape == (3, 5, 4, 2)
    c = a.moveaxis(source=1, destination=-1)
    assert c.get_total_shape() == (2, 4, 5, 3)


def test_transpose_1():
    a = yast.ones(config=config_U1_R, s=(-1, -1, -1, 1, 1, 1),
                    t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                    D=[(2, 3), (4, 5), (6, 7), (6, 5), (4, 3), (2, 1)])
    assert a.get_total_shape() == (5, 9, 13, 11, 7, 3)
    b = a.transpose(axes=(1, 2, 3, 0, 5, 4))
    bs = b.get_total_shape()
    assert b.to_numpy().shape == (9, 13, 11, 5, 3, 7)
    assert bs == (9, 13, 11, 5, 3, 7)
    c = a.moveaxis(source=1, destination=4)
    cs = c.get_total_shape()
    assert c.to_numpy().shape == (5, 13, 11, 7, 9, 3)
    assert cs == (5, 13, 11, 7, 9, 3)


def test_transpose_2():
    t1 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.ones(config=config_Z2_U1_R, s=(-1, -1, 1, 1),
                    t=[t1, t1, t1, t1],
                    D=[(7, 3, 4, 5), (5, 4, 3, 2), (3, 4, 5, 6), (1, 2, 3, 4)])
    assert a.get_total_shape() == (19, 14, 18, 10)
    b = a.transpose(axes=(1, 2, 3, 0)).to_numpy()
    assert b.shape == (14, 18, 10, 19)
    a.moveaxis(source=-1, destination=-3, inplace=True)
    assert a.get_total_shape() == (19, 10, 14, 18)

if __name__ == '__main__':
    test_transpose_0()
    test_transpose_1()
    test_transpose_2()
