try:
    import yast
except ModuleNotFoundError:
    import fix_path
    import yast
import config_dense_R
import config_U1_R
from math import isclose

tol = 1e-12


def run_to_number(a, b):
    ax = tuple(range(a.get_ndim()))  # here a.get_ndim() == b.get_ndim()
    t0 = yast.tensordot(a, b, axes=(ax, ax), conj=(1, 0))  # 0-dim tensor with 1 element, i.e., a number

    nb0 = t0.to_number()  # this is backend-type number
    it0 = t0.item()  # this is python float (or int)

    tDa = {ii: a.get_leg_structure(ii) for ii in range(a.get_ndim())}  # info on charges and dimensions on all legs
    tDb = {ii: b.get_leg_structure(ii) for ii in range(b.get_ndim())}
    na = a.to_numpy(tDb)  # use tDb to fill in missing zero blocks to make sure that na and nb match
    nb = b.to_numpy(tDa)
    ns = na.conj().reshape(-1) @ nb.reshape(-1)  # this is numpy scalar

    assert isclose(it0, ns, rel_tol=tol)
    assert isclose(it0, it0, rel_tol=tol)
    assert isinstance(it0, (float, int))  # in the examples it is real
    assert type(it0) is not type(nb0)


def test_to_number_0():
    a = yast.rand(config=config_dense_R, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))
    b = yast.rand(config=config_dense_R, s=(-1, 1, 1, -1), D=(2, 3, 4, 5))

    run_to_number(a, b)


def test_to_number_1():
    a = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                  t=((-1, 1, 0), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    b = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                  t=((-2, 1, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    c = yast.rand(config=config_U1_R, s=(-1, 1, 1, -1),
                  t=((-2, 2), (-1, 1, 2), (-1, 1, 2), (-1, 1, 2)),
                  D=((1, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
    run_to_number(a, b)
    run_to_number(a, c)
    run_to_number(b, c)


if __name__ == '__main__':
    test_to_number_0()
    test_to_number_1()
