""" yast.linalg.qr() """
import numpy as np
from itertools import product
import yast
try:
    from .configs import config_dense, config_U1, config_Z2xU1, config_Z3
except ImportError:
    from configs import config_dense, config_U1, config_Z2xU1, config_Z3

tol = 1e-10  #pylint: disable=invalid-name


def qr_combine(a):
    """ decompose and contracts tensor using qr decomposition """
    Q, R = yast.linalg.qr(a, axes=((3, 1), (2, 0)))
    QR = yast.tensordot(Q, R, axes=(2, 0))
    QR = QR.transpose(axes=(3, 1, 2, 0))
    assert yast.norm(a - QR) < tol  # == 0.0
    assert Q.is_consistent()
    assert R.is_consistent()
    check_diag_R_nonnegative(R.fuse_legs(axes=(0, (1, 2)), mode='hard'))

    # changes signature of new leg; and position of new leg
    Q2, R2 = yast.qr(a, axes=((3, 1), (2, 0)), sQ=-1, Qaxis=0, Raxis=-1)
    QR2 = yast.tensordot(R2, Q2, axes=(2, 0)).transpose(axes=(1, 3, 0, 2))
    assert yast.norm(a - QR2) < tol  # == 0.0
    assert Q2.is_consistent()
    assert R2.is_consistent()
    check_diag_R_nonnegative(R2.fuse_legs(axes=((0, 1), 2), mode='hard'))


def check_diag_R_nonnegative(R):
    """ checks that diagonal of R is selected to be non-negative """
    for t in R.struct.t:
        assert all(R.config.backend.diag_get(R.real()[t]) >= 0)
        assert all(R.config.backend.diag_get(R.imag()[t]) == 0)


def test_qr_basic():
    """ test qr decomposition for various symmetries """
    # dense
    a = yast.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    qr_combine(a)

    # U1
    a = yast.rand(config=config_U1, s=(-1, -1, 1, 1), n=1,
                  t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                  D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
    qr_combine(a)

    # Z2xU1
    t1 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.ones(config=config_Z2xU1, s=(1, 1, -1, -1),
                  t=[t1, t1, t1, t1],
                  D=[(2, 3, 4, 5), (5, 4, 3, 2), (3, 4, 5, 6), (1, 2, 3, 4)])
    qr_combine(a)


def test_qr_Z3():
    # Z3
    sset = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    nset = (0, 1, 2)
    sQset = (-1, 1)
    for s, n, sQ in product(sset, nset, sQset):
        a = yast.rand(config=config_Z3, s=s, n=n, t=[(0, 1, 2), (0, 1, 2)], D=[(2, 5, 3), (5, 2, 3)], dtype='complex128')
        Q, R = yast.linalg.qr(a, axes=(0, 1), sQ=sQ)
        assert yast.norm(a - Q @ R) < tol  # == 0.0
        assert Q.is_consistent()
        assert R.is_consistent()
        check_diag_R_nonnegative(R)

if __name__ == '__main__':
    test_qr_basic()
    test_qr_Z3()
