
from .context import yast
from .context import config_dense, config_U1, config_Z2_U1

tol = 1e-10


def svd_combine(a):
    U, S, V = yast.linalg.svd(a, axes=((3, 1), (2, 0)), sU=-1)
    US = yast.tensordot(U, S, axes=(2, 0))
    USV = yast.tensordot(US, V, axes=(2, 0))
    USV = USV.transpose(axes=(3, 1, 2, 0))
    assert a.norm_diff(USV) < tol  # == 0.0
    assert a.is_consistent()
    assert U.is_consistent()
    assert S.is_consistent()
    assert V.is_consistent()


def svd_order_combine(a):
    U, S, V = yast.linalg.svd(a, axes=((3, 1), (2, 0)), sU=1, Uaxis=0, Vaxis=-1)
    US = yast.tensordot(S, U, axes=(0, 0))
    USV = yast.tensordot(US, V, axes=(0, 2))
    USV = USV.transpose(axes=(3, 1, 2, 0))
    assert a.norm_diff(USV) < tol  # == 0.0
    assert U.is_consistent()
    assert S.is_consistent()
    assert V.is_consistent()


def qr_combine(a):
    Q, R = yast.linalg.qr(a, axes=((3, 1), (2, 0)))
    QR = yast.tensordot(Q, R, axes=(2, 0))
    QR = QR.transpose(axes=(3, 1, 2, 0))
    assert a.norm_diff(QR) < tol  # == 0.0
    assert Q.is_consistent()
    assert R.is_consistent()


def qr_order_combine(a):
    Q, R = yast.linalg.qr(a, axes=((3, 1), (2, 0)), sQ=-1, Qaxis=-2, Raxis=1)
    QR = yast.tensordot(Q, R, axes=(1, 1))
    QR = QR.transpose(axes=(3, 1, 2, 0))
    assert a.norm_diff(QR) < tol  # == 0.0
    assert Q.is_consistent()
    assert R.is_consistent()


def eigh_combine(a):
    a2 = yast.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))
    S, U = yast.linalg.eigh(a2, axes=((0, 1), (2, 3)))
    US = yast.tensordot(U, S, axes=(2, 0))
    USU = yast.tensordot(US, U, axes=(2, 2), conj=(0, 1))
    assert a2.norm_diff(USU) < tol  # == 0.0
    assert U.is_consistent()
    assert S.is_consistent()


def eigh_order_combine(a):
    a2 = yast.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))
    S, U = yast.linalg.eigh(a2, axes=((0, 1), (2, 3)), Uaxis=0, sU=-1)
    US = yast.tensordot(S, U, axes=(0, 0))
    USU = yast.tensordot(US, U, axes=(0, 0), conj=(0, 1))
    assert a2.norm_diff(USU) < tol  # == 0.0
    assert U.is_consistent()
    assert S.is_consistent()


def test_svd_0():
    a = yast.rand(config=config_dense, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    svd_combine(a)
    svd_order_combine(a)
    qr_combine(a)
    qr_order_combine(a)
    eigh_combine(a)
    eigh_order_combine(a)


def test_svd_1():
    a = yast.rand(config=config_U1, s=(-1, -1, 1, 1), n=1,
                  t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                  D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
    svd_combine(a)
    svd_order_combine(a)
    qr_combine(a)
    qr_order_combine(a)
    eigh_combine(a)
    eigh_order_combine(a)


def test_svd_2():
    t1 = [(0, 0), (0, 2), (1, 0), (1, 2)]
    a = yast.ones(config=config_Z2_U1, s=(-1, -1, 1, 1),
                  t=[t1, t1, t1, t1],
                  D=[(2, 3, 4, 5), (5, 4, 3, 2), (3, 4, 5, 6), (1, 2, 3, 4)])
    svd_combine(a)
    svd_order_combine(a)
    qr_combine(a)
    qr_order_combine(a)
    eigh_combine(a)
    eigh_order_combine(a)


if __name__ == '__main__':
    test_svd_0()
    test_svd_1()
    test_svd_2()
