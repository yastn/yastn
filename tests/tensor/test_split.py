import yamps.tensor as tensor
import settings_full
import settings_U1
import settings_Z2_U1
import settings_U1_U1
import pytest


def svd_combine(a):
    U, S, V = a.split_svd(axes=((3, 1), (2, 0)), sU=-1)
    US = U.dot_diag(S, axis=2)
    USV = US.dot(V, axes=(2, 0))
    USV = USV.transpose(axes=(3, 1, 2, 0))
    assert pytest.approx(a.norm_diff(USV), rel=1e-8, abs=1e-8) == 0
    assert a.is_symmetric()
    assert U.is_symmetric()
    assert S.is_symmetric()
    assert V.is_symmetric()


def svd_order_combine(a):
    U, S, V = a.split_svd(axes=((3, 1), (2, 0)), sU=1, Uaxis=0, Vaxis=-1)
    US = U.dot_diag(S, axis=0)
    USV = US.dot(V, axes=(0, 2))
    USV = USV.transpose(axes=(3, 1, 2, 0))
    assert pytest.approx(a.norm_diff(USV), rel=1e-8, abs=1e-8) == 0
    assert U.is_symmetric()
    assert S.is_symmetric()
    assert V.is_symmetric()


def qr_combine(a):
    Q, R = a.split_qr(axes=((3, 1), (2, 0)))
    QR = Q.dot(R, axes=(2, 0))
    QR = QR.transpose(axes=(3, 1, 2, 0))
    assert pytest.approx(a.norm_diff(QR), rel=1e-8, abs=1e-8) == 0
    assert Q.is_symmetric()
    assert R.is_symmetric()


def qr_order_combine(a):
    Q, R = a.split_qr(axes=((3, 1), (2, 0)), sQ=-1, Qaxis=-2, Raxis=1)
    QR = Q.dot(R, axes=(1, 1))
    QR = QR.transpose(axes=(3, 1, 2, 0))
    assert pytest.approx(a.norm_diff(QR), rel=1e-8, abs=1e-8) == 0
    assert Q.is_symmetric()
    assert R.is_symmetric()


def eigh_combine(a):
    a2 = a.dot(a, axes=((0, 1), (0, 1)), conj=(0, 1))
    S, U = a2.split_eigh(axes=((0, 1), (2, 3)))
    US = U.dot_diag(S, axis=2)
    USU = US.dot(U, axes=(2, 2), conj=(0, 1))
    assert pytest.approx(a2.norm_diff(USU), rel=1e-8, abs=1e-8) == 0
    assert U.is_symmetric()
    assert S.is_symmetric()


def eigh_order_combine(a):
    a2 = a.dot(a, axes=((0, 1), (0, 1)), conj=(0, 1))
    S, U = a2.split_eigh(axes=((0, 1), (2, 3)), Uaxis=0, sU=-1)
    US = U.dot_diag(S, axis=0)
    USU = US.dot(U, axes=(0, 0), conj=(0, 1))
    assert pytest.approx(a2.norm_diff(USU), rel=1e-8, abs=1e-8) == 0
    assert U.is_symmetric()
    assert S.is_symmetric()


def test_svd0():
    a = tensor.rand(settings=settings_full, s=(-1, 1, -1, 1), D=[11, 12, 13, 21])
    svd_combine(a)
    svd_order_combine(a)
    qr_combine(a)
    qr_order_combine(a)
    eigh_combine(a)
    eigh_order_combine(a)


def test_svd1():
    a = tensor.rand(settings=settings_U1, s=(-1, 1, -1, 1), n=1,
                    t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                    D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
    svd_combine(a)
    svd_order_combine(a)
    qr_combine(a)
    qr_order_combine(a)
    eigh_combine(a)
    eigh_order_combine(a)


def test_svd2():
    a = tensor.ones(settings=settings_Z2_U1, s=(-1, -1, 1, 1),
                    t=[(0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2), (0, 1), (0, 2)],
                    D=[(2, 3), (3, 2), (4, 5), (5, 4), (4, 3), (3, 4), (2, 3), (3, 2)])
    svd_combine(a)
    svd_order_combine(a)
    qr_combine(a)
    qr_order_combine(a)
    eigh_combine(a)
    eigh_order_combine(a)


if __name__ == '__main__':
    test_svd0()
    test_svd1()
    test_svd2()
