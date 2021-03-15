import yast
import config_U1_R
import numpy as np

tol = 1e-12

def test_split_svd_sparse():
    a=yast.Tensor(config=config_U1_R, s=(-1, -1, -1, 1, 1, 1), n=0)
    a.set_block(ts=(1, 0, 0, 1, 0, 0), Ds=(2, 2, 2, 2, 2, 2), val='rand')
    a.set_block(ts=(0, 1, 0, 0, 1, 0), Ds=(2, 2, 2, 2, 2, 2), val='rand')
    a.set_block(ts=(0, 0, 1, 0, 0, 1), Ds=(2, 2, 2, 2, 2, 2), val='rand')
    axes=((1, 2, 0), (5, 3, 4))
    U, S, V = a.split_svd(axes)
    assert U.is_consistent()
    assert S.is_consistent()
    assert V.is_consistent()

    ll = len(axes[0])
    US = yast.tensordot(U, S, axes=(ll, 0))
    USV = yast.tensordot(US, V, axes=(ll, 0))
    b = a.transpose(axes=axes[0]+axes[1])
    assert b.norm_diff(USV) < tol  # == 0.0


def test_split_svd_truncate():
    a = yast.rand(config=config_U1_R, s=(1, 1, -1, -1), n=1,
                    t=[(0, 1), (-1, 0), (-1, 0, 1), (-1, 0, 1)],
                    D=[(5, 6), (5, 6), (2, 3, 4), (2, 3, 4)])
    U, S, V = a.split_svd(axes=((0, 1), (2, 3)), sU=-1) 
    S.set_block(ts = (-2, -2), Ds = 4, val = np.diag(np.array([2**(-ii-6) for ii in range(4)])))
    S.set_block(ts = (-1, -1), Ds = 12, val = np.diag(np.array([2**(-ii-2) for ii in range(12)])))
    S.set_block(ts = (0, 0), Ds = 25, val = np.diag(np.array([2**(-ii-1) for ii in range(25)])) )
    US = yast.tensordot(U, S, axes=(2, 0))
    a = yast.tensordot(US, V, axes=(2, 0))

    
    opts={'tol':0.01, 'D_block':100, 'D_total':12, 'truncated_svd':False}
    U1, S1, V1 = a.split_svd(axes=((0, 1), (2, 3)), sU=-1, **opts)
    assert S1.get_shape() == (12, 12)


def test_split_svd_division():
    a = yast.rand(config=config_U1_R, s=(-1, -1, 1, 1), n=3,
                    t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                    D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
    U1, S1, V1 = a.split_svd(axes=((0, 1), (2, 3)), nU=True, sU=1)
    US1 = yast.tensordot(U1, S1, axes=(2, 0))
    USV1 = yast.tensordot(US1, V1, axes=(2, 0))
    assert a.norm_diff(USV1) < tol  # == 0.0
    assert U1.n-3 < tol  # == 0.0
    assert V1.n-0 < tol  # == 0.0 

    U2, S2, V2 = a.split_svd(axes=((0, 1), (2, 3)), nU=False, sU=1)
    US2 = yast.tensordot(U2, S2, axes=(2, 0))
    USV2 = yast.tensordot(US2, V2, axes=(2, 0))
    assert a.norm_diff(USV2) < tol  # == 0.0
    assert U2.n == 0
    assert V2.n == 3


if __name__ == '__main__':
    test_split_svd_sparse()
    test_split_svd_truncate()
    test_split_svd_division()

