import yast
import config_dense_R
import config_U1_R
import config_Z2_U1_R
from math import isclose
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
    US = U.dot(S, axes=(ll, 0))
    USV = US.dot(V, axes=(ll, 0))
    b = a.transpose(axes=axes[0]+axes[1])
    assert isclose(b.norm_diff(USV), 0, rel_tol=tol, abs_tol=tol)


def test_split_svd_truncate():
    a = yast.rand(config=config_U1_R, s=(1, 1, -1, -1), n=1,
                    t=[(0, 1), (-1, 0), (-1, 0, 1), (-1, 0, 1)],
                    D=[(5, 6), (5, 6), (2, 3, 4), (2, 3, 4)])
    U, S, V = a.split_svd(axes=((0, 1), (2, 3)), sU=-1) 
    S.set_block(ts = (-2, -2), Ds = 4, val = np.diag(np.array([2**(-ii-6) for ii in range(4)]))) 	
    S.set_block(ts = (-1, -1), Ds = 12, val = np.diag(np.array([2**(-ii-2) for ii in range(12)]))) 	
    S.set_block(ts = (0, 0), Ds = 25, val = np.diag(np.array([2**(-ii-1) for ii in range(25)])) )	
    a = (U.dot(S, axes=(2, 0))).dot(V, axes=(2, 0))
    
    opts={'tol':0.01, 'D_block':100, 'D_total':12, 'truncated_svd':False} 	
    U1, S1, V1 = a.split_svd(axes=((0, 1), (2, 3)), sU=-1, **opts)
    assert S1.get_shape() == (12, 12)


def test_split_svd_division():
    a = yast.rand(config=config_U1_R, s=(-1, -1, 1, 1), n=3,
                    t=[(-1, 0, 1), (-2, 0, 2), (-2, -1, 0, 1, 2), (0, 1)],
                    D=[(2, 3, 4), (5, 6, 7), (6, 5, 4, 3, 2), (2, 3)])
    U1, S1, V1 = a.split_svd(axes=((0, 1), (2, 3)), nU=True, sU=1)
    USV1 = U1.dot(S1, axes=(2, 0)).dot(V1, axes=(2, 0))
    assert isclose(a.norm_diff(USV1), 0, rel_tol=tol, abs_tol=tol)
    assert isclose(U1.n-3, 0, rel_tol=tol, abs_tol=tol)
    assert isclose(V1.n-0, 0, rel_tol=tol, abs_tol=tol) 

    U2, S2, V2 = a.split_svd(axes=((0, 1), (2, 3)), nU=False, sU=1)
    USV2 = U2.dot(S2, axes=(2, 0)).dot(V2, axes=(2, 0))
    assert isclose(a.norm_diff(USV2), 0, rel_tol=tol, abs_tol=tol)
    assert isclose(U2.n-0, 0, rel_tol=tol, abs_tol=tol)
    assert isclose(V2.n-3, 0, rel_tol=tol, abs_tol=tol) 


if __name__ == '__main__':
    # test_split_svd_sparse()
    test_split_svd_truncate()
    # test_split_svd_division()

