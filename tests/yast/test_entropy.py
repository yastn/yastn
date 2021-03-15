import yast
import config_U1_R
from math import isclose
import numpy as np

tol = 1e-12

def test_entropy():
    a = yast.rand(config=config_U1_R, s=(1, 1, -1, -1), n=1,
                    t=[(0, 1), (-1, 0), (-1, 0, 1), (-1, 0, 1)],
                    D=[(5, 6), (5, 6), (2, 3, 4), (2, 3, 4)])
    U, S, V = a.split_svd(axes=((0, 1), (2, 3)), sU=-1) 
    S.set_block(ts=-2, Ds=4, val='ones')
    S.set_block(ts=-1, Ds=12, val='ones')
    S.set_block(ts=0, Ds=25, val='ones')
    US = yast.tensordot(U, S, axes=(2, 0))
    a = yast.tensordot(US, V, axes=(2, 0))
    
    entropy, Smin, normalization = a.entropy(axes=((0, 1), (2, 3)))
    assert isclose(entropy, np.log2(41), rel_tol=tol)
    assert isclose(Smin, 1, rel_tol=tol)
    assert isclose(normalization, np.sqrt(41), rel_tol=tol)

if __name__ == '__main__':
    test_entropy()

