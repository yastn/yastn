import yamps.tensor.yast as yast
import config_U1_R
import pytest
import numpy as np


def test_entropy():
    a = yast.rand(config=config_U1_R, s=(1, 1, -1, -1), n=1,
                    t=[(0, 1), (-1, 0), (-1, 0, 1), (-1, 0, 1)],
                    D=[(5, 6), (5, 6), (2, 3, 4), (2, 3, 4)])
    U, S, V = a.split_svd(axes=((0, 1), (2, 3)), sU=-1) 
    S.set_block(ts=-2, Ds=4, val='ones') 	
    S.set_block(ts=-1, Ds=12, val='ones') 	
    S.set_block(ts=0, Ds=25, val='ones')	
    a = (U.dot(S, axes=(2, 0))).dot(V, axes=(2, 0))
    
    entropy, Smin, normalization = a.entropy(axes=((0, 1), (2, 3)))
    assert pytest.approx(entropy) == np.log2(41)
    assert pytest.approx(Smin) == 1.
    assert pytest.approx(normalization) == np.sqrt(41)

if __name__ == '__main__':
    test_entropy()

