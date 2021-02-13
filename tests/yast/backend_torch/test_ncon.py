""" 
Test yast.ncon
"""

import yamps.yast as yast
import config_U1_R
import config_dense_R
from math import isclose

tol = 1e-12

def test_ncon_0():
    a=yast.rand(s=(1, 1, 1), D=(20, 3, 1), config=config_dense_R)
    b=yast.rand(s=(1, 1, -1), D=(4, 2, 20), config=config_dense_R)
    c=yast.rand(s=(-1, 1, 1, -1), D=(20, 30, 10, 10), config=config_dense_R)
    d=yast.rand(s=(1, 1, 1, -1), D=(30, 10, 20, 10), config=config_dense_R)
    e=yast.rand(s=(1, 1), D=(3, 1), config=config_dense_R)
    f=yast.rand(s=(-1, -1), D=(2, 4), config=config_dense_R)
    g=yast.rand(s=(1,), D=(5, ), config=config_dense_R)
    h=yast.rand(s=(-1,), D=(6, ), config=config_dense_R)
    
    x = yast.ncon([a, b], [[1, -2, -4], [-1, -3, 1]])
    assert x.get_shape() == (4, 3, 2, 1)
   
    y = yast.ncon([a, b, c, d], [[4, -3, -1], [-4, -2, 5], [4, 3, 1, 1], [3, 2, 5, 2]], [0, 1, 0, 1]) 
    assert y.get_shape() == (1, 2, 3, 4)

    z1 = yast.ncon([e, f], [[-3, -1], [-2, -4]], [0, 1]) 
    z2 = yast.ncon([f, e], [[-2, -4], [-3, -1]], [1, 0])     
    z3 = yast.ncon([e, f], [[-3, -1], [-2, -4]], [0, 0]) 
    z4 = yast.ncon([f, e], [[-2, -4], [-3, -1]], [1, 1]) 
    assert z1.get_shape() == (1, 2, 3, 4)
    assert z2.get_shape() == (1, 2, 3, 4)
    assert z3.get_shape() == (1, 2, 3, 4)
    assert z4.get_shape() == (1, 2, 3, 4)
    assert isclose(z1.norm_diff(z2), 0, rel_tol=tol, abs_tol=tol)
    assert isclose((z3-z4.conj()).norm(), 0, rel_tol=tol, abs_tol=tol)

    y1 = yast.ncon([a, b, c, d, g, h], [[4, -3, -1], [-4, -2, 5], [4, 3, 1, 1], [3, 2, 5, 2], [-5], [-6]], [1, 0, 1, 0, 1, 0]) 
    assert y1.get_shape() == (1, 2, 3, 4, 5, 6)

    y2 = yast.ncon([a, g, a], [[1, 2, 3], [-1], [1, 2, 3]], [0, 0, 1])
    assert y2.get_shape() == (5,)

    y3 = yast.ncon([a, g, a, b, a], [[1, 2, 3], [-5], [1, 2, 3], [-4, -2, 4], [4, -3, -1]], [0, 0, 1, 0, 0])
    assert y3.get_shape() == (1, 2, 3, 4, 5)

    y4 = yast.ncon([a, a, b, b], [[1, 2, 3], [1, 2, 3], [6, 5, 4], [6, 5, 4]], [0, 1, 0, 1])
    y5 = yast.ncon([a, a, b, b], [[6, 5, 4], [6, 5, 4], [1, 3, 2], [1, 3, 2]], [0, 1, 0, 1]) 
    # assert isinstance(y4, complex)
    # assert isinstance(y5, complex)
    assert isclose(abs(y4-y5), 0, rel_tol=tol, abs_tol=tol)

def test_ncon_1():
    a = yast.rand(config=config_U1_R, s=[-1, 1, -1], n=0, 
                  D=((20, 10) ,(3, 3) ,(1, 1)), t=((1, 0), (1, 0), (1, 0)))
    b = yast.rand(config=config_U1_R, s=[1, 1, 1], n=1,
                  D=((4, 4), (2, 2), (20, 10)), t=((1, 0), (1, 0), (1, 0))) 
    c = yast.rand(config=config_U1_R, s=[1, 1, 1, -1], n=1,
                  D=((20, 10), (30, 20) ,(10, 5), (10, 5)), t=((1, 0), (1, 0), (1, 0), (1, 0))) 
    d = yast.rand(config=config_U1_R, s=[1, 1, -1, -1], n=0,
                  D=((30, 20), (10, 5) ,(20, 10), (10, 5)),  t=((1, 0), (1, 0), (1, 0), (1, 0))) 
    
    e = yast.ncon([a, b], [[1, -2, -4], [-1, -3, 1]]) 		
    assert e.get_shape() == (8, 6, 4, 2)
    assert e.is_consistent()
    h = yast.ncon([a, b, c, d], [[4, -3, -1], [-4, -2, 5], [4, 3, 1, 1], [3, 2, 5, 2]], [0, 1, 0, 1]) 		
    assert h.get_shape() == (2, 4, 6, 8)
    assert h.is_consistent()
    g = yast.ncon([a, a, a, b], [[1, 2, 3], [1, 2, 3], [4, -3, -1], [-4, -2, 4]], [0, 1, 1, 1]) 
    assert g.get_shape() == (2, 4, 6, 8)
    assert g.is_consistent()

if __name__ == '__main__':
    test_ncon_0()
    test_ncon_1()
