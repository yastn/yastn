import yast
import config_U1_C
import numpy as np

tol = 1e-12

def test_real_1():
    a = yast.rand(config=config_U1_C, s=(-1, 1),
                  t=[(0, 1), (0, 1)],
                  D=[(1, 2), (1, 2)])
    assert np.iscomplexobj(a.to_numpy())
    assert a.config.dtype == 'complex128'
    
    b = a.real()
    assert np.iscomplexobj(b.to_numpy())
    assert b.config.dtype == 'complex128'

    c = yast.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))
    
    number_c = c.to_number()
    number_r = c.real().to_number()

    assert isinstance(number_c.item(), complex)
    assert not isinstance(number_r.item(), complex)
