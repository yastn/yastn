""" yast.real yast.imag """
import numpy as np
import yast
if __name__ == '__main__':
    from configs import config_U1
else:
    from .configs import config_U1

tol = 1e-12


def test_real_1():
    a = yast.rand(config=config_U1, s=(-1, 1), t=[(0, 1), (0, 1)], D=[(1, 2), (1, 2)], dtype='complex128')

    assert np.iscomplexobj(a.to_numpy())

    b = a.real()
    assert np.isrealobj(b.to_numpy())  # to_dense, to_numpy use config.dtype

    c = yast.tensordot(a, a, axes=((0, 1), (0, 1)), conj=(0, 1))  # ndim = 0
    num_c = c.to_number()  # dtype same as that of data in tensor c
    num_r1 = c.real().to_number()  # dtype same as that of data in tensor c.real()
    num_r2 = c.to_number('real')  # takes real part of to_number (backend ascetic)

    assert isinstance(num_c.item(), complex)
    assert isinstance(num_r1.item(), float)
    assert isinstance(num_r2.item(), float)
    assert abs(num_c.real - num_r1) < tol
    assert abs(num_c.real - num_r2) < tol

    ar = a.real()
    ai = a.imag()
    # TODO assert yast.are_independent(ar, a)
    # TODO assert yast.are_independent(ai, a)
    assert yast.norm_diff(ar + 1j * ai, a) < tol


if __name__ == '__main__':
    test_real_1()
