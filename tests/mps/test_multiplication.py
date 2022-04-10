""" examples for multiplication of the Mps-s """
import numpy as np
import pytest
import yamps
import yast
try:
    from . import ops_dense
except ImportError:
    import ops_dense


def test_full_addition():
    multiplier = 2 + 1j
    psi0 = ops_dense.mps_random(N=8, Dmax=15, d=3)
    psi0.canonize_sweep(to='first')
    psi1 = psi0.copy()
    psi0.A[0] = multiplier*psi0.A[0]

    out0 = yamps.x_a_times_b(a=psi0, b=psi1, axes=(1, 1), axes_fin=((0,2,), (1,3,)), conj=(0, 1), x=1.)
    norm = np.identity(1)
    for it in range(out0.N):
        norm = np.matmul(norm, out0.A[it].to_numpy())
    assert abs(multiplier - norm[0,0]) < 1e-14

if __name__ == "__main__":
    test_full_addition()
