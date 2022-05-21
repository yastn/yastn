""" examples for addition of the Mps-s """
import numpy as np
import pytest
import yamps
try:
    from . import ops_dense
    from . import ops_Z2
except ImportError:
    import ops_dense
    import ops_Z2


tol = 1e-6


def test_full_addition():
    """create two Mps-s and add them to each other"""
    psi0 = ops_dense.mps_random(N=8, Dmax=15, d=1)
    psi1 = ops_dense.mps_random(N=8, Dmax=19, d=1)

    out0 = yamps.apxb(a=psi0, b=psi1, common_legs=(1,), x=2.)
    out1 = yamps.add(tens=[psi0, psi1], amp=[1., 2.], common_legs=(1,))
    assert yamps.measure_overlap(out0, out1) > 1 - tol


def test_Z2_addition():
    """create two Mps-s and add them to each other"""
    psi0 = ops_Z2.mps_random(N=8, Dblock=8, total_parity=0)
    psi1 = ops_Z2.mps_random(N=8, Dblock=12, total_parity=0)

    out0 = yamps.apxb(a=psi0, b=psi1, common_legs=(1,), x=2.)
    out1 = yamps.add(tens=[psi0, psi1], amp=[1., 2.], common_legs=(1,))
    assert yamps.measure_overlap(out0, out1) > 1 - tol


if __name__ == "__main__":
    test_full_addition()
    test_Z2_addition()
