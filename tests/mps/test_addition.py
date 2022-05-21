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


def check_add(psi0, psi1):
    """ test yamps.add using overlaps"""
    out1 = yamps.add(tens=[psi0, psi1], amp=[1., 2.], common_legs=(1,))
    o1 = yamps.measure_overlap(out1, out1)
    p0 = yamps.measure_overlap(psi0, psi0)
    p1 = yamps.measure_overlap(psi1, psi1)
    p01 = yamps.measure_overlap(psi0, psi1)
    p10 = yamps.measure_overlap(psi1, psi0)
    assert abs(o1 - p0 - 4 * p1 - 2 * p01 - 2 * p10) < tol


def test_addition():
    """create two Mps-s and add them to each other"""
    psi0 = ops_dense.mps_random(N=8, Dmax=15, d=1)
    psi1 = ops_dense.mps_random(N=8, Dmax=19, d=1)
    check_add(psi0, psi1)

    psi0 = ops_Z2.mps_random(N=8, Dblock=8, total_parity=0)
    psi1 = ops_Z2.mps_random(N=8, Dblock=12, total_parity=0)
    check_add(psi0, psi1)


if __name__ == "__main__":
    test_addition()
