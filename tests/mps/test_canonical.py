""" basic procedures of single mps """
import numpy as np
import pytest
import yast
import yamps
try:
    from . import ops_dense
    from . import ops_Z2
except ImportError:
    import ops_dense
    import ops_Z2

tol = 1e-12


def is_left_canonical(psi):
    """ Assert if each mps tensor is left canonical. """
    cl = (0, 1) if psi.nr_phys == 1 else (0, 1, 2)
    for n in range(psi.N):
        x = psi.A[n].tensordot(psi.A[n], axes=(cl, cl), conj=(1, 0))
        x0 = yast.match_legs(tensors=[x, x], legs=[0, 1], isdiag=True, val='ones', conjs=[1, 1])
        assert yast.norm(x - x0.diag()) < tol  # == 0
    assert psi.pC is None


def is_right_canonical(psi):
    """ Assert if each mps tensor is right canonical. """
    cl = (1, 2) if psi.nr_phys == 1 else (1, 2, 3)
    for n in range(psi.N):
        x = psi.A[n].tensordot(psi.A[n], axes=(cl, cl), conj=(0, 1))
        x0 = yast.match_legs(tensors=[x, x], legs=[0, 1], isdiag=True, val='ones', conjs=[1, 1])
        assert yast.norm(x - x0.diag()) < tol  # == 0
    assert psi.pC is None


def check_canonize(psi):
    """ Canonize mps to left and right, running tests if it is canonical. """
    psi.canonize_sweep(to='last')
    is_left_canonical(psi)
    psi.canonize_sweep(to='first')
    is_right_canonical(psi)


def test_full_canonize():
    """ Initialize random mps of full tensors and checks canonization. """
    psi1 = ops_dense.mps_random(N=16, Dmax=9, d=2)
    check_canonize(psi1)
    psi2 = ops_dense.mps_random(N=16, Dmax=19, d=[2, 3])
    check_canonize(psi2)
    psi3 = ops_dense.mpo_random(N=16, Dmax=36, d=[2, 3], d_out=[2, 1])
    check_canonize(psi3)


def test_Z2_canonize():
    """ Initialize random mps of full tensors and checks canonization. """
    psi1 = ops_Z2.mps_random(N=16, Dblock=11, total_parity=0)
    check_canonize(psi1)
    psi2 = ops_Z2.mps_random(N=16, Dblock=12, total_parity=1)
    check_canonize(psi2)
    psi3 = ops_Z2.mpo_random(N=16, Dblock=3, total_parity=1)
    check_canonize(psi3)
    psi4 = ops_Z2.mpo_random(N=16, Dblock=4, total_parity=0, t_out=(0,))
    check_canonize(psi4)


if __name__ == "__main__":
    test_full_canonize()
    test_Z2_canonize()
