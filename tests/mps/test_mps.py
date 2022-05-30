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
        x = yast.tensordot(psi.A[n], psi.A[n], axes=(cl, cl), conj=(1, 0))
        x0 = yast.eye(config=x.config, legs=x.get_leg([0, 1]))
        assert yast.norm(x - x0.diag()) < tol  # == 0
    assert psi.pC is None


def is_right_canonical(psi):
    """ Assert if each mps tensor is right canonical. """
    cl = (1, 2) if psi.nr_phys == 1 else (1, 2, 3)
    for n in range(psi.N):
        x = yast.tensordot(psi.A[n], psi.A[n], axes=(cl, cl), conj=(0, 1))
        x0 = yast.eye(config=x.config, legs=x.get_leg([0, 1]))
        assert yast.norm(x - x0.diag()) < tol  # == 0
    assert psi.pC is None


def check_canonize(psi):
    """ Canonize mps to left and right, running tests if it is canonical. """
    psi.canonize_sweep(to='last')
    is_left_canonical(psi)
    psi.canonize_sweep(to='first')
    is_right_canonical(psi)


def env2_measure(psi1, psi2):
    """ Test if different overlaps of psi1 and psi2 give consistent results. """
    N = psi1.N
    env = yamps.Env2(bra=psi1, ket=psi2)
    env.setup(to='first')
    env.setup(to='last')

    results = [env.measure()]
    for n in range(N - 1):
        results.append(env.measure(bd=(n, n + 1)))
    results.append(env.measure(bd=(N - 1, N)))
    results.append(env.measure(bd=(N, N - 1)))
    for n in range(N - 1, 0, -1):
        results.append(env.measure(bd=(n, n - 1)))
    results.append(env.measure(bd=(0, -1)))

    env2 = yamps.Env2(bra=psi2, ket=psi1)
    env2.setup(to='last')
    results.append(env2.measure(bd=(N, N - 1)).conj())

    results.append(yamps.measure_overlap(bra=psi1, ket=psi2))
    results.append(yamps.measure_overlap(bra=psi2, ket=psi1).conj())
    results = [x.item() for x in results]  # added for cuda
    assert np.std(results) / abs(np.mean(results)) < tol


def env3_measure(psi1, op, psi2):
    """ Test if different overlaps of psi1 and psi2 give consistent results. """
    N = psi1.N
    env = yamps.Env3(bra=psi1, op=op, ket=psi2)
    env.setup(to='first')
    env.setup(to='last')

    results = [env.measure()]
    for n in range(N - 1):
        results.append(env.measure(bd=(n, n + 1)))
    results.append(env.measure(bd=(N - 1, N)))
    results.append(env.measure(bd=(N, N - 1)))
    for n in range(N - 1, 0, -1):
        results.append(env.measure(bd=(n, n - 1)))
    results.append(env.measure(bd=(0, -1)))
    results.append(yamps.measure_mpo(bra=psi1, op=op, ket=psi2))
    results = [x.item() for x in results]  # added for cuda
    assert np.std(results) / abs(np.mean(results)) < tol


def env2_cononize(psi):
    """ Test if state is normalized after canonization """
    psi.canonize_sweep(to='last')
    env = yamps.Env2(ket=psi).setup(to='first')
    assert abs(env.measure() - 1) < tol


def check_copy(psi1, psi2):
    """ Test if two mps-s have the same tensors (velues). """
    for n in psi1.sweep():
        assert np.allclose(psi1.A[n].to_numpy(), psi2.A[n].to_numpy())


def test_full_copy():
    """ Initialize random mps of full tensors and checks copying. """
    psi = ops_dense.mps_random(N=16, Dmax=15, d=2)
    phi = psi.copy()
    check_copy(psi, phi)

    psi = ops_dense.mps_random(N=16, Dmax=19, d=[2, 3])
    phi = psi.copy()
    check_copy(psi, phi)

    psi = ops_dense.mpo_random(N=16, Dmax=25, d=[2, 3], d_out=[2, 1])
    phi = psi.copy()
    check_copy(psi, phi)


def test_full_canonize():
    """ Initialize random mps of full tensors and checks canonization. """
    psi1 = ops_dense.mps_random(N=16, Dmax=9, d=2)
    check_canonize(psi1)
    psi2 = ops_dense.mps_random(N=16, Dmax=19, d=[2, 3])
    check_canonize(psi2)
    psi3 = ops_dense.mpo_random(N=16, Dmax=36, d=[2, 3], d_out=[2, 1])
    check_canonize(psi3)


def test_full_env2_update():
    """ Initialize random mps' and check if overlaps are calculated consistently. """
    N = 13
    psi1 = ops_dense.mps_random(N=N, Dmax=15, d=3)
    psi2 = ops_dense.mps_random(N=N, Dmax=7, d=3)
    env2_measure(psi1, psi2)
    env2_cononize(psi1)
    env2_cononize(psi2)


def test_full_env3_update():
    """ Initialize random mps' and check if overlaps are calculated consistently. """
    N = 13
    psi1 = ops_dense.mps_random(N=N, Dmax=15, d=3)
    psi2 = ops_dense.mps_random(N=N, Dmax=7, d=3)
    op = ops_dense.mpo_random(N=N, Dmax=5, d=3)
    env3_measure(psi1, op, psi2)


def test_Z2_copy():
    """ Initialize random mps of full tensors and checks copying. """
    psi = ops_Z2.mps_random(N=16, Dblock=25, total_parity=0)
    phi = psi.copy()
    check_copy(psi, phi)

    psi = ops_Z2.mps_random(N=16, Dblock=25, total_parity=1)
    phi = psi.copy()
    check_copy(psi, phi)


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


def test_Z2_env2_update():
    """ Initialize random mps' and check if overlaps are calculated consistently. """
    psi1 = ops_Z2.mps_random(N=16, Dblock=11, total_parity=0)
    psi2 = ops_Z2.mps_random(N=16, Dblock=12, total_parity=0)
    psi3 = ops_Z2.mpo_random(N=16, Dblock=13, total_parity=1)
    psi4 = ops_Z2.mpo_random(N=16, Dblock=5, total_parity=1)

    env2_measure(psi1, psi2)
    env2_cononize(psi1)
    env2_cononize(psi2)
    env2_measure(psi3, psi4)
    env2_cononize(psi3)
    env2_cononize(psi4)


def test_Z2_env3_update():
    """ Initialize random mps' and check if overlaps are calculated consistently. """
    psi1 = ops_Z2.mps_random(N=16, Dblock=11, total_parity=0)
    psi2 = ops_Z2.mps_random(N=16, Dblock=12, total_parity=0)
    op = ops_Z2.mpo_random(N=16, Dblock=3, total_parity=0)
    env3_measure(psi1, op, psi2)


if __name__ == "__main__":
    test_full_copy()
    test_full_canonize()
    test_full_env2_update()
    test_full_env3_update()
    test_Z2_copy()
    test_Z2_canonize()
    test_Z2_env2_update()
    test_Z2_env3_update()
