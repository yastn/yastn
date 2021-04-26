import ops_full
import ops_Z2
import yast.mps as mps
import yast
import numpy as np
import pytest


def is_left_canonical(psi):
    """
    Assert if each mps tensor is left canonical
    """
    cl = (0, 1) if psi.nr_phys == 1 else (0, 1, 2)
    for n in range(psi.N):
        x = psi.A[n].tensordot(psi.A[n], axes=(cl, cl), conj=(1, 0))
        x0 = yast.match_legs(tensors=[x, x], legs=[0, 1], isdiag=True, val='ones', conjs=[1, 1])
        assert pytest.approx(x.norm_diff(x0.diag())) == 0


def is_right_canonical(psi):
    """
    Assert if each mps tensor is right canonical
    """
    cl = (1, 2) if psi.nr_phys == 1 else (1, 2, 3)
    for n in range(psi.N):
        x = psi.A[n].tensordot(psi.A[n], axes=(cl, cl), conj=(0, 1))
        x0 = yast.match_legs(tensors=[x, x], legs=[0, 1], isdiag=True, val='ones', conjs=[1, 1])
        assert pytest.approx(x.norm_diff(x0.diag())) == 0


def check_canonize(psi):
    """
    Canonize mps to left and right, running tests if it is canonical
    """
    psi.canonize_sweep(to='last')
    is_left_canonical(psi)
    psi.canonize_sweep(to='first')
    is_right_canonical(psi)


def env2_measure(psi1, psi2):
    """
    Test if different overlaps of psi1 and psi2 give consistent results
    """
    N = psi1.g.N
    env = mps.Env2(bra=psi1, ket=psi2)
    env.setup_to_first()
    env.setup_to_last()

    results = [env.measure()]
    for n in range(N - 1):
        results.append(env.measure(bd=(n, n + 1)))
    results.append(env.measure(bd=(N - 1, None)))
    results.append(env.measure(bd=(None, N - 1)))
    for n in range(N - 1, 0, -1):
        results.append(env.measure(bd=(n, n - 1)))
    results.append(env.measure(bd=(0, None)))

    env2 = mps.Env2(bra=psi2, ket=psi1)
    env2.setup_to_last()
    results.append(np.conj(env2.measure(bd=(None, N - 1))))

    results.append(mps.measure_overlap(bra=psi1, ket=psi2))
    results.append(mps.measure_overlap(bra=psi2, ket=psi1).conj())
    assert(np.std(results) / abs(np.mean(results)) < 1e-12)


def env3_measure(psi1, op, psi2):
    """
    Test if different overlaps of psi1 and psi2 give consistent results
    """
    N = psi1.N
    env = mps.Env3(bra=psi1, op=op, ket=psi2)
    env.setup_to_first()
    env.setup_to_last()

    results = [env.measure()]
    for n in range(N - 1):
        results.append(env.measure(bd=(n, n + 1)))
    results.append(env.measure(bd=(N - 1, None)))
    results.append(env.measure(bd=(None, N - 1)))
    for n in range(N - 1, 0, -1):
        results.append(env.measure(bd=(n, n - 1)))
    results.append(env.measure(bd=(0, None)))

    results.append(mps.measure_mpo(bra=psi1, op=op, ket=psi2))
    assert(np.std(results) / abs(np.mean(results)) < 1e-12)


def env2_cononize(psi):
    psi.canonize_sweep(to='last')
    env = mps.Env2(ket=psi)
    env.setup_to_first()
    assert(abs(env.measure() - 1) < 1e-12)


def check_copy(psi1, psi2):
    assert psi1.g is psi2.g
    for n in psi1.g.sweep():
        assert np.allclose(psi1.A[n].to_numpy(), psi2.A[n].to_numpy())


def test_full_copy():
    """
    Initialize random mps of full tensors and checks copying.
    """
    psi = ops_full.mps_random(N=16, Dmax=15, d=2)
    phi = psi.copy()
    check_copy(psi, phi)

    psi = ops_full.mps_random(N=16, Dmax=19, d=[2, 3])
    phi = psi.copy()
    check_copy(psi, phi)

    psi = ops_full.mpo_random(N=16, Dmax=25, d=[2, 3], d_out=[2, 1])
    phi = psi.copy()
    check_copy(psi, phi)


def test_full_canonize():
    """
    Initialize random mps of full tensors and checks canonization.
    """
    psi1 = ops_full.mps_random(N=16, Dmax=9, d=2)
    check_canonize(psi1)
    psi2 = ops_full.mps_random(N=16, Dmax=19, d=[2, 3])
    check_canonize(psi2)
    psi3 = ops_full.mpo_random(N=16, Dmax=36, d=[2, 3], d_out=[2, 1])
    check_canonize(psi3)


def test_full_env2_update():
    """
    Initialize random mps' and check if overlaps are calculated consistently.
    """
    N = 13
    psi1 = ops_full.mps_random(N=N, Dmax=15, d=3)
    psi2 = ops_full.mps_random(N=N, Dmax=7, d=3)
    env2_measure(psi1, psi2)
    env2_cononize(psi1)
    env2_cononize(psi2)


def test_full_env3_update():
    """
    Initialize random mps' and check if overlaps are calculated consistently.
    """
    N = 13
    psi1 = ops_full.mps_random(N=N, Dmax=15, d=3)
    psi2 = ops_full.mps_random(N=N, Dmax=7, d=3)
    op = ops_full.mpo_random(N=N, Dmax=5, d=3)
    env3_measure(psi1, op, psi2)


def test_Z2_copy():
    """
    Initialize random mps of full tensors and checks copying.
    """
    psi = ops_Z2.mps_random(N=16, Dblock=25, total_parity=0)
    phi = psi.copy()
    check_copy(psi, phi)

    psi = ops_Z2.mps_random(N=16, Dblock=25, total_parity=1)
    phi = psi.copy()
    check_copy(psi, phi)


def test_Z2_canonize():
    """
    Initialize random mps of full tensors and checks canonization
    """
    psi1 = ops_Z2.mps_random(N=16, Dblock=11, total_parity=0)
    check_canonize(psi1)
    psi2 = ops_Z2.mps_random(N=16, Dblock=12, total_parity=1)
    check_canonize(psi2)
    psi3 = ops_Z2.mpo_random(N=16, Dblock=3, total_parity=1)
    check_canonize(psi3)
    psi4 = ops_Z2.mpo_random(N=16, Dblock=4, total_parity=0, t_out=(0,))
    check_canonize(psi4)


def test_Z2_env2_update():
    """
    Initialize random mps' and check if overlaps are calculated consistently.
    """
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
    """
    Initialize random mps' and check if overlaps are calculated consistently.
    """
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