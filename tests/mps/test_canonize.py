""" basic procedures of single mps """
import numpy as np
import yast
import yamps


tol = 1e-12


def test_canonize():
    """ Initialize random mps and checks canonization. """
    operators = yast.operators.Spin1(sym='Z3')
    generate = yamps.Generator(N=16, operators=operators)

    for n in (0, 1, 2):
        psi = generate.random_mps(n=n, D_total=16)
        check_canonize(psi)
    psi = generate.random_mpo(D_total=8, dtype='complex128')
    check_canonize(psi)

    operators = yast.operators.Spin12(sym='dense')
    generate = yamps.Generator(N=16, operators=operators)
    psi = generate.random_mps(D_total=16, dtype='complex128')
    check_canonize(psi)
    psi = generate.random_mpo(D_total=8)
    check_canonize(psi)


def test_env2_update():
    """ Initialize random mps' and check if overlaps are calculated consistently. """
    operators = yast.operators.Spin12(sym='U1')
    generate = yamps.Generator(N=12, operators=operators)
    psi1 = generate.random_mps(D_total=15)
    psi2 = generate.random_mps(D_total=7)
    check_env2_measure(psi1, psi2)

    operators = yast.operators.SpinlessFermions(sym='Z2')
    generate = yamps.Generator(N=13, operators=operators)

    psi1 = generate.random_mps(D_total=11, n=1)
    psi2 = generate.random_mps(D_total=15, n=1)
    psi3 = generate.random_mpo(D_total=10)
    psi4 = generate.random_mpo(D_total=8)
    check_env2_measure(psi1, psi2)
    check_env2_measure(psi3, psi4)


def test_env3_update():
    """ Initialize random mps' and check if overlaps are calculated consistently. """
    operators = yast.operators.SpinfulFermions(sym='U1xU1')
    generate = yamps.Generator(N=13, operators=operators)
    psi1 = generate.random_mps(D_total=11, n=(7, 7))
    psi2 = generate.random_mps(D_total=15, n=(7, 7))
    op = generate.random_mpo(D_total=10)
    check_env3_measure(psi1, op, psi2)


def check_canonize(psi):
    """ Canonize mps to left and right, running tests if it is canonical. """
    psi.canonize_sweep(to='last')
    assert psi.is_canonical(to='last', tol=tol)
    assert abs(yamps.measure_overlap(psi, psi) - 1) < tol
    psi.canonize_sweep(to='first')
    assert psi.is_canonical(to='first', tol=tol)
    assert abs(yamps.measure_overlap(psi, psi) - 1) < tol


def check_env2_measure(psi1, psi2):
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


def check_env3_measure(psi1, op, psi2):
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


if __name__ == "__main__":
    test_canonize()
    test_env2_update()
    test_env3_update()
