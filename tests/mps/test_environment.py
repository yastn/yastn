# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
""" basic methods of single Mps """
import numpy as np
import pytest
import yastn
import yastn.tn.mps as mps


def test_env2_update(config_kwargs, tol=1e-12):
    """ Initialize random mps' and check if overlaps are calculated consistently. """
    N = 12
    for sym, n in [('U1', 0), ('Z2', 1), ('Z2', 0)]:
        ops = yastn.operators.Spin12(sym=sym, **config_kwargs)
        ops.random_seed(seed=0)
        I = mps.product_mpo(ops.I(), N)
        psi1 = mps.random_mps(I, D_total=15, n=n)
        psi2 = mps.random_mps(I, D_total=7, n=n)
        H1 = mps.random_mpo(I, D_total=10)
        H2 = mps.random_mpo(I, D_total=8)
        #
        check_env2_measure(psi1, psi2, tol)
        check_env2_measure(H1, H2, tol)
        check_env2_measure(psi1, [[H1, psi1], [psi2]], tol)


def check_env2_measure(psi1, psi2, tol):
    """ Test if different overlaps of psi1 and psi2 give consistent results. """
    N = psi1.N
    env = mps.Env(psi1, psi2)
    env.setup_(to='first')
    env.setup_(to='last')

    results = [env.measure()]
    results.append(mps.measure_overlap(bra=psi1, ket=psi2))

    for n1 in range(-1, N+1):
        for n2 in range(-1, N+1):
            if n1 != n2:
                results.append(env.measure(bd=(n1, n2)))

    if not isinstance(psi2, list):
        env2 = mps.Env(psi2, psi1)
        env2.setup_(to='last')
        results.append(env2.measure(bd=(N, N - 1)).conj())
        results.append(mps.measure_overlap(bra=psi2, ket=psi1).conj())

    results = [x.item() for x in results]  # added for cuda
    assert np.std(results) / abs(np.mean(results)) < tol


def test_env3_update(config_kwargs, tol=1e-12):
    """ Initialize random mps' and check if overlaps are calculated consistently. """
    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **config_kwargs)
    ops.random_seed(seed=0)
    N = 7
    I = mps.product_mpo(ops.I(), N=N)
    psi1 = mps.random_mps(I, D_total=11, n=(4, 2))
    psi2 = mps.random_mps(I, D_total=15, n=(4, 2))
    op = mps.random_mpo(I, D_total=16)
    #
    check_env3_measure(psi1, op, psi2, tol)
    check_env3_measure(op, op, op, tol)
    #
    # also for MpoPBC
    op_pbc = mps.Mpo(N, periodic=True)
    for n in op_pbc.sweep():
        op_pbc[n] = op[(n + 5) % N].copy()
    check_env3_measure(psi1, op_pbc, psi2, tol)


def check_env3_measure(psi1, op, psi2, tol):
    """ Test if different overlaps of psi1 and psi2 give consistent results. """
    N = psi1.N
    env = mps.Env(psi1, [op, psi2])
    env.setup_(to='first')
    env.setup_(to='last')

    results = [env.measure()]
    for n1 in range(-1, N+1):
        for n2 in range(-1, N+1):
            if n1 != n2:
                results.append(env.measure(bd=(n1, n2)))
    results.append(mps.measure_mpo(bra=psi1, op=op, ket=psi2))
    results = [x.item() for x in results]  # added for cuda
    assert np.std(results) / abs(np.mean(results)) < tol


def test_env_setup(config_kwargs):
    """ Simple test of minimal setup and measure. """
    ops = yastn.operators.SpinfulFermions(sym='Z2', **config_kwargs)
    N = 10
    I = mps.product_mpo(ops.I(), N)
    env = mps.Env(I, I)
    assert len(env.F) == 2
    env.measure((-1, 10))
    #
    env.setup_(to=(0, 9))
    assert len(env.F) == 4
    env.measure((0, 9))
    #
    env.setup_(to=5)
    assert len(env.F) == 11
    env.measure((4, 6))


@pytest.mark.parametrize('sym', ['Z2', 'U1'])
def test_env_hole(config_kwargs, sym, tol=1e-12):
    """
    Env(bra, [op, ket]).hole(n) is the environment of the MPO tensor op[n] in <bra|op|ket>,
    with legs ordered as in op[n]. It is used in Monte Carlo sampling of PEPS.
    """
    ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)
    N = 6
    I = mps.product_mpo(ops.I(), N)
    terms = [mps.Hterm(1.0, [i, j], [ops.cp(), ops.c()]) for i in range(N) for j in range(N) if abs(i - j) == 1]
    terms += [mps.Hterm(0.3, [i], [ops.n()]) for i in range(N)]
    H0 = mps.generate_mpo(I, terms)
    HP = mps.Mpo(N, periodic=True)  # the same operator as MpoPBC
    for i in range(N):
        HP[(i + N // 2) % N] = H0[i].copy()
    n = (N // 2,) if sym == 'U1' else (1,)

    for H, env_class in [(H0, 'Env_mps_mpo_mps'), (HP, 'Env_mps_mpopbc_mps')]:
        H = 0.7 * H  # non-trivial factors of op, bra, and ket
        bra = 1.3 * mps.random_mps(I, n=n, D_total=6)
        ket = 2 * mps.random_mps(I, n=n, D_total=5)
        env = mps.Env(bra, [H, ket]).setup_(to='first').setup_(to='last')
        assert type(env).__name__ == env_class

        ref = mps.vdot(bra, H, ket)
        for k in range(N):
            hole = env.hole(k)
            val = hole.tensordot(H[k], axes=((0, 1, 2, 3), (0, 1, 2, 3))).to_number() * env.factor()
            assert abs(val - ref) < tol * abs(ref)
            #
            # hole is a linear functional of the tensor at site k
            X = yastn.rand(H.config, legs=H[k].get_legs(), n=H[k].n)
            H2 = H.shallow_copy()
            H2[k] = X
            val = hole.tensordot(X, axes=((0, 1, 2, 3), (0, 1, 2, 3))).to_number() * env.factor()
            ref2 = mps.vdot(bra, H2, ket)
            assert abs(val - ref2) < tol * abs(ref2)


def test_env_raise(config_kwargs):
    ops = yastn.operators.SpinfulFermions(sym='Z2', **config_kwargs)
    I12 = mps.product_mpo(ops.I(), 12)
    H12 = mps.random_mpo(I12, D_total=10)
    psi12 = mps.random_mps(I12, D_total=15)

    I13 = mps.product_mpo(ops.I(), 13)
    psi13 = mps.random_mps(I13, D_total=4)

    # miscalcullus
    env = mps.Env(psi12, [[H12, H12], psi12])  # Env_sum
    assert env.factor() == 1

    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, [psi12, psi12])
        # Env: MPO operator should have 2 physical legs.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, [H12, H12])
        # Env: bra and ket should have the same number of physical legs.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, [H12, H12, psi12])
        # Env: Input cannot be parsed.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, [1, psi12])
        # Env: Input cannot be parsed.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, psi13)
        # Env: bra and ket should have the same number of physical legs.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi13, [H12, psi13])
        # Env: MPO operator, bra and ket should have the same number of sites.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, H12)
        # Env: bra and ket should have the same number of physical legs.
    with pytest.raises(yastn.YastnError):
        mps.Env(psi12, [[H12, psi12], psi12, [psi12]])
        # Env: Input cannot be parsed.
    with pytest.raises(yastn.YastnError):
        H12pbc = mps.Mpo(12, periodic=True)
        mps.Env(H12, [H12pbc, H12])
        # Env: Application of MpoPBC on Mpo is not supported. Contact developers to add this functionality.
    for precompute in (False, True):
        env = mps.Env(psi12, [H12, psi12], precompute=precompute).setup_(to='first')
        with pytest.raises(yastn.YastnError, match="to should be 'first' or 'last'"):
            env.update_env_(0, to='right')
            # Env: to should be 'first' or 'last'.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
