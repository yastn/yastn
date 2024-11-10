# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import pytest
import numpy as np
import yastn
import yastn.tn.mps as mps
import time


def build_mpo_hopping_Hterm(J, sym, config_kwargs):
    """
    Fermionic hopping Hamiltonian on N sites with hoppings at arbitrary range.

    The upper triangular part of N x N matrix J defines hopping amplitudes,
    and the diagonal defines on-site chemical potentials.
    """
    ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)
    c, cp, occ = ops.c(), ops.cp(), ops.n()
    #
    Hterms = []  # list of Hterm(amplitude, positions, operators)
    #
    # Each Hterm corresponds to a single product of local operators.
    # Hamiltonian is a sum of such products.
    #
    N = len(J)
    #
    # chemical potential on site n
    #
    for n in range(N):
        if abs(J[n][n]) > 0:
            Hterms.append(mps.Hterm(amplitude=J[n][n],
                                    positions=[n],
                                    operators=[occ]))
    #
    # hopping term between sites m and n
    for m in range(N):
        for n in range(m + 1, N):
            if abs(J[m][n]) > 0:
                Hterms.append(mps.Hterm(amplitude=J[m][n],
                                        positions=(m, n),
                                        operators=(cp, c)))
                Hterms.append(mps.Hterm(amplitude=np.conj(J[m][n]),
                                        positions=(n, m),
                                        operators=(cp, c)))
    #
    # We need an identity MPO operator.
    #
    I = mps.product_mpo(ops.I(), N)
    #
    # Generate MPO for Hterms
    #
    H = mps.generate_mpo(I, Hterms)
    #
    return H


def bench_mpo_generator(config_kwargs={}, N=64):
    J = np.triu(np.random.rand(N, N))
    #
    # we make it hermitian
    J += np.tril(J.T, 1).conj()

    # N^2 terms in Hamiltonian makes it expensive for this generator.
    t0 = time.time()
    H = build_mpo_hopping_Hterm(J, 'U1', config_kwargs)
    t1 = time.time()

    print("N =", N)
    print("Bond dimensions:", H.get_bond_dimensions())
    print("Time [sek]:", t1 - t0)


def test_build_mpo_hopping_Hterm(config_kwargs, tol=1e-12):
    """ test example generating mpo using Hterm """
    N = 25
    J = np.triu(np.random.rand(N, N))
    #
    # we make it hermitian
    J += np.tril(J.T, 1).conj()

    for sym, n in [('Z2', (0,)), ('U1', (N // 2,))]:
        H = build_mpo_hopping_Hterm(J, sym=sym, config_kwargs=config_kwargs)
        ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)
        I = mps.product_mpo(ops.I(), N)
        psi = mps.random_mps(I, D_total=16, n=n).canonize_(to='last').canonize_(to='first')

        E1 = mps.vdot(psi, H, psi)

        cp, c = ops.cp(), ops.c()
        ecpc = mps.measure_2site(psi, cp, c, psi, bonds='a')
        assert len(ecpc) == N * N
        E2 = sum(J[n1, n2] * v for (n1, n2), v in ecpc.items())

        assert pytest.approx(E1.item(), rel=tol) == E2.item()

        eccp = mps.measure_2site(psi, c, cp, psi, bonds="<")  # defaulf bonds
        assert all(abs(eccp[k].conj() + ecpc[k]) < tol for k in eccp)
        assert len(eccp) == N * (N - 1) / 2


def test_generate_mpo_basic(config_kwargs):
    """ test generate_mpo on simple fermionic examples """
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)

    I1 = mps.product_mpo(ops.I(), N=4)  # identity as single operator
    I2 = mps.generate_mpo([ops.I(), ops.I()], N=4)  # no terms provided; identity as a list
    I3 = mps.generate_mpo([ops.I(), ops.I(), ops.I(), ops.I()], terms=[])  # empty terms; identity as a list; N not provided
    I4 = mps.generate_mpo(ops.I(), terms=[mps.Hterm(amplitude=4)], N=4)  # single term with no operators

    assert (I1 - I2).norm() < 1e-12
    assert (I1 - I3).norm() < 1e-12
    assert (4 * I1 - I4).norm() < 1e-12

    # local operators and state for testing
    c, cp, n = ops.c(), ops.cp(), ops.n()
    v0, v1 = ops.vec_n(0), ops.vec_n(1)
    v1101 = mps.product_mps([v1, v1, v0, v1])
    v0111 = mps.product_mps([v0, v1, v1, v1])
    psi = v0111 + v1101

    # test product operator with fermionic anticommutation relations
    Hterms1 = [mps.Hterm(1., positions=[0, 3, 2], operators=[cp, n, c])]
    O1 = mps.generate_mpo(ops.I(), Hterms1, N=4)
    Hterms2 = [mps.Hterm(1., positions=[2, 3, 0, 3], operators=[c, cp, cp, c])]
    O2 = mps.generate_mpo(I1, Hterms2)
    assert abs(O1.norm() - 2 ** 0.5) < 1e-12
    assert abs(O2.norm() - 2 ** 0.5) < 1e-12
    assert (O1 - O2).norm() < 1e-12
    assert abs(mps.vdot(psi, O1, psi) + 1) < 1e-12

    # two identical terms
    O12 = mps.generate_mpo(ops.I(), Hterms1 + Hterms2, N=4)
    assert (2 * O1 - O12).norm() < 1e-12
    O12 = mps.generate_mpo(ops.I(), Hterms1 + Hterms1, N=4)
    assert (2 * O1 - O12).norm() < 1e-12

    # make O hetmitian
    Hterms = [mps.Hterm(1., positions=[2, 3, 0, 3], operators=[c, cp, cp, c]),
               mps.Hterm(1., positions=[0, 3, 2, 3], operators=[c, cp, cp, c])]
    O = mps.generate_mpo(I1, Hterms)
    assert abs(mps.vdot(psi, O, psi) + 2) < 1e-12
    psir = mps.random_mps(I1, n=2, D_total=16, dtype='complex128')
    tmp = mps.vdot(psir, O, psir).item()
    assert abs(tmp.imag) < 1e-12  # expectation value is real for hermitian O

    # more terms changing particle number
    v1100 = mps.product_mps([v1, v1, v0, v0])
    v1001 = mps.product_mps([v1, v0, v0, v1])
    v0110 = mps.product_mps([v0, v1, v1, v0])
    psi2 = v1100 - v1001 - 1j * v0110
    # psi = v0111 + v1101
    Hterms = [mps.Hterm(1., positions=[2, 0, 1], operators=[c, cp, c]), # 0111 + 1101 -> 0011 - 1001 -> 1011 -> -1001
              mps.Hterm(1j, positions=[0, 2, 3], operators=[cp, c, c]), # 0111 + 1101 -> 0110 + 1100 -> -0100 -> -1100
              mps.Hterm(1, positions=[2, 3, 2], operators=[cp, c, c])]  # 0111 + 1101 -> -0101 -> 0100 -> -0110

    O23 = mps.generate_mpo(I1, Hterms)
    assert abs(mps.vdot(psi2, O23, psi) - 1 + 2j) < 1e-12


def test_generate_mpo_raise(config_kwargs):
    ops = yastn.operators.Spin12(sym='U1', **config_kwargs)
    I = mps.product_mpo(ops.I(), N=7)

    with pytest.raises(yastn.YastnError,
                       match="Hterm: positions should be in 0, 1, ..., N-1."):
        Hterms = [mps.Hterm(1., positions=[20], operators=[ops.sz()])]
        mps.generate_mpo(I, Hterms)

    with pytest.raises(yastn.YastnError,
                       match="Hterm: numbers of provided positions and operators do not match."):
        Hterms = [mps.Hterm(1., positions=[2], operators=[ops.sz(), ops.sz()])]
        mps.generate_mpo(I, Hterms)

    with pytest.raises(yastn.YastnError,
                       match="Hterm: operator should be a Tensor with ndim=2 and signature matching identity I at the corresponding site."):
        Hterms = [mps.Hterm(1., positions=[2], operators=[ops.sz().conj()])]
        mps.generate_mpo(I, Hterms)

    with pytest.raises(yastn.YastnError,
                       match="generate_mpo: provided terms do not all add up to the same total charge."):
        Hterms = [mps.Hterm(1., positions=[3], operators=[ops.sp()]),
                  mps.Hterm(1., positions=[3], operators=[ops.sm()])]
        mps.generate_mpo(I, Hterms)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
    bench_mpo_generator(N=64)
