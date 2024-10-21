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
try:
    from .configs import config_dense as cfg
except ImportError:
    from configs import config_dense as cfg
# pytest modifies cfg to inject different backends and devices during tests


def build_mpo_hopping_Hterm(J, sym="U1", config=None):
    """
    Fermionic hopping Hamiltonian on N sites with hoppings at arbitrary range.

    The upper triangular part of N x N matrix J defines hopping amplitudes,
    and the diagonal defines on-site chemical potentials.
    """
    opts_config = {} if config is None else \
                  {'backend': config.backend,
                   'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)
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


def bench_mpo_generator(N=64):
    J = np.triu(np.random.rand(N, N))
    #
    # we make it hermitian
    J += np.tril(J.T, 1).conj()

    # N^2 terms in Hamiltonian makes it expensive for this generator.
    t0 = time.time()
    H = build_mpo_hopping_Hterm(J, sym='U1', config=cfg)
    t1 = time.time()

    print("N =", N)
    print("Bond dimensions:", H.get_bond_dimensions())
    print("Time [sek]:", t1 - t0)


def test_build_mpo_hopping_Hterm(config=cfg, tol=1e-12):
    """ test example generating mpo using Hterm """
    opts_config = {} if config is None else \
                  {'backend': config.backend, 'default_device': config.default_device}

    N = 25
    J = np.triu(np.random.rand(N, N))
    #
    # we make it hermitian
    J += np.tril(J.T, 1).conj()

    for sym, n in [('Z2', (0,)), ('U1', (N // 2,))]:
        H = build_mpo_hopping_Hterm(J, sym=sym, config=config)
        ops = yastn.operators.SpinlessFermions(sym=sym, **opts_config)
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


def test_generate_mpo_basic(config=cfg):
    """ test generate_mpo on simple fermionic examples """
    opts_config = {} if config is None else \
                  {'backend': config.backend, 'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.SpinlessFermions(sym='U1', **opts_config)
    I = mps.product_mpo(ops.I(), N=4)
    I2 = mps.generate_mpo(ops.I(), N=4)
    assert (I - I2).norm() < 1e-12

    c, cp, n = ops.c(), ops.cp(), ops.n()
    v0, v1 = ops.vec_n(0), ops.vec_n(1)
    v1101 = mps.product_mps([v1, v1, v0, v1])
    v0111 = mps.product_mps([v0, v1, v1, v1])
    psi = v0111 + v1101

    # test product operator with fermionic anti-commutation relations
    Hterms1 = [mps.Hterm(1., positions=[0, 3, 2], operators=[cp, n, c])]
    O1 = mps.generate_mpo(ops.I(), Hterms1, N=4)
    Hterms2 = [mps.Hterm(1., positions=[2, 3, 0, 3], operators=[c, cp, cp, c])]
    O2 = mps.generate_mpo(I, Hterms2)
    assert abs(O1.norm() - 2 ** 0.5) < 1e-12
    assert abs(O2.norm() - 2 ** 0.5) < 1e-12
    assert (O1 - O2).norm() < 1e-12
    assert abs(mps.vdot(psi, O1, psi) + 1) < 1e-12

    # two identical terms
    O12 = mps.generate_mpo([ops.I(), ops.I()], Hterms1 + Hterms2, N=4)
    assert (2 * O1 - O12).norm() < 1e-12

    # make O hetmitian
    Hterms = [mps.Hterm(1., positions=[2, 3, 0, 3], operators=[c, cp, cp, c]),
               mps.Hterm(1., positions=[0, 3, 2, 3], operators=[c, cp, cp, c])]
    O = mps.generate_mpo(I, Hterms)
    assert abs(mps.vdot(psi, O, psi) + 2) < 1e-12
    psir = mps.random_mps(I, n=2, D_total=16, dtype='complex128')
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

    O23 = mps.generate_mpo(I, Hterms)
    assert abs(mps.vdot(psi2, O23, psi) - 1 + 2j) < 1e-12


def test_generate_mpo_raise(config=cfg):
    opts_config = {} if config is None else \
                  {'backend': config.backend, 'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.Spin12(sym='U1', **opts_config)
    I = mps.product_mpo(ops.I(), N=7)

    with pytest.raises(yastn.YastnError):
        Hterms = [mps.Hterm(1., positions=[20], operators=[ops.sz()])]
        mps.generate_mpo(I, Hterms)
        # position in Hterm should be in 0, 1, ..., N-1
    with pytest.raises(yastn.YastnError):
        Hterms = [mps.Hterm(1., positions=[2], operators=[ops.sz(), ops.sz()])]
        mps.generate_mpo(I, Hterms)
        # Hterm: numbers of positions and operators do not match.
    with pytest.raises(yastn.YastnError):
        Hterms = [mps.Hterm(1., positions=[2], operators=[ops.sz().conj()])]
        mps.generate_mpo(I, Hterms)
        # operator in Hterm should be a matrix with signature matching I at given site
    with pytest.raises(yastn.YastnError):
        Hterms = [mps.Hterm(1., positions=2, operators=ops.sz())]
        mps.generate_mpo(I, Hterms)
        # Hterm: positions and operators should be provided as lists or tuples.
    with pytest.raises(yastn.YastnError):
        Hterms = [mps.Hterm(1., positions=[3], operators=[ops.sp()]),
                  mps.Hterm(1., positions=[3], operators=[ops.sm()])]
        mps.generate_mpo(I, Hterms)
        # generate_mpo: Provided terms do not all have the same total charge.

if __name__ == "__main__":
    test_generate_mpo_basic()
    test_build_mpo_hopping_Hterm()
    test_generate_mpo_raise()
    bench_mpo_generator(N=64)
