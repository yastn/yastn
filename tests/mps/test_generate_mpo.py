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
                                    operators=[ops.n()]))
    #
    # hopping term between sites m and n
    for m in range(N):
        for n in range(m + 1, N):
            if abs(J[m][n]) > 0:
                Hterms.append(mps.Hterm(amplitude=J[m][n],
                                        positions=(m, n),
                                        operators=(ops.cp(), ops.c())))
                Hterms.append(mps.Hterm(amplitude=np.conj(J[m][n]),
                                        positions=(n, m),
                                        operators=(ops.cp(), ops.c())))
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


def test_build_mpo_hopping_Hterm(config=cfg, tol=1e-12):
    """ test example generating mpo using Hterm """
    opts_config = {} if config is None else \
                  {'backend': config.backend, 'default_device': config.default_device}

    N = 35
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


def test_generate_mpo_raise(config=cfg):
    opts_config = {} if config is None else \
                  {'backend': config.backend, 'default_device': config.default_device}
    # pytest uses config to inject various backends and devices for testing
    ops = yastn.operators.Spin12(sym='dense', **opts_config)
    I = mps.product_mpo(ops.I(), N=7)

    II = mps.generate_mpo(I, [])
    assert (I - II).norm() < 1e-12

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
        # operator in Hterm should be a matrix with signature (1, -1)
    with pytest.raises(yastn.YastnError):
        Hterms = [mps.Hterm(1., positions=20, operators=ops.sz())]
        mps.generate_mpo(I, Hterms)
        # Hterm: positions and operators should be provided as lists or tuples.


if __name__ == "__main__":
    test_build_mpo_hopping_Hterm()
    test_generate_mpo_raise()
