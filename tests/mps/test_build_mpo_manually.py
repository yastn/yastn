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
import numpy as np
import pytest
import yastn
import yastn.tn.mps as mps


def build_mpo_nn_hopping_manually(N, t, mu, sym, config_kwargs):
    """
    Nearest-neighbor hopping Hamiltonian on N sites
    with hopping amplitude t and chemical potential mu,
    H = t * sum_{n=1}^{N-1} (cp_n c_{n+1} + cp_{n+1} c_n)
        + mu * sum_{n=1}^{N} cp_n c_n

    Initialize MPO symmetric tensors by hand with sym = 'dense', 'Z2', or 'U1'.
    Config is used to inject non-default backend and default_device.
    """
    #
    # Build empty MPO for system of N sites
    #
    H = mps.Mpo(N)
    #
    # Depending on the symmetry, define elements of on-site tensor
    #
    # We chose signature convention for indices of the MPO tensor as follows
    #         |
    #         ^(-1)
    #         |
    # (-1)-<-|T|-<-(+1)
    #         |
    #         ^(+1)
    #         |
    #
    # We set fermionic=True for consistency as in tests
    # MPO is compared with operators.SpinlessFermions.
    # backend and device is inherited from config to automitize pytest testing.


    config = yastn.make_config(fermionic=True, sym=sym, **config_kwargs)

    if sym == 'dense':  # no symmetry
        # Basic rank-2 blocks (matrices) of on-site tensors
        cp = np.array([[0, 0], [1, 0]])
        c  = np.array([[0, 1], [0, 0]])
        nn = np.array([[0, 0], [0, 1]])
        ee = np.array([[1, 0], [0, 1]])
        oo = np.array([[0, 0], [0, 0]])

        for n in H.sweep(to='last'):
            # empty tensors
            H[n] = yastn.Tensor(config=config, s=(-1, 1, 1, -1))
            # that gets filled-in
            if n == H.first:
                tmp = np.block([[mu * nn, t * cp, t * c, ee]])
                H[n].set_block(val=tmp, Ds=(1, 2, 4, 2))
            elif n == H.last:
                tmp = np.block([[ee], [c], [cp], [mu * nn]])
                H[n].set_block(val=tmp, Ds=(4, 2, 1, 2))
            else:
                tmp = np.block([[ee, oo, oo, oo],
                                [c, oo, oo, oo],
                                [cp, oo, oo, oo],
                                [mu * nn, t * cp, t * c, ee]])
                H[n].set_block(val=tmp, Ds=(4, 2, 4, 2))

    elif sym == 'Z2':  # Z2 symmetry
        for n in H.sweep(to='last'):
            H[n] = yastn.Tensor(config=config, s=(-1, 1, 1, -1), n=0)
            if n == H.first:
                H[n].set_block(ts=(0, 0, 0, 0),
                               Ds=(1, 1, 2, 1), val=[0, 1])
                H[n].set_block(ts=(0, 1, 0, 1),
                               Ds=(1, 1, 2, 1), val=[mu, 1])
                H[n].set_block(ts=(0, 0, 1, 1),
                               Ds=(1, 1, 2, 1), val=[t, 0])
                H[n].set_block(ts=(0, 1, 1, 0),
                               Ds=(1, 1, 2, 1), val=[0, t])
            elif n == H.last:
                H[n].set_block(ts=(0, 0, 0, 0),
                               Ds=(2, 1, 1, 1), val=[1, 0])
                H[n].set_block(ts=(0, 1, 0, 1),
                               Ds=(2, 1, 1, 1), val=[1, mu])
                H[n].set_block(ts=(1, 1, 0, 0),
                               Ds=(2, 1, 1, 1), val=[1, 0])
                H[n].set_block(ts=(1, 0, 0, 1),
                               Ds=(2, 1, 1, 1), val=[0, 1])
            else:
                H[n].set_block(ts=(0, 0, 0, 0),
                               Ds=(2, 1, 2, 1), val=[[1, 0], [0, 1]])
                H[n].set_block(ts=(0, 1, 0, 1),
                               Ds=(2, 1, 2, 1), val=[[1, 0], [mu, 1]])
                H[n].set_block(ts=(0, 0, 1, 1),
                               Ds=(2, 1, 2, 1), val=[[0, 0], [t, 0]])
                H[n].set_block(ts=(0, 1, 1, 0),
                               Ds=(2, 1, 2, 1), val=[[0, 0], [0, t]])
                H[n].set_block(ts=(1, 1, 0, 0),
                               Ds=(2, 1, 2, 1), val=[[1, 0], [0, 0]])
                H[n].set_block(ts=(1, 0, 0, 1),
                               Ds=(2, 1, 2, 1), val=[[0, 0], [1, 0]])

    elif sym == 'U1':  # U1 symmetry
        for n in H.sweep(to='last'):
            H[n] = yastn.Tensor(config=config, s=(-1, 1, 1, -1), n=0)
            if n == H.first:
                H[n].set_block(ts=(0, 0, 0, 0),
                                 val=[0, 1], Ds=(1, 1, 2, 1))
                H[n].set_block(ts=(0, 1, 0, 1),
                                 val=[mu, 1], Ds=(1, 1, 2, 1))
                H[n].set_block(ts=(0, 0, 1, 1),
                                 val=[t], Ds=(1, 1, 1, 1))
                H[n].set_block(ts=(0, 1, -1, 0),
                                 val=[t], Ds=(1, 1, 1, 1))
            elif n == H.last:
                H[n].set_block(ts=(0, 0, 0, 0),
                                 Ds=(2, 1, 1, 1), val=[1, 0])
                H[n].set_block(ts=(0, 1, 0, 1),
                                 Ds=(2, 1, 1, 1), val=[1, mu])
                H[n].set_block(ts=(1, 1, 0, 0),
                                 Ds=(1, 1, 1, 1), val=[1])
                H[n].set_block(ts=(-1, 0, 0, 1),
                                 Ds=(1, 1, 1, 1), val=[1])
            else:
                H[n].set_block(ts=(0, 0, 0, 0),
                                 Ds=(2, 1, 2, 1), val=[[1, 0], [0, 1]])
                H[n].set_block(ts=(0, 1, 0, 1),
                                 Ds=(2, 1, 2, 1), val=[[1, 0], [mu, 1]])
                H[n].set_block(ts=(0, 0, 1, 1),
                                 Ds=(2, 1, 1, 1), val=[0, t])
                H[n].set_block(ts=(0, 1, -1, 0),
                                 Ds=(2, 1, 1, 1), val=[0, t])
                H[n].set_block(ts=(1, 1, 0, 0),
                                 Ds=(1, 1, 2, 1), val=[1, 0])
                H[n].set_block(ts=(-1, 0, 0, 1),
                                 Ds=(1, 1, 2, 1), val=[1, 0])
    return H


def test_build_mpo_nn_hopping_manually(config_kwargs, tol=1e-12):
    """ test example generating mpo by hand """
    N, t, mu = 10, 1.0, 0.1
    H = {}
    for sym in ['dense', 'Z2', 'U1']:
        H[sym] = build_mpo_nn_hopping_manually(N, t, mu, sym, config_kwargs)

    H_Z2_dense = mps.Mpo(N=N)
    H_U1_dense = mps.Mpo(N=N)
    for n in range(N):
        H_Z2_dense[n] = H['Z2'][n].to_nonsymmetric()
        H_U1_dense[n] = H['U1'][n].to_nonsymmetric()

    assert (H_Z2_dense - H['dense']).norm() < tol
    assert (H_U1_dense - H['dense']).norm() < tol

    # test mpo energy with direct calculation of all terms
    for sym, n in [('Z2', (0,)), ('U1', (N // 2,))]:
        # SpinlessFermions do not support 'dense'
        ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)
        I = mps.product_mpo(ops.I(), N)
        psi = mps.random_mps(I, D_total=16, n=n).canonize_(to='last').canonize_(to='first')

        E1 = mps.measure_mpo(psi, H[sym], psi)

        cp, c = ops.cp(), ops.c()
        epm = mps.measure_2site(psi, cp, c, psi, bonds='r1')
        en = mps.measure_1site(psi, cp @ c, psi)
        E2 = t * sum(2 * epm[(n, n+1)].real for n in range(N - 1))
        E2 += mu * sum(en[n] for n in range(N))
        assert pytest.approx(E1.item(), rel=tol) == E2.item()

        emp = mps.measure_2site(psi, c, cp, psi, bonds='r1')
        assert all(abs(emp[k].conj() + epm[k]) < tol for k in emp)
        assert len(emp) == len(epm) == N - 1

        psi_dense = mps.Mps(N=N)  # test also dense Hamiltonian casting down state psi to dense tensors
        # this cannot be used with measure_2site as it employs swap_gates for fermions
        for n in range(N):
            psi_dense[n] = psi[n].to_nonsymmetric()
        E3 = mps.measure_mpo(psi_dense, H['dense'], psi_dense)
        assert pytest.approx(E1.item(), rel=tol) == E3.item()


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
