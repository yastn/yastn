# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
""" Test evolution of PEPS using an example of quantum Fourier transform. """
import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.mps as mps

tol = 1e-12  #pylint: disable=invalid-name


def mpo_controlled_Rns(ops, N, pc, *args):
    r"""
    Rank-2 MPO, controlled by site pc, building QFT.

    N is the number of sites, pc is control qbit, and *args lists (position, n) of Rn gates.
    """
    I, Z, X = ops.I(), ops.z(), ops.x()
    P0, P1 = (I + Z) / 2, (I - Z) / 2
    Rn = lambda n: P0 + np.exp(2 * 1j * np.pi / 2 ** n) * P1
    H = (X + Z) / np.sqrt(2)  # Hadamard gate
    HP0, HP1 = P0 @ H, P1 @ H

    terms = [mps.Hterm(1 + 0j, [pc], [HP0]),
             mps.Hterm(1 + 0j, [pc] + [p for p, _ in args], [HP1] + [Rn(n) for _, n in args])]
    return mps.generate_mpo(I, terms, N=N)


def mpo_Hadamard(ops, N, p):
    """
    MPO for Hadamard gate at site p.
    """
    I, Z, X = ops.I(), ops.z(), ops.x()
    H = (X + Z) / np.sqrt(2)  # Hadamard gate
    terms = [I] * p + [H] + [I] * (N - p - 1)
    return mps.product_mpo(terms)


def mpo_swap(ops, N, p0, p1):
    """
    MPO for swap gate between sites p0 and p1.
    """
    I, Z, Sp, Sm = ops.I(), ops.z(), ops.sp(), ops.sm()
    P0, P1 = (I + Z) / 2, (I - Z) / 2

    terms = [mps.Hterm(1, [p0, p1], [P0, P0]),
             mps.Hterm(1, [p0, p1], [Sp, Sm]),
             mps.Hterm(1, [p0, p1], [Sm, Sp]),
             mps.Hterm(1, [p0, p1], [P1, P1])]

    return mps.generate_mpo(I, terms, N=N)


def qft_matrix(N):
    """
    Generate MPO of swap gate between sites p0 and p1.
    """
    NN = 2 ** N
    w = np.exp(2 * np.pi * 1j / NN)
    qftm = np.zeros((NN, NN), dtype=np.complex128)

    for x in range(NN):
        for k in range(NN):
            qftm[x, k] = w ** (x * k)
    return qftm / np.sqrt(NN)


def swap_fuse_numpy(ten):
    """
    Bit reversal on legs 1,3,...2N-1 and fuse into matrix

              1     3        2N-1
              |     |          |
           ┌──┴─────┴── ... ───┴──┐
    ten =  |          MPO         |
           └──┬─────┬── ... ───┬──┘
              |     |          |
              0     2        2N-2
    """
    N = ten.ndim // 2
    axes = ([2 * ( N - n - 1) for n in range(N)], [2 * n + 1 for n in range(N)])
    return ten.fuse_legs(axes=axes).to_numpy()


def test_qft_mpo(config_kwargs, N=6):
    """
    Test that MPO representation of QFT is correct.
    """
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    #
    # MPOs representing QFT circuit
    Rns = []
    for n in range(N - 1):
        args = [(p, m) for m, p in enumerate(range(n + 1, N), start=2)]
        Rns.append(mpo_controlled_Rns(ops, N, n, *args))
    HNm1 = mpo_Hadamard(ops, N, N-1)
    #
    # produce QFT = H(N-1) @ R(N-2) @ .. @ R(1) @ R(0) (up to bit reversal)
    qft = HNm1
    for Rn in Rns[::-1]:
        qft = qft @ Rn
    #
    # compress bond dimension
    qft.canonize_(to='last', normalize=False)
    qft.truncate_(to='first', opts_svd={'tol': 1e-12}, normalize=False)
    #
    # dense representation, doing bit reversal
    qftt = swap_fuse_numpy(qft.to_tensor())
    #
    # exact reference
    qft_ref = qft_matrix(N)
    assert np.allclose(qft_ref, qftt)
    #
    # swaps = [mpo_swap(ops, N, n, N - 1 - n) for n in range(N // 2)]
    # swap = swaps[0]
    # for op in swaps[1:]:
    #     swap = swap @ op


def test_peps_evolution_qft(config_kwargs):
    """
    Generate PEPS encoding QFT of 6 sites by application of few MPO gates.
    """
    #
    ops = yastn.operators.Spin12(sym='dense', **config_kwargs)
    #
    # initialized PEPS in a product state
    g = fpeps.SquareLattice(dims=(2, 3), boundary='obc')
    sites = g.sites()
    N = len(sites)
    #
    # gates that will span peps sites
    gates = []
    Rn0 = mpo_controlled_Rns(ops, 6, 0, (1, 3), (2, 5), (3, 6), (4, 4), (5, 2))
    gates.append(fpeps.Gate(Rn0, [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]))
    #
    Rn1 = mpo_controlled_Rns(ops, 6, 3, (0, 4), (1, 2), (4, 3), (5, 5))
    gates.append(fpeps.Gate(Rn1, [(0, 2), (0, 1), (0, 0), (1, 0), (1, 1), (1, 2)]))
    #
    Rn2 = mpo_controlled_Rns(ops, 4, 1, (0, 2), (2, 3), (3, 4))
    gates.append(fpeps.Gate(Rn2, [(1, 1), (0, 1), (0, 2), (1, 2)]))
    #
    Rn3 = mpo_controlled_Rns(ops, 3, 0, (1, 3), (2, 2))
    gates.append(fpeps.Gate(Rn3, [(1, 1), (1, 2), (0, 2)]))
    #
    Rn4 = mpo_controlled_Rns(ops, 2, 1, (0, 2))
    gates.append(fpeps.Gate(Rn4, [(1, 2), (0, 2)]))
    #
    H5 = mpo_Hadamard(ops, 1, 0)
    gates.append(fpeps.Gate(H5, [(1, 2)]))
    #
    qft_ref = qft_matrix(N)
    #
    # test application of gate without performing truncation
    psi = fpeps.product_peps(g, ops.I())
    for gate in gates:
        psi.apply_gate_(gate)
    psit = swap_fuse_numpy(psi.to_tensor())
    assert np.allclose(qft_ref, psit)
    #
    # test evolution with EnvNTU
    for method in ['NN', 'mpo']:
        psi = fpeps.product_peps(g, ops.I())
        env = fpeps.EnvNTU(psi, which='NN')
        fpeps.evolution_step_(env, gates, symmetrize=False, opts_svd={'D_total': 16}, method=method)
        psit = swap_fuse_numpy(psi.to_tensor())
        psit = psit * qft_ref[0, 0] / psit[0, 0]  # evolution_step_ does not keep the norm
        assert np.allclose(qft_ref, psit)
        #
        # test evolution with EnvBP
        psi = fpeps.product_peps(g, ops.I())
        env = fpeps.EnvBP(psi, which='BP')
        env.iterate_(max_sweeps=5)
        fpeps.evolution_step_(env, gates, symmetrize=False, opts_svd={'D_total': 16}, method=method)
        psit = swap_fuse_numpy(psi.to_tensor())
        psit = psit * qft_ref[0, 0] / psit[0, 0]  # evolution_step_ does not keep the norm
        assert np.allclose(qft_ref, psit)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
