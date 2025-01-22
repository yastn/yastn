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
""" examples for addition of the Mps-s """
import pytest
import numpy as np
import yastn
import yastn.tn.mps as mps


def build_aklt_state_manually(config_kwargs, N=5, lvec=(1, 0), rvec=(0, 1)):
    """
    Example for Spin-1 AKLT state of ``N`` sites.
    Initialize MPS tensors by hand.
    Allow inputing boundary vectors ``lvec`` an ``rvec`` for an MPS with OBC.
    """
    # Prepare rank-3 on-site tensor with virtual dimensions 2
    # and physical dimension dim(Spin-1)=3
    #                _
    # dim(left)=2 --|A|-- dim(right)=2
    #                |
    #           dim(Spin-1)=3
    #
    config = yastn.make_config(sym='none', **config_kwargs)
    A = yastn.Tensor(config, s=(-1, 1, 1))
    s13, s23 = np.sqrt(1. / 3), np.sqrt(2. / 3)
    A.set_block(Ds=(2, 3, 2), val=[[[0, s23], [-s13, 0], [0, 0]],
                                   [[0, 0], [0, s13], [-s23, 0]]])
    #
    # as well as left and right boundary vectors
    #
    LV = yastn.Tensor(config, s=(-1, 1))
    LV.set_block(Ds=(1, 2), val=lvec)
    RV = yastn.Tensor(config, s=(-1, 1))
    RV.set_block(Ds=(2, 1), val=rvec)
    #
    # We initialize empty MPS for N sites
    # and assign its on-site tensors one-by-one.
    #
    psi = mps.Mps(N)
    for n in range(N):
        psi[n] = A.copy()
    #
    # Due to open boundary conditions, the first and the last
    # MPS tensors have left and right virtual indices projected to dimension 1
    #
    psi[psi.first] = LV @ psi[psi.first]
    psi[psi.last] = psi[psi.last] @ RV
    #
    # The resulting MPS is not normalized, nor in any canonical form.
    #
    return psi


def test_measure_mps_aklt(config_kwargs, tol=1e-12):
    """ Test measuring MPS expectation values in AKLT state"""
    #
    # AKLT state with open boundary conditions and N = 31 sites.
    #
    N = 31
    psi = build_aklt_state_manually(config_kwargs, N, lvec=(1, 0), rvec=(1, 0))
    #
    # We verify transfer matrix in the middle of AKLT state
    # to specify the effect of lvec and rvec
    #
    A = psi[N // 2]  # read MPS tensor
    T = yastn.ncon((A, A.conj()), ((-0, 1, -2), (-1, 1, -3)))
    T = T.fuse_legs(axes=((0, 1), (2, 3)))
    Tref = np.array([[1,  0,  0,  2],
                     [0, -1,  0,  0],
                     [0,  0, -1,  0],
                     [2,  0,  0,  1]])
    assert np.allclose(T.to_numpy(), Tref / 3, atol=tol)
    #
    # psi1 is not normalized.
    # Norm is very close to sqrt(0.5) for this system size.
    #
    norm = psi.norm()  # use canonization to calculate norm
    norm2 = mps.vdot(psi, psi)  # directly contract mps squared
    #
    assert abs(norm - np.sqrt(0.5)) < tol
    assert abs(norm2 - 0.5) < tol
    #
    # Initialize dense Spin1 operators for expectation values calculations
    #
    ops = yastn.operators.Spin1(sym='dense', **config_kwargs)
    #
    # Measure Sz at the first and last sites
    #
    ez = mps.measure_1site(psi, {0: ops.sz(), N-1: ops.sz()}, psi)
    assert len(ez) == 2 and isinstance(ez, dict)
    assert abs(ez[0] - 1./3) < tol  # psi is not normalized
    assert abs(ez[N - 1] + 1./3) < tol
    #
    #  the same with other syntaxes
    #
    ez_bis = mps.measure_1site(psi, ops.sz(), psi, sites=[0, N-1])
    assert ez == ez_bis
    ez0 = mps.measure_1site(psi, ops.sz(), psi, sites=0)  # this return a number
    assert abs(ez0 - 1./3) < tol
    #
    #  with normalised state
    psin = psi / norm
    ez = mps.measure_1site(psin, ops.sz(), psin)  # calculate sz on all sites
    assert len(ez) == N and isinstance(ez, dict)
    assert abs(ez[0] - 2. / 3) < tol  # psin is normalized
    #
    # Calculate <Sz_i Sz_j> for (i, j) = (n, n+r) with r=1
    ezz = mps.measure_2site(psin, ops.sz(), ops.sz(), psin, bonds='r1')
    assert len(ezz) == N - 1 and isinstance(ezz, dict)
    assert all(abs(ezznn + 4. / 9) < tol for ezznn in ezz.values())
    #
    # here calculate a single bond, returning a number
    ezz12 = mps.measure_2site(psin, ops.sz(), ops.sz(), psin, bonds=(1, 2))
    assert abs(ezz12 + 4. / 9) < tol
    #
    #  Measure string operator; exp (i pi * Sz) = I - 2 * abs(Sz)
    #
    esz = ops.I() - 2 * abs(ops.sz())
    Hterm = mps.Hterm(amplitude=1,
                      positions=(10, 11, 12, 13, 14, 15),
                      operators=(ops.sz(), esz, esz, esz, esz, ops.sz()))
    I = mps.product_mpo(ops.I(), N)
    Ostring = mps.generate_mpo(I, [Hterm])  # MPO a string correlator
    Estring = mps.vdot(psin, Ostring, psin)
    assert abs(Estring + 4. / 9) < tol


@pytest.mark.parametrize('sym', ['dense', 'Z3', 'U1'])
def test_mps_spectrum_ghz(config_kwargs, sym, tol=1e-12):
    """ Test measuring Schmidt spectrum and entropy of mps. """
    N = 8
    # consider 3-dimensional local Hilbert space
    ops = yastn.operators.Spin1(sym=sym, **config_kwargs)
    #
    # take 3 orthonormal product states
    #
    psi1 = mps.product_mps(ops.vec_z(val=+0), N)
    psi2 = mps.product_mps([ops.vec_z(val=+1), ops.vec_z(val=-1)], N)
    psi3 = mps.product_mps([ops.vec_z(val=-1), ops.vec_z(val=+1)], N)
    #
    # For Z3 and U1, their sum has a well-defined
    # global charge only for even N
    #
    psi = mps.add(psi1, psi2, psi3, amplitudes=(0.5, 0.25, -0.25))
    #
    # state psi is not normalized
    #
    assert abs(psi.norm() - np.sqrt(0.375)) < tol
    #
    # calculate Schmidt values; Schmidt values are properly normalized
    #
    schmidt_spectra = psi.get_Schmidt_values()
    assert len(schmidt_spectra) == N + 1 and isinstance(schmidt_spectra, list)
    assert all(abs(sv.norm() - 1) < tol for sv in schmidt_spectra)
    #
    bds = (1,) + (3,) * (N-1) + (1,)  # (1, 3, 3, 3, 3, 3, 3, 3, 1) for N = 8
    assert psi.get_bond_dimensions() == bds
    # schmidt values are stored as diagonal yastn.Tensors
    assert all(sv.isdiag and sv.get_shape() == (bd, bd)
               for sv, bd in zip(schmidt_spectra, bds))
    #
    # mps.get_Schmidt_values is used by mps.get_entropy
    #
    entropies = psi.get_entropy()
    assert len(entropies) == N + 1 and isinstance(entropies, list)
    #
    # by default calculate von Neuman entropy (base-2 log)
    #
    assert all(abs(ent - (np.log2(6) - 4/3)) < tol for ent in entropies[1:-1])
    assert abs(entropies[0]) < tol and abs(entropies[-1]) < tol
    #
    # Renyi entropy with alpha=2 (and base-2 log)
    #
    entropies2 = psi.get_entropy(alpha=2)
    assert all(abs(ent - 1.) < tol for ent in entropies2[1:-1])
    assert abs(entropies2[0]) < tol and abs(entropies2[-1]) < tol
    #
    # the state psi is not affected by calculations
    #
    assert abs(psi.norm() - np.sqrt(0.375)) < tol
    #
    # similar thing can be done for MPO


@pytest.mark.parametrize('sym', ['dense', 'Z3', 'U1'])
def test_mpo_spectrum(config_kwargs, sym, tol=1e-12):
    """ Test measuring Schmidt spectrum and entropy of mps. """
    N = 8
    ops = yastn.operators.Spin1(sym=sym, **config_kwargs)
    #
    # take 3 orthogonal operator-product states
    #
    psi1 = mps.product_mpo(ops.sz(), N)
    psi2 = mps.product_mpo([ops.sm(), ops.sp()], N)
    psi3 = mps.product_mpo([ops.sp(), ops.sm()], N)
    #
    # they are not normalized
    assert abs(psi1.norm() - 2. ** (N / 2)) < tol
    assert abs(psi2.norm() - 2. ** N) < tol
    assert abs(psi3.norm() - 2. ** N) < tol
    #
    # For Z2 and U1, their sum has a well-defined
    # global charge only for even N
    #
    psi = mps.add(psi1, psi2, psi3, amplitudes=(2. ** (N / 2), 1, 1))
    #
    # state psi is not normalized
    #
    assert abs(psi.norm() - np.sqrt(3) * 2. ** N) < tol * 2. ** N
    #
    # calculate Schmidt values; Schmidt values are properly normalized
    #
    schmidt_spectra = psi.get_Schmidt_values()
    assert len(schmidt_spectra) == N + 1 and isinstance(schmidt_spectra, list)
    assert all(abs(sv.norm() - 1) < tol for sv in schmidt_spectra)
    #
    bds = (1,) + (3,) * (N - 1) + (1,)  # (1, 3, 3, 3, 3, 3, 3, 3, 1) for N = 8
    assert psi.get_bond_dimensions() == bds
    #
    # schmidt values are stored as diagonal yastn.Tensors
    #s
    assert all(sv.isdiag and sv.get_shape() == (bd, bd) for sv, bd in zip(schmidt_spectra, bds))
    #
    # mps.get_Schmidt_values is used by mps.get_entropy
    # Renyi entropy with alpha=0.5 (and base-2 log)
    #
    entropies = psi.get_entropy(alpha=0.5)
    #
    # here all Schmidt values are the same
    #
    assert all(abs(ent - np.log2(3)) < tol for ent in entropies[1:-1])
    assert abs(entropies[0]) < tol and abs(entropies[-1]) < tol


@pytest.mark.parametrize("sym", ['Z2', 'U1'])
def test_measure_fermions_and_unbalanced(config_kwargs, sym, tol=1e-12):
    """ Initialize small MPS and measure fermionic correlators. Additionally text measure_2site syntax. """

    if sym == 'U1' or sym == 'Z2':
        ops = yastn.operators.SpinlessFermions(sym=sym, **config_kwargs)
        v0, v1 = ops.vec_n(0), ops.vec_n(1)
    elif sym == 'U1xU1':
        # here two spicies commute; we can have particles in "d" and still match spinless case
        ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
        v0, v1 = ops.vec_n((0, 1)), ops.vec_n((1, 1))
    elif sym == 'U1xU1xZ2':
        # here two spicies anti-commute; no particles in "d" to match spinless case
        ops = yastn.operators.SpinfulFermions(sym=sym, **config_kwargs)
        v0, v1 = ops.vec_n((0, 0)), ops.vec_n((1, 0))
    I = mps.product_mpo(ops.I(), 4)

    v1101 = mps.product_mps([v1, v1, v0, v1])
    v0111 = mps.product_mps([v0, v1, v1, v1])
    psi3 = v1101 + v0111
    npsi3 = mps.vdot(psi3, psi3)
    assert abs(npsi3 - 2.0) < tol
    #
    # psi3 is not normalized;  measure_2site, measure_1site do not normalise the state.
    #
    psi3_cp_c_psi3 = np.array([[ 1.0, 0.0, -1.0, 0.0],
                               [ 0.0, 2.0,  0.0, 0.0],
                               [-1.0, 0.0,  1.0, 0.0],
                               [ 0.0, 0.0,  0.0, 2.0]])
    #
    en33 = mps.measure_1site(psi3, ops.n(), psi3)  # all sites
    ecpc33 = mps.measure_2site(psi3, ops.cp(), ops.c(), psi3, bonds='a')  # all pairs of sites
    eccp33 = mps.measure_2site(psi3, ops.c(), ops.cp(), psi3, bonds='a')  # bonds with i <= j  # move to syntax
    assert len(en33) == len(psi3)
    assert len(ecpc33) == len(psi3) * len(psi3)
    assert len(eccp33) == len(psi3) * len(psi3)
    assert all(abs(v - psi3_cp_c_psi3[k, k]) < tol for k, v in en33.items())
    for k in eccp33:
        OP = mps.generate_mpo(I, [mps.Hterm(1, k, [ops.c(), ops.cp()])])
        assert abs(eccp33[k] - mps.vdot(psi3, OP, psi3)) < tol
        assert abs(eccp33[k] - npsi3 * (k[0] == k[1]) + psi3_cp_c_psi3[k].conjugate()) < tol
        assert abs(ecpc33[k] - psi3_cp_c_psi3[k]) < tol
    #
    # transitions between states with different partice number
    #
    v0101 = mps.product_mps([v0, v1, v0, v1])
    v1001 = mps.product_mps([v1, v0, v0, v1])
    psi2 = 1j * v0101 - v1001
    assert abs(mps.vdot(psi2, psi2) - 2.0) < tol
    #
    ec23 = mps.measure_1site(psi2, ops.c(), psi3)
    ecp32 = mps.measure_1site(psi3, ops.cp(), psi2)
    psi2_c_psi3 = [-1.0j, 1.0, 1.0j, 0.0]
    #
    for k in ec23:
        assert abs(psi2_c_psi3[k] - ec23[k]) < tol
        OP = mps.generate_mpo(I, [mps.Hterm(1, [k], [ops.c()])])
        assert abs(ec23[k] - mps.vdot(psi2, OP, psi3)) < tol
        OP = mps.generate_mpo(I, [mps.Hterm(1, [k], [ops.cp()])])
        assert abs(psi2_c_psi3[k] - ecp32[k].conj()) < tol
        assert abs(ecp32[k] - mps.vdot(psi3, OP, psi2)) < tol
    #
    v0100 = mps.product_mps([v0, v1, v0, v0])
    v0010 = mps.product_mps([v0, v0, v1, v0])
    v0001 = mps.product_mps([v0, v0, v0, v1])
    psi1 = v0100 + v0010 + v0001
    assert abs(mps.vdot(psi1, psi1) - 3.0) < tol
    #
    psi1_c_c_psi3 = np.array([[ 0.0, -1.0,  0.0,  1.0],
                              [ 1.0,  0.0, -1.0,  1.0],
                              [ 0.0,  1.0,  0.0, -1.0],
                              [-1.0, -1.0,  1.0,  0.0]])
    #
    ecc13 = mps.measure_2site(psi1, ops.c(), ops.c(), psi3, bonds='a')
    ecpcp31 = mps.measure_2site(psi3, ops.cp(), ops.cp(), psi1, bonds='a')

    for k in ecc13:
        if k[0] != k[1]:  # here exclude diagonal, as there is problem with resolving zero operator
            OP = mps.generate_mpo(I, [mps.Hterm(1, k, [ops.c(), ops.c()])])
            assert abs(ecc13[k] - mps.vdot(psi1, OP, psi3)) < tol
        assert abs(ecc13[k] - psi1_c_c_psi3[k]) < tol
        assert abs(ecpcp31[k] + psi1_c_c_psi3[k].conjugate()) < tol


def test_measure_syntax_raises(config_kwargs):
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    v0, v1 = ops.vec_n(0), ops.vec_n(1)
    v1101 = mps.product_mps([v1, v1, v0, v1])
    v0111 = mps.product_mps([v0, v1, v1, v1])
    psi3 = v1101 + v0111
    N = 4
    #
    #  measure_2site syntax
    #
    out = mps.measure_2site(psi3, {2: ops.cp()}, {0: ops.c(), 2: ops.c(), 3: ops.c()}, psi3, bonds='a')  # limited bonds
    assert all(k in out for k in [(2, 0), (2, 2), (2, 3)]) and len(out) == 3
    #
    out = mps.measure_2site(psi3, ops.cp(), ops.c(), psi3, bonds='r-1p')  # step -1 and PBC
    assert all(k in out for k in [(1, 0), (2, 1), (3, 2), (0, 3)]) and len(out) == 4
    #
    out = mps.measure_2site(psi3, ops.cp(), ops.c(), psi3, bonds='r-3r-2')  # step -3 and -2
    assert all(k in out for k in [(3, 0), (3, 1), (2, 0)])  and len(out) == 3
    #
    out = mps.measure_2site(psi3, ops.cp(), ops.c(), psi3, bonds='<=')  # i <= j
    assert all(k[0] <= k[1] for k in out ) and len(out) == N * (N + 1) // 2
    #
    out = mps.measure_2site(psi3, ops.cp(), ops.c(), psi3, bonds='>=')  # i >= j
    assert all(k[0] >= k[1] for k in out ) and len(out) == N * (N + 1) // 2
    #
    out = mps.measure_2site(psi3, ops.cp(), ops.c(), psi3)  # i < j  (default)
    assert all(k[0] < k[1] for k in out ) and len(out) == N * (N - 1) // 2
    #
    out = mps.measure_2site(psi3, ops.cp(), ops.c(), psi3, bonds=(0, 2))
    assert abs(out + 1.0) < 1e-12  # out is a number
    #
    out = mps.measure_2site(psi3, ops.cp(), ops.c(), psi3, bonds=[(0, 2)])
    assert abs(out[(0, 2)] + 1.0) < 1e-12 and len(out) == 1 # out is a dict
    #
    out = mps.measure_2site(psi3, {1: ops.cp()}, ops.c(), psi3, bonds=[(0, 2)])
    assert len(out) == 0 # empty dict
    #
    out = mps.measure_2site(psi3, {1: ops.cp()}, ops.c(), psi3, bonds=(0, 2))
    assert len(out) == 0 # empty dict as well
    #
    #  measure_1site syntax
    #
    out = mps.measure_1site(psi3, ops.n(), psi3, sites=[0, 0, 4, 5])  # repeated or out-of-bound sites are dropped
    assert len(out) == 1
    #
    out = mps.measure_1site(psi3, {1: ops.n()}, psi3, sites=[0, 2])
    assert len(out) == 0
    #
    out = mps.measure_1site(psi3, ops.n(), psi3, sites=1)
    assert isinstance(out.item(), float)
    #
    #  measure_1site measure_2site raises
    #
    with pytest.raises(yastn.YastnError):
        out = mps.measure_1site(psi3, {1: ops.n(), 2: ops.c()}, psi3)
        # In mps.measure_1site, all operators in O should have the same charge.
    with pytest.raises(yastn.YastnError):
        out = mps.measure_2site(psi3, {1: ops.n(), 2: ops.c()}, ops.cp(), psi3)
        # In mps.measure_2site, all operators in O should have the same charge.
    with pytest.raises(yastn.YastnError):
        out = mps.measure_2site(psi3, ops.cp(), {1: ops.n(), 2: ops.c()}, psi3)
        # In mps.measure_2site, all operators in P should have the same charge.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])

# if __name__ == "__main__":
#     measure_mps_aklt()
#     for sym in ['dense', 'Z3', 'U1']:
#         mps_spectrum_ghz(sym=sym)
#         test_mpo_spectrum(sym=sym, config=cfg)
#     for sym in ['Z2', 'U1', 'U1xU1', 'U1xU1xZ2']:
#         test_measure_fermions_and_unbalanced(sym=sym, config=cfg)
#     test_measure_syntax_raises()
