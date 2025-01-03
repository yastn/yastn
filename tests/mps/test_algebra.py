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
import numpy as np
import pytest
import yastn
import yastn.tn.mps as mps


def test_addition_example(config_kwargs, tol=1e-12):
    # Define random MPS's without any symmetry
    ops = yastn.operators.Qdit(d=2, **config_kwargs)
    I = mps.product_mpo(ops.I(), N=8)
    psi0 = mps.random_mps(I, D_total=5)
    psi1 = mps.random_mps(I, D_total=3, dtype='complex128')

    # We want to calculate: res = psi0 + 2 * psi1.
    # There is a couple of ways:

    # A/
    resA = mps.add(psi0, 2.0 * psi1)

    # B/
    resB = mps.add(psi0, psi1, amplitudes=[1.0, 2.0])

    # C/
    resC = psi0 + 2.0 * psi1

    nresA, nresB, nresC = resA.norm(), resB.norm(), resC.norm()
    assert abs(mps.vdot(resA, resB) / (nresA * nresB) - 1) < tol
    assert abs(mps.vdot(resA, resC) / (nresA * nresC) - 1) < tol
    assert (x.get_bond_dimensions() == (1, 8, 8, 8, 8, 8, 8, 8, 1)
            for x in (resA, resB, resC))


def test_add(config_kwargs, tol=1e-8):
    ops = yastn.operators.SpinfulFermions(sym='U1xU1', **config_kwargs)
    generate = mps.Generator(N=9, operators=ops)

    psi0 = generate.random_mps(D_total=15, n=(3, 5))
    psi1 = generate.random_mps(D_total=19, n=(3, 5))
    check_add(psi0, psi1, tol)

    psi0 = generate.random_mpo(D_total=12)
    psi1 = generate.random_mpo(D_total=11)
    check_add(psi0, psi1, tol)

    psi_zero = psi0 * 0
    assert psi_zero.norm() < tol
    assert psi_zero.factor == 0  # multiply by 0, included in psi.factor


def check_add(psi0, psi1, tol):
    """ series of test of mps.add performed on provided psi0 and psi1"""
    out1 = mps.add(psi0, psi1, amplitudes=[1., 2.])
    out2 = (1.0 * psi0) + (2.0 * psi1)
    p0 = mps.measure_overlap(psi0, psi0)
    p1 = mps.measure_overlap(psi1, psi1)
    p01 = mps.measure_overlap(psi0, psi1)
    p10 = mps.measure_overlap(psi1, psi0)
    for out in (out1, out2):
        o1 = mps.measure_overlap(out, out)
        assert abs(o1 - p0 - 4 * p1 - 2 * p01 - 2 * p10) < tol


def test_multiply(config_kwargs):
    # Define random MPS's without any symmetry
    N = 4
    psi = mps.random_dense_mps(N=N, D=5, d=2, **config_kwargs)
    H1 = mps.random_dense_mpo(N=N, D=3, d=2, **config_kwargs)
    H2 = mps.random_dense_mpo(N=N, D=4, d=2, **config_kwargs)

    H1psi = H1 @ psi
    assert H1psi.get_bond_dimensions() == (1, 15, 15, 15, 1)
    assert (H1psi[n].s == (-1, 1, 1) for n in H1psi.sweep())
    assert H1psi.nr_phys == 1
    H1H2 = H1 @ H2
    assert H1H2.get_bond_dimensions() == (1, 12, 12, 12, 1)
    assert (H1H2[n].s == (-1, 1, 1, -1) for n in H1H2.sweep())
    assert H1H2.nr_phys == 2


def test_multiplication_example_gs(config_kwargs, tol=1e-12):
    """
    Calculate ground state and tests, within eigen-condition,
    functions mps.multiply, __mul__, mps.zipper and mps.add

    This test presents multiplication as part of the DMRG study.
    We use multiplication to get expectation values from a state.
    Knowing the exact solution, we will compare it to the value we obtain.
    """
    N = 7
    Eng = -3.427339492125
    t, mu = 1.0, 0.2
    #
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    #
    # The Hamiltonian is obtained using Hterm.
    #
    I = mps.product_mpo(ops.I(), N)  # identity MPO
    terms = []
    for i in range(N - 1):
        terms.append(mps.Hterm(t, (i, i+1), (ops.cp(), ops.c())))
        terms.append(mps.Hterm(t, (i+1, i), (ops.cp(), ops.c())))
    for i in range(N):
        terms.append(mps.Hterm(mu, [i], [ops.cp() @ ops.c()]))
    H = mps.generate_mpo(I, terms)
    #
    # To standardize this test we fix a seed for random MPS we use.
    #
    ops.random_seed(seed=0)
    #
    # In this example we use yastn.Tensor with U1 symmetry.
    #
    total_charge = 3
    psi = mps.random_mps(I, D_total=8, n=total_charge)
    #
    # We set truncation for DMRG and run the algorithm in '2site' version.
    #
    opts_svd = {'D_total': 8}
    opts_dmrg = {'max_sweeps': 20, 'Schmidt_tol': 1e-14}
    out = mps.dmrg_(psi, H, method='2site', **opts_dmrg, opts_svd=opts_svd)
    out = mps.dmrg_(psi, H, method='1site', **opts_dmrg)
    #
    # Test if we obtained the exact solution for energy?:
    #
    assert abs(out.energy - Eng) < tol
    #
    # We should have the ground state.
    # Now we calculate the variation of energy <H^2>-<H>^2=<(H-Eng)^2>
    # to check if DMRG converged properly to tol.
    # There are a few ways to do that:
    #
    # case 1/
    # Find approximate Mps representing H @ psi. This is done in two steps.
    # First, zipper function gives a good first approximation
    # though a series of svd.
    #
    Hpsi = mps.zipper(H, psi, opts_svd={"D_total": 8})  # here, Hpsi is normalized
    #
    # Second, compression is optimizing the overlap
    # between approximation and a product H @ psi.
    #
    mps.compression_(Hpsi, (H, psi), method='2site', max_sweeps=5,
                     Schmidt_tol=1e-6, opts_svd={"D_total": 8},
                     normalize=False)  # norm of Hpsi approximates norm of H @ psi
    #
    # Use mps.norm() to get variation.
    # Norm canonizes a copy of the state and, close to zero,
    # is more precise than direct contraction using vdot.
    #
    p0 = Eng * psi - Hpsi
    assert p0.norm() < tol
    #
    # case 2/
    # Here H @ psi is calculated exactly with resulting
    # bond dimension being a product of bond dimensions of H and psi.
    #
    Hpsi = mps.multiply(H, psi)
    p0 = Eng * psi - Hpsi
    assert p0.norm() < tol
    #
    # Equivalently we can call.
    Hpsi = H @ psi
    p0 = Eng * psi - Hpsi
    assert p0.norm() < tol


def test_add_multiply_raise(config_kwargs):
    # Define random MPS's without any symmetry
    ops = yastn.operators.Qdit(d=2, **config_kwargs)

    I = ops.I()
    psi8 = mps.random_mps(mps.product_mpo(I, N=8), D_total=5)
    H8 = mps.random_mpo(mps.product_mpo(I, N=8), D_total=3)
    psi7 = mps.random_mps(mps.product_mpo(I, N=7), D_total=7)

     # numpy is creating some issues with __mul__ that has to be circumvented
    assert H8.factor == 1
    tmp = np.int64(4) * H8
    assert tmp.factor == 4

    with pytest.raises(yastn.YastnError):
        H8 @ 1
        # multiply requres two MpsMpoOBC-s.
    with pytest.raises(yastn.YastnError):
        np.int64(4) + H8
        # Only np.float * Mps is supported; add was called.
    with pytest.raises(yastn.YastnError):
        H8 + 1
        # All added states should be MpsMpoOBC-s.
    with pytest.raises(yastn.YastnError):
        mps.add(psi8, psi8, amplitudes=[2, 3, 4])
        # Number of Mps-s to add must be equal to the number of coefficients in amplitudes.
    with pytest.raises(yastn.YastnError):
        mps.add(psi7, psi8)
        # All MpsMpoOBC to add must have equal number of sites.
    with pytest.raises(yastn.YastnError):
        mps.add(H8, psi8)
        #  All states to add should be either Mps or Mpo.
    H8c = H8.shallow_copy()
    H8c.orthogonalize_site_(4, to='last')
    with pytest.raises(yastn.YastnError):
        mps.add(H8c, H8c)
        #  Absorb central block of MpsMpoOBC-s before calling add.

    with pytest.raises(yastn.YastnError):
        H8 @ psi7
        #  MpsMpoOBC-s to multiply must have equal number of sites.
    with pytest.raises(yastn.YastnError):
        H8c @ psi8
        # Absorb central blocks of MpsMpoOBC-s before calling multiply.
    with pytest.raises(yastn.YastnError):
        psi8 @ H8
        # Multiplication by MPS from left is not supported.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
