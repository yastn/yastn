# Copyright 2026 The YASTN Authors. All Rights Reserved.
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
"""Tests for biorthogonalize_left in yastn.tn.mps._umps.

Covers:
  * dense tensors (sym='none'), N = 1, 2, 3
  * U(1)-symmetric tensors with Spin-1 physical space, N = 1, 2, 3
  * two test scenarios each: MPS biorthogonalized against itself (self-case)
    and two independently drawn random MPS (different-case).
"""
import pytest
import yastn
import numpy as np
from yastn.tn.mps import Mps
from yastn.tn.mps._umps import biorthogonalize_left, verify_biorth
from yastn.initialize import eye

tol= 1e-8

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_dense_umps(N, D, d, cfg, seed):
    """Return a list of N random dense uMPS tensors with bond dim D, phys dim d."""
    cfg.backend.random_seed(seed)
    res= Mps(N)
    for i in range(N):
        res[i] = yastn.rand(cfg, s=(-1, 1, 1), D=(D, d, D))
    return res


def _make_u1_umps(N, D, ops, seed):
    """Return a list of N random U(1) uMPS tensors using the Spin-1 physical space.

    *D* is the bond dimension per charge sector; the physical leg is taken
    from ``ops.space()``, which carries charges t ∈ {−1, 0, 1} (Spin-1, U(1)).
    The uMPS tensor has signature s=(−1, 1, 1) and total charge n=0.
    """
    cfg = ops.config
    phys_leg = ops.space()                                        # s=1, t={-1,0,1}
    bond_leg = yastn.Leg(cfg, s=-1, t=(-1, 0, 1), D=(D, D, D))  # left bond
    cfg.backend.random_seed(seed)
    res= Mps(N)
    for i in range(N):
        res[i]= yastn.rand(cfg, legs=[bond_leg, phys_leg, bond_leg.conj()], n=0)
    return res


def _assert_biorthogonal(P_L, Pbar_L, tol=tol):
    """Assert the left biorthogonality condition.

    After convergence of biorthogonalize_left the following should hold::

        sum_s  P_L[s] · Pbar_L[s]^T  =  λ · I

    which in yastn reads

        P_L.tensordot(Pbar_L, axes=((0, 1), (0, 1)))  ∝  eye

    We test proportionality in a sign-invariant way by normalising both sides
    and checking the minimum of (M/|M| ± I/|I|), which handles real λ of
    either sign (the dominant eigenvalue of the transfer matrix is real for
    physical MPS).
    """
    M = P_L.tensordot(Pbar_L, axes=((0, 1), (0, 1)))
    I_ref = eye(P_L.config, legs=M.get_legs(), isdiag=False)
    M_norm = M.norm()
    assert M_norm > 1e-14, "M is numerically zero — something went wrong."
    # remove global complex phase: M/|M| should equal e^{iφ}·I/|I|
    # compute phase from the inner product <M, I_ref>
    lambda_I_ref = M.tensordot(I_ref, axes=(1,0)) # lambda * I_ref
    delta= (M-lambda_I_ref).norm() / I_ref.norm()  # relative error compared to I_ref
    assert delta < tol, (
        f"Biorthogonality not satisfied: δ = {delta:.4e} > tol={tol}"
    )


# ---------------------------------------------------------------------------
# dense (sym='none') tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [1, 2, 3])
def test_biorthogonalize_left_dense_self(config_kwargs, N):
    """Dense uMPS: biorthogonalize a random MPS against itself (self-case).

    When both top and bottom layers are the same MPS the biorthogonality
    condition reduces to a left-isometry condition.
    """
    cfg = yastn.make_config(sym='none', **config_kwargs)
    D, d = 3, 2 # bond dim, physical dim (matches dense Spin-1)
    umps = _make_dense_umps(N, D, d, cfg, seed=42)

    P_L, Pbar_L, C_LU, C_DL = biorthogonalize_left(umps, umps.conj())

    assert P_L.ndim == 3,   "P_L should be a rank-3 tensor"
    assert Pbar_L.ndim == 3, "Pbar_L should be a rank-3 tensor"
    assert C_LU.ndim == 2,  "C_LU should be a rank-2 gauge matrix"
    assert C_DL.ndim == 2,  "C_DL should be a rank-2 gauge matrix"
    _assert_biorthogonal(P_L, Pbar_L)

    passed, res= verify_biorth(umps, umps.conj(), P_L, Pbar_L, C_LU, C_DL, tol=tol)
    assert passed, f"Biorthogonality verification failed: {res}"


@pytest.mark.parametrize("N", [1, 2, 3])
def test_biorthogonalize_left_dense_different(config_kwargs, N):
    """Dense uMPS: biorthogonalize two independently drawn random MPS."""
    cfg = yastn.make_config(sym='none', **config_kwargs)
    D, d = 3, 2
    umps_top = _make_dense_umps(N, D, d, cfg, seed=42)
    umps_bot = _make_dense_umps(N, D, d, cfg, seed=99).conj()

    P_L, Pbar_L, C_LU, C_DL = biorthogonalize_left(umps_top, umps_bot)

    assert P_L.ndim == 3
    assert Pbar_L.ndim == 3
    assert C_LU.ndim == 2
    assert C_DL.ndim == 2
    _assert_biorthogonal(P_L, Pbar_L)

    passed, res= verify_biorth(umps_top, umps_bot, P_L, Pbar_L, C_LU, C_DL, tol=tol)
    assert passed, f"Biorthogonality verification failed: {res}"


@pytest.mark.parametrize("N", [1,])
def test_biorthogonalize_left_dense_differentDs(config_kwargs, N):
    """Dense uMPS: biorthogonalize two independently drawn random MPS."""
    cfg = yastn.make_config(sym='none', **config_kwargs)
    D, d = 3, 2
    umps_top = _make_dense_umps(N, D, d, cfg, seed=42)
    umps_bot = _make_dense_umps(N, D+1, d, cfg, seed=99).conj()

    P_L, Pbar_L, C_LU, C_DL = biorthogonalize_left(umps_top, umps_bot)

    assert P_L.ndim == 3
    assert Pbar_L.ndim == 3
    assert C_LU.ndim == 2
    assert C_DL.ndim == 2
    _assert_biorthogonal(P_L, Pbar_L)

    passed, res= verify_biorth(umps_top, umps_bot, P_L, Pbar_L, C_LU, C_DL, tol=tol)
    assert passed, f"Biorthogonality verification failed: {res}"

# ---------------------------------------------------------------------------
# U(1) Spin-1 tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [1, 2, 3])
def test_biorthogonalize_left_u1_self(config_kwargs, N):
    """U(1)/Spin-1 uMPS: biorthogonalize a random MPS against itself (self-case)."""
    ops = yastn.operators.Spin1(sym='U1', **config_kwargs)
    D = 2  # bond dimension per charge sector; total virtual dim = 6
    umps = _make_u1_umps(N, D, ops, seed=42)

    P_L, Pbar_L, C_LU, C_DL = biorthogonalize_left(umps, umps.conj())

    assert P_L.ndim == 3
    assert Pbar_L.ndim == 3
    assert C_LU.ndim == 2
    assert C_DL.ndim == 2
    _assert_biorthogonal(P_L, Pbar_L)


@pytest.mark.parametrize("N", [1, 2, 3])
def test_biorthogonalize_left_u1_different(config_kwargs, N):
    """U(1)/Spin-1 uMPS: biorthogonalize two independently drawn random MPS."""
    ops = yastn.operators.Spin1(sym='U1', **config_kwargs)
    D = 2
    umps_top = _make_u1_umps(N, D, ops, seed=42)
    umps_bot = _make_u1_umps(N, D, ops, seed=99).conj()

    P_L, Pbar_L, C_LU, C_DL = biorthogonalize_left(umps_top, umps_bot)

    assert P_L.ndim == 3
    assert Pbar_L.ndim == 3
    assert C_LU.ndim == 2
    assert C_DL.ndim == 2
    _assert_biorthogonal(P_L, Pbar_L)

    passed, res= verify_biorth(umps_top, umps_bot, P_L, Pbar_L, C_LU, C_DL, tol=tol)
    assert passed, f"Biorthogonality verification failed: {res}"


@pytest.mark.parametrize("N", [1,])
def test_biorthogonalize_left_u1_different_large(config_kwargs, N):
    """U(1)/Spin-1 uMPS: biorthogonalize two independently drawn random MPS."""
    ops = yastn.operators.Spin1(sym='U1', **config_kwargs)
    D = 20
    umps_top = _make_u1_umps(N, D, ops, seed=42)
    umps_bot = _make_u1_umps(N, D+10, ops, seed=99).conj()

    P_L, Pbar_L, C_LU, C_DL = biorthogonalize_left(umps_top, umps_bot)

    assert P_L.ndim == 3
    assert Pbar_L.ndim == 3
    assert C_LU.ndim == 2
    assert C_DL.ndim == 2
    _assert_biorthogonal(P_L, Pbar_L)