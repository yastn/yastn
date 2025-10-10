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

torch_test = pytest.mark.skipif("'torch' not in config.getoption('--backend')",
                                reason="Test torch backend.linalg routines.")


@torch_test
def test_SVDSYMARNOLDI_random():
    import torch
    from yastn.backend.linalg.torch_svds_scipy import SVDSYMARNOLDI

    m = 50
    k = 10
    M = torch.rand((m, m), dtype=torch.float64)
    M = 0.5 * (M + M.t())

    D0 = torch.linalg.eigvalsh(M)
    S0, _ = torch.sort(torch.abs(D0), descending=True)

    U, S, V = SVDSYMARNOLDI.apply(M, k)
    # |M|=\sqrt{Tr(MM^t)}=\sqrt{Tr(D^2)} =>
    # |M-US_kV^t|=\sqrt{Tr(D^2)-Tr(S^2)}=\sqrt{\sum_i>k D^2_i}
    assert torch.norm(M - U @ torch.diag(S) @ V.t()) - torch.sqrt(torch.sum(S0[k:] ** 2)) < S0[0]*(m**2)*1e-14


@torch_test
def test_SVDARNOLDI_random():
    import torch
    from yastn.backend.linalg.torch_svds_scipy import SVDS_SCIPY

    m = 50
    k = 10
    M = torch.rand((m, m), dtype=torch.float64)
    U0, S0, V0 = torch.svd(M)

    U , S, V = SVDS_SCIPY.apply(M, k)
    # |M|=\sqrt{Tr(MM^t)}=\sqrt{Tr(D^2)} =>
    # |M-US_kV^t|=\sqrt{Tr(D^2)-Tr(S^2)}=\sqrt{\sum_i>k D^2_i}
    assert abs(torch.norm(M - U @ torch.diag(S) @ V) - torch.sqrt(torch.sum(S0[k:] ** 2))) < S0[0] * m**2 * 1e-14


@torch_test
def test_SVDARNOLDI_rank_deficient():
    import torch
    from yastn.backend.linalg.torch_svds_scipy import SVDS_SCIPY

    m = 50
    k = 15
    for r in [25, 35, 40, 45]:
        M = torch.rand((m, m), dtype=torch.float64)
        U, S0, V = torch.svd(M)
        S0[-r:] = 0
        M = U @ torch.diag(S0) @ V.t()

        U, S, V = SVDS_SCIPY.apply(M, k)
        assert abs(torch.norm(M - U @ torch.diag(S) @ V) - torch.sqrt(torch.sum(S0[k:] ** 2))) < S0[0] * m**2 * 1e-14


@torch_test
def test_SYMEIG_random():
    import torch
    from yastn.backend.linalg.torch_eig_sym import SYMEIG

    m = 50
    M = torch.rand(m, m, dtype=torch.float64)
    M = 0.5 * (M + M.t())

    ad_decomp_reg = torch.tensor(1.0e-12)

    D, U = SYMEIG.apply(M, ad_decomp_reg)
    assert torch.norm(M - U @ torch.diag(D) @ U.t()) < abs(D[0]) * (m ** 2) * 1e-14

    # since we always assume matrix M to be symmetric, the finite difference
    # perturbations should be symmetric as well
    M.requires_grad_(True)

    def force_sym_eig(M):
        M = 0.5 * (M + M.t())
        return SYMEIG.apply(M, ad_decomp_reg)

    assert(torch.autograd.gradcheck(force_sym_eig, M, eps=1e-6, atol=1e-4))


@torch_test
def test_SYMEIG_3x3degenerate():
    import torch
    from yastn.backend.linalg.torch_eig_sym import SYMEIG

    M = torch.zeros((3, 3), dtype=torch.float64)
    M[0, 1] = M[0, 2] = M[1, 2] = 1.
    M = 0.5 * (M + M.t())
    ad_decomp_reg = torch.tensor(1.0e-12)

    D, U = SYMEIG.apply(M, ad_decomp_reg)

    assert torch.norm(M - U @ torch.diag(D) @ U.t()) < abs(D[0]) * (M.size()[0] ** 2) * 1e-14

    M.requires_grad_(True)
    torch.set_printoptions(precision=9)

    def force_sym_eig(M):
        M = 0.5 * (M + M.t())
        print(M)
        D, U = SYMEIG.apply(M, ad_decomp_reg)
        return U

    pytest.xfail("To fix")
    assert torch.autograd.gradcheck(force_sym_eig, M, eps=1e-6, atol=1e-4)


@torch_test
def test_SYMEIG_rank_deficient():
    import torch
    from yastn.backend.linalg.torch_eig_sym import SYMEIG

    m = 50
    r = 10
    M = torch.rand((m, m), dtype=torch.float64)
    M = M + M.t()
    D, U = torch.linalg.eigh(M)
    D[-r:] = 0
    M = U @ torch.diag(D) @ U.t()

    ad_decomp_reg = torch.tensor(1.0e-12)
    D, U = SYMEIG.apply(M, ad_decomp_reg)

    assert torch.norm(M - U @ torch.diag(D) @ U.t()) < abs(D[0]) * (M.size()[0] ** 2) * 1e-14

    M.requires_grad_(True)

    def force_sym_eig(M):
        M = 0.5 * (M + M.t())
        D, U = SYMEIG.apply(M, ad_decomp_reg)
        return U

    pytest.xfail("To fix")
    assert torch.autograd.gradcheck(force_sym_eig, M, eps=1e-6, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch"])
