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
import pytest

torch_test = pytest.mark.skipif("'torch' not in config.getoption('--backend')",
                                reason="Test torch backend.linalg routines.")

long_test = pytest.mark.skipif("not config.getoption('long_tests')",
                               reason="long duration tests are skipped")



@torch_test
@long_test
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
@long_test
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
@long_test
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
@long_test
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
@long_test
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
@long_test
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
    pytest.main([__file__, "-vs", "--durations=0", "--backend", "torch", "--long_tests"])
