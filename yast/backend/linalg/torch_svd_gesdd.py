import torch

def safe_inverse(x, epsilon=1E-12):
    return x / (x**2 + epsilon)

def safe_inverse_2(x, epsilon):
    x[abs(x) < epsilon] = float('inf')
    return x.pow(-1)

class SVDGESDD(torch.autograd.Function):
    @staticmethod
    def forward(self, A, ad_decomp_reg):
        U, S, V = torch.svd(A)
        self.save_for_backward(U, S, V, ad_decomp_reg)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        # https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/FunctionsManual.cpp
        # Tensor svd_backward(const std::vector<torch::autograd::Variable> &grads, const Tensor& self,
        #   bool some, bool compute_uv, const Tensor& raw_u, const Tensor& sigma, const Tensor& raw_v) {
        # TORCH_CHECK(compute_uv,
        #    "svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
        #    "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");

        #auto m = self.size(-2); # first dim of original tensor A = USV^\dag 
        #auto n = self.size(-1); # second dim of A
        #auto k = sigma.size(-1); # size of singular value vector
        # auto gsigma = grads[1]; # dS
        U, S, V, ad_decomp_reg = self.saved_tensors
        m = U.size(0)
        n = V.size(0)
        k = S.size(0)
        sigma = S
        sigma_scale = sigma[0]
        gsigma = dS

        # auto u = raw_u;
        # auto v = raw_v;
        # auto gu = grads[0];
        # auto gv = grads[2];
        u = U
        v = V
        gu = dU
        gv = dV

        # some is always True here
        # if (!some) {
        #     // We ignore the free subspace here because possible base vectors cancel
        #     // each other, e.g., both -v and +v are valid base for a dimension.
        #     // Don't assume behavior of any particular implementation of svd.
        #     u = raw_u.narrow(-1, 0, k);
        #     v = raw_v.narrow(-1, 0, k);
        #     if (gu.defined()) {
        #       gu = gu.narrow(-1, 0, k);
        #     }
        #     if (gv.defined()) {
        #       gv = gv.narrow(-1, 0, k);
        #     }
        # }
        # auto vh = v.conj().transpose(-2, -1);
        vh= v.conj().transpose(-2,-1)

        # Tensor sigma_term;
        # if (gsigma.defined()) {
        #     gsigma = gsigma.to(self.dtype());
        #     // computes u @ diag(gsigma) @ vh
        #     sigma_term = at::matmul(u * gsigma.unsqueeze(-2), vh);
        # } else {
        #     sigma_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        # }
        # // in case that there are no gu and gv, we can avoid the series of kernel
        # // calls below
        # if (!gv.defined() && !gu.defined()) {
        #     return sigma_term;
        # }
        # we always compute gu, gv here
        sigma_term = u * gsigma.unsqueeze(-2) @ vh

        # auto uh = u.conj().transpose(-2, -1);
        # auto sigma_inv = sigma.pow(-1);
        # auto sigma_sq = sigma.pow(2);
        # auto F = sigma_sq.unsqueeze(-2) - sigma_sq.unsqueeze(-1);
        uh = u.conj().transpose(-2, -1)
        # sigma_inv= sigma.pow(-1)
        sigma_inv = safe_inverse_2(sigma.clone(), sigma_scale * 1.0e-12)
        sigma_sq = sigma.pow(2)
        F = sigma_sq.unsqueeze(-2) - sigma_sq.unsqueeze(-1)

        # F_ij = 1/(S^2_i - S^2_j)
        # // The following two lines invert values of F, and fills the diagonal with 0s.
        # // Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
        # // first to prevent nan from appearing in backward of this function.
        # F.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(INFINITY);
        # F = F.pow(-1);
        F.diagonal(0, -2, -1).fill_(float('inf'))
        # F= F.pow(-1)
        F = safe_inverse_2(F, sigma_scale * 1.0e-12)

        # Tensor u_term, v_term;
        # if (gu.defined()) {
        #     auto guh = gu.conj().transpose(-2, -1);
        #     u_term = at::matmul(u, F.mul(at::matmul(uh, gu) - at::matmul(guh, u)) * sigma.unsqueeze(-2));
        #     if (m > k) {
        #         // projection operator onto subspace orthogonal to span(U) defined as I - UU^H
        #         auto proj_on_ortho_u = -at::matmul(u, uh);
        #         proj_on_ortho_u.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).add_(1);
        #         u_term = u_term + proj_on_ortho_u.matmul(gu * sigma_inv.unsqueeze(-2));
        #     }
        #     u_term = at::matmul(u_term, vh);
        # } else {
        #     u_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        # }
        # gu is always defined here
        guh = gu.conj().transpose(-2, -1)
        u_term = u @ (F.mul( uh @ gu - guh @ u) * sigma.unsqueeze(-2))
        if m > k:
            # projection operator onto subspace orthogonal to span(U) defined as I - UU^H
            proj_on_ortho_u = -u@uh
            proj_on_ortho_u.diagonal(0, -2, -1).add_(1)
            u_term = u_term + proj_on_ortho_u @ (gu * sigma_inv.unsqueeze(-2))
        u_term = u_term @ vh

        # if (gv.defined()) {
        #     auto gvh = gv.conj().transpose(-2, -1);
        #     v_term = sigma.unsqueeze(-1) * at::matmul(F.mul(at::matmul(vh, gv) - at::matmul(gvh, v)), vh);
        #     if (n > k) {
        #         // projection operator onto subspace orthogonal to span(V) defined as I - VV^H
        #         auto proj_on_v_ortho = -at::matmul(v, vh);
        #         proj_on_v_ortho.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).add_(1);
        #         v_term = v_term + sigma_inv.unsqueeze(-1) * at::matmul(gvh, proj_on_v_ortho);
        #     }
        #     v_term = at::matmul(u, v_term);
        # } else {
        #     v_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        # }
        # gv is always defined here
        gvh = gv.conj().transpose(-2, -1)
        v_term = sigma.unsqueeze(-1) * (F.mul(vh @ gv - gvh @ v) @ vh)
        if n > k:
            # projection operator onto subspace orthogonal to span(V) defined as I - VV^H
            proj_on_v_ortho = -v @ vh
            proj_on_v_ortho.diagonal(0, -2, -1).add_(1)
            v_term = v_term + sigma_inv.unsqueeze(-1) * (gvh @ proj_on_v_ortho)
        v_term = u @ v_term

        # // for complex-valued input there is an additional term
        # // https://giggleliu.github.io/2019/04/02/einsumbp.html
        # // https://arxiv.org/abs/1909.02659
        # if (self.is_complex() && gu.defined()) {
        #     Tensor L = at::matmul(uh, gu).diagonal(0, -2, -1);
        #     at::real(L).zero_();
        #     at::imag(L).mul_(sigma_inv);
        #     Tensor imag_term = at::matmul(u * L.unsqueeze(-2), vh);
        #     return u_term + sigma_term + v_term + imag_term;
        # }
        if U.is_complex() or V.is_complex():
            L= (uh @ gu).diagonal(0,-2,-1)
            L.real.zero_()
            L.imag.mul_(sigma_inv)
            imag_term= (u * L.unsqueeze(-2)) @ vh
            return u_term + sigma_term + v_term + imag_term, None

        return u_term + sigma_term + v_term, None