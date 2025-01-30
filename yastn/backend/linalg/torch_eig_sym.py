'''
Implementation taken from https://arxiv.org/abs/1903.09650
which follows derivation given in https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
'''
import torch


def safe_inverse(x, epsilon=1e-12):
    return x / (x ** 2 + epsilon)


class SYMEIG(torch.autograd.Function):
    @staticmethod
    def forward(A, ad_decomp_reg):
        r"""
        :param A: square symmetric matrix
        :type A: torch.tensor
        :return: eigenvalues values D, eigenvectors vectors U
        :rtype: torch.tensor, torch.tensor

        Computes symmetric decomposition :math:`M= UDU^T`.
        """
        # input validation (A is square and symmetric) is provided by torch.symeig

        D, U = torch.linalg.eigh(A)
        # torch.symeig returns eigenpairs ordered in the ascending order with
        # respect to eigenvalues. Reorder the eigenpairs by abs value of the eigenvalues
        # abs(D)
        absD, p = torch.sort(torch.abs(D), descending=True)
        D = D[p]
        U = U[:, p]
        return D, U

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        _, ad_decomp_reg = inputs
        D, U = output
        ctx.save_for_backward(D, U, ad_decomp_reg)

    @staticmethod
    def backward(ctx, dD, dU):
        D, U, ad_decomp_reg = ctx.saved_tensors
        Ut = U.t()

        F = (D - D[:, None])
        F = safe_inverse(F, epsilon=ad_decomp_reg)
        # F= 1/F
        F.diagonal().fill_(0)
        # F[abs(F) > 1.0e+8]=0

        dA = U @ (torch.diag(dD) + F * (Ut @ dU)) @ Ut
        return dA, None
