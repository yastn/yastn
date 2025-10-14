import torch
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator


class SVDSYMARNOLDI(torch.autograd.Function):
    @staticmethod
    def forward(self, M, k):
        r"""
        :param M: square symmetric matrix :math:`N \times N`
        :param k: desired rank (must be smaller than :math:`N`)
        :type M: torch.tensor
        :type k: int
        :return: leading k left eigenvectors U, singular values S, and right
                 eigenvectors V
        :rtype: torch.tensor, torch.tensor, torch.tensor

        **Note:** `depends on scipy`

        Return leading k-singular triples of a matrix M, where M is symmetric
        :math:`M=M^T`, by computing the symmetric decomposition :math:`M= UDU^T`
        up to rank k. Partial eigendecomposition is done through Arnoldi method.
        """
        # input validation (M is square and symmetric) is provided by
        # the scipy.sparse.linalg.eigsh

        # get M as numpy ndarray and wrap back to torch
        # allow for mat-vec ops to be carried out on GPU
        def mv(v):
            V= torch.as_tensor(v,dtype=M.dtype,device=M.device)
            V= torch.mv(M,V)
            return V.detach().cpu().numpy()

        # M_nograd = M.clone().detach().cpu().numpy()
        M_nograd= LinearOperator(M.size(), matvec=mv)

        D, U= scipy.sparse.linalg.eigsh(M_nograd, k=k)

        D= torch.as_tensor(D)
        U= torch.as_tensor(U)

        # reorder the eigenpairs by the largest magnitude of eigenvalues
        S,p= torch.sort(torch.abs(D),descending=True)
        U= U[:,p]

        # 1) M = UDU^t = US(sgn)U^t = U S (sgn)U^t = U S V^t
        # (sgn) is a diagonal matrix with signs of the eigenvales D
        V= U@torch.diag(torch.sign(D[p]))

        if M.is_cuda:
            U= U.cuda()
            V= V.cuda()
            S= S.cuda()

        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        raise Exception("backward not implemented")
        U, S, V = self.saved_tensors
        dA= None
        return dA, None


class SVDS_SCIPY(torch.autograd.Function):
    @staticmethod
    def forward(self, M, k, thresh=0.1, solver='arpack', **kwargs):
        # TODO should threshold be handled here or at call site ?
        r"""
        :param M: square matrix :math:`N \times N`
        :param k: desired rank (must be smaller than :math:`N`)
        :param thresh: threshold for applying partial SVD (svds) instead of full SVD
        :param solver: solver for scipy.sparse.linalg.svds
        :param kwargs: additional keyword arguments for the scipy.sparse.linalg.svds
        :type M: torch.Tensor
        :type k: int
        :type thresh: float
        :type solver: str
        :return: leading k left eigenvectors U, singular values S, and right
                 eigenvectors V
        :rtype: torch.Tensor, torch.Tensor, torch.Tensor

        **Note:** `depends on scipy`

        Return leading k-singular triples of a matrix M via SciPy. See
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html
        for details on solver and solver options.
        """
        # input validation is provided by the scipy.sparse.linalg.eigsh /
        # scipy.sparse.linalg.svds

        # ----- Option 0
        # M_nograd = M.clone().detach()
        # MMt= M_nograd@M_nograd.t().conj()

        # def mv(v):
        #     B= torch.as_tensor(v,dtype=M.dtype,device=M.device)
        #     B= torch.mv(MMt,B)
        #     return B.detach().cpu().numpy()
        # if k == M.size(dim=0):
        #     D, U= scipy.linalg.eigh(MMt)
        # else:
        #     MMt_op= LinearOperator(M.size(), matvec=mv)
        #     D, U= scipy.sparse.linalg.eigsh(MMt_op, k=k)
        # D= torch.as_tensor(D,device=M.device)
        # U= torch.as_tensor(U,device=M.device)

        # # reorder the eigenpairs by the largest magnitude of eigenvalues
        # S,p= torch.sort(torch.abs(D),descending=True)
        # S= torch.sqrt(S)
        # U= U[:,p]

        # # compute right singular vectors as Mt = V.S.Ut /.U => Mt.U = V.S
        # if U.dtype == torch.complex128:
        #     V = M_nograd.t().conj() @ U * (1/S.to(dtype=torch.complex128))
        # else:
        #     V = M_nograd.t().conj() @ U * (1/S)
        # V = Functional.normalize(V, p=2, dim=0)

        # k is already passed as an argument
        # propack default 10*k, arpack default min(M.size) * 10 as per scipy docs
        kwargs.pop('k', None)
        maxiter= kwargs.pop('maxiter', k*10 if solver == 'propack' else 10 * min(M.shape))

        # ----- Option 1
        if min(M.shape)*thresh < k: # k / matrix size is too large for speed-up by iterative solver
            U, S, Vh = torch.linalg.svd(M, driver='gesvd' if M.is_cuda else None)
            U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
            self.save_for_backward(U, S, Vh)
            return U, S, Vh

        elif M.device != torch.device('cpu'): # assume accelerator for matrix-vector products
            # TODO consider circulant matrix [[0,M],[M^\dag,0]] and solve via eigsh

            def mv(v):
                B= torch.as_tensor(v,dtype=M.dtype,device=M.device)
                B= torch.matmul(M,B)
                return B.detach().cpu().numpy()
            def vm(v):
                B= torch.as_tensor(v,dtype=M.dtype,device=M.device)
                B= torch.matmul(M.t().conj(),B)
                return B.detach().cpu().numpy()

            M_nograd= LinearOperator(M.size(), matvec=mv, rmatvec=vm)
            U, S, Vh= scipy.sparse.linalg.svds(M_nograd, k=k, solver=solver, maxiter=maxiter, **kwargs)

        else: # solve in numpy
            U, S, Vh= scipy.sparse.linalg.svds(M.detach().cpu().numpy(), k=k, solver=solver, maxiter=maxiter, **kwargs)

        # the order of singular values is not guaranteed to be descending (assumme its either ascending or descending)
        if S[0]<S[-1]:
            # reorder the singular values and vectors
            U, S, Vh= U[:,::-1], S[::-1], Vh[::-1,:]

        neg_strides= lambda x: any([s for s in x.strides if s < 0])
        S= torch.as_tensor(S.copy() if neg_strides(S) else S).to(device=M.device)
        U= torch.as_tensor(U.copy() if neg_strides(U) else U).to(device=M.device)
        Vh= torch.as_tensor(Vh.copy() if neg_strides(Vh) else Vh).to(device=M.device)

        self.save_for_backward(U, S, Vh)
        return U, S, Vh

    @staticmethod
    def backward(self, dU, dS, dV):
        r"""
        The backward is not implemented.
        """
        raise Exception("backward not implemented")
        U, S, V = self.saved_tensors
        dA= None
        return dA, None, None, None
