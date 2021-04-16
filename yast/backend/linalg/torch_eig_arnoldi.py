import torch
try:
    import scipy.sparse.linalg
    from scipy.sparse.linalg import LinearOperator
except ImportError:
    import warnings
    warnings.warn("fbpca not available", Warning)


class SYMARNOLDI(torch.autograd.Function):
    @staticmethod
    def forward(self, M, k):
        r"""
        :param M: square symmetric matrix :math:`N \times N`
        :param k: desired rank (must be smaller than :math:`N`)
        :type M: torch.tensor
        :type k: int
        :return: eigenvalues D, leading k eigenvectors U
        :rtype: torch.tensor, torch.tensor

        **Note:** `depends on scipy`

        Return leading k-eigenpairs of a matrix M, where M is symmetric
        :math:`M=M^T`, by computing the symmetric decomposition :math:`M= UDU^T`
        up to rank k. Partial eigendecomposition is done through Arnoldi method.
        """
        # input validation (M is square and symmetric) is provided by
        # the scipy.sparse.linalg.eigsh

        # get M as numpy ndarray and wrap back to torch
        # allow for mat-vec ops to be carried out on GPU
        def mv(v):
            V = torch.as_tensor(v, dtype=M.dtype, device=M.device)
            V = torch.mv(M, V)
            return V.detach().cpu().numpy()
        M_nograd = LinearOperator(M.size(), matvec=mv)

        D, U = scipy.sparse.linalg.eigsh(M_nograd, k=k)
        D = torch.as_tensor(D, dtype=M.dtype, device=M.device)
        U = torch.as_tensor(U, dtype=M.dtype, device=M.device)

        # reorder the eigenpairs by the largest magnitude of eigenvalues
        absD, p = torch.sort(torch.abs(D), descending=True)
        D = D[p]
        U = U[:, p]

        self.save_for_backward(D, U)
        return D, U


class SYMARNOLDI_2C(torch.autograd.Function):
    @staticmethod
    def forward(self, M, Mhc, k):
        r"""
        :param M: matrix :math:`N \times M`
        :param Mhc: matrix :math:`M \times N`
        :param k: desired rank (must be smaller than :math:`N+M`)
        :type M: torch.tensor
        :type Mhc: torch.tensor
        :type k: int
        :return: eigenvalues D, leading k eigenvectors U
        :rtype: torch.tensor, torch.tensor

        **Note:** `depends on scipy`

        Return leading k-eigenpairs of a 2-cyclic symmetric matrix A, formed by
        two non-zero blocks ``M`` and ``Mhc`` where :math:`M^\dag = M_{hc}` as follows::

            A = [0   M]
                [Mhc 0]

        by computing the symmetric decomposition :math:`M= UDU^T` up to rank k.
        Partial eigendecomposition is done through Arnoldi method. The eigenvectors
        and eigenvalues of A are related to SVD of M as follows::

            M = U S V^\dag  where U = [u_0, u_1, ...], u_i being column vectors
                                  V = [v_0, ...]     , v_i being column vectors

        then for solution Ax=wx the following holds::

            w_i = +/-S_i and x_i = 1/\sqrt(2)[+/-u_i]
                                             [   v_i]
        """

        # get M as numpy ndarray and wrap back to torch
        # allow for mat-vec ops to be carried out on GPU
        #
        # [0   M][v0] = [M v1   ]
        # [Mhc 0][v1] = [Mhc v0 ]
        def mv(v):
            V = torch.as_tensor(v, dtype=M.dtype, device=M.device)
            R = torch.zeros_like(V)
            R[:M.size(0)] = torch.mv(M, V[-M.size(1):])
            R[-Mhc.size(0):] = torch.mv(Mhc, V[:Mhc.size(1)])
            return R.detach().cpu().numpy()
        M_nograd = LinearOperator((M.size(0) + Mhc.size(0), (M.size(1) + Mhc.size(1))), matvec=mv)

        D, U = scipy.sparse.linalg.eigsh(M_nograd, k=k)
        D = torch.as_tensor(D, dtype=M.dtype, device=M.device)
        U = torch.as_tensor(U, dtype=M.dtype, device=M.device)

        # reorder the eigenvalues in descending fashion
        D, p = torch.sort(D, descending=True)
        U = U[:, p]

        self.save_for_backward(D, U)
        return D, U

    @staticmethod
    def backward(self, dD, dU):
        raise Exception("backward not implemented")
        D, U = self.saved_tensors
        dA = None
        return dA, None
