from yamps.tensor import ncon
from .geometry import Geometry

###################################
#     basic operations on MPS     #
###################################


class MpsError(Exception):
    pass


class Mps:
    """
    The class implements the structure of mps/mpo and some basic operations within a single mps.
    Order of legs for a single mps tensor is (left virtual, 1st physical, 2nd physical, right virtual).
    mps tensors are index with :math:`0, 1, 2, 3, \\ldots, N-1` (with :math:`0` corresponding to the leftmost site).
    A central block (associated with a bond) is indexed using ordered tuple, e.g. (n, n+1)
    and (None, n) if placed outside of a leaf. Maximally one central block is allowed.
    """

    def __init__(self, N=2, nr_phys=1, nr_aux=0):
        r"""
        Initialize basic structure for matrix product state/operator/purification.

        Parameters
        ----------
        N : int, geometry
            number of sites of mps, or directly an instance of geometry class
        nr_phys : int
            number of physical legs: _1_ for mps; _2_ for mpo;
        """
        self.g = Geometry(N) if isinstance(N, int) else N  # g is a number (length) or directly a geometry class
        self.normalize = True # if the state should be normalized 
        self.N = self.g.N
        self.A = {}  # dict of mps tensors; indexed by integers
        self.pC = None  # index of the central site, None if it does not exist
        self.norma = 1. # norm of a tensor
        self.nr_phys = nr_phys  # number of physical legs
        self.nr_aux = nr_aux  # number of auxilliary legs
        self.left = (0,)  # convention which leg is left-virtual (connected to site with smaller index)
        self.right = (nr_phys + 1,)  # convention which leg is right-virtual leg (connected to site with larger index)
        self.phys = tuple(ii for ii in range(1, nr_phys + 1))  # convention which legs are physical
        self.aux = tuple(ii for ii in range(1 + nr_phys, nr_aux + nr_phys + 1))
  
    def copy(self):
        r"""
        Makes a copy of mps.

        Use when retaining "old" mps is neccesary -- operations on mps are done in-place.

        Returns
        -------
        Copied mps : mps
        """
        phi = Mps(g=self.g, nr_phys=self.nr_phys)
        for ind in self.A:
            phi.A[ind] = self.A[ind].copy()
        phi.pC = self.pC
        return phi

    def orthogonalize_site(self, n, towards):
        r"""
        Orthogonalize n-th site in the direction given by towards. Generate normalized central site.

        Parameters
        ----------
        n : int
            index of site to be ortogonalized.
        towards : int
            index of site toward which to orthogonalize.
        normalize : bool
            true if central site should be normalized

        Returns
        -------
        normC : float
            Norm of the central site.
        """

        if self.pC is not None:
            raise MpsError('Only one central site is possible.')

        nnext, leg, _ = self.g.from_site(n, towards)

        if nnext is not None:
            self.pC = (n, nnext)
            if leg == 0:  # ortogonalize from right to left (last to first)
                Q, R = self.A[n].split_qr(axes=(self.phys + self.right, self.left), sQ=1, Qaxis=0, Raxis=-1)
            else:  # leg == 1 or leg is None:  # ortogonalize from left to right (first to last)
                Q, R = self.A[n].split_qr(axes=(self.left + self.phys, self.right), sQ=-1)

            self.A[n] = Q
            normC = R.norm() if self.normalize else 1.
            self.A[self.pC] = (1 / normC) * R
        else:
            normC = self.A[n].norm() if self.normalize else 1.
            self.A[n] = (1 / normC) * self.A[n]
        return normC

    def diagonalize_central(self, opts={}):
        r"""
        Perform svd of central site -- truncating according to opts.

        Attach U and V to respective left and right sites.

        Parameters
        ----------
        opts : dict
            Options passed for svd -- this includes truncation.
        """
        if self.pC is not None:
            U, S, V = self.A[self.pC].svd(axis=(0, 1), opts=opts)
        self.A[self.pC] = S

        nnext, leg, nprev = self.g.from_bond(self.pC, towards=self.g.first)
        if nprev is None:
            UV = U.dot(V, axis=(0, 1))
            if leg == 0:  # first site
                self.A[nprev] = UV.dot(self.A[nnext], axis=(1, self.left))
            else:  # leg == 1:  last site
                self.A[nprev] = self.A[nnext].dot(UV, axis=(self.right, 0))
        else:
            self.A[nnext] = self.A[nnext].dot(U, axis=(self.right, 0))
            self.A[nprev] = V.dot(self.A[nprev], axis=(0, self.left))

    def absorb_central(self, towards):
        r"""
        Absorb central site. Do nothing if is does not exist.

        Parameters
        ----------
        towards : int
            index of site toward which to absorb.
        """
        if self.pC is not None:
            C = self.A.pop(self.pC)
            nnext, leg, _ = self.g.from_bond(self.pC, towards)
            self.pC = None

            if leg == 1:
                self.A[nnext] = self.A[nnext].dot(C, axes=(self.right, 0))
            else:  # leg == 0:
                self.A[nnext] = C.dot(self.A[nnext], axes=(1, self.left))

    def canonize_sweep(self, to='last'):
        r"""
        Left or right canonize and normalize mps if normalize is True.
        """
        if to == 'last':
            for n in self.g.sweep(to='last'):
                norma = self.orthogonalize_site(n=n, towards=self.g.last)
                self.absorb_central(towards=self.g.last)
        elif to == 'first':
            for n in self.g.sweep(to='first'):
                norma = self.orthogonalize_site(n=n, towards=self.g.first)
                self.absorb_central(towards=self.g.first)
        else:
            raise MpsError("mps/canonize_sweep: Option ", to, " is not defined.")
        if not self.normalize:
            self.norma = norma

    def merge_mps(self, n):
        r"""
        Merge two neighbouring mps sites.

        Parameters
        ----------
        n : ind
            Index of the left site.

        Returns
        ----------
        out : Tensor
            tensor formed from A[n] and A[nr]
        """
        _, nr = self.g.order_neighbours(n)
        if self.nr_phys == 1:
            return self.A[n].dot(self.A[nr], axes=(self.right, self.left))
        else:  # self.nr_phys == 2:
            return ncon([self.A[n], self.A[nr]], ((-1, -2, -4, 1), (1, -3, -5, -6)), (0, 0))

    def get_D(self):
        r"""
        Returns bond dimensions of mps.

        Returns
        -------
        Ds : list
            list of bond dimensions on virtual legs from left to right,
            including "trivial" leftmost and rightmost virtual indices.
        """
        Ds = [None]*(self.N+1)
        for n in range(self.N):
            DAn = self.A[n].get_shape()[0]
            Ds[n] = DAn[self.left[0]] 
        Ds[n+1] = DAn[self.right[0]]
        return Ds

    def measuring(self,list_of_ops, norm=None):
        if norm:
            norm.setup_to_first()
            norm = norm.measure().real
        else:
            norm = 1.
        out = [None]*len(list_of_ops)
        for n in range(len(out)):
            list_of_ops[n].setup_to_first()
            out[n] = list_of_ops[n].measure()/norm
        return out