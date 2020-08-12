import logging
from yamps.tensor import ncon
from .geometry import Geometry


class FatalError(Exception):
    pass


logger = logging.getLogger('yamps.mps.mps')


###################################
#     basic operations on MPS     #
###################################


class Mps:
    """
    The class implements the structure of mps/mpo and some basic operations on a single mps.
    Order of legs for a single mps tensor is (left virtual, 1st physical, 2nd physical, right virtual).
    mps tensors are index with :math:`0, 1, 2, 3, \\ldots, N-1` (with :math:`0` corresponding to the leftmost site).
    A central block (associated with a bond) is indexed using ordered tuple, e.g. (n, n+1)
    and (None, n) if placed outside of a leaf. Maximally one central block is allowed.
    """

    def __init__(self, N, nr_phys=1):
        r"""
        Initialize basic structure for matrix product state/operator/purification.

        Parameters
        ----------
        N : int, geometry
            number of sites of mps, or directly an instance of geometry class
        nr_phys : int
            number of physical legs: _1_ for mps; _2_ for mpo;
        """
        # g is a number (length) or directly a geometry class
        self.g = Geometry(N) if isinstance(N, int) else N
        self.N = self.g.N
        self.A = {}  # dict of mps tensors; indexed by integers
        self.pC = None  # index of the central site, None if it does not exist
        self.nr_phys = nr_phys  # number of physical legs
        # convention which leg is left virtual(connected to site with smaller index)
        self.left = (0,)
        # convention which leg is a right virtual leg (connected to site with larger index)
        self.right = (nr_phys + 1,)
        # convention which legs are physical
        self.phys = tuple(ii for ii in range(1, nr_phys + 1))

    def copy(self):
        r"""
        Makes a copy of mps.

        Use when retaining "old" mps is neccesary -- operations on mps are done in-place.

        Returns
        -------
        Copied mps : mps
        """
        phi = Mps(N=self.g, nr_phys=self.nr_phys)
        for ind in self.A:
            phi.A[ind] = self.A[ind].copy()
        phi.pC = self.pC
        return phi

    def orthogonalize_site(self, n, towards, normalize=True):
        r"""
        Orthogonalize n-th site in the direction given by towards.

        Parameters
        ----------
        n : int
            index of site to be orthogonalized.
        towards : int
            index of site toward which to orthogonalize.
        normalize : bool
            If true, central sites is normalized to 1 according to standard 2-norm.
        """

        if self.pC is not None:
            logger.exception('Only one central site is possible.')
            raise FatalError

        nnext, leg, _ = self.g.from_site(n, towards)

        if nnext is not None:
            self.pC = (n, nnext)
            if leg == 0:  # orthogonalize from right to left (last to first)
                Q, R = self.A[n].split_qr(
                    axes=(self.phys + self.right, self.left), sQ=1, Qaxis=0, Raxis=-1)
            else:  # leg == 1 or leg is None:  # orthogonalize from left to right (first to last)
                Q, R = self.A[n].split_qr(
                    axes=(self.left + self.phys, self.right), sQ=-1)
            self.A[n] = Q
            self.A[self.pC] = (1 / R.norm()) * R if normalize else R
        else:
            self.A[n] = (1 / self.A[n].norm()) * self.A[n] if normalize else self.A[n]

    def diagonalize_central(self, opts={}, normalize=True):
        r"""
        Perform svd of central site C = U S V -- truncating according to opts.

        Attach U and V respective to the left and right sites.

        Parameters
        ----------
        opts : dict
            Options passed for svd -- including information how to truncate.

        normalize : bool
            If true, S is normalized to 1 according to the standard 2-norm.

        Return
        ------
        discarded : float
            norm of discarded singular value, normalized by the remining ones
        """
        if self.pC is not None:
            normC = self.A[self.pC].norm()
            U, S, V = self.A[self.pC].split_svd(axes=(0, 1), **opts)
            normS = S.norm()

            self.A[self.pC] = (1 / normS) * S if normalize else S

            nnext, leg, nprev = self.g.from_bond(self.pC, towards=self.g.first)
            if nprev is None:
                UV = U.dot(V, axes=(0, 1))
                if leg == 0:  # first site
                    self.A[nprev] = UV.dot(self.A[nnext], axes=(1, self.left))
                else:  # leg == 1:  last site
                    self.A[nprev] = self.A[nnext].dot(UV, axes=(self.right, 0))
            else:
                self.A[nnext] = self.A[nnext].dot(U, axes=(self.right, 0))
                self.A[nprev] = V.dot(self.A[nprev], axes=(1, self.left))
            return normC / normS - 1
        else:
            return 0.

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

    def canonize_sweep(self, to='last', normalize=True):
        r"""
        Left or right canonize.

        Parameters
        ----------
        to : str
            'last' or 'first'.

        normalize : bool
            If true, S is normalized to 1 according to the standard 2-norm.
        """
        if to == 'last':
            for n in self.g.sweep(to='last'):
                self.orthogonalize_site(n=n, towards=self.g.last, normalize=normalize)
                self.absorb_central(towards=self.g.last)
        elif to == 'first':
            for n in self.g.sweep(to='first'):
                self.orthogonalize_site(n=n, towards=self.g.first, normalize=normalize)
                self.absorb_central(towards=self.g.first)
        else:
            logger.exception("canonize_sweep: Option " + to + " is not defined.")
            raise FatalError

    def sweep_truncate(self, to='last', opts={}, normalize=True):
        r"""
        Left or right canonize, and truncate using svd.

        For truncation to make sense, mps should be in the cannonical form in the oposite direction to that of the current sweep.

        Parameters
        ----------
        to : str
            'last' or 'first'.

        opts : dict
            Options passed for svd -- including information how to truncate.

        normalize : bool
            If true, S is normalized to 1 according to the standard 2-norm.

        Return
        ------
        discarded : float
            maximal norm of discarded singular values
        """

        discarded_max = 0.
        if to == 'last':
            for n in self.g.sweep(to='last'):
                self.orthogonalize_site(n=n, towards=self.g.last, normalize=normalize)
                discarded = self.diagonalize_central(opts=opts, normalize=normalize)
                discarded_max = max(discarded_max, discarded)
                self.absorb_central(towards=self.g.last)
        elif to == 'first':
            for n in self.g.sweep(to='first'):
                self.orthogonalize_site(n=n, towards=self.g.first, normalize=normalize)
                discarded = self.diagonalize_central(opts=opts, normalize=normalize)
                discarded_max = max(discarded_max, discarded)
                self.absorb_central(towards=self.g.first)
        else:
            logger.exception("canonize_sweep: Option " + to +  " is not defined.")
            raise FatalError

        return discarded_max

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
        Ds = []
        for n in self.g.sweep(to='last'):
            DAn = self.A[n].get_shape()[0]
            Ds.append(DAn[self.left[0]])
        Ds.append(DAn[self.right[0]])
        return Ds
