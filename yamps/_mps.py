""" Mps structure and its basic manipulations. """

class YampsError(Exception):
    pass


###################################
#     basic operations on MPS     #
###################################


class Mps:
    """
    The basic structure of mps (for nr_phys=1) and mpo (for nr_phys=2) and some basic operations on a single mps.
    Order of legs for a single mps tensor is (left virtual, 1st physical, 2nd physical, right virtual).
    Mps tensors are index with :math:`0, 1, 2, 3, \\ldots, N-1` (with :math:`0` corresponding to the first site).
    A central block (associated with a bond) is indexed using ordered tuple (n, n+1).
    Maximally one central block is allowed.
    """

    def __init__(self, N, nr_phys=1):
        r"""
        Initialize basic structure for matrix product state/operator/purification.

        Parameters
        ----------
        N : int
            number of sites of mps, or directly an instance of geometry class
        nr_phys : int
            number of physical legs: _1_ for mps; _2_ for mpo;
        """
        if not isinstance(N, int) or N <= 0:
            raise YampsError("Number of Mps sites N should be a positive integer.")
        if nr_phys not in (1, 2):
            raise YampsError("Number of physical legs of Mps, nr_phys, should be equal to 1 or 2.")
        self.N = N
        self.nr_phys = nr_phys
        self.A = {}  # dict of mps tensors; indexed by integers
        self.pC = None  # index of the central site, None if it does not exist
        self.left = (0,)  # convention which leg is left virtual(connected to site with smaller index)
        self.right = (nr_phys + 1,)  # convention which leg is a right virtual leg (connected to site with larger index)
        self.phys = (1,) if nr_phys == 1 else (1, 2)  # convention which legs are physical
        self.first = 0
        self.last = self.N - 1


    def sweep(self, to='last', df=0, dl=0):
        r"""
        Generator of indices of all sites going from first to last or vice-versa

        Parameters
        ----------
        to : str
            'first' or 'last'.
        df, dl : int
            shift iterator by df >= 0 and dl >= 0 from the first and the last site, respectively.
        """
        if to == 'last':
            return range(df, self.N - dl)
        if to == 'first':
            return range(self.N - 1 - dl, df - 1, -1)
        raise YampsError('Argument "to" should be in ("first", "last")')


    def clone(self):
        r"""
        Makes a copy of mps. Copy all mps tensors, tracking gradients.

        Use when retaining "old" mps is neccesary -- other operations on mps are often done in-place.

        Returns
        -------
        Cloned mps : Mps
        """
        phi = Mps(N=self.N, nr_phys=self.nr_phys)
        phi.A = {ind: ten.clone() for ind, ten in self.A.items()}
        phi.pC = self.pC
        return phi


    def copy(self):
        r"""
        Makes a copy of mps. Copy all mps tensors.

        Warning, this might break autograd if you are using it.
        Use when retaining "old" mps is neccesary -- other operations on mps are often done in-place.

        Returns
        -------
        Copied mps : Mps
        """
        phi = Mps(N=self.N, nr_phys=self.nr_phys)
        phi.A = {ind: ten.copy() for ind, ten in self.A.items()}
        phi.pC = self.pC
        return phi


    def orthogonalize_site(self, n, to='last', normalize=True):
        r"""
        Orthogonalize n-th site to the first site.

        Parameters
        ----------
            n : int
                index of site to be ortogonalized

            to : str
                'last' or 'first'.

            normalize : bool
                If true, central sites is normalized to 1 according to standard 2-norm.
        """
        if self.pC is not None:
            raise YampsError('Only one central block is possible. Attach the existing central block first.')

        if to == 'first':
            self.pC = (n - 1, n)
            self.A[n], R = self.A[n].qr(axes=(self.phys + self.right, self.left), sQ=1, Qaxis=0, Raxis=-1)
            self.A[self.pC] = R / R.norm() if normalize else R
        elif to == 'last':
            self.pC = (n, n + 1)
            self.A[n], R = self.A[n].qr(axes=(self.left + self.phys, self.right), sQ=-1)
            self.A[self.pC] = R / R.norm() if normalize else R
        else:
            raise YampsError('Argument "to" should be in ("first", "last")')


    def diagonalize_central(self, opts=None, normalize=True):
        r"""
        Perform svd of the central site C = U S V -- truncating according to opts which is pased into svd.

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
            norm of discarded singular values normalized by the remining ones
        """
        if self.pC is not None:
            normC = self.A[self.pC].norm()
            if opts is None:
                opts = {'tol': 1e-12}
            U, S, V = self.A[self.pC].svd(axes=(0, 1), sU=-1, **opts)

            normS = S.norm()
            self.A[self.pC] = S / normS if normalize else S
            n1, n2 = self.pC

            if n1 >= 0:
                self.A[n1] = self.A[n1].tensordot(U, axes=(self.right, 0))
            else:
                self.A[self.pC] = U.tensordot(self.A[self.pC], axes=(1, 0))

            if n2 <= self.N - 1:
                self.A[n2] = V.tensordot(self.A[n2], axes=(1, self.left))
            else:
                self.A[self.pC] = self.A[self.pC].tensordot(V, axes=(1, 0))

            return (normC -  normS) / normS
        return 0.


    def remove_central(self):
        r"""
        Removes (ignores) the central site. Do nothing if is does not exist.
        """
        if self.pC is not None:
            del self.A[self.pC]
            self.pC = None


    def absorb_central(self, to='last'):
        r"""
        Absorb central site towards the first or the last site.

        If the central site is outside of the chain, it is goes in the one direction possible.
        Do nothing if central does not exist.

        Parameters
        ----------
        to : str
            'last' or 'first'.
        """
        if self.pC is not None:
            C = self.A.pop(self.pC)
            n1, n2 = self.pC
            self.pC = None

            if (to == 'first' and n1 >= 0) or n2 >= self.N:
                self.A[n1] = self.A[n1].tensordot(C, axes=(self.right, 0))
            else:  # (to == 'last' and n2 < self.N) or n1 < 0
                self.A[n2] = C.tensordot(self.A[n2], axes=(1, self.left))


    def canonize_sweep(self, to='last', normalize=True):
        r"""
        Sweep though the mps and cannonize it toward the first of last site.

        At the end, attach the trivial central block to the end of the chain, so that there is no central block left.

        Parameters
        ----------
        to : str
            'last' or 'first'.

        normalize : bool
            If true, S is normalized to 1 according to the standard 2-norm.
        """
        self.absorb_central(to=to)
        for n in self.sweep(to=to):
            self.orthogonalize_site(n=n, to=to, normalize=normalize)
            self.absorb_central(to=to)
        return self


    def truncate_sweep(self, to='last', normalize=True, opts=None):
        r"""
        Sweep though the mps, cannonize it toward the first or last site truncating via svd.

        Truncation makes sense if mps is in the cannonical form in the oposite direction to that of the current sweep.

        Parameters
        ----------
        to : str
            'last' or 'first'.

        normalize : bool
            If true, S is normalized to 1 according to the standard 2-norm.

        opts : dict
            Options passed for svd; includes information how to truncate.

        Return
        ------
        discarded : float
            maximal norm of discarded singular values normalized by the remining ones
        """
        discarded_max = 0.
        if opts is None:
            opts = {'tol': 1e-12}
        for n in self.sweep(to=to):
            self.orthogonalize_site(n=n, to=to, normalize=normalize)
            discarded = self.diagonalize_central(opts=opts, normalize=normalize)
            discarded_max = max(discarded_max, discarded)
            self.absorb_central(to=to)
        return discarded_max


    def merge_two_sites(self, bd):
        r"""
        Merge two neighbouring mps sites and return the resulting tensor. 

        Fuse physical indices.

        Parameters
        ----------
        bd : tuple
            (n, n + 1), index of two sites to merge.

        Returns
        ----------
        out : Tensor
            tensor formed from A[n] and A[n + 1]
        """
        nl, nr = bd
        AA = self.A[nl].tensordot(self.A[nr], axes=(self.right, self.left))
        axes = (0, (1, 2), 3) if self.nr_phys == 1 else (0, (1, 3), (2, 4), 5)
        AA.fuse_legs_hard(axes=axes, inplace=True)
        return AA


    def unmerge_two_sites(self, AA, bd, opts_svd):
        r"""
        Unmerge tensor into two neighbouring mps sites and a central block using svd to trunctate the bond dimension.

        Provided tensor should be fused consistently with `merge_two_sites`.

        Parameters
        ----------
        AA : Tensor
            Tensor to be unmerged. It gets unfused in place during the operation.

        bd : tuple
            (n, n + 1), index of two sites to merge.

        Returns
        ----------
        out : Tensor
            tensor formed from A[n] and A[n + 1]
        """
        nl, nr = bd
        axes = (1,) if self.nr_phys == 1 else (1, 2)
        AA.unfuse_legs_hard(axes=axes, inplace=True)
        axes = ((0, 1), (2, 3)) if self.nr_phys == 1 else ((0, 1, 3), (2, 4, 5))
        self.pC = bd
        self.A[nl], self.A[bd], self.A[nr] = AA.svd(axes=axes, sU=-1, **opts_svd)


    def get_bond_dimensions(self):
        r"""
        Returns bond dimensions of mps.

        Returns
        -------
        Ds : list
            list of bond dimensions on virtual legs from first to last,
            including "trivial" leftmost and rightmost virtual indices.
        """
        Ds = [self.A[n].get_shape(self.left[0]) for n in self.sweep(to='last')]
        Ds.append(self.A[self.last].get_shape(self.right[0]))
        return Ds


    def get_bond_charges_dimensions(self):
        r"""
        Returns charges and dimensions of all virtual mps bonds.

        Returns
        -------
        tDs : list
            list of charges and corresponding dimensions on virtual mps bonds from first to last,
            including "trivial" leftmost and rightmost virtual indices.
        """
        tDs = [self.A[n].get_leg_structure(self.left[0]) for n in self.sweep(to='last')]
        tDs.append(self.A[self.last].get_leg_structure(self.right[0]))
        return tDs


    def get_entropy(self, alpha=1):
        r"""
        Entropy and spectral information on a cut.

        Returns
        -------
        Ds : list
            set of bond dimension on cuts
        Schmidt_spectrum : list
            Schmidt spectrum on each bond
        Smin : list
            smallest singular value on a cut
        Entropy : list
            list of bond entropies on virtual legs.
        SV : list
            list of Schmidt values saved as a directory
        """
        pass


def add(states=None, weights=None):
    pass
