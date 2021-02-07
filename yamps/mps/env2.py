import logging
from yamps.yast import match_legs

class FatalError(Exception):
    pass


logger = logging.getLogger('yamps.mps.env2')


################################################
#     environment for <bra|ket> operations     #
################################################


class Env2:
    """
    The class combines environments of mps+mps for calculation of expectation values, overlaps, etc.
    """

    def __init__(self, bra=None, ket=None):
        r"""
        Initialize structure for :math:`\langle {\rm bra} | {\rm ket} \rangle` related operations.

        Parameters
        ----------
        bra : mps
            mps for :math:`| {\rm bra} \rangle`. If None, it is the same as ket.
        ket : mps
            mps for :math:`| {\rm ket} \rangle`.
        """
        self.ket = ket
        self.bra = bra if bra is not None else ket
        self.g = ket.g
        self.nr_phys = self.ket.nr_phys
        self.F = {}  # dict for environments

        if self.bra.nr_phys != self.ket.nr_phys:
            logger.error('bra and ket should have the same number of physical legs.')
            raise FatalError

        # set environments at boundaries
        ff = self.g.first
        ll = self.g.last

        # left boundary
        self.F[(None, ff)] = match_legs(tensors=[self.bra.A[ff], self.ket.A[ff]],
                                        legs=[self.bra.left[0], self.ket.left[0]],
                                        conjs=[1, 0], val='ones')

        # right boundary
        self.F[(None, ll)] = match_legs(tensors=[self.ket.A[ll], self.bra.A[ll]],
                                        legs=[self.ket.right[0], self.bra.right[0]],
                                        conjs=[0, 1], val='ones')

    def update(self, n, towards):
        r"""
        Update environment including site n, in the direction given by towards.

        Parameters
        ----------
        n: int
            index of site to include to the environment
        towards : int
            towards which site (end) is the environment facing.
        """

        nnext, leg, nprev = self.g.from_site(n, towards)

        if self.nr_phys == 1 and leg == 1:
            self.F[(n, nnext)] = self.F[(nprev, n)].dot(self.bra.A[n], axes=((0),(0)), conj=(0,1)).dot(self.ket.A[n], axes=((0,1),(0,1)))
        elif self.nr_phys == 1 and leg == 0:
            self.F[(n, nnext)] = self.ket.A[n].dot(self.F[(nprev, n)], axes=((2),(0))).dot(self.bra.A[n], axes=((1,2),(1,2)), conj=(0,1))
        elif self.nr_phys == 2 and leg == 1:
            self.F[(n, nnext)] = self.F[(nprev, n)].dot(self.bra.A[n], axes=((0),(0)), conj=(0,1)).dot(self.ket.A[n], axes=((0,1,2),(0,1,2)))
        else:  # self.nr_phys == 2 and leg == 0:
            self.F[(n, nnext)] = self.ket.A[n].dot(self.F[(nprev, n)], axes=((3),(0))).dot(self.bra.A[n], axes=((1,2,3),(1,2,3)), conj=(0,1))

    def setup_to_last(self):
        r"""
        Setup all environments in the direction from first site to last site
        """
        for n in self.g.sweep(to='last'):
            self.update(n, towards=self.g.last)

    def setup_to_first(self):
        r"""
        Setup all environments in the direction from last site to first site
        """
        for n in self.g.sweep(to='first'):
            self.update(n, towards=self.g.first)

    def clear_site(self, n):
        r"""
        Clear environments pointing from site n.
        """
        nl, nr = self.g.order_neighbours(n)
        self.F.pop((n, nl), None)
        self.F.pop((n, nr), None)

    def overlap(self):
        r"""
        Sweep from last site to first updating environments and calculates overlap.

        Returns
        -------
        overlap : float or complex
        """
        n0 = None
        for n in self.g.sweep(to='first'):
            self.update(n, towards=self.g.first)
            if n0 is not None:
                self.F.pop((n0, n))
            n0 = n
        return self.F[(None, n0)].dot(self.F[(n0, None)], axes=((0, 1), (1, 0))).to_number()  

    def measure(self, bd=None):
        r"""
        Calculate overlap between environments at nn bond

        Parameters
        ----------
        bd: tuple
            index of bond at which to calculate overlap.
            If None, it is measured at bond (outside, first site)

        Returns
        -------
        overlap : float or complex
        """
        if bd is None:
            bd = (None, self.g.first)
        return self.F[bd].dot(self.F[bd[::-1]], axes=((0, 1), (1, 0))).to_number()

    def project_ket_on_bra(self, n):
        r"""Project ket on a n-th site of bra.

        It is equall to the overlap <bra|ket> up to the contribution from n-th site of bra.

        Parameters
        ----------
        n : int
            index of site

        Returns
        -------
        out : tensor
            result of projection
        """

        nl, nr = self.g.order_neighbours(n)

        if self.nr_phys == 1:
            return self.F[(nl, n)].dot(self.ket.A[n], axes=(1, 0)).dot(self.F[(nr, n)], axes=(2, 0))
        else:
            return self.F[(nl, n)].dot(self.ket.A[n], axes=(1, 0)).dot(self.F[(nr, n)], axes=(3, 0))
