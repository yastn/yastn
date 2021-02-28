""" Environments for the <mps| mpo |mps> contractions. """
import logging
from .. import ncon, match_legs


class FatalError(Exception):
    pass


logger = logging.getLogger('yast.mps.geometry')


def measure_mpo(bra, op, ket):
    r"""
    Calculate overlap <bra|ket>

    Returns
    -------
    overlap : float or complex
    """
    env = Env3(bra=bra, op=op, ket=ket)
    return env.overlap()


class Env3:
    """
    The class combines environments of mps+mps for calculation of expectation values, overlaps, etc.
    """

    def __init__(self, bra=None, op=None, ket=None, on_aux=False):
        r"""
        Initialize structure for :math:`\langle {\rm bra} | {\rm opp} | {\rm ket} \rangle` related operations.

        Parameters
        ----------
        bra : mps
            mps for :math:`| {\rm bra} \rangle`. If None, it is the same as ket.
        opp : mps
            mps for opperator opp.
        ket : mps
            mps for :math:`| {\rm ket} \rangle`.
        """
        self.ket = ket
        self.bra = bra if bra is not None else ket
        self.op = op
        self.g = ket.g
        self.nr_phys = self.ket.nr_phys
        self.on_aux = on_aux
        self.F = {}  # dict for environments
        if self.bra.nr_phys != self.ket.nr_phys:
            logger.error(
                'bra and ket should have the same number of physical legs.')
            raise FatalError

        # set environments at boundaries
        ff = self.g.first
        ll = self.g.last

        # left boundary
        self.F[(None, ff)] = match_legs(tensors=[self.bra.A[ff], self.op.A[ff], self.ket.A[ff]],
                                        legs=[self.bra.left[0], self.op.left[0], self.ket.left[0]],
                                        conjs=[1, 0, 0], val='ones')

        # right boundary
        self.F[(None, ll)] = match_legs(tensors=[self.ket.A[ll], self.op.A[ll], self.bra.A[ll]],
                                        legs=[self.ket.right[0], self.op.right[0], self.bra.right[0]],
                                        conjs=[0, 0, 1], val='ones')

    def update(self, n, towards):
        r"""
        Update environment including site n, in the direction given by torawds.

        Parameters
        ----------
        n: int
            index of site to include to the environment
        towards : int
            towards which site (end) is the environment facing.
        """

        nnext, leg, nprev = self.g.from_site(n, towards)

        if self.nr_phys == 1 and leg == 1:
            self.F[(n, nnext)] = self.F[(nprev, n)].dot(self.bra.A[n], axes=((0), (0)), conj=(0, 1)).dot(
                self.op.A[n], axes=((0, 2), (0, 1))).dot(self.ket.A[n], axes=((0, 2), (0, 1)))
        elif self.nr_phys == 1 and leg == 0:
            self.F[(n, nnext)] = self.ket.A[n].dot(self.F[(nprev, n)], axes=((2), (0))).dot(
                self.op.A[n], axes=((1, 2), (2, 3))).dot(self.bra.A[n], axes=((3, 1), (1, 2)), conj=(0, 1))
        elif self.nr_phys == 2 and leg == 1 and not self.on_aux:
            self.F[(n, nnext)] = ncon([self.bra.A[n], self.F[(nprev, n)], self.ket.A[n], self.op.A[n]], ((
                4, 5, 6, -1), (4, 2, 1), (1, 3, 6, -3), (2, 5, 3, -2)), (1, 0, 0, 0))
        elif self.nr_phys == 2 and leg == 0 and not self.on_aux:
            self.F[(n, nnext)] = ncon([self.ket.A[n], self.F[(nprev, n)], self.op.A[n], self.bra.A[n]],
                                      ((-1, 2, 6, 1), (1, 3, 5), (-2, 4, 2, 3), (-3, 4, 6, 5)), (0, 0, 0, 1))
        elif self.nr_phys == 2 and leg == 1 and self.on_aux:
            self.F[(n, nnext)] = ncon([self.bra.A[n], self.F[(nprev, n)], self.ket.A[n], self.op.A[n]], ((
                4, 6, 5, -1), (4, 2, 1), (1, 6, 3, -3), (2, 5, 3, -2)), (1, 0, 0, 0))
        else:  # self.nr_phys == 2 and leg == 0 and self.on_aux:
            self.F[(n, nnext)] = ncon([self.ket.A[n], self.F[(nprev, n)], self.op.A[n], self.bra.A[n]],
                                      ((-1, 6, 2, 1), (1, 3, 5), (-2, 4, 2, 3), (-3, 6, 4, 5)), (0, 0, 0, 1))

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
        return self.F[(None, n0)].dot(self.F[(n0, None)], axes=((0, 1, 2), (2, 1, 0))).to_number()

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
        return self.F[bd].dot(self.F[bd[::-1]], axes=((0, 1, 2), (2, 1, 0))).to_number()

    def Heff0(self, C, bd, conj=False):
        r"""
        Action of Heff on central site.

        Parameters
        ----------
        C : tensor
            a central site
        bd : tuple
            and a bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]
        conj : boolean
            True to calculate Heff^\daggerC

        Returns
        -------
        out : tensor
            Heff0 * C
        """
        bd = self.g.order_bond(bd)
        if conj:
            return self.F[bd].dot(C, axes=((0), (0)), conj=(0, 1)).dot(self.F[bd[::-1]], axes=((0, 2), (1, 2))).conj()
        else:
            return self.F[bd].dot(C, axes=((2), (0))).dot(self.F[bd[::-1]], axes=((1, 2), (1, 0)))

    def Heff1(self, A, n, conj=False):
        r"""Action of Heff on central site.

        Parameters
        ----------
        A : tensor
            site tensor
        n : int
            index of corresponding site
        conj : boolean
            True to calculate Heff^\daggerA = (Heff^T A.conj).conj

        Returns
        -------
        out : tensor
            Heff1 * A
        """
        nl, nr = self.g.order_neighbours(n)
        if conj:
            if self.nr_phys == 1:
                return self.F[(nl, n)].dot(A, axes=((0), (0)), conj=(0,1)).dot(self.op.A[n], axes=((0, 2), (0, 1))).dot(self.F[(nr, n)], axes=((1, 3), (2, 1))).conj()
            elif self.nr_phys == 2:
                return ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]], ((1, 2, -1), (1, 3, -3, 4), (2, 3, -2, 5), (-4, 5, 4)), (0, 1, 0, 0)).conj()
            else:  # self.nr_phys == 2 and self.on_aux:
                return ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]], ((1, 2, -1), (1, -2, 3, 4), (2, 3, -3, 5), (-4, 5, 4)), (0, 1, 0, 0)).conj()
        else:
            if self.nr_phys == 1:
                return self.F[(nl, n)].dot(A, axes=((2), (0))).dot(self.op.A[n], axes=((1, 2), (0, 2))).dot(self.F[(nr, n)], axes=((1, 3), (0, 1)))
            elif self.nr_phys == 2 and not self.on_aux:
                return ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]], ((-1, 2, 1), (1, 3, -3, 4), (2, -2, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
            else:  # self.nr_phys == 2 and self.on_aux:
                return ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]], ((-1, 2, 1), (1, -2, 3, 4), (2, -3, 3, 5), (4, 5, -4)), (0, 0, 0, 0))

    def Heff2(self, AA, n, conj=False):
        r"""Action of Heff on central site.

        Parameters
        ----------
        AA : tensor
            merged tensor for 2 sites
        n : int
            index of the left site
        conj : boolean
            True to calculate Heff^\daggerAA

        Returns
        -------
        out : tensor
            Heff2 * AA
        """
        nl, nn = self.g.order_neighbours(n)
        _, nr = self.g.order_neighbours(nn)
        OO = self.op.A[n].dot(self.op.A[nn], axes=((3), (0)))
        if conj:
            if self.nr_phys == 1:
                return self.F[(nl, n)].dot(AA, axes=((0), (0)), conj=(0,1)).dot(OO, axes=((0,2,3), (0,1,3))).dot(self.F[(nr, nn)], axes=((1,4), (2,1))).conj()
            elif self.nr_phys == 2 and not self.on_aux:
                return ncon([self.F[(nl, n)], AA, OO, self.F[(nr, nn)]], ((1, 2, -1), (1, 3, 4, -4, -5, 5), (2, 3, 4, -2, -3, 6), (-6, 6, 5)), (0, 1, 0, 0)).conj()
            # else: self.nr_phys == 2 and self.on_aux:
            return ncon([self.F[(nl, n)], AA, OO, self.F[(nr, nn)]], ((1, 2, -1), (1, -2, -3, 3, 4, 5), (2, 3, 4, -4, -5, 6), (-6, 6, 5)), (0, 1, 0, 0)).conj()
        else:
            if self.nr_phys == 1:
                return self.F[(nl, n)].dot(AA, axes=((2), (0))).dot(OO, axes=((1,2,3), (0,2,4))).dot(self.F[(nr, nn)], axes=((1,4), (0,1)))
            elif self.nr_phys == 2 and not self.on_aux:
                return ncon([self.F[(nl, n)], AA, OO, self.F[(nr, nn)]], ((-1, 2, 1), (1, 3, 4, -4, -5, 5), (2, -2, 3, -3, 4, 6), (5, 6, -6)), (0, 0, 0, 0))
            #else: self.nr_phys == 2 and self.on_aux:
            return ncon([self.F[(nl, n)], AA, OO, self.F[(nr, nn)]], ((-1, 2, 1), (1, -2, -3, 3, 4, 5), (2, -4, 3, -5, 4, 6), (5, 6, -6)), (0, 0, 0, 0))

    def Heff2_group(self, AA, n, conj=False):
        r"""Action of Heff on central site.

        Parameters
        ----------
        AA : tensor
            merged tensor for 2 sites
        n : int
            index of the left site
        conj : boolean
            True to calculate Heff^\daggerAA

        Returns
        -------
        out : tensor
            Heff2 * AA
        """
        nl, nn = self.g.order_neighbours(n)
        _, nr = self.g.order_neighbours(nn)

        if not hasattr(self, 'op_merged'):
            self.op_merged = {}
        if not (n, nn) in self.op_merged:
            OO = self.op.A[n].dot(self.op.A[nn], axes=((3), (0)))
            OO = OO.fuse_legs(axes=(0, (1, 3), (2, 4), 5))
            # OO, _ = OO.group_legs(axes=(2, 4), new_s=-1)
            # OO, _ = OO.group_legs(axes=(1, 3), new_s=1)
            self.op_merged[(n, nn)] = OO
        OO = self.op_merged[(n, nn)]
        if conj:
            if self.nr_phys == 1:
                return self.F[(nl, n)].dot(AA.conj(), axes=((0), (0))).dot(OO, axes=((0, 2), (0, 1))).dot(self.F[(nr, nn)], axes=((1, 3), (2, 1))).conj()
        else:
            if self.nr_phys == 1:
                return ncon([self.F[(nl, n)], AA, OO, self.F[(nr, nn)]], ((-1, 2, 1), (1, 3, 4), (2, -2, 3, 5), (4, 5, -3)), (0, 0, 0, 0))

    def project_ket_on_bra(self, n):
        r"""Project ket on a n-th site of bra.

        It is equall to the overlap <bra|op|ket> up to the contribution from n-th site of bra.

        Parameters
        ----------
        n : int
            index of site

        Returns
        -------
        out : tensor
            result of projection
        """
        return self.Heff1(self, self.ket.A[n], n, conj=False)
