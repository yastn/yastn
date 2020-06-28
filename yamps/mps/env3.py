from yamps.tensor import ncon
from .mps import MpsError

####################################################
#     environment for <bra|opp|ket> operations     #
####################################################


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
            raise MpsError('bra and ket should have the same number of physical legs.')

        # set environments at boundaries
        ff = self.g.first
        ll = self.g.last
        tn = self.ket.A[ff]

        # left boundary
        self.F[(None, ff)] = tn.match_legs(tensors=[self.bra.A[ff], self.op.A[ff], self.ket.A[ff]],
                                           legs=[self.bra.left[0], self.op.left[0], self.ket.left[0]],
                                           conjs=[1, 0, 0], val='ones')

        # right boundary
        self.F[(None, ll)] = tn.match_legs(tensors=[self.ket.A[ll], self.op.A[ll], self.bra.A[ll]],
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
            self.F[(n, nnext)] = ncon([self.bra.A[n], self.F[(nprev, n)], self.ket.A[n], self.op.A[n]], ((4, 5, -1), (4, 2, 1), (1, 3, -3), (2, 5, 3, -2)), (1, 0, 0, 0))
        elif self.nr_phys == 1 and leg == 0:
            self.F[(n, nnext)] = ncon([self.ket.A[n], self.F[(nprev, n)], self.op.A[n], self.bra.A[n]], ((-1, 2, 1), (1, 3, 5), (-2, 4, 2, 3), (-3, 4, 5)), (0, 0, 0, 1))
        elif self.nr_phys == 2 and leg == 1 and not self.on_aux:
            self.F[(n, nnext)] = ncon([self.bra.A[n], self.F[(nprev, n)], self.ket.A[n], self.op.A[n]], ((4, 5, 6, -1), (4, 2, 1), (1, 3, 6, -3), (2, 5, 3, -2)), (1, 0, 0, 0))
        elif self.nr_phys == 2 and leg == 0 and not self.on_aux:
            self.F[(n, nnext)] = ncon([self.ket.A[n], self.F[(nprev, n)], self.op.A[n], self.bra.A[n]], ((-1, 2, 6, 1), (1, 3, 5), (-2, 4, 2, 3), (-3, 4, 6, 5)), (0, 0, 0, 1))
        elif self.nr_phys == 2 and leg == 1 and self.on_aux:
            self.F[(n, nnext)] = ncon([self.bra.A[n], self.F[(nprev, n)], self.ket.A[n], self.op.A[n]], ((4, 6, 5, -1), (4, 2, 1), (1, 6, 3, -3), (2, 5, 3, -2)), (1, 0, 0, 0))
        else:  # self.nr_phys == 2 and leg == 0 and self.on_aux:
            self.F[(n, nnext)] = ncon([self.ket.A[n], self.F[(nprev, n)], self.op.A[n], self.bra.A[n]], ((-1, 6, 2, 1), (1, 3, 5), (-2, 4, 2, 3), (-3, 6, 4, 5)), (0, 0, 0, 1))

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

    def Heff0(self, C, bd):
        r"""
        Action of Heff on central site.

        Parameters
        ----------
        C : tensor
            a central site
        bd : tuple
            and a bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]

        Returns
        -------
        out : tensor
            Heff0 * C
        """
        bd = self.g.order_bond(bd)
        return ncon([self.F[bd], C, self.F[bd[::-1]]], ((-1, 2, 1), (1, 3), (3, 2, -2)), (0, 0, 0))

    def Heff1(self, A, n):
        r"""Action of Heff on central site.

        Parameters
        ----------
        A : tensor
            site tensor
        n : int
            index of corresponding site

        Returns
        -------
        out : tensor
            Heff1 * A
        """
        nl, nr = self.g.order_neighbours(n)

        if self.nr_phys == 1:
            return ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]], ((-1, 2, 1), (1, 3, 4), (2, -2, 3, 5), (4, 5, -3)), (0, 0, 0, 0))
        elif self.nr_phys == 2 and not self.on_aux:
            return ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]], ((-1, 2, 1), (1, 3, -3, 4), (2, -2, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
        else:  # self.nr_phys == 2 and self.on_aux:
            return ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]], ((-1, 2, 1), (1, -2, 3, 4), (2, -3, 3, 5), (4, 5, -4)), (0, 0, 0, 0))

    def Heff2(self, AA, n):
        r"""Action of Heff on central site.

        Parameters
        ----------
        AA : tensor
            merged tensor for 2 sites
        n : int
            index of the left site

        Returns
        -------
        out : tensor
            Heff2 * AA
        """
        nl, nn = self.g.order_neighbours(n)
        _, nr = self.g.order_neighbours(nn)

        OO = ncon([self.op.A[n], self.op.A[nn]], ((-1, -2, -4, 1), (1, -3, -5, -6)))

        if self.nr_phys == 1:
            return ncon([self.F[(nl, n)], AA, OO, self.F[(nr, nn)]], ((-1, 2, 1), (1, 3, 4, 5), (2, -2, -3, 3, 4, 6), (5, 6, -4)), (0, 0, 0, 0))
        elif self.nr_phys == 2 and not self.on_aux:
            return ncon([self.F[(nl, n)], AA, OO, self.F[(nr, nn)]], ((-1, 2, 1), (1, 3, 4, -4, -5, 5), (2, -2, -3, 3, 4, 6), (5, 6, -6)), (0, 0, 0, 0))
        else:  # self.nr_phys == 2 and self.on_aux:
            return ncon([self.F[(nl, n)], AA, OO, self.F[(nr, nn)]], ((-1, 2, 1), (1, -2, -3, 3, 4, 5), (2, -4, -5, 3, 4, 6), (5, 6, -4)), (0, 0, 0, 0))

    def Heff2_group(self, AA, n):
        r"""Action of Heff on central site.

        Parameters
        ----------
        AA : tensor
            merged tensor for 2 sites
        n : int
            index of the left site

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
            OO = ncon([self.op.A[n], self.op.A[nn]], ((-1, -2, -4, 1), (1, -3, -5, -6)))
            OO, _ = OO.group_legs(axes=(3, 4), new_s=-1)
            OO, _ = OO.group_legs(axes=(1, 2), new_s=1)
            self.op_merged[(n, nn)] = OO

        if self.nr_phys == 1:
            return ncon([self.F[(nl, n)], AA, self.op_merged[(n, nn)], self.F[(nr, nn)]], ((-1, 2, 1), (1, 3, 4), (2, -2, 3, 5), (4, 5, -3)), (0, 0, 0, 0))
