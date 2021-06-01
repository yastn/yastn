""" Environments for the <mps| mpo |mps> contractions. """
from yast import ncon, match_legs, tensordot
from ._mps import YampsError


def measure_mpo(bra, op, ket):
    r"""
    Calculate overlap <bra|ket>

    Returns
    -------
    overlap : float or complex
    """
    env = Env3(bra=bra, op=op, ket=ket)
    env.setup(to='first')
    return env.measure(bd=(-1, 0))


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
        self.nr_phys = self.ket.nr_phys
        self.on_aux = on_aux
        self.N = self.ket.N
        self.F = {}  # dict for environments

        if self.bra.nr_phys != self.ket.nr_phys:
            raise YampsError('bra and ket should have the same number of physical legs.')
        if self.bra.N != self.N or self.op.N != self.N:
            raise YampsError('bra, op, ket should have the same number of sites.')

        # left boundary
        self.F[(-1, 0)] = match_legs(tensors=[self.bra.A[0], self.op.A[0], self.ket.A[0]],
                                        legs=[self.bra.left[0], self.op.left[0], self.ket.left[0]],
                                        conjs=[1, 0, 0], val='ones')
        # right boundary
        ll = self.N - 1
        self.F[(ll + 1, ll)] = match_legs(tensors=[self.ket.A[ll], self.op.A[ll], self.bra.A[ll]],
                                        legs=[self.ket.right[0], self.op.right[0], self.bra.right[0]],
                                        conjs=[0, 0, 1], val='ones')


    from ._env import setup, clear_site


    def update(self, n, to='last'):
        r"""
        Update environment including site n, in the direction given by to.

        Parameters
        ----------
        n: int
            index of site to include to the environment

        to : str
            'first' or 'last'.
        """

        if to == 'last':
            if self.nr_phys == 1:
                self.F[(n, n + 1)] = ncon([self.bra.A[n], self.F[(n - 1, n)], self.ket.A[n], self.op.A[n]],
                                            ((4, 5, -1), (4, 2, 1), (1, 3, -3), (2, 5, 3, -2)), conjs=(1, 0, 0, 0))
            elif self.nr_phys == 2 and not self.on_aux:
                self.F[(n, n + 1)] = ncon([self.bra.A[n], self.F[(n - 1, n)], self.ket.A[n], self.op.A[n]],
                                            ((4, 5, 6, -1), (4, 2, 1), (1, 3, 6, -3), (2, 5, 3, -2)), (1, 0, 0, 0))
            else:  # self.nr_phys == 2 and self.on_aux:
                self.F[(n, n + 1)] = ncon([self.bra.A[n], self.F[(n - 1, n)], self.ket.A[n], self.op.A[n]],
                                            ((4, 6, 5, -1), (4, 2, 1), (1, 6, 3, -3), (2, 5, 3, -2)), (1, 0, 0, 0))
        elif to == 'first':
            if self.nr_phys == 1:
                self.F[(n, n - 1)] = ncon([self.ket.A[n], self.F[(n + 1, n)], self.op.A[n], self.bra.A[n]],
                                            ((-1, 2, 1), (1, 3, 5), (-2, 4, 2, 3), (-3, 4, 5)), (0, 0, 0, 1))
            elif self.nr_phys == 2 and not self.on_aux:
                self.F[(n, n - 1)] = ncon([self.ket.A[n], self.F[(n + 1, n)], self.op.A[n], self.bra.A[n]],
                                            ((-1, 2, 6, 1), (1, 3, 5), (-2, 4, 2, 3), (-3, 4, 6, 5)), (0, 0, 0, 1))
            else:  # self.nr_phys == 2 and self.on_aux:
                self.F[(n, n - 1)] = ncon([self.ket.A[n], self.F[(n + 1, n)], self.op.A[n], self.bra.A[n]],
                                            ((-1, 6, 2, 1), (1, 3, 5), (-2, 4, 2, 3), (-3, 6, 4, 5)), (0, 0, 0, 1))


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
            bd = (-1, 0)
        return tensordot(self.F[bd], self.F[bd[::-1]], axes=((0, 1, 2), (2, 1, 0))).to_number()


    def Heff0(self, C, bd):
        r"""
        Action of Heff on central site.

        Parameters
        ----------
        C : tensor
            a central site
        bd : tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]

        Returns
        -------
        out : tensor
            Heff0 * C
        """
        bd, ibd = (bd[::-1], bd) if bd[1] < bd[0] else (bd, bd[::-1])
        return tensordot(tensordot(self.F[bd], C, axes=(2, 0)), self.F[ibd], axes=((1, 2), (1, 0)))


    def Heff1(self, A, n):
        r"""
        Action of Heff on a single site mps tensor.

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
        nl, nr = n - 1, n + 1
        if self.nr_phys == 1:
            return ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]],
                        ((-1, 2, 1), (1, 3, 4), (2, -2, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
        if self.nr_phys == 2 and not self.on_aux:
            return ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]],
                        ((-1, 2, 1), (1, 3, -3, 4), (2, -2, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
        return ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]],
                    ((-1, 2, 1), (1, -2, 3, 4), (2, -3, 3, 5), (4, 5, -4)), (0, 0, 0, 0))


    def Heff2(self, AA, bd):
        r"""
        Action of Heff on merged two-sites mps tensors.

        Parameters
        ----------
        AA : tensor
            merged tensor for two sites
        bd : tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]

        Returns
        -------
        out : tensor
            Heff2 * AA
        """
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        nl, nr = n1 - 1, n2 + 1

        OO = tensordot(self.op.A[n1], self.op.A[n1], axes=(3, 0)).transpose(axes=(0, 1, 3, 2, 4, 5))

        if self.nr_phys == 1:
            return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
                        ((-1, 2, 1), (1, 3, 4, 5), (2, -2, -3, 3, 4, 6), (5, 6, -4)), (0, 0, 0, 0))
        if self.nr_phys == 2 and not self.on_aux:
            return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
                        ((-1, 2, 1), (1, 3, 4, -4, -5, 5), (2, -2, -3, 3, 4, 6), (5, 6, -6)), (0, 0, 0, 0))
        # else: self.nr_phys == 2 and self.on_aux:
        return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
                    ((-1, 2, 1), (1, -2, -3, 3, 4, 5), (2, -4, -5, 3, 4, 6), (5, 6, -6)), (0, 0, 0, 0))


    def Heff2_group(self, AA, bd):
        r"""Action of Heff on central site.

        Parameters
        ----------
        AA : tensor
            merged tensor for 2 sites.
            Physical legs should be fused turning it effectivly into 1-site update.
        bd : tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]

        Returns
        -------
        out : tensor
            Heff2 * AA
        """
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        bd, nl, nr = (n1, n2), n1 - 1, n2 + 1

        if not hasattr(self, 'op_merged'):
            self.op_merged = {}
        if bd not in self.op_merged:
            OO = tensordot(self.op.A[n1], self.op.A[n2], axes=(3, 0))
            OO = OO.fuse_legs(axes=(0, (1, 3), (2, 4), 5))
            self.op_merged[bd] = OO
        OO = self.op_merged[bd]

        if self.nr_phys == 1:
            return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
                        ((-1, 2, 1), (1, 3, 4), (2, -2, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
        if self.nr_phys == 2 and not self.on_aux:
            return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
                        ((-1, 2, 1), (1, 3, -3, 4), (2, -2, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
        return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
                    ((-1, 2, 1), (1, -2, 3, 4), (2, -3, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
