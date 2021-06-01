""" Environments for the <mps|mps> contractions. """
from yast import match_legs, tensordot
from ._mps import YampsError


def measure_overlap(bra, ket):
    r"""
    Calculate overlap <bra|ket>

    Returns
    -------
    overlap : float or complex
    """
    env = Env2(bra=bra, ket=ket)
    env.setup(to='first')
    return env.measure(bd=(-1, 0))


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
        self.N = ket.N
        self.nr_phys = ket.nr_phys
        self.F = {}  # dict for environments

        if self.bra.nr_phys != self.ket.nr_phys:
            raise YampsError('bra and ket should have the same number of physical legs.')
        if self.bra.N != self.ket.N:
            raise YampsError('bra and ket should have the same number of sites.')

        # set environments at boundaries
        ll = self.N - 1
        # left boundary
        self.F[(-1, 0)] = match_legs(tensors=[self.bra.A[0], self.ket.A[0]],
                                        legs=[self.bra.left[0], self.ket.left[0]],
                                        conjs=[1, 0], val='ones')
        # right boundary
        self.F[(ll + 1, ll)] = match_legs(tensors=[self.ket.A[ll], self.bra.A[ll]],
                                        legs=[self.ket.right[0], self.bra.right[0]],
                                        conjs=[0, 1], val='ones')


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
        if to == 'first':
            if self.nr_phys == 1:
                T1 = tensordot(self.ket.A[n], self.F[(n + 1, n)], axes=(2, 0))
                self.F[(n, n - 1)] = tensordot(T1, self.bra.A[n], axes=((1, 2), (1, 2)), conj=(0, 1))
            else:
                T1 = tensordot(self.ket.A[n], self.F[(n + 1, n)], axes=(3, 0))
                self.F[(n, n - 1)] = tensordot(T1, self.bra.A[n], axes=((1, 2, 3), (1, 2, 3)), conj=(0, 1))
        elif to == 'last':
            if self.nr_phys == 1:
                T1 = tensordot(self.F[(n - 1, n)], self.bra.A[n], axes=(0, 0), conj=(0, 1))
                self.F[(n, n + 1)] = tensordot(T1, self.ket.A[n], axes=((0, 1), (0, 1)))
            else:
                T1 = tensordot(self.F[(n - 1, n)], self.bra.A[n], axes=(0, 0), conj=(0, 1))
                self.F[(n, n + 1)] = tensordot(T1, self.ket.A[n], axes=((0, 1, 2), (0, 1, 2)))


    def measure(self, bd=None):
        r"""
        Calculate overlap between environments at nn bond

        Parameters
        ----------
        bd: tuple
            index of bond at which to calculate overlap.
            If None, it is measured at bond (-1, 0) before the first site

        Returns
        -------
        overlap : float or complex
        """
        if bd is None:
            bd = (-1, 0)
        return self.F[bd].tensordot(self.F[bd[::-1]], axes=((0, 1), (1, 0))).to_number()


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

        nl, nr = n - 1, n + 1

        if self.nr_phys == 1:
            return self.F[(nl, n)].tensordot(self.ket.A[n], axes=(1, 0)).tensordot(self.F[(nr, n)], axes=(2, 0))
        return self.F[(nl, n)].tensordot(self.ket.A[n], axes=(1, 0)).tensordot(self.F[(nr, n)], axes=(3, 0))
