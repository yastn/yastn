import logging


class FatalError(Exception):
    pass


logger = logging.getLogger('yast.mps.geometry')


##################################
#     geometry of the lattice    #
##################################


class Geometry:
    """
    Operations encoding geometry. This is a complete overkill in the context of mps itself.
    Sites of mps are numbered with :math:`0, 1, 2, 3, \\ldots, N-1` with :math:`0` corresponds to the leftmost site.
    """
 

    def from_site(self, n, towards):
        r"""
        Next site, leg connecting to that site, and other neighbouring sites.

        If n == towards == leaf, than return direction going outside of the network.

        Parameters
        ----------
        n : int
            index of the reference site.
        towards : int
            index of the site towards which to go.

        Returns
        -------
        nnext, leg, nprev : int, int, ins(s)
        """
        if towards < n:
            nnext, leg = n - 1, 0
            nprev = n + 1 if n < self.last else None
        elif n < towards:
            nnext, leg = n + 1, 1
            nprev = n - 1 if n > self.first else None
        elif n == self.first:  # and n == towards:
            nnext, leg = None, 0
            nprev = n + 1
        elif n == self.last:  # and n == towards:
            nnext, leg = None, 1
            nprev = n - 1
        else:
            nnext, leg = None, None
            nprev = [n - 1, n + 1]
        return nnext, leg, nprev


    def from_bond(self, bd, towards):
        r"""
        Returns index of the site going to given towards, leg connecting to that bond, and other site in the bond.

        Parameters
        ----------
        bond : tuple
            index of the reference bond.
        towards : int
            index of the site towards which to go.

        Returns
        -------
        nnext, leg, nprev : int, int, int
        """
        if None in bd:  # the other site in the bond is a leaf -- point toward leaf
            if self.first in bd:
                return self.first, 0, None
            elif self.last in bd:  # else:
                return self.last, 1, None
        elif towards <= bd[0] and towards <= bd[1]:
            return min(bd), 1, max(bd)
        elif bd[0] <= towards and bd[1] <= towards:  # else:
            return max(bd), 0, min(bd)


    def order_bond(self, bd):
        return (min(bd), max(bd))


    def order_neighbours(self, n):
        if n == self.first:
            return None, n + 1
        elif n == self.last:
            return n - 1, None
        else:
            return n - 1, n + 1


    def sweep(self, to='last', df=0, dl=0):
        r"""
        Generator of all sites going from first to last or vice-versa

        Parameters
        ----------
        to : str
            'first or 'last'.
        df, dl : int
            shift by df and dl from first and last site respectively.
        """
        df = abs(df)
        dl = abs(dl)
        if to == 'last':
            return range(self.first + df, self.last + 1 - dl)
        elif to == 'first':
            return range(self.last - dl, self.first + df - 1, -1)
