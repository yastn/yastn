""" Common functions for Env2 and Env3. """

def setup(self, to='last'):
    r"""
    Setup all environments in the direction given by to.

    Parameters
    ----------
    to : str
        'first' or 'last'.
    """
    for n in self.ket.sweep(to=to):
        self.update(n, to=to)


def clear_site(self, n):
    r""" Clear environments pointing from site n. """
    self.F.pop((n, n - 1), None)
    self.F.pop((n, n + 1), None)
