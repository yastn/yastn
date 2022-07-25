from ..initialize import make_config
from ..sym import sym_Z2, sym_U1
from ..tensor import YastError, Tensor

class SpinlessFermions:
    """ Predefine operators for spinless fermions. """

    def __init__(self, sym='U1', **kwargs):
        """ 
        Generator of standard operators for single fermionic species (2-dimensional Hilbert space).

        Predefine identity, rising and lowering operators, and density operators.

        Other config parameters can be provided, see :meth:`yast.make_config`

        fermionic is set to True.

        Parameters
        ----------
        sym : str
            Should be 'Z2', or 'U1'.

        Notes
        -----
        Assume the following conventions:
        For both Z2 and U1, charge t=0 <=> |0>, t=1 <=> |1>
        """
        if not sym in ('Z2', 'U1'):
            raise YastError("For SpinlessFermions sym should be in ('Z2', 'U1').")
        self._sym = sym
        kwargs['fermionic'] = True
        import_sym = {'Z2': sym_Z2, 'U1': sym_U1}
        kwargs['sym'] = import_sym[sym]
        self.config = make_config(**kwargs)
        self.s = (1, -1)

    def I(self):
        """ Identity operator. """
        I = Tensor(config=self.config, s=self.s, n=0)
        I.set_block(ts=(0, 0), Ds=(1, 1), val=1)
        I.set_block(ts=(1, 1), Ds=(1, 1), val=1)
        return I

    def n(self):
        """ Particle number operator. """
        n = Tensor(config=self.config, s=self.s, n=0)
        n.set_block(ts=(0, 0), Ds=(1, 1), val=0)
        n.set_block(ts=(1, 1), Ds=(1, 1), val=1)
        return n

    def cp(self):
        """ Rising operator. """
        cp = Tensor(config=self.config, s=self.s, n=1)
        cp.set_block(ts=(1, 0), Ds=(1, 1), val=1)
        return cp

    def c(self):
        """ Lowering operator. """
        n = 1 if self._sym == 'Z2' else -1
        c = Tensor(config=self.config, s=self.s, n=n)
        c.set_block(ts=(0, 1), Ds=(1, 1), val=1)
        return c
