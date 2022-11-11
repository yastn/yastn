from ..initialize import make_config
from ..sym import sym_Z2, sym_U1xU1, sym_U1xU1xZ2
from ..tensor import YastError, Tensor
from ._meta_operators import meta_operators

class SpinfulFermions(meta_operators):
    """ Predefine operators for spinful fermions. """

    def __init__(self, sym='Z2', **kwargs):
        """
        Generator of standard operators for local Hilbert space with two fermionic species (4-dimensional Hilbert space).

        Predefine identity, rising, lowering, and density operators.

        Other config parameters can be provided, see :meth:`yast.make_config`

        Parameters
        ----------
        sym : str
            Should be 'Z2', 'U1xU1' or 'U1xU1xZ2'.

        Notes
        -----
        For 'Z2' and 'U1xU1xZ2', the two species (spin-up and spin-down) are treated as indistinguishable.
        In that case, rising and lowering operators of the two species anti-commute (fermionic statistics is encoded in the Z2 channel)

        For 'U1xU1' the two species (spin-up and spin-down) are treated as distinguishable.
        In that case, rising and lowering operators of the two species commute.
        """
        if not sym in ('Z2', 'U1xU1', 'U1xU1xZ2'):
            raise YastError("For SpinfulFermions sym should be in ('Z2', 'U1xU1', 'U1xU1xZ2').")
        kwargs['fermionic'] = (False, False, True) if sym == 'U1xU1xZ2' else True
        import_sym = {'Z2': sym_Z2, 'U1xU1': sym_U1xU1, 'U1xU1xZ2': sym_U1xU1xZ2}
        kwargs['sym'] = import_sym[sym]
        super().__init__(**kwargs)
        self._sym = sym
        self.operators = ('I', 'n', 'c', 'cp')


    def I(self):
        """ Identity operator. """
        if self._sym == 'Z2':
            I = Tensor(config=self.config, s=self.s, n=0)
            for t in [0, 1]:
                I.set_block(ts=(t, t), Ds=(2, 2), val=[[1, 0], [0, 1]])
        if self._sym == 'U1xU1xZ2':
            I = Tensor(config=self.config, s=self.s, n=(0, 0, 0))
            for t in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
                I.set_block(ts=(t, t), Ds=(1, 1), val=1)
        if self._sym == 'U1xU1':
            I = Tensor(config=self.config, s=self.s, n=(0, 0))
            for t in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                I.set_block(ts=(t, t), Ds=(1, 1), val=1)
        return I

    def n(self, spin='u'):
        """
        Particle number operator.

        Parameters
        ----------
        spin : str
            'u' for spin-up, 'd' for spin-down.
        """
        return self.cp(spin=spin) @ self.c(spin=spin)

    def cp(self, spin='u'):
        """
        Rising operator.

        Parameters
        ----------
        spin : str
            'u' for spin-up, 'd' for spin-down.
        """
        if self._sym == 'Z2' and spin == 'u':  # charges: 0 <-> (|00>, |11>); <-> (|10>, |01>)
            cp = Tensor(config=self.config, s=self.s, n=1)
            cp.set_block(ts=(0, 1), Ds=(2, 2), val=[[0, 0], [0, 1]])
            cp.set_block(ts=(1, 0), Ds=(2, 2), val=[[1, 0], [0, 0]])
        elif self._sym == 'Z2' and spin == 'd':
            cp = Tensor(config=self.config, s=self.s, n=1)
            cp.set_block(ts=(0, 1), Ds=(2, 2), val=[[0, 0], [-1, 0]])
            cp.set_block(ts=(1, 0), Ds=(2, 2), val=[[0, 0], [1, 0]])
        elif self._sym == 'U1xU1xZ2' and spin == 'u':
            cp = Tensor(config=self.config, s=self.s, n=(1, 0, 1))
            cp.set_block(ts=((1, 0, 1), (0, 0, 0)), Ds=(1, 1), val=1)
            cp.set_block(ts=((1, 1, 0), (0, 1, 1)), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1xZ2' and spin == 'd':
            cp = Tensor(config=self.config, s=self.s, n=(0, 1, 1))
            cp.set_block(ts=((0, 1, 1), (0, 0, 0)), Ds=(1, 1), val=1)
            cp.set_block(ts=((1, 1, 0), (1, 0, 1)), Ds=(1, 1), val=-1)
        elif self._sym == 'U1xU1' and spin == 'u':
            cp = Tensor(config=self.config, s=self.s, n=(1, 0))
            cp.set_block(ts=((1, 0), (0, 0)), Ds=(1, 1), val=1)
            cp.set_block(ts=((1, 1), (0, 1)), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1' and spin == 'd':
            cp = Tensor(config=self.config, s=self.s, n=(0, 1))
            cp.set_block(ts=((0, 1), (0, 0)), Ds=(1, 1), val=1)
            cp.set_block(ts=((1, 1), (1, 0)), Ds=(1, 1), val=1)
        else:
            raise YastError("spin shoul be equal 'u' or 'd'.")
        return cp

    def c(self, spin='u'):
        """
        Lowering operator.
        
        Parameters
        ----------
        spin : str
            'u' for spin-up, 'd' for spin-down.
        """
        if self._sym == 'Z2' and spin == 'u': # charges: 0 <-> (|00>, |11>); 1 <-> (|10>, |01>)
            c = Tensor(config=self.config, s=self.s, n=1)
            c.set_block(ts=(0, 1), Ds=(2, 2), val=[[1, 0], [0, 0]])
            c.set_block(ts=(1, 0), Ds=(2, 2), val=[[0, 0], [0, 1]])
        elif self._sym == 'Z2' and spin == 'd':
            c = Tensor(config=self.config, s=self.s, n=1)
            c.set_block(ts=(0, 1), Ds=(2, 2), val=[[0, 1], [0, 0]])
            c.set_block(ts=(1, 0), Ds=(2, 2), val=[[0, -1], [0, 0]])
        elif self._sym == 'U1xU1xZ2' and spin == 'u': # charges <-> (ocupation up, occupation down, total_parity)
            c = Tensor(config=self.config, s=self.s, n=(-1, 0, 1))
            c.set_block(ts=((0, 0, 0), (1, 0, 1)), Ds=(1, 1), val=1)
            c.set_block(ts=((0, 1, 1), (1, 1, 0)), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1xZ2' and spin == 'd':
            c = Tensor(config=self.config, s=self.s, n=(0, -1, 1))
            c.set_block(ts=((0, 0, 0), (0, 1, 1)), Ds=(1, 1), val=1)
            c.set_block(ts=((1, 0, 1), (1, 1, 0)), Ds=(1, 1), val=-1)
        elif self._sym == 'U1xU1' and spin == 'u':  # charges <-> (ocupation up, occupation down)
            c = Tensor(config=self.config, s=self.s, n=(-1, 0))
            c.set_block(ts=((0, 0), (1, 0)), Ds=(1, 1), val=1)
            c.set_block(ts=((0, 1), (1, 1)), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1' and spin == 'd':
            c = Tensor(config=self.config, s=self.s, n=(0, -1))
            c.set_block(ts=((0, 0), (0, 1)), Ds=(1, 1), val=1)
            c.set_block(ts=((1, 0), (1, 1)), Ds=(1, 1), val=1)
        else:
            raise YastError("spin shoul be equal 'u' or 'd'.")
        return c

    def to_dict(self):
        return {'I': lambda j: self.I(),
                'nu': lambda j: self.n(spin='u'),
                'cu': lambda j: self.c(spin='u'),
                'cpu': lambda j: self.cp(spin='u'),
                'nd': lambda j: self.n(spin='d'),
                'cd': lambda j: self.c(spin='d'),
                'cpd': lambda j: self.cp(spin='d')}