from ..sym import sym_Z2, sym_U1xU1, sym_U1xU1xZ2
from ..tensor import YastnError, Tensor
from ._meta_operators import meta_operators

class SpinfulFermions(meta_operators):
    """ Predefine operators for spinful fermions with double occupation states excluded . """

    def __init__(self, sym='Z2', **kwargs):
        """
        Generator of operators for local Hilbert space with two fermionic species within Gutzwiller Projection where double occupation is prohibitted.
        This ammount to 3-dimensional Hilbert space.

        Predefine identity, creation, annihilation, and density operators.

        Parameters
        ----------
        sym : str
            Should be 'Z2', 'U1xU1' or 'U1xU1xZ2'. Fixes symmetry and fermionic fields in config.

        **kwargs : any
            Passed to :meth:`yastn.make_config` to change backend, default_device or other config parameters.

        Notes
        -----
        For 'Z2' and 'U1xU1xZ2', the two species (spin-up and spin-down) are treated as indistinguishable.
        In that case, creation and annihilation operators of the two species anti-commute
        (fermionic statistics is encoded in the Z2 channel).

        For 'U1xU1' the two species (spin-up and spin-down) are treated as distinguishable.
        In that case, creation and annihilation operators of the two species commute.
        """
        if sym not in ('U1xU1xZ2',):  # 'Z2', 'U1xU1',
            raise YastnError("For SpinfulFermions sym should be in ('U1xU1xZ2',).")  # 'Z2', 'U1xU1',
        kwargs['fermionic'] = (False, False, True) if sym == 'U1xU1xZ2' else True
        import_sym = {'U1xU1xZ2': sym_U1xU1xZ2}  # 'Z2': sym_Z2, 'U1xU1': sym_U1xU1,
        kwargs['sym'] = import_sym[sym]
        super().__init__(**kwargs)
        self._sym = sym
        self.operators = ('I', 'n', 'c', 'cp')


    def I(self):
        """ Identity operator in 3-dimensional Hilbert space. """
        if self._sym == 'U1xU1xZ2':
            I = Tensor(config=self.config, s=self.s, n=(0, 0, 0))
            for t in [(0, 0, 0), (0, 1, 1), (1, 0, 1)]:
                I.set_block(ts=(t, t), Ds=(1, 1), val=1)
        return I

    def n(self, spin='u'):
        """ Particle number operator, with spin='u' for spin-up, and 'd' for spin-down. """
        return (self.cp(spin=spin) @ self.c(spin=spin)).remove_zero_blocks()

    def cp(self, spin='u'):
        """ Creation operator, with spin='u' for spin-up, and 'd' for spin-down. """
        if self._sym == 'U1xU1xZ2' and spin == 'u':  # charges: 0 <-> (|00>); <-> (|10>, |01>)
            cp = Tensor(config=self.config, s=self.s, n=(1, 0, 1))   # charges <-> (ocupation up, occupation down, total_parity)
            cp.set_block(ts=((1, 0, 1), (0, 0, 0)), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1xZ2' and spin == 'd':
            cp = Tensor(config=self.config, s=self.s, n=(0, 1, 1))
            cp.set_block(ts=((0, 1, 1), (0, 0, 0)), Ds=(1, 1), val=1)
        else:
            raise YastnError("spin shoul be equal 'u' or 'd'.")
        return cp

    def c(self, spin='u'):
        """ Annihilation operator, with spin='u' for spin-up, and 'd' for spin-down. """
        # charges: 0 <-> (|00>,); 1 <-> (|10>, |01>)
        if self._sym == 'U1xU1xZ2' and spin == 'u':
            c = Tensor(config=self.config, s=self.s, n=(-1, 0, 1))
            c.set_block(ts=((0, 0, 0), (1, 0, 1)), Ds=(1, 1), val=1)  # charges <-> (ocupation up, occupation down, total_parity)
        elif self._sym == 'U1xU1xZ2' and spin == 'd':
            c = Tensor(config=self.config, s=self.s, n=(0, -1, 1))
            c.set_block(ts=((0, 0, 0), (0, 1, 1)), Ds=(1, 1), val=1)
        else:
            raise YastnError("spin shoul be equal 'u' or 'd'.")
        return c

    def to_dict(self):
        return {'I': lambda j: self.I(),
                'nu': lambda j: self.n(spin='u'),
                'cu': lambda j: self.c(spin='u'),
                'cpu': lambda j: self.cp(spin='u'),
                'nd': lambda j: self.n(spin='d'),
                'cd': lambda j: self.c(spin='d'),
                'cpd': lambda j: self.cp(spin='d')}
