""" Generator of basic local spingful-fermion operators. """
from __future__ import annotations
from ..sym import sym_Z2, sym_U1xU1, sym_U1xU1xZ2
from ..tensor import YastnError, Tensor, Leg
from ._meta_operators import meta_operators

class SpinfulFermions(meta_operators):
    """ Predefine operators for spinful fermions. """

    def __init__(self, sym='Z2', **kwargs):
        r"""
        Generator of standard operators for local Hilbert space with two fermionic species and 4-dimensional Hilbert space.

        Predefine identity, creation, annihilation, and density operators.
        Defines vectors with possible occupations, and local Hilbert space as a :class:`yastn.Leg`.

        Parameters
        ----------
        sym : str
            Explicit symmetry to used. Allowed options are :code:`'Z2'`, :code:`'U1xU1'`, or :code:`'U1xU1xZ2'`.

        **kwargs : any
            Passed to :meth:`yastn.make_config` to change backend, default_device or other config parameters.

        Notes
        -----
        Fermionic field in config is fixed such that:

            * For :code:`'Z2'` and :code:`'U1xU1xZ2'`, the two species (spin-up and spin-down)
              are treated as indistinguishable. In that case, creation and annihilation operators
              of the two species anti-commute (fermionic statistics is encoded in the Z2 channel).
            * For :code:`'U1xU1'` the two species (spin-up and spin-down) are treated as distinguishable.
              In that case, creation and annihilation operators of the two species commute.
        """
        if sym not in ('Z2', 'U1xU1', 'U1xU1xZ2'):
            raise YastnError("For SpinfulFermions sym should be in ('Z2', 'U1xU1', 'U1xU1xZ2').")
        kwargs['fermionic'] = (False, False, True) if sym == 'U1xU1xZ2' else True
        import_sym = {'Z2': sym_Z2, 'U1xU1': sym_U1xU1, 'U1xU1xZ2': sym_U1xU1xZ2}
        kwargs['sym'] = import_sym[sym]
        super().__init__(**kwargs)
        self._sym = sym
        self.operators = ('I', 'n', 'c', 'cp')

    def space(self) -> yastn.Leg:
        r""" :class:`yastn.Leg` describing local Hilbert space. """
        if self._sym == 'Z2':
            leg = Leg(self.config, s=1, t=(0, 1), D=(2, 2))
        if self._sym == 'U1xU1xZ2':
            leg = Leg(self.config, s=1, t=((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)), D=(1, 1, 1 ,1))
        if self._sym == 'U1xU1':
            leg = Leg(self.config, s=1, t=((0, 0), (0, 1), (1, 0), (1, 1)), D=(1, 1, 1, 1))
        return leg

    def I(self) -> yastn.Tensor:
        r""" Identity operator in 4-dimensional Hilbert space. """
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

    def n(self, spin='u') -> yastn.Tensor:
        r""" Particle number operator, with spin='u' for spin-up, and 'd' for spin-down. """
        return (self.cp(spin=spin) @ self.c(spin=spin)).remove_zero_blocks()

    def vec_n(self, val=(0, 0)) -> yastn.Tensor:
        r""" State with occupation given by tuple (nu, nd). """
        if self._sym == 'Z2' and val == (0, 0):
            vec = Tensor(config=self.config, s=(1,), n=0)
            vec.set_block(ts=(0,), Ds=(2,), val=[1, 0])
        elif self._sym == 'Z2' and val == (1, 0):
            vec = Tensor(config=self.config, s=(1,), n=1)
            vec.set_block(ts=(1,), Ds=(2,), val=[1, 0])
        elif self._sym == 'Z2' and val == (0, 1):
            vec = Tensor(config=self.config, s=(1,), n=1)
            vec.set_block(ts=(1,), Ds=(2,), val=[0, 1])
        elif self._sym == 'Z2' and val == (1, 1):
            vec = Tensor(config=self.config, s=(1,), n=0)
            vec.set_block(ts=(0,), Ds=(2,), val=[0, 1])
        elif self._sym == 'U1xU1' and val == (0, 0):
            vec = Tensor(config=self.config, s=(1,), n=(0, 0))
            vec.set_block(ts=((0, 0),), Ds=(1,), val=[1])
        elif self._sym == 'U1xU1' and val == (1, 0):
            vec = Tensor(config=self.config, s=(1,), n=(1, 0))
            vec.set_block(ts=((1, 0),), Ds=(1,), val=[1])
        elif self._sym == 'U1xU1' and val == (0, 1):
            vec = Tensor(config=self.config, s=(1,), n=(0, 1))
            vec.set_block(ts=((0, 1),), Ds=(1,), val=[1])
        elif self._sym == 'U1xU1' and val == (1, 1):
            vec = Tensor(config=self.config, s=(1,), n=(1, 1))
            vec.set_block(ts=((1, 1),), Ds=(1,), val=[1])
        elif self._sym == 'U1xU1xZ2' and val == (0, 0):
            vec = Tensor(config=self.config, s=(1,), n=(0, 0, 0))
            vec.set_block(ts=((0, 0, 0),), Ds=(1,), val=[1])
        elif self._sym == 'U1xU1xZ2' and val == (1, 0):
            vec = Tensor(config=self.config, s=(1,), n=(1, 0, 1))
            vec.set_block(ts=((1, 0, 1),), Ds=(1,), val=[1])
        elif self._sym == 'U1xU1xZ2' and val == (0, 1):
            vec = Tensor(config=self.config, s=(1,), n=(0, 1, 1))
            vec.set_block(ts=((0, 1, 1),), Ds=(1,), val=[1])
        elif self._sym == 'U1xU1xZ2' and val == (1, 1):
            vec = Tensor(config=self.config, s=(1,), n=(1, 1, 0))
            vec.set_block(ts=((1, 1, 0),), Ds=(1,), val=[1])
        else:
            raise YastnError('Occupations given by val should be (0, 0), (1, 0), (0, 1), or (1, 1).')
        return vec

    def cp(self, spin='u') -> yastn.Tensor:
        r""" Creation operator, with spin='u' for spin-up, and 'd' for spin-down. """
        if self._sym == 'Z2' and spin == 'u':  # charges: 0 <-> (|00>, |11>); 1 <-> (|10>, |01>)
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
            raise YastnError("spin shoul be equal 'u' or 'd'.")
        return cp

    def c(self, spin='u') -> yastn.Tensor:
        r""" Annihilation operator, with spin='u' for spin-up, and 'd' for spin-down. """
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
