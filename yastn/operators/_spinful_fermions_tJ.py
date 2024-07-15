""" Generator of basic local spinful-fermion operators for tJ model. """
from __future__ import annotations
from ..tensor import YastnError, Tensor, Leg
from ._meta_operators import meta_operators
import yastn

class SpinfulFermions_tJ(meta_operators):
    """ Predefine operators for spinful fermions. """

    def __init__(self, sym, **kwargs):
        r"""
        Generator of standard operators for local Hilbert space with two fermionic species and 3-dimensional Hilbert space for tJ model (double occupancy is excluded).
        
        Symmetry 'U1xU1xZ2' is forced to used.

        Predefine identity, creation, annihilation, and density operators.
        Defines vectors with possible occupations.

        Parameters
        ----------

        **kwargs : any
            Passed to :meth:`yastn.make_config` to change backend, default_device or other config parameters.

        Notes
        -----
        For 'U1xU1xZ2', the two species (spin-up and spin-down) are treated as indistinguishable.
        In that case, creation and annihilation operators of the two species anti-commute
        (fermionic statistics is encoded in the Z2 channel).
        """
        # sym = 'U1xU1xZ2'
        kwargs['fermionic'] = (False, False, True) if sym == 'U1xU1xZ2' else True
        kwargs['sym'] = sym
        super().__init__(**kwargs)
        self._sym = sym
        self.operators = ('I', 'n', 'c', 'cp')
        
    def space(self) -> yastn.Leg:
        r""" :class:`yastn.Leg` describing local Hilbert space. """
        if self._sym == 'U1xU1xZ2':
            leg = Leg(self.config, s=1, t=((0, 0, 0), (0, 1, 1), (1, 0, 1)), D=(1, 1, 1))
        if self._sym == 'Z2':
            leg = Leg(self.config, s=1, t=(0, 1), D=(1, 2))
        if self._sym == 'U1xU1':
            leg = Leg(self.config, s=1, t=((0, 0), (0, 1), (1, 0)), D=(1, 1, 1))

        return leg


    def I(self) -> yastn.Tensor:
        r""" Identity operator in 4-dimensional Hilbert space. """
        if self._sym == 'U1xU1xZ2':
            I = Tensor(config=self.config, s=self.s, n=(0, 0, 0))
            for t in [(0, 0, 0), (0, 1, 1), (1, 0, 1)]:
                I.set_block(ts=(t, t), Ds=(1, 1), val=1)
        if self._sym == 'Z2':
            I = Tensor(config=self.config, s=self.s, n=0)
            I.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            I.set_block(ts=(1, 1), Ds=(2, 2), val=[[1, 0], [0, 1]])
        if self._sym == 'U1xU1':
            I = Tensor(config=self.config, s=self.s, n=(0, 0))
            for t in [(0, 0), (0, 1), (1, 0)]:
                I.set_block(ts=(t, t), Ds=(1, 1), val=1)
        return I

    def n(self, spin='u'):
        """ Particle number operator, with spin='u' for spin-up, and 'd' for spin-down. """
        # n = Tensor(config=self.config, s=self.s, n=(0, 0, 0))
        # if spin == 'u':
        #     n.set_block(ts=((1, 0, 1), (1, 0, 1)), Ds=(1, 1), val=1)
        # elif spin == 'd':
        #     n.set_block(ts=((0, 1, 1), (0, 1, 1)), Ds=(1, 1), val=1)
        # return n
        return (self.cp(spin=spin) @ self.c(spin=spin)).remove_zero_blocks()
    

    def h(self):
        """ hole number operator"""
        if self._sym == 'U1xU1xZ2':
            h = Tensor(config=self.config, s=self.s, n=(0, 0, 0))
            h.set_block(ts=((0, 0, 0), (0, 0, 0)), Ds=(1, 1), val=1)
        if self._sym == 'U1xU1':
            h = Tensor(config=self.config, s=self.s, n=(0, 0))
            h.set_block(ts=((0, 0), (0, 0)), Ds=(1, 1), val=1)
        if self._sym == 'Z2':
            h = Tensor(config=self.config, s=self.s, n=0)
            h.set_block(ts=(0, 0), Ds=(1, 1), val=1)
        return h

    def vec_n(self, val=(0, 0)) -> yastn.Tensor:
        r""" Vector with occupation (u, d). """
        if self._sym == 'U1xU1xZ2' and val == (0, 0):
            vec = Tensor(config=self.config, s=(1,), n=(0, 0, 0))
            vec.set_block(ts=((0, 0, 0),), Ds=(1,), val=[1])
        elif self._sym == 'U1xU1xZ2' and val == (1, 0):
            vec = Tensor(config=self.config, s=(1,), n=(1, 0, 1))
            vec.set_block(ts=((1, 0, 1),), Ds=(1,), val=[1])
        elif self._sym == 'U1xU1xZ2' and val == (0, 1):
            vec = Tensor(config=self.config, s=(1,), n=(0, 1, 1))
            vec.set_block(ts=((0, 1, 1),), Ds=(1,), val=[1])
        elif self._sym == 'Z2' and val == (0, 0):
            vec = Tensor(config=self.config, s=(1,), n=0)
            vec.set_block(ts=(0,), Ds=(1,), val=1)
        elif self._sym == 'Z2' and val == (1, 0):
            vec = Tensor(config=self.config, s=(1,), n=1)
            vec.set_block(ts=(1,), Ds=(2,), val=[1, 0])
        elif self._sym == 'Z2' and val == (0, 1):
            vec = Tensor(config=self.config, s=(1,), n=1)
            vec.set_block(ts=(1,), Ds=(2,), val=[0, 1])
        elif self._sym == 'U1xU1' and val == (0, 0):
            vec = Tensor(config=self.config, s=(1,), n=(0, 0))
            vec.set_block(ts=((0, 0),), Ds=(1,), val=[1])
        elif self._sym == 'U1xU1' and val == (1, 0):
            vec = Tensor(config=self.config, s=(1,), n=(1, 0))
            vec.set_block(ts=((1, 0),), Ds=(1,), val=[1])
        elif self._sym == 'U1xU1' and val == (0, 1):
            vec = Tensor(config=self.config, s=(1,), n=(0, 1))
            vec.set_block(ts=((0, 1),), Ds=(1,), val=[1])
        else:
            raise YastnError('For SpinfulFermions val in vec_n should be in [(0, 0), (1, 0), (0, 1)].')
        return vec

    def cp(self, spin='u') -> yastn.Tensor:
        r""" Creation operator, with spin='u' for spin-up, and 'd' for spin-down. """
        if self._sym == 'U1xU1xZ2' and spin == 'u':
            cp = Tensor(config=self.config, s=self.s, n=(1, 0, 1))
            cp.set_block(ts=((1, 0, 1), (0, 0, 0)), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1xZ2' and spin == 'd':
            cp = Tensor(config=self.config, s=self.s, n=(0, 1, 1))
            cp.set_block(ts=((0, 1, 1), (0, 0, 0)), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1' and spin == 'u':
            cp = Tensor(config=self.config, s=self.s, n=(1, 0))
            cp.set_block(ts=((1, 0), (0, 0)), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1' and spin == 'd':
            cp = Tensor(config=self.config, s=self.s, n=(0, 1))
            cp.set_block(ts=((0, 1), (0, 0)), Ds=(1, 1), val=1)
        elif self._sym == 'Z2' and spin == 'u':  # charges: 0 <-> (|00>, |11>); 1 <-> (|10>, |01>)
            cp = Tensor(config=self.config, s=self.s, n=1)
            cp.set_block(ts=(1, 0), Ds=(2, 1), val=[[1,], [0,]])
        elif self._sym == 'Z2' and spin == 'd':
            cp = Tensor(config=self.config, s=self.s, n=1)
            cp.set_block(ts=(1, 0), Ds=(2, 1), val=[[0,], [1,]])
        else:
            raise YastnError("spin shoul be equal 'u' or 'd'.")
        return cp

    def c(self, spin='u') -> yastn.Tensor:
        r""" Annihilation operator, with spin='u' for spin-up, and 'd' for spin-down. """
        if self._sym == 'U1xU1xZ2' and spin == 'u': # charges <-> (ocupation up, occupation down, total_parity)
            c = Tensor(config=self.config, s=self.s, n=(-1, 0, 1))
            c.set_block(ts=((0, 0, 0), (1, 0, 1)), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1xZ2' and spin == 'd':
            c = Tensor(config=self.config, s=self.s, n=(0, -1, 1))
            c.set_block(ts=((0, 0, 0), (0, 1, 1)), Ds=(1, 1), val=1)
        elif self._sym == 'Z2' and spin == 'u': # charges: 0 <-> (|00>, |11>); 1 <-> (|10>, |01>)
            c = Tensor(config=self.config, s=self.s, n=1)
            c.set_block(ts=(0, 1), Ds=(1, 2), val=[[1, 0]])
        elif self._sym == 'Z2' and spin == 'd':
            c = Tensor(config=self.config, s=self.s, n=1)
            c.set_block(ts=(0, 1), Ds=(1, 2), val=[[0, 1]])
        elif self._sym == 'U1xU1' and spin == 'u':  # charges <-> (ocupation up, occupation down)
            c = Tensor(config=self.config, s=self.s, n=(-1, 0))
            c.set_block(ts=((0, 0), (1, 0)), Ds=(1, 1), val=1)
        elif self._sym == 'U1xU1' and spin == 'd':
            c = Tensor(config=self.config, s=self.s, n=(0, -1))
            c.set_block(ts=((0, 0), (0, 1)), Ds=(1, 1), val=1)
        else:
            raise YastnError("spin shoul be equal 'u' or 'd'.")
        return c
    
    def Sz(self) -> yastn.Tensor:
        """ Return Sz operator for spin-1/2 fermions
        Returns:
            yastn.Tensor: _description_
        """
        return 0.5 * (self.n('u') - self.n('d'))
    
    def Sp(self) -> yastn.Tensor:
        """ Return Sp operator for spin-1/2 fermions
        Returns:
            yastn.Tensor: _description_
        """
        return self.cp('u') @ self.c('d')
    
    def Sm(self) -> yastn.Tensor:
        """ Return Sm operator for spin-1/2 fermions
        Returns:
            yastn.Tensor: _description_
        """
        return self.cp('d') @ self.c('u')

    def to_dict(self):
        return {'I': lambda j: self.I(),
                'nu': lambda j: self.n(spin='u'),
                'cu': lambda j: self.c(spin='u'),
                'cpu': lambda j: self.cp(spin='u'),
                'nd': lambda j: self.n(spin='d'),
                'cd': lambda j: self.c(spin='d'),
                'cpd': lambda j: self.cp(spin='d'),
                'h': lambda j: self.h(),
                'Sz': lambda j: self.Sz(),
                'Sp': lambda j: self.Sp(),
                'Sm': lambda j: self.Sm()}
