from ..initialize import make_config
from ..sym import sym_none, sym_Z2, sym_U1
from ..tensor import YastError, Tensor

class Spin12:
    """ Predefine operators for spin-1/2 system. """

    def __init__(self, sym='dense', **kwargs):
        """
        Generator of standard operators for 2-dimensional Hilbert space.

        Predefine identity, rising and lowering operators, and Pauli matrices (if allowed by symmetry).

        Other config parameters can be provided, see :meth:`yast.make_config`
        fermionic is set to False.

        Parameters
        ----------
        sym : str
            Should be 'dense', 'Z2', or 'U1'.

        Notes
        -----
        Assume the following conventions:
        For dense, the basis order is (z=+1, z=-1)
        For Z2, charge t=0 <=> z=1, t=1 <=> z=-1; i.e., z = e^i pi t
        For U1, charge t=-1 <=> z=-1, t=1 <=> z=1; i.e., z = t

        Using :meth:`yast.to_numpy`, U1 additionally requires reverse=True to obtain the standard matrix representation,
        as by default the charges get ordered in the increasing order.
        """
        if not sym in ('dense', 'Z2', 'U1'):
            raise YastError("For Spin12 sym should be in ('dense', 'Z2', 'U1').")
        self._sym = sym
        kwargs['fermionic'] = False
        import_sym = {'dense': sym_none, 'Z2': sym_Z2, 'U1': sym_U1}
        kwargs['sym'] = import_sym[sym]
        self.config = make_config(**kwargs)
        self.s = (1, -1)
        self.operators = ('I', 'x', 'y', 'z', 'sx', 'sy', 'sz', 'sp', 'sm')

    def I(self):
        """ Identity operator. """
        if self._sym == 'dense':
            I = Tensor(config=self.config, s=self.s)
            I.set_block(val=[[1, 0], [0, 1]], Ds=(2, 2))
        if self._sym in 'Z2':
            I = Tensor(config=self.config, s=self.s, n=0)
            I.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            I.set_block(ts=(1, 1), Ds=(1, 1), val=1)
        if self._sym in 'U1':
            I = Tensor(config=self.config, s=self.s, n=0)
            I.set_block(ts=(1, 1), Ds=(1, 1), val=1)
            I.set_block(ts=(-1, -1), Ds=(1, 1), val=1)
        return I

    def x(self):
        """ Pauli sigma_x operator. """
        if self._sym == 'dense':
            x = Tensor(config=self.config, s=self.s)
            x.set_block(val=[[0, 1], [1, 0]], Ds=(2, 2))
        if self._sym == 'Z2':
            x = Tensor(config=self.config, s=self.s, n=1)
            x.set_block(ts=(1, 0), Ds=(1, 1), val=1)
            x.set_block(ts=(0, 1), Ds=(1, 1), val=1)
        if self._sym == 'U1':
            raise YastError('Cannot define sigma_x operator for U(1) symmetry.')
        return x

    def y(self):
        """ Pauli sigma_y operator. """
        if self._sym == 'dense':
            y = Tensor(config=self.config, s=self.s, dtype='complex128')
            y.set_block(val=[[0, -1j], [1j, 0]], Ds=(2, 2))
        if self._sym == 'Z2':
            y = Tensor(config=self.config, s=self.s, n=1, dtype='complex128')
            y.set_block(ts=(0, 1), Ds=(1, 1), val=-1j)
            y.set_block(ts=(1, 0), Ds=(1, 1), val=1j)
        if self._sym == 'U1':
            raise YastError('Cannot define sigma_y operator for U(1) symmetry.')
        return y

    def z(self):
        """ Pauli sigma_z operator. """
        if self._sym == 'dense':
            z = Tensor(config=self.config, s=self.s)
            z.set_block(val=[[1, 0], [0, -1]], Ds=(2, 2))
        if self._sym == 'Z2':
            z = Tensor(config=self.config, s=self.s, n=0)
            z.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            z.set_block(ts=(1, 1), Ds=(1, 1), val=-1)
        if self._sym == 'U1':
            z = Tensor(config=self.config, s=self.s, n=0)
            z.set_block(ts=(1, 1), Ds=(1, 1), val=1)
            z.set_block(ts=(-1, -1), Ds=(1, 1), val=-1)
        return z

    def sx(self):
        """ Spin-1/2 sx operator """
        return self.x() / 2

    def sy(self):
        """ Spin-1/2 sy operator """
        return self.y() / 2

    def sz(self):
        """ Spin-1/2 sz operator """
        return self.z() / 2

    def sp(self):
        """ Rising operator. """
        if self._sym == 'dense':
            sp = Tensor(config=self.config, s=self.s)
            sp.set_block(val=[[0, 1], [0, 0]], Ds=(2, 2))
        if self._sym == 'Z2':
            sp = Tensor(config=self.config, s=self.s, n=1)
            sp.set_block(ts=(0, 1), Ds=(1, 1), val=1)
        if self._sym == 'U1':
            sp = Tensor(config=self.config, s=self.s, n=2)
            sp.set_block(ts=(1, -1), Ds=(1, 1), val=1)
        return sp

    def sm(self):
        """ Lowering operator. """
        if self._sym == 'dense':
            sm = Tensor(config=self.config, s=self.s)
            sm.set_block(val=[[0, 0], [1, 0]], Ds=(2, 2))
        if self._sym == 'Z2':
            sm = Tensor(config=self.config, s=self.s, n=1)
            sm.set_block(ts=(1, 0), Ds=(1, 1), val=1)
        if self._sym == 'U1':
            sm = Tensor(config=self.config, s=self.s, n=-2)
            sm.set_block(ts=(-1, 1), Ds=(1, 1), val=1)
        return sm

    def to_dict(self):
        return {'I': self.I(),
                'x': self.x(),
                'y': self.y(),
                'z': self.z(),
                'sx': self.sx(),
                'sy': self.sy(),
                'sz': self.sz(),
                'sp': self.sp(),
                'sm': self.sm()}
