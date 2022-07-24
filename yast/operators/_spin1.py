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
        For dense, basis order is (Z=+1, Z=-1)
        For Z2, charge t=0 <=> Z=1, t=1 <=> Z=-1; i.e., Z = e^i pi t
        For U1, charge t=-1 <=> Z=-1, t=1 <=> Z=1; i.e., Z = t

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

    def X(self):
        """ Pauli sigma_x operator. """
        if self._sym == 'dense':
            X = Tensor(config=self.config, s=self.s)
            X.set_block(val=[[0, 1], [1, 0]], Ds=(2, 2))
        if self._sym == 'Z2':
            X = Tensor(config=self.config, s=self.s, n=1)
            X.set_block(ts=(1, 0), Ds=(1, 1), val=1)
            X.set_block(ts=(0, 1), Ds=(1, 1), val=1)
        if self._sym == 'U1':
            raise YastError('Cannot define sigma_x operator for U(1) symmetry.')
        return X

    def Y(self):
        """ Pauli sigma_y operator. """
        if self._sym == 'dense':
            Y = Tensor(config=self.config, s=self.s, dtype='complex128')
            Y.set_block(val=[[0, -1j], [1j, 0]], Ds=(2, 2))
        if self._sym == 'Z2':
            Y = Tensor(config=self.config, s=self.s, n=1, dtype='complex128')
            Y.set_block(ts=(0, 1), Ds=(1, 1), val=-1j)
            Y.set_block(ts=(1, 0), Ds=(1, 1), val=1j)
        if self._sym == 'U1':
            raise YastError('Cannot define sigma_y operator for U(1) symmetry.')
        return Y

    def Z(self):
        """ Pauli sigma_z operator. """
        if self._sym == 'dense':
            Z = Tensor(config=self.config, s=self.s)
            Z.set_block(val=[[1, 0], [0, -1]], Ds=(2, 2))
        if self._sym == 'Z2':
            Z = Tensor(config=self.config, s=self.s, n=0)
            Z.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            Z.set_block(ts=(1, 1), Ds=(1, 1), val=-1)
        if self._sym == 'U1':
            Z = Tensor(config=self.config, s=self.s, n=0)
            Z.set_block(ts=(1, 1), Ds=(1, 1), val=1)
            Z.set_block(ts=(-1, -1), Ds=(1, 1), val=-1)
        return Z

    def Sp(self):
        """ Rising operator. """
        if self._sym == 'dense':
            Sp = Tensor(config=self.config, s=self.s)
            Sp.set_block(val=[[0, 1], [0, 0]], Ds=(2, 2))
        if self._sym == 'Z2':
            Sp = Tensor(config=self.config, s=self.s, n=1)
            Sp.set_block(ts=(0, 1), Ds=(1, 1), val=1)
        if self._sym == 'U1':
            Sp = Tensor(config=self.config, s=self.s, n=2)
            Sp.set_block(ts=(1, -1), Ds=(1, 1), val=1)
        return Sp

    def Sm(self):
        """ Lowering operator. """
        if self._sym == 'dense':
            Sm = Tensor(config=self.config, s=self.s)
            Sm.set_block(val=[[0, 0], [1, 0]], Ds=(2, 2))
        if self._sym == 'Z2':
            Sm = Tensor(config=self.config, s=self.s, n=1)
            Sm.set_block(ts=(1, 0), Ds=(1, 1), val=1)
        if self._sym == 'U1':
            Sm = Tensor(config=self.config, s=self.s, n=-2)
            Sm.set_block(ts=(-1, 1), Ds=(1, 1), val=1)
        return Sm
