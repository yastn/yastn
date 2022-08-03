import numpy as np
from ..initialize import make_config
from ..sym import sym_none, sym_Z3, sym_U1
from ..tensor import YastError, Tensor

class Spin1:
    """ Predefine operators for spin-1 system. """

    def __init__(self, sym='dense', **kwargs):
        """
        Generator of standard operators for 3-dimensional Hilbert space.

        Predefine identity, rising and lowering operators, and spin-1 matrices (if allowed by symmetry).

        Other config parameters can be provided, see :meth:`yast.make_config`
        fermionic is set to False.

        Parameters
        ----------
        sym : str
            Should be 'dense', 'Z3', or 'U1'.

        Notes
        -----
        Assume the following conventions:
        For dense, basis order is (sz=+1, sz=0, sz=-1)
        For Z3, charge t=0 <=> Z=1, t=1 <=> Z=0; t=2 <=> Z=-1;
        For U1, charge t=-1 <=> sz=-1, t=0 <=> sz=0, t=1 <=> sz=1; i.e., sz = t

        Using :meth:`yast.to_numpy`, U1 additionally requires reverse=True to obtain the standard matrix representation,
        as by default the charges get ordered in the increasing order.
        """
        if not sym in ('dense', 'Z3', 'U1'):
            raise YastError("For Spin1 sym should be in ('dense', 'Z3', 'U1').")
        self._sym = sym
        kwargs['fermionic'] = False
        import_sym = {'dense': sym_none, 'Z3': sym_Z3, 'U1': sym_U1}
        kwargs['sym'] = import_sym[sym]
        self.config = make_config(**kwargs)
        self.s = (1, -1)
        self.operators = ('I', 'sx', 'sy', 'sz', 'sp', 'sm')


    def I(self):
        """ Identity operator. """
        if self._sym == 'dense':
            I = Tensor(config=self.config, s=self.s)
            I.set_block(val=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], Ds=(3, 3))
        if self._sym in 'Z3':
            I = Tensor(config=self.config, s=self.s, n=0)
            I.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            I.set_block(ts=(1, 1), Ds=(1, 1), val=1)
            I.set_block(ts=(2, 2), Ds=(1, 1), val=1)
        if self._sym in 'U1':
            I = Tensor(config=self.config, s=self.s, n=0)
            I.set_block(ts=(1, 1), Ds=(1, 1), val=1)
            I.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            I.set_block(ts=(-1, -1), Ds=(1, 1), val=1)
        return I

    def sx(self):
        """ Spin-1 sx operator. """
        isq2 = 1 / np.sqrt(2)
        if self._sym == 'dense':
            sx = Tensor(config=self.config, s=self.s)
            sx.set_block(val=[[0, isq2, 0], [isq2, 0, isq2], [0, isq2, 0]], Ds=(3, 3))
        if self._sym in ('Z3', 'U1'):
            raise YastError('Cannot define sx operator for U(1) or Z3 symmetry.')
        return sx

    def sy(self):
        """ Spin-1 sy operator. """
        iisq2 = 1j / np.sqrt(2)
        if self._sym == 'dense':
            sy = Tensor(config=self.config, s=self.s, dtype='complex128')
            sy.set_block(val=[[0, -iisq2, 0], [iisq2, 0, -iisq2], [0, iisq2, 0]], Ds=(3, 3))
        if self._sym in ('Z3', 'U1'):
            raise YastError('Cannot define sy operator for U(1) or Z3 symmetry.')
        return sy

    def sz(self):
        """ Spin-1 sz operator. """
        if self._sym == 'dense':
            sz = Tensor(config=self.config, s=self.s)
            sz.set_block(val=[[1, 0, 0], [0, 0, 0], [0, 0, -1]], Ds=(3, 3))
        if self._sym == 'Z3':
            sz = Tensor(config=self.config, s=self.s, n=0)
            sz.set_block(ts=(0, 0), Ds=(1, 1), val=1)
            sz.set_block(ts=(1, 1), Ds=(1, 1), val=0)
            sz.set_block(ts=(2, 2), Ds=(1, 1), val=-1)
        if self._sym == 'U1':
            sz = Tensor(config=self.config, s=self.s, n=0)
            sz.set_block(ts=(1, 1), Ds=(1, 1), val=1)
            sz.set_block(ts=(0, 0), Ds=(1, 1), val=0)
            sz.set_block(ts=(-1, -1), Ds=(1, 1), val=-1)
        return sz

    def sp(self):
        """ Spin-1 rising operator. """
        sq2 = np.sqrt(2)
        if self._sym == 'dense':
            sp = Tensor(config=self.config, s=self.s)
            sp.set_block(val=[[0, sq2, 0], [0, 0, sq2], [0, 0, 0]], Ds=(3, 3))
        if self._sym == 'Z3':
            sp = Tensor(config=self.config, s=self.s, n=2)
            sp.set_block(ts=(0, 1), Ds=(1, 1), val=sq2)
            sp.set_block(ts=(1, 2), Ds=(1, 1), val=sq2)
        if self._sym == 'U1':
            sp = Tensor(config=self.config, s=self.s, n=1)
            sp.set_block(ts=(0, -1), Ds=(1, 1), val=sq2)
            sp.set_block(ts=(1, 0), Ds=(1, 1), val=sq2)
        return sp

    def sm(self):
        """ Spin-1 lowering operator. """
        sq2 = np.sqrt(2)
        if self._sym == 'dense':
            sm = Tensor(config=self.config, s=self.s)
            sm.set_block(val=[[0, 0, 0], [sq2, 0, 0], [0, sq2, 0]], Ds=(3, 3))
        if self._sym == 'Z3':
            sm = Tensor(config=self.config, s=self.s, n=1)
            sm.set_block(ts=(1, 0), Ds=(1, 1), val=sq2)
            sm.set_block(ts=(2, 1), Ds=(1, 1), val=sq2)
        if self._sym == 'U1':
            sm = Tensor(config=self.config, s=self.s, n=-1)
            sm.set_block(ts=(0, 1), Ds=(1, 1), val=sq2)
            sm.set_block(ts=(-1, 0), Ds=(1, 1), val=sq2)
        return sm

    def to_dict(self):
        return {'I': self.I(),
                'sx': self.sx(),
                'sy': self.sy(),
                'sz': self.sz(),
                'sp': self.sp(),
                'sm': self.sm()}