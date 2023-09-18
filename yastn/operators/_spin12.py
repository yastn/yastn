from ..sym import sym_none, sym_Z2, sym_U1
from ..tensor import YastnError, Tensor
from ._meta_operators import meta_operators

class Spin12(meta_operators):
    # Predefined set of Pauli operators and spin-1/2 operators.
    def __init__(self, sym='dense', **kwargs):
        r"""
        A set of standard operators for 2-dimensional Hilbert space. Defines identity,
        :math:`S^z,\ S^x,\ S^y` operators and :math:`S^+,\ S^-` raising and lowering operators,
        and Pauli matrices (if allowed by symmetry). Define eigenvectors of :math:`S^z`

        Parameters
        ----------
        sym : str
            Explicit symmetry to used. Allowed options are :code:`'dense'`, ``'Z2'``, or ``'U1'``.

        kwargs
            Other YASTN configuration parameters can be provided, see :meth:`yastn.make_config`.

        Notes
        -----
        The following basis ordering and charge conventions are assumed

            * For :code:`sym='dense'`, the basis order is (z=+1, z=-1).
            * For :code:`sym='Z2'`, charge t=0 -> z=1, t=1 -> z=-1; i.e., :math:`z = e^{i \pi t}`.
            * For :code:`sym='U1'`, charge t=-1 -> z=-1, t=1 -> z=1; i.e., z = t.

        Default configuration sets :code:`fermionic` to :code:`False`.

        When using :meth:`yastn.to_numpy` to recover usual dense representation of the algebra
        for :code:`sym='U1'` symmetry, :code:`reverse=True` is required
        since by default the charges are ordered in the increasing order.
        """
        if sym not in ('dense', 'Z2', 'U1'):
            raise YastnError("For Spin12 sym should be in ('dense', 'Z2', 'U1').")
        kwargs['fermionic'] = False
        import_sym = {'dense': sym_none, 'Z2': sym_Z2, 'U1': sym_U1}
        kwargs['sym'] = import_sym[sym]
        super().__init__(**kwargs)
        self._sym = sym
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
        r""" Pauli :math:`\sigma^x` operator. """
        if self._sym == 'dense':
            x = Tensor(config=self.config, s=self.s)
            x.set_block(val=[[0, 1], [1, 0]], Ds=(2, 2))
        if self._sym == 'Z2':
            x = Tensor(config=self.config, s=self.s, n=1)
            x.set_block(ts=(1, 0), Ds=(1, 1), val=1)
            x.set_block(ts=(0, 1), Ds=(1, 1), val=1)
        if self._sym == 'U1':
            raise YastnError('Cannot define sigma_x operator for U(1) symmetry.')
        return x

    def y(self):
        r""" Pauli :math:`\sigma^y` operator. """
        if self._sym == 'dense':
            y = Tensor(config=self.config, s=self.s, dtype='complex128')
            y.set_block(val=[[0, -1j], [1j, 0]], Ds=(2, 2))
        if self._sym == 'Z2':
            y = Tensor(config=self.config, s=self.s, n=1, dtype='complex128')
            y.set_block(ts=(0, 1), Ds=(1, 1), val=-1j)
            y.set_block(ts=(1, 0), Ds=(1, 1), val=1j)
        if self._sym == 'U1':
            raise YastnError('Cannot define sigma_y operator for U(1) symmetry.')
        return y

    def z(self):
        r""" Pauli :math:`\sigma^z` operator. """
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

    def vec_z(self, val=1):
        """ Normalized eigenvectors of :math:`\sigma^z. """
        if val not in (-1, 1):
            raise YastnError('val should be in (-1, 1)')
        if self._sym == 'dense' and val == 1:
            vec = Tensor(config=self.config, s=(1,))
            vec.set_block(val=[1, 0], Ds=(2,))
        elif self._sym == 'dense' and val == -1:
            vec = Tensor(config=self.config, s=(1,))
            vec.set_block(val=[0, 1], Ds=(2,))
        elif self._sym == 'Z2' and val == 1:
            vec = Tensor(config=self.config, s=(1,), n=(0,))
            vec.set_block(val=[1], ts=(0,), Ds=(1,))
        elif self._sym == 'Z2' and val == -1:
            vec = Tensor(config=self.config, s=(1,), n=(1,))
            vec.set_block(val=[1], ts=(1,), Ds=(1,))
        elif self._sym == 'U1' and val == 1:
            vec = Tensor(config=self.config, s=(1,), n=(1,))
            vec.set_block(val=[1], ts=(1,), Ds=(1,))
        elif self._sym == 'U1' and val == -1:
            vec = Tensor(config=self.config, s=(1,), n=(-1,))
            vec.set_block(val=[1], ts=(-1,), Ds=(1,))
        else:
            raise YastnError('For Spin12 val in vec_z should be in (-1, 1).')
        return vec

    def sx(self):
        """ Spin-1/2 :math:`S^x` operator """
        return self.x() / 2

    def sy(self):
        """ Spin-1/2 :math:`S^y` operator """
        return self.y() / 2

    def sz(self):
        """ Spin-1/2 :math:`S^z` operator """
        return self.z() / 2

    def sp(self):
        """ Spin-1/2 raising operator :math:`S^+=S^x + iS^y`. """
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
        """ Spin-1/2 lowering operator :math:`S^-=S^x - iS^y`. """
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
        """
        Returns
        -------
        dict(str,yastn.Tensor)
            a map from strings to operators
        """
        return {'I': lambda j: self.I(),
                'x': lambda j: self.x(),
                'y': lambda j: self.y(),
                'z': lambda j: self.z(),
                'sx': lambda j: self.sx(),
                'sy': lambda j: self.sy(),
                'sz': lambda j: self.sz(),
                'sp': lambda j: self.sp(),
                'sm': lambda j: self.sm()}
