from ..sym import sym_none
from ..initialize import make_config, eye
from .. import diag

class Qdit():
    
    def __init__(self, d=2, **kwargs):
        r"""
        Algebra of d-dimensional Hilbert space with only identity operator.

        Parameters
        ----------
        d : int
            Hilbert space dimension.

        kwargs
            Other YAST configuration parameters can be provided, see :meth:`yast.make_config`.

        Notes
        -----
        Default configuration sets :code:`fermionic` to :code:`False`.
        """
        self._d= d
        self._sym= 'dense'
        kwargs['fermionic'] = False
        kwargs['sym'] = sym_none
        self.config = make_config(**kwargs)
        self.s = (1, -1)
        self.operators = ('I',)

    def I(self):
        """ Identity operator. """
        return diag(eye(config=self.config, s=self.s, D=self._d))

    def to_dict(self):
        """
        Returns
        -------
        dict(str,yast.Tensor)
            a map from strings to operators
        """
        return {'I': lambda j: self.I()}