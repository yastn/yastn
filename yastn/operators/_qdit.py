from __future__ import annotations
from ..initialize import eye
from .. import diag
from ..tensor import Leg
from ._meta_operators import meta_operators

class Qdit(meta_operators):
    # Predefine dense operators with set dimension of the local space.
    def __init__(self, d=2, **kwargs):
        r"""
        Algebra of d-dimensional Hilbert space with only identity operator, and local Hilbert space as a :class:`yastn.Leg`.

        Parameters
        ----------
        d : int
            Hilbert space dimension.

        kwargs
            Other YASTN configuration parameters can be provided, see :meth:`yastn.make_config`.

        Notes
        -----
        Default configuration sets :code:`fermionic` to :code:`False`.
        """
        kwargs['fermionic'] = False
        kwargs['sym'] = 'dense'
        super().__init__(**kwargs)
        self._d = d
        self._sym = 'dense'
        self.operators = ('I',)

    def space(self) -> yastn.Leg:
        r""" :class:`yastn.Leg` describing local Hilbert space. """
        return Leg(self.config, s=1, D=(self._d,))

    def I(self) -> yastn.Tensor:
        """ Identity operator. """
        return diag(eye(config=self.config, s=self.s, D=self._d))

    def to_dict(self):
        """
        Returns
        -------
        dict(str,yastn.Tensor)
            a map from strings to operators
        """
        return {'I': lambda j: self.I()}
