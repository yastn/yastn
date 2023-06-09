"""Z2 symmetry"""
import numpy as np
from .sym_abelian import sym_abelian

class sym_Z2(sym_abelian):
    """Z2 symmetry"""

    SYM_ID = 'Z2'
    NSYM = 1

    @classmethod
    def fuse(cls, charges, signatures, new_signature):
        """
        Fusion rule for :math:`Z_2` symmetry.

        Parameters
        ----------
            charges: numpy.ndarray
                rank-3 integer tensor with shape (k, n, NSYM)

            signatures: numpy.ndarray
                integer vector with `n` +1 or -1 elements

            new_signature: int

        Returns
        -------
            teff: numpy.ndarray
                integer matrix with shape (k,NSYM) of fused charges and multiplied by ``new_signature``
        """
        return np.mod(new_signature * (charges.swapaxes(1, 2) @ signatures), 2)
