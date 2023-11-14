""" Define rules for Z2 symmetry"""
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
            charges: numpy.ndarray(int)
                `k x m x nsym` matrix, where `k` is the number of
                independent blocks, and `m` is the number of fused legs.

            signatures: numpy.ndarray(int)
                integer vector with `m` elements in `{-1, +1}`

            new_signature: int

        Returns
        -------
            numpy.ndarray(int)
                integer matrix with shape (k,NSYM) of fused charges;
                includes multiplication by ``new_signature``
        """
        return np.mod(new_signature * (charges.swapaxes(1, 2) @ signatures), 2)
