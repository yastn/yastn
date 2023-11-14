""" Define rules for Z3 symmetry"""
import numpy as np
from .sym_abelian import sym_abelian

class sym_Z3(sym_abelian):
    """Z3 symmetry"""

    SYM_ID = 'Z3'
    NSYM = 1  # single int is used to distinguish symmetry sectors

    @classmethod
    def fuse(cls, charges, signatures, new_signature):
        """
        Fusion rule for Z3 symmetry

        Parameters
        ----------
            charges: nparray(int)
                `k x m x nsym` matrix, where `k` is the number of
                independent blocks, and `m` is the number of fused legs.

            signatures: numpy.ndarray(int)
                integer vector with `m` elements in `{-1, +1}`

            new_signature: int

        Returns
        -------
            teff: nparray(int)
                integer matrix with shape (k,NSYM) of fused charges;
                includes multiplication by ``new_signature``
        """
        return np.mod(new_signature * (charges.swapaxes(1, 2) @ signatures), 3)
