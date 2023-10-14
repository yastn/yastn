""" Define rules for U(1) x U(1) x Z2 symmetry"""
import numpy as np
from .sym_abelian import sym_abelian

class sym_U1xU1xZ2(sym_abelian):
    """ U(1) x U(1) x Z2 -- here Z2 will be used to account for global fermionic parity. """

    SYM_ID = "U(1)xU(1)xZ2"
    NSYM = 3  # three ints used to distinguish symmetry sectors

    @classmethod
    def fuse(cls, charges, signatures, new_signature):
        """
        Fusion rule for ... symmetry

        Parameters
        ----------
            charges: nparray(int)
                `k x m x nsym` matrix, where `k` is the number of independent blocks,
                and `m` is the number of fused legs.

            signatures: numpy.ndarray(int)
                integer vector with `m` elements in `{-1, +1}`

            new_signature: int

        Returns
        -------
            nparray(int)
                integer matrix with shape (k,NSYM) of fused charges;
                includes multiplication by ``new_signature``
        """
        teff = new_signature * (charges.swapaxes(1,2) @ signatures)
        teff[:, 2] = np.mod(teff[:, 2], 2)
        return teff
