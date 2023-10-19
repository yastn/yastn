""" Define trivial rules for dense tensor"""
from .sym_abelian import sym_abelian

class sym_none(sym_abelian):
    """No symmetry"""

    SYM_ID = 'dense'
    NSYM = 0  # nothing to distinguish symmetry sector

    @classmethod
    def fuse(cls, charges, signatures, new_signature):
        """
        Full tensor

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
        # charges is an empty matrix
        # swap to properly match non-zero dimensions of returned tset
        return charges.swapaxes(1, 2) @ signatures
