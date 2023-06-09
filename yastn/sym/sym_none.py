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
                k x number_legs x nsym matrix, where k is the number of independent blocks.

            signatures: nparray(ind)
                vector with number_legs elements

            new_signature: int

        Returns
        -------
            teff: nparray(int)
                matrix of effective fused charges of size k x nsym for new signature
        """
        # charges is an empty matrix
        # swap to properly match non-zero dimensions of returned tset
        return charges.swapaxes(1, 2) @ signatures
