""" Define rules for U(1)xU(1) symmetry"""
from .sym_abelian import sym_abelian

class sym_U1xU1(sym_abelian):
    """U(1)xU(1) symmetry"""

    SYM_ID = 'U(1)xU(1)'
    NSYM = 2

    @classmethod
    def fuse(cls, charges, signatures, new_signature):
        """
        Fusion rule for U(1) x U(1) symmetry

        Parameters
        ----------
            charges: numpy.ndarray(int)
                rank-3 integer tensor with shape (k, n, NSYM)

            signatures: numpy.ndarray(int)
                integer vector with `n` +1 or -1 elements

            new_signature: int

        Returns
        -------
            numpy.ndarray(int)
                integer matrix with shape (k,NSYM) of fused charges; includes multiplication by ``new_signature``
        """
        return new_signature * (charges.swapaxes(1, 2) @ signatures)
