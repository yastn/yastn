"""U(1) symmetry"""

SYM_ID= 'U(1)'
NSYM= 1

def fuse(charges, signatures, new_signature):
    """
    Fusion rule for U(1) symmetry.

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
    return new_signature * (charges.swapaxes(1, 2) @ signatures)
