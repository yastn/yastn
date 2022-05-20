import numpy as np

SYM_ID = "Z2xU1"
NSYM = 2

def fuse(charges, signatures, new_signature):
    """
    Fusion rule for Z2xU(1) symmetry

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
    teff = new_signature * (charges.swapaxes(1,2) @ signatures)
    teff[:, 0] = np.mod(teff[:, 0], 2)
    return teff
