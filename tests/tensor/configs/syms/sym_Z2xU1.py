import numpy as np

SYM_ID = "Z2xU1"
NSYM = 2  # two ints used to distinguish symmetry sectors

def fuse(charges, signatures, new_signature):
    """
    Fusion rule for ... symmetry

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
    teff = new_signature * (charges.swapaxes(1,2) @ signatures)
    teff[:, 0] = np.mod(teff[:, 0], 2)
    return teff
