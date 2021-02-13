""" Define trivial rules for dense tensor"""

nsym = 0  # nothing to distinguish symmetry sector

name = 'Dense'

def fuse(charges, signatures, new_signature):
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
    return charges.swapaxes(1,2) @ signatures  # swap to properly match non-zero dimensions of tset;
