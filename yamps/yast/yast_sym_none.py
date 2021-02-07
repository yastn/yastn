nsym = 0  # trivial symmetry sector

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
    return (charges.swapaxes(1,2) @ signatures) # just to match non-zero dimensions; it is an empty matrix
