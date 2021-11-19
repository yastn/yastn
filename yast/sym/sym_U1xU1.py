""" Define rules for U(1) x U(1) symmetry"""

SYM_ID = 'U(1)xU(1)'
NSYM = 2  # single int is used to distinguish symmetry sectors


def fuse(charges, signatures, new_signature):
    """
    Fusion rule for U(1) x U(1) symmetry

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
    return new_signature * (charges.swapaxes(1, 2) @ signatures)
