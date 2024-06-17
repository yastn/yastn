import numpy as np
from yastn.sym.sym_abelian import sym_abelian

class sym_Z2xU1(sym_abelian):
    """ Z2xU(1) symmetry; for testing"""
    SYM_ID = "Z2xU1"
    NSYM = 2

    def fuse(charges, signatures, new_signature):
        """ Fusion rule for Z2xU(1) symmetry. """
        teff = new_signature * (charges.swapaxes(1,2) @ signatures)
        teff[:, 0] = np.mod(teff[:, 0], 2)
        return teff
