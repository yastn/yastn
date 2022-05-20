r"""
YAST specifies symmetry through any object be it plain Python module, :class:`types.SimpleNamespace` 
or class instance which defines

* ``SYM_ID`` string label for the symmetry
* ``NSYM`` number of elements in the charge vector. For example, `NSYM=1` for U(1) or :math:`Z_2`
  group. For product groups such as U(1)xU(1) `NSYM=2`.
* addition of charges through function `fuse`
"""

SYM_ID= 'symmetry-name'
NSYM= len('length-of-charge-vector')

def fuse(charges, signatures, new_signature):
    """
    Fusion rule for abelian symmetry. An `i`-th row ``charges[i,:,:]`` contains `n` length-`NSYM`
    charge vectors. For each row, the charge vectors are added up (fused) with selected ``signature`` 
    according to the group addition rules.

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
    pass
