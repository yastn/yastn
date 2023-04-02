class sym_meta(type):
    def __str__(cls):
        return cls.SYM_ID

    # def __repr__(cls):
    #     """ standard path to predefined symmetries """
    #     return "yastn.sym.sym_" + cls.SYM_ID


class sym_abelian(metaclass=sym_meta):
    """
    Interface to be subclassed for concrete symmetry implementations.
    """
    SYM_ID = 'symmetry-name'
    NSYM = len('length-of-charge-vector')

    @classmethod
    def fuse(cls, charges, signatures, new_signature):
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
        raise NotImplementedError("Subclasses need to override the fuse function")  # pragma: no cover
