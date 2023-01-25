class General:
    """ Predefine operators for spinless fermions. """

    def __init__(self, basis_operators):
        """ 
        Generator of any arbritrary form. The operators are provided by user.
        
        No predefined operators.

        fermionic is False unless set to the other by a user

        Parameters
        ----------
        basis_operators : dictionary
            Should be a dictionary with elements in a form: 
            name: lambda j: tensor
            with name: str, name of the operator,
                j: index, single index,
                tensor: yast.Tensor, Tensor with bra and ket phisical indicies.
            All yast.Tensor-s have to have the same symmetry.

        Notes
        -----
        There are no checkpoints to check the validity of the input. It is user's responsibility
        to make sure that he/she provides a correct format of the input.
        
        basis_operators have to include an identity operators defined under a key 'I'.
        """
        
        self.config = basis_operators['I'](0).config
        self._sym = self.config.sym.SYM_ID
        self.s = (1, -1)
        self.operators =  list(basis_operators.keys())
        self.basis_operators = basis_operators

    def I(self):
        """ Identity operator. """
        return self.basis_operators['I'](0)

    def to_dict(self):
        return self.basis_operators