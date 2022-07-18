import yast

class YampsError(Exception):
    pass

def use_default_basis(config, basis_type):
        r""""
        Use a basis for single particle operators compatible with config.

        "identity" = identity operator
        "c"        = annihilation operator
        "cp"       = creation operator
        
        The basis is written to fullbasis which means that it will overwrite the basis if keys overlap.
        """
        fullbasis = {}
        if basis_type == 'creation_annihilation':
                if config.sym.SYM_ID == 'dense':
                        Ds, s = (2, 2), (1, -1)

                        CP = yast.Tensor(config=config, s=s)
                        CP.set_block(Ds=Ds, val=[[0, 0], [1, 0]])

                        C = yast.Tensor(config=config, s=s)
                        C.set_block(Ds=Ds, val=[[0, 1], [0, 0]])

                        EE = yast.Tensor(config=config, s=s)
                        EE.set_block(Ds=Ds, val=[[1, 0], [0, 1]])

                elif config.sym.SYM_ID == 'U(1)':
                        Ds, s = (1, 1), (1, -1)

                        C = yast.Tensor(config=config, s=s, n=-1)
                        C.set_block(Ds=Ds, val=1, ts=(0, 1))

                        CP = yast.Tensor(config=config, s=s, n=1)
                        CP.set_block(Ds=Ds, val=1, ts=(1, 0))

                        EE = yast.Tensor(config=config, s=s, n=0)
                        EE.set_block(Ds=Ds, val=1, ts=(0, 0))
                        EE.set_block(Ds=Ds, val=1, ts=(1, 1))
                elif config.sym.SYM_ID == 'Z2':
                        Ds, s = (1, 1), (1, -1)

                        X = yast.Tensor(config=config, s=s, n=1)
                        X.set_block(Ds=Ds, val=1, ts=(1, 0))
                        X.set_block(Ds=Ds, val=1, ts=(0, 1))

                        Y = yast.Tensor(config=config, s=s, n=1)
                        Y.set_block(Ds=Ds, val=1j, ts=(1, 0))
                        Y.set_block(Ds=Ds, val=-1j, ts=(0, 1))

                        C = 0.5*(X + 1j*Y)
                        CP = 0.5*(X - 1j*Y)

                        EE = yast.Tensor(config=config, s=s, n=0)
                        EE.set_block(Ds=Ds, val=1, ts=(0, 0))
                        EE.set_block(Ds=Ds, val=1, ts=(1, 1))
                else:
                        raise YampsError("Entry is not defined.")
                fullbasis['identity'] = lambda j: EE
                fullbasis['c'] = lambda j: C
                fullbasis['cp'] = lambda j: CP
        elif basis_type == 'pauli_matrices':
                if config.sym.SYM_ID == 'dense':
                        Ds, s = (2, 2), (1, -1)

                        X = yast.Tensor(config=config, s=s)
                        X.set_block(Ds=Ds, val=[[0, 1], [1, 0]])

                        Y = yast.Tensor(config=config, s=s)
                        Y.set_block(Ds=Ds, val=[[0, -1j], [1j, 0]])

                        Z = yast.Tensor(config=config, s=s)
                        Z.set_block(Ds=Ds, val=[[1, 0], [0, -1]])

                        EE = yast.Tensor(config=config, s=s)
                        EE.set_block(Ds=Ds, val=[[1, 0], [0, 1]])
                elif config.sym.SYM_ID == 'Z2':
                        Ds, s = (1, 1), (1, -1)

                        X = yast.Tensor(config=config, s=s, n=1)
                        X.set_block(Ds=Ds, val=1, ts=(1, 0))
                        X.set_block(Ds=Ds, val=1, ts=(0, 1))

                        Y = yast.Tensor(config=config, s=s, n=1)
                        Y.set_block(Ds=Ds, val=1j, ts=(1, 0))
                        Y.set_block(Ds=Ds, val=-1j, ts=(0, 1))

                        Z = yast.Tensor(config=config, s=s, n=0)
                        Z.set_block(Ds=Ds, val=1, ts=(0, 0))
                        Z.set_block(Ds=Ds, val=-1, ts=(1, 1))

                        EE = yast.Tensor(config=config, s=s, n=0)
                        EE.set_block(Ds=Ds, val=1, ts=(0, 0))
                        EE.set_block(Ds=Ds, val=1, ts=(1, 1))
                else:
                        raise YampsError("Entry is not defined.")
                fullbasis['identity'] = lambda j: EE
                fullbasis['x'] = lambda j: X
                fullbasis['y'] = lambda j: Y
                fullbasis['z'] = lambda j: Z               
        else:
                raise YampsError("Entry is not defined.")
        return fullbasis

