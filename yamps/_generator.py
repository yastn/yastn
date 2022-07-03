from ._mps import Mpo
from ._auxliary import add
import yast
from itertools import product

class GenerateMpo():
        def __init__(self, N, identity, annihilation, creation, opts={'tol': 1e-14}):
                self.N = N
                self.opts = opts
                # prepare identity
                id = identity.copy().add_leg(axis=0, s=1).add_leg(axis=-1, s=-1)
                self.identity = Mpo(self.N)
                for n in range(self.N):
                        self.identity.A[n] = id.copy()
                self.annihilation = annihilation
                self.creation = creation

        def c(self, j):
                return j, self.annihilation.copy()

        def cp(self, j):
                return j, self.creation.copy()

        def prod(self, *multiplied):
                product = self.identity.copy()
                for im in range(len(multiplied)):
                        j, op = multiplied[im]
                        operator = op.copy().add_leg(axis=0, s=1)
                        leg = operator.get_legs(0)
                        cfg = operator.config
                        product.A[j] = yast.ncon([product.A[j], operator], [(-1, 1, -4, -5), (-2, -3, 1)])
                        product.A[j] = product.A[j].fuse_legs(axes=((0, 1), 2, 3, 4), mode='hard')
                        for j3 in range(j):
                                virtual = yast.ones(config=cfg, legs=[leg, leg.conj()])
                                product.A[j-1-j3] = yast.ncon([product.A[j-1-j3], virtual], [(-1, -3, -4, -5), (-2, -6)])
                                product.A[j-1-j3] = product.A[j-1-j3].swap_gate(axes=(1, 2))
                                product.A[j-1-j3] = product.A[j-1-j3].fuse_legs(axes=((0, 1), 2, 3, (4, 5)), mode='hard')
                for n in range(self.N):
                        product.A[n] = product.A[n].drop_leg_history(axis=(0,3))
                return product

        def sum(self, f, *ranges):
                result = []
                for js in product(*ranges):
                        element = f(*js)
                        result.append(element)
                M = add(*result)
                if self.opts:
                        M.canonize_sweep(to='last', normalize=False)
                        M.truncate_sweep(to='first', opts=self.opts, normalize=False)
                return M

def generateMpo(N, config, opts={'tol': 1e-14}):
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
        # które to creacji a które anihilacji?  W tej wersji działa dobrze
        return GenerateMpo(N, identity=EE, annihilation=CP, creation=C, opts=opts)