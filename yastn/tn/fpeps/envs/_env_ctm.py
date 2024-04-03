from typing import NamedTuple
from dataclasses import dataclass
from .... import rand, ones, YastnError, Leg, tensordot
from ... import mps
from .._peps import Peps, Peps2Layers
from ..gates import match_ancilla_1s, apply_gate
from .._geometry import Bond


class ctm_window(NamedTuple):
    """ elements of a 2x2 window for the CTM algorithm. """
    nw : tuple  # north-west
    ne : tuple  # north-east
    sw : tuple  # south-west
    se : tuple  # south-east


class EnvCTM(Peps):
    r""" Geometric information about the lattice provided to ctm tensors """
    def __init__(self, psi):
        super().__init__(psi.geometry)
        self.psi = Peps2Layers(psi) if psi.has_physical() else psi

        windows = []
        for site in self.sites():
            win = [self.nn_site(site, d=d) for d in ('r', 'b', 'br')]
            if None not in win:
                windows.append(ctm_window(site, *win))
        self._windows = tuple(windows)
        self.init_()

    def copy(self):
        env = EnvCTM(self.psi)
        env._data = {k : v.copy() for k, v in self._data.items()}
        return env

    def tensors_CtmEnv(self):
        return self._windows

    def boundary_mps(self, n, dirn):
        """ Convert environmental tensors of Ctm to an MPS """
        if dirn == 'b':
            H = mps.Mps(N=self.Ny)
            for ny in range(self.Ny):
                H.A[ny] = self[n, ny].b.transpose(axes=(2, 1, 0))
            H = H.conj()
        elif dirn == 'r':
            H = mps.Mps(N=self.Nx)
            for nx in range(self.Nx):
                H.A[nx] = self[nx, n].r
        elif dirn == 't':
            H = mps.Mps(N=self.Ny)
            for ny in range(self.Ny):
                H.A[ny] = self[n, ny].t
        elif dirn == 'l':
            H = mps.Mps(N=self.Nx)
            for nx in range(self.Nx):
                H.A[nx] = self[nx, n].l.transpose(axes=(2, 1, 0))
            H = H.conj()
        return H


    def init_(self, t0=None, D0=1, type='rand'):
        """ Initialize random CTMRG environments of peps tensors A. """
        if type not in ('rand', 'ones'):
            raise YastnError(f"type = {type} not recognized in EnvCTM init. Should be in ('rand', 'ones')")

        config = self.psi.config
        tensor_init = rand if type == 'rand' else ones

        if t0 is None:
            t0 = (0,) * config.sym.NSYM
        leg0 = Leg(config, s=1, t=[t0], D=[D0])

        for site in self.sites():
            legsA = self.psi[site].get_legs()

            self[site] = Local_CTMEnv()
            self[site].tl = tensor_init(config, legs=[leg0, leg0.conj()])
            self[site].tr = tensor_init(config, legs=[leg0, leg0.conj()])
            self[site].bl = tensor_init(config, legs=[leg0, leg0.conj()])
            self[site].br = tensor_init(config, legs=[leg0, leg0.conj()])
            self[site].t = tensor_init(config, legs=[leg0, legsA[0].conj(), leg0.conj()])
            self[site].b = tensor_init(config, legs=[leg0, legsA[2].conj(), leg0.conj()])
            self[site].l = tensor_init(config, legs=[leg0, legsA[1].conj(), leg0.conj()])
            self[site].r = tensor_init(config, legs=[leg0, legsA[3].conj(), leg0.conj()])

    def measure_1site(self, op, site=None):
        r"""
        dictionary containing site coordinates as keys and their corresponding expectation values

        Parameters
        ----------
        env: class CtmEnv
            class containing ctm environment tensors along with lattice structure data
        op: single site operator
        """
        if site is None:
            return {site: self.measure_1site(op, site) for site in self.sites()}

        lenv = self[site]
        ten = self.psi[site]
        val_no = measure_one(lenv, ten)

        op_aux = match_ancilla_1s(op, ten.A)
        ten.A = ten.A @ op_aux.T

        val_op = measure_one(lenv, ten)
        return val_op / val_no


    def measure_nn(self, O0, O1, bond=None):
        if bond is None:
             return {bond: self.measure_nn(O0, O1, bond) for bond in self.bonds()}

        bond = Bond(*bond)
        env0 = self[bond.site0]
        env1 = self[bond.site1]
        ten0 = self.psi[bond.site0]
        ten1 = self.psi[bond.site1]

        if bond.dirn == 'h':
            vecl = (env0.bl @ env0.l) @ (env0.tl @ env0.t)
            vecr = (env1.tr @ env1.r) @ (env1.br @ env1.b)

            tmp0 = ten0._attach_01(vecl)
            tmp0 = tensordot(env0.b, tmp0, axes=((2, 1), (0, 1)))
            tmp1 = ten1._attach_23(vecr)
            tmp1 = tensordot(env1.t, tmp1, axes=((2, 1), (0, 1)))
            val_no = tensordot(tmp0, tmp1, axes=((0, 1, 2), (1, 0, 2))).to_number()

            ten0.A = apply_gate(ten0.A, O0, dir='l')
            ten1.A = apply_gate(ten1.A, O1, dir='r')

            tmp0 = ten0._attach_01(vecl)
            tmp0 = tensordot(env0.b, tmp0, axes=((2, 1), (0, 1)))
            tmp1 = ten1._attach_23(vecr)
            tmp1 = tensordot(env1.t, tmp1, axes=((2, 1), (0, 1)))
            val_op = tensordot(tmp0, tmp1, axes=((0, 1, 2), (1, 0, 2))).to_number()
        else:
            vect = (env0.l @ env0.tl) @ (env0.t @ env0.tr)
            vecb = (env1.r @ env1.br) @ (env1.b @ env1.bl)

            tmp0 = ten0._attach_01(vect)
            tmp0 = tensordot(tmp0, env0.r, axes=((2, 3), (0, 1)))
            tmp1 = ten1._attach_23(vecb)
            tmp1 = tensordot(tmp1, env1.l, axes=((2, 3), (0, 1)))
            val_no = tensordot(tmp0, tmp1, axes=((0, 1, 2), (2, 1, 0))).to_number()

            ten0.A = apply_gate(ten0.A, O0, dir='t')
            ten1.A = apply_gate(ten1.A, O1, dir='b')

            tmp0 = ten0._attach_01(vect)
            tmp0 = tensordot(tmp0, env0.r, axes=((2, 3), (0, 1)))
            tmp1 = ten1._attach_23(vecb)
            tmp1 = tensordot(tmp1, env1.l, axes=((2, 3), (0, 1)))
            val_op = tensordot(tmp0, tmp1, axes=((0, 1, 2), (2, 1, 0))).to_number()

        return val_op / val_no


    def update_(method="2site"):  # single sweep
        pass



def measure_one(lenv, ten):
    vect = (lenv.l @ lenv.tl) @ (lenv.t @ lenv.tr)
    vect = ten._attach_01(vect)
    vect = tensordot(vect, lenv.r, axes=((2, 3), (0, 1)))
    vecb = (lenv.br @ lenv.b) @ lenv.bl
    return tensordot(vect, vecb, axes=((0, 1, 2), (2, 1, 0))).to_number()


@dataclass()
class Local_CTMEnv():
    """ data class for CTM environment tensors associated with each peps tensor """

    tl : any = None # top-left
    tr : any = None # top-right
    bl : any = None # bottom-left
    br : any = None # bottom-right
    t : any = None  # top
    b : any = None  # bottom
    l : any = None  # left
    r : any = None  # right

    def copy(self):
        return Local_CTMEnv(tl=self.tl, tr=self.tr, bl=self.bl, br=self.br, t=self.t, b=self.b, l=self.l, r=self.r)
