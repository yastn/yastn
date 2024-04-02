from typing import NamedTuple
from dataclasses import dataclass
from .... import rand, ones, YastnError
from ... import mps
from .._peps import Peps, Peps2Layers


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


    def init(self, tc, Dc, type='rand'):
        """ Initialize random CTMRG environments of peps tensors A. """


        if type not in ('rand', 'ones'):
            raise YastnError(f"type = {type} not recognized in EnvCTM init. Should be in ('rand', 'ones')")

        tensor_init = rand if type == 'rand' else ones

        config = self.psi.config

        # for ms in self.sites():
        #     B = A[ms].copy()
        #     B = B.unfuse_legs(axes=(0,1))
        #     env[ms] = Local_CTMEnv()
        #     env[ms].tl = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        #     env[ms].tr = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        #     env[ms].bl = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        #     env[ms].br = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        #     legs = [env[ms].tl.get_legs(1).conj(),
        #             B.get_legs(0).conj(),
        #             B.get_legs(0),
        #             env[ms].tr.get_legs(0).conj()]
        #     env[ms].t = rand(config=config, legs=legs)
        #     legs = [env[ms].br.get_legs(1).conj(),
        #             B.get_legs(2).conj(),
        #             B.get_legs(2),
        #             env[ms].bl.get_legs(0).conj()]
        #     env[ms].b = rand(config=config, legs=legs)
        #     legs = [env[ms].bl.get_legs(1).conj(),
        #             B.get_legs(1).conj(),
        #             B.get_legs(1),
        #             env[ms].tl.get_legs(0).conj()]
        #     env[ms].l = rand(config=config, legs=legs)
        #     legs = [env[ms].tr.get_legs(1).conj(),
        #             B.get_legs(3).conj(),
        #             B.get_legs(3),
        #             env[ms].br.get_legs(0).conj()]
        #     env[ms].r = rand(config=config, legs=legs)
        #     env[ms].t = env[ms].t.fuse_legs(axes=(0, (1, 2), 3))
        #     env[ms].b = env[ms].b.fuse_legs(axes=(0, (1, 2), 3))
        #     env[ms].l = env[ms].l.fuse_legs(axes=(0, (1, 2), 3))
        #     env[ms].r = env[ms].r.fuse_legs(axes=(0, (1, 2), 3))

        # return env


@dataclass()
class Local_ProjectorEnv():
    """ data class for projectors labelled by a single lattice site calculated during ctm renormalization step """

    hlt : any = None # horizontal left top
    hlb : any = None # horizontal left bottom
    hrt : any = None # horizontal right top
    hrb : any = None # horizontal right bottom
    vtl : any = None # vertical top left
    vtr : any = None # vertical top right
    vbl : any = None # vertical bottom left
    vbr : any = None # vertical bottom right

    def copy(self):
        return Local_ProjectorEnv(hlt=self.hlt, hlb=self.hlb, hrt=self.hrt, hrb=self.hrb, vtl=self.vtl, vtr=self.vtr, vbl=self.vbl, vbr=self.vbr)


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
