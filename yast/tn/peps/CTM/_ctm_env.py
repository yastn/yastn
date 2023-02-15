from typing import NamedTuple, Tuple
import numpy as np
from itertools import accumulate
from dataclasses import dataclass
from ....tn import mps
#from yast.tn.peps import Lattice
from yast.tn.peps import Peps
from yast.tn.peps.operators.gates import match_ancilla_1s
from yast import rand, tensordot
import copy

class ctm_window(NamedTuple):
    """ elements of a 2x2 window for the CTM algorithm. """
    nw : tuple
    ne : tuple
    sw : Tuple
    se : Tuple

class CtmEnv(Peps):
    r""" Geometric information about the lattice provided to ctm tensors """
    def __init__(self, lattice='checkerboard', dims=(2, 2), boundary='infinite'):
        super().__init__(lattice=lattice, dims=dims, boundary=boundary)
        #self.ket = ket
        #self.bra = ket if bra is None else bra
        ## assert that ket and bra are matching ....
        inds = set(self.site2index(site) for site in self._sites)
        self._data = {ind: None for ind in inds}  # I don't know what to do with ket and bra; for now I am importing the double peps tensors 
                                                  # within the CtmEnv structure

    def __getitem__(self, site):
        assert site in self._sites, "Site is inconsistent with lattice"
        return self._data[self.site2index(site)]


    def __setitem__(self, site, local_env):
        assert site in self._sites, "Site is inconsistent with lattice"
        self._data[self.site2index(site)] = local_env

    def copy(self):
        env = CtmEnv(lattice=self.lattice, dims=self.dims, boundary=self.boundary)
        env._data = {k : v.copy() for k, v in self._data.items()}        
        return env


    def tensors_CtmEnv(self, trajectory):
        """ 
        Choosing 2x2 ctm windows for horizontal and vertical moves of the CTM of a mxn lattice
        """

        ss = []

        if trajectory == 'h':  # create columns
            order = []

            if self.boundary == 'infinite':
                for n in range(self.Ny):
                    hor = []
                    for m in range(self.Nx): 
                        hor.append(tuple((m, n)))
                    order.append(hor)

            elif self.boundary == 'finite':
                for n in range(self.Ny-1):
                    hor =[]
                    for m in range(self.Nx-1):
                        hor.append(tuple((m+1, n+1)))
                    order.append(hor)

        elif trajectory == 'v':
            order = []

            if self.boundary == 'infinite':
                for m in range(self.Nx):
                    ver = []
                    for n in range(self.Ny):
                        ver.append(tuple((m, n))) 
                    order.append(ver)
            elif self.boundary == 'finite':
                for m in range(self.Nx-1):
                    ver = []
                    for n in range(self.Ny-1):
                        ver.append(tuple((m+1, n+1))) 
                    order.append(ver)
            print(order)

        
        for ms in order:
            s = []
            for xs in ms:
                site_se = xs
                site_ne = self.nn_site(site_se, d='t')
                site_sw = self.nn_site(site_se, d='l')
                site_nw = self.nn_site(site_se, d='tl')
                s.append(ctm_window(site_nw, site_ne, site_sw, site_se))
            ss.append(s)
        
    
        if self.lattice == 'checkerboard':
            ss.clear()
            ss = [[ctm_window(nw=(0, 0), ne=(0, 1), sw=(1, 0), se=(1, 1)),ctm_window(nw=(1, 0), ne=(1, 1), sw=(0, 0), se=(0, 1))]]

        return ss


class Proj(Peps):
    def __init__(self, lattice='checkerboard', dims=(2, 2), boundary='infinite'):
        super().__init__(lattice=lattice, dims=dims, boundary=boundary)
        #self.ket = ket
        #self.bra = ket if bra is None else bra
        ## assert that ket and bra are matching ....
        inds = set(self.site2index(site) for site in self._sites)
        self._data = {ind: None for ind in inds}  # I don't know what to do with ket and bra; for now I am importing the double peps tensors 
                                                  # within the CtmEnv structure

    def __getitem__(self, site):
        assert site in self._sites, "Site is inconsistent with lattice"
        return self._data[self.site2index(site)]


    def __setitem__(self, site, local_env):
        assert site in self._sites, "Site is inconsistent with lattice"
        self._data[self.site2index(site)] = local_env

    def copy(self):
        proj_new = Proj(lattice=self.lattice, dims=self.dims, boundary=self.boundary)
        proj_new._data = {k : v.copy() for k, v in self._data.items()}        
        return proj_new


@dataclass()
class Local_Projector_Env(): # no more variables than the one given 
    # data class for projectors labelled by a single lattice site calculate during ctm renormalization step

    hlt : any = None   # horizontal left top
    hlb : any = None   # horizontal left bottom
    hrt : any = None    # horizontal right top
    hrb : any = None    # horizontal right bottom
    vtl : any = None     # vertical top left
    vtr : any = None    # vertical top right
    vbl : any = None    # vertical bottom left
    vbr : any = None     # vertical bottom right

    def copy(self):
        return Local_Projector_Env(hlt=self.hlt, hlb=self.hlb, hrt=self.hrt, hrb=self.hrb, vtl=self.vtl, vtr=self.vtr, vbl=self.vbl, vbr=self.vbr)


@dataclass()
class Local_CTM_Env(): # no more variables than the one given 
    tl : any = None
    tr : any = None
    bl : any = None
    br : any = None
    t : any = None
    b : any = None
    l : any = None
    r : any = None

    def copy(self):
        return Local_CTM_Env(tl=self.tl, tr=self.tr, bl=self.bl, br=self.br, t=self.t, b=self.b, l=self.l, r=self.r)


def CtmEnv2Mps(net, env, index, index_type):
    """ Convert environmental tensors of Ctm to an MPS """
    if index_type == 'b':
        nx = index
        H = mps.Mps(N=net.Ny)
        for ny in range(net.Ny):
            site = (nx, ny)
            H.A[nx] = env[site].b
    elif index_type == 'r':
        ny = index
        H = mps.Mps(N=net.Nx)
        for nx in range(net.Nx):
            site = (nx, ny)
            H.A[nx] = env[site].r
    elif index_type == 't':
        nx = index
        H = mps.Mps(N=net.Ny)
        for ny in range(net.Ny):
            site = (nx, ny)
            H.A[ny] = env[site].t
    elif index_type == 'l':
        ny = index
        H = mps.Mps(N=net.Nx)
        for nx in range(net.Nx):
            site = (nx, ny)
            H.A[nx] = env[site].l
    return H


def init_rand(A, tc, Dc):
    """ Initialize random CTMRG environments of peps tensors A. """

    config = A[(0,0)].config 
    env= CtmEnv(A.lattice, A.dims, A.boundary)

    for ms in A.sites():
        B = A[ms].copy()
        B = B.unfuse_legs(axes=(0,1))
        env[ms] = Local_CTM_Env()
        env[ms].tl = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        env[ms].tr = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        env[ms].bl = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        env[ms].br = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        legs = [env[ms].tl.get_legs(1).conj(),
                B.get_legs(0).conj(),
                B.get_legs(0),
                env[ms].tr.get_legs(0).conj()]
        env[ms].t = rand(config=config, legs=legs)
        legs = [env[ms].br.get_legs(1).conj(),
                B.get_legs(2).conj(),
                B.get_legs(2),
                env[ms].bl.get_legs(0).conj()]
        env[ms].b = rand(config=config, legs=legs)
        legs = [env[ms].bl.get_legs(1).conj(),
                B.get_legs(1).conj(),
                B.get_legs(1),
                env[ms].tl.get_legs(0).conj()]
        env[ms].l = rand(config=config, legs=legs)
        legs = [env[ms].tr.get_legs(1).conj(),
                B.get_legs(3).conj(),
                B.get_legs(3),
                env[ms].br.get_legs(0).conj()]
        env[ms].r = rand(config=config, legs=legs)
        env[ms].t = env[ms].t.fuse_legs(axes=(0, (1, 2), 3))
        env[ms].b = env[ms].b.fuse_legs(axes=(0, (1, 2), 3))
        env[ms].l = env[ms].l.fuse_legs(axes=(0, (1, 2), 3))
        env[ms].r = env[ms].r.fuse_legs(axes=(0, (1, 2), 3))

    return env

def sample(state, CTMenv, projectors, opts_svd=None, opts_var=None):
    """ 
    Sample a random configuration from a finite peps. 
   
    Takes  CTM emvironments and a complete list of projectors to sample from.
    """

    config = state[0, 0].config
    rands = (config.backend.rand(state.Nx * state.Ny) + 1) / 2

    out = {}
    count = 0
    vR = CtmEnv2Mps(state, CTMenv, index=state.Ny-1, index_type='r')

    for ny in range(state.Ny - 1, -1, -1):

        Os = state.mpo(index=ny, index_type='column')
        vL = CtmEnv2Mps(state, CTMenv, index=ny, index_type='l').conj()

        env = mps.Env3(vL, Os, vR).setup(to = 'first')

        for nx in range(0, state.Nx):
            dpt = Os[nx].copy()
            loc_projectors = [match_ancilla_1s(pr, dpt.A) for pr in projectors]
            prob = []
            norm_prob = env.measure(bd=(nx - 1, nx))
            for proj in loc_projectors:
                dpt_pr = dpt.copy()
                dpt_pr.A = tensordot(dpt_pr.A, proj, axes=(4, 1))
                Os[nx] = dpt_pr
                env.update_env(nx, to='last')
                prob.append(env.measure(bd=(nx, nx+1)) / norm_prob)

            assert abs(sum(prob) - 1) < 1e-12
            rand = rands[count]
            ind = sum(apr < rand for apr in accumulate(prob))
            out[(nx, ny)] = ind
            dpt.A = tensordot(dpt.A, loc_projectors[ind], axes=(4, 1))
            Os[nx] = dpt
            env.update_env(nx, to='last')
            count += 1

        if opts_svd is None:
            opts_svd = {'D_total': max(vL.get_bond_dimensions())}

        vRnew = mps.zipper(Os, vR, opts=opts_svd)
        if opts_var is None:
            opts_var = {}
        mps.variational_(vRnew, Os, vR, method='1site', **opts_var)
        vR = vRnew
    return out

