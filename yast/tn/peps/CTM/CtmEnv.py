from typing import NamedTuple, Tuple
from itertools import accumulate
from dataclasses import dataclass
from ....tn import mps
#from yast.tn.peps import Lattice
from yast.tn.peps import Peps
from yast.tn.peps.operators.gates import match_ancilla_1s
from yast import rand, tensordot

class ctm_window(NamedTuple):
    """ elements of a 2x2 window for the CTM algorithm. """
    nw : tuple
    ne : tuple
    sw : Tuple
    se : Tuple

class CtmEnv(Peps):
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


    def tensors_CtmEnv(self, trajectory):
        """ 
        Choosing 2x2 ctm windows for horizontal and vertical moves of the CTM of a mxn lattice
        """
        
        order = []
        s = []
        self.boundary = 'infinite'
        if trajectory == 'h':
            for n in range(self.Ny):
                for m in range(self.Nx): 
                    order.append(tuple((m, n)))
        elif trajectory == 'v':
            for m in range(self.Nx):
                for n in range(self.Ny):
                    order.append(tuple((m, n)))
 
        for ms in order:
            site_se = ms
            site_ne = self.nn_site(site_se, d='t')
            site_sw = self.nn_site(site_se, d='l')
            site_nw = self.nn_site(site_sw, d='t')
            s.append(ctm_window(site_nw, site_ne, site_sw, site_se))

        return s

@dataclass()
class Local_CTM_Env: # no more variables than the one given 
    tl : any = None
    tr : any = None
    bl : any = None
    br : any = None
    t : any = None
    b : any = None
    l : any = None
    b : any = None


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


def init_rand(AAb, tc, Dc):
    """ Initialize random CTMRG environments of peps tensors A. """

    config = AAb[(0,0)].A.config 
    env= {}

    for ms in AAb.sites():
        env[ms] = Local_CTM_Env()
        env[ms].tl = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        env[ms].tr = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        env[ms].bl = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        env[ms].br = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        legs = [env[ms].tl.get_legs(1).conj(),
                AAb[ms].A.get_legs(0).conj(),
                AAb[ms].A.get_legs(0),
                env[ms].tr.get_legs(0).conj()]
        env[ms].t = rand(config=config, legs=legs)
        legs = [env[ms].br.get_legs(1).conj(),
                AAb[ms].A.get_legs(2).conj(),
                AAb[ms].A.get_legs(2),
                env[ms].bl.get_legs(0).conj()]
        env[ms].b = rand(config=config, legs=legs)
        legs = [env[ms].bl.get_legs(1).conj(),
                AAb[ms].A.get_legs(1).conj(),
                AAb[ms].A.get_legs(1),
                env[ms].tl.get_legs(0).conj()]
        env[ms].l = rand(config=config, legs=legs)
        legs = [env[ms].tr.get_legs(1).conj(),
                AAb[ms].A.get_legs(3).conj(),
                AAb[ms].A.get_legs(3),
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
   
    Takes a CTM emvironments and a complet list of projectors to sample from.
    """

    config = state[0, 0].config
    rands = (config.backend.rand(state.Nx * state.Ny) + 1) / 2


    out = {}
    count = 0
    vR = CtmEnv2Mps(state, CTMenv, index=state.Ny - 1, index_type='r')

    for ny in range(state.Ny - 1, -1, -1):
        O = state.mpo(index=ny, index_type='column')
        vL = CtmEnv2Mps(state, CTMenv, index=ny, index_type='l').conj()
        env = mps.Env3(vL, O, vR).setup(to = 'first')

        for nx in range(0, state.Nx):
            dpt = O[nx].copy()
            loc_projectors = [match_ancilla_1s(pr, dpt.A) for pr in projectors]
            prob = []
            norm_prob = env.measure(bd=(nx - 1, nx))
            for proj in loc_projectors:
                dpt_pr = dpt.copy()
                dpt_pr.A = tensordot(dpt_pr.A, proj, axes=(4, 1))
                O[nx] = dpt_pr
                env.update_env(nx, to='last')
                prob.append(env.measure(bd=(nx, nx+1)) / norm_prob)
            assert abs(sum(prob) - 1) < 1e-12
            rand = rands[count]
            ind = sum(apr < rand for apr in accumulate(prob))
            out[(nx, ny)] = ind
            dpt.A = tensordot(dpt.A, loc_projectors[ind], axes=(4, 1))
            O[nx] = dpt
            env.update_env(nx, to='last')
            count += 1

        if opts_svd is None:
            opts_svd = {'D_total': max(vL.get_bond_dimensions())}

        vRnew = mps.zipper(O, vR, opts=opts_svd)
        if opts_var is None:
            opts_var = {}
        mps.variational_(vRnew, O, vR, method='1site', **opts_var)
        vR = vRnew
    return out
