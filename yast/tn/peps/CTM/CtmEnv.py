from yaps import Lattice
from typing import NamedTuple, Tuple
from dataclasses import dataclass
from yast import tensordot, svd_with_truncation, rand, ones

class ctm_window(NamedTuple):
    """ elements of a 2x2 window for the CTM algorithm. """
    nw : tuple
    ne : tuple
    sw : Tuple
    se : Tuple

class CtmEnv(Lattice):
    def __init__(self, ket, bra=None):
        super().__init__(lattice=ket.lattice, dims=ket.dims, boundary=ket.boundary)
        self.ket = ket
        self.bra = ket if bra is None else bra
        ## assert that ket and bra are matching ....
        inds = set(self.site2index(site) for site in self._sites)
        self._data = {ind: None for ind in inds}

    def __getitem__(self, site):
        assert site in self._sites, "Site is inconsistent with lattice"
        return self._data[self.site2index(site)]

    def __setitem__(self, site, local_env):
        assert site in self._sites, "Site is inconsistent with lattice"
        self._data[self.site2index(site)] = local_env


    def tensors_CtmEnv(self, trajectory):
        """ 
        We can traverse through the lattice in different trajectories     
        Order Left move: site[0, 0], site[1, 0], site[0, 1], site[1, 1]
        Order Right move: sites[0, 1], site[1, 1], site[0, 0], site[1, 0]
        Order Top Move: site[0, 0], site[0, 1], site[1, 0], site[1, 1]
        Order Bottom Move: site[1, 0], site[1, 1], site[0, 0], site[0, 1]
        """
        
        order = []
        s = []
        self.boundary = 'infinite'
        if trajectory == 'h':
            for n in range(self.Nx):
                for m in range(self.Ny): 
                    order.append(tuple((m, n)))
        elif trajectory == 'v':
            for m in range(self.Ny):
                for n in range(self.Nx):
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

def init_rand(A, tc, Dc, lattice):
    """ Initialize random CTMRG environments of peps tensors A. """
    config = A[0, 0].config
  
    list_sites = lattice.sites()

    env= {}

    for ms in list_sites:
        env[ms] = Local_CTM_Env()

    for ms in list_sites:
        env[ms].tl = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        env[ms].tr = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        env[ms].bl = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        env[ms].br = rand(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))


        legs = [env[ms].tl.get_legs(1).conj(),
                A[ms].get_legs(0).conj(),
                A[ms].get_legs(0),
                env[ms].tr.get_legs(0).conj()]
        env[ms].t = rand(config=config, legs=legs)

        legs = [env[ms].br.get_legs(1).conj(),
                A[ms].get_legs(2).conj(),
                A[ms].get_legs(2),
                env[ms].bl.get_legs(0).conj()]
        env[ms].b = rand(config=config, legs=legs)

        legs = [env[ms].bl.get_legs(1).conj(),
                A[ms].get_legs(1).conj(),
                A[ms].get_legs(1),
                env[ms].tl.get_legs(0).conj()]
        env[ms].l = rand(config=config, legs=legs)

        legs = [env[ms].tr.get_legs(1).conj(),
                A[ms].get_legs(3).conj(),
                A[ms].get_legs(3),
                env[ms].br.get_legs(0).conj()]
        env[ms].r = rand(config=config, legs=legs)

        env[ms].t = env[ms].t.fuse_legs(axes=(0, (1, 2), 3))
        env[ms].b = env[ms].b.fuse_legs(axes=(0, (1, 2), 3))
        env[ms].l = env[ms].l.fuse_legs(axes=(0, (1, 2), 3))
        env[ms].r = env[ms].r.fuse_legs(axes=(0, (1, 2), 3))

    return env



