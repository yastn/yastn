from typing import NamedTuple, Tuple
from itertools import accumulate
from dataclasses import dataclass
from ....tn import mps
from .... import Tensor
from yastn.tn.fpeps import Lattice
from yastn.tn.fpeps.operators.gates import match_ancilla_1s
from yastn import rand, tensordot, ones
from .._auxiliary import transfer_mpo


class ctm_window(NamedTuple):
    """ elements of a 2x2 window for the CTM algorithm. """
    nw : tuple # north-west
    ne : tuple # north-east
    sw : Tuple # south-west
    se : Tuple # south-east


class CtmEnv(Lattice):
    r""" Geometric information about the lattice provided to ctm tensors """
    def __init__(self, peps):
        super().__init__(lattice=peps.lattice, dims=peps.dims, boundary=peps.boundary)

        if self.lattice == 'checkerboard':
            windows = (ctm_window(nw=(0, 0), ne=(0, 1), sw=(0, 1), se=(0, 0)),
                       ctm_window(nw=(0, 1), ne=(0, 0), sw=(0, 0), se=(0, 1)))
        else:
            windows = []
            for site in self.sites():
                win = [site]
                for d in ('r', 'b', 'br'):
                    win.append(self.nn_site(site, d=d))
                if None not in win:
                    windows.append(ctm_window(*win))
        self._windows = tuple(windows)


    def copy(self):
        env = CtmEnv(self)
        env._data = {k : v.copy() for k, v in self._data.items()}
        return env

    def tensors_CtmEnv(self):
        return self._windows

    def env2mps(self, index, index_type):
        """ Convert environmental tensors of Ctm to an MPS """
        if index_type == 'b':
            nx = index
            H = mps.Mps(N=self.Ny)
            for ny in range(self.Ny):
                site = (nx, ny)
                H.A[nx] = self[site].b.transpose(axes=(2,1,0))
            H = H.conj()
        elif index_type == 'r':
            ny = index
            H = mps.Mps(N=self.Nx)
            for nx in range(self.Nx):
                site = (nx, ny)
                H.A[nx] = self[site].r
        elif index_type == 't':
            nx = index
            H = mps.Mps(N=self.Ny)
            for ny in range(self.Ny):
                site = (nx, ny)
                H.A[ny] = self[site].t
        elif index_type == 'l':
            ny = index
            H = mps.Mps(N=self.Nx)
            for nx in range(self.Nx):
                site = (nx, ny)
                H.A[nx] = self[site].l.transpose(axes=(2,1,0))
            H = H.conj()
        return H


@dataclass()
class Local_ProjectorEnv(): # no more variables than the one given
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
class Local_CTMEnv(): # no more variables than the one given
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



def init_rand(A, tc, Dc):
    """ Initialize random CTMRG environments of peps tensors A. """

    config = A[(0,0)].config
    env= CtmEnv(A)

    for ms in A.sites():
        B = A[ms].copy()
        B = B.unfuse_legs(axes=(0,1))
        env[ms] = Local_CTMEnv()
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


def init_ones(A):
    """ Initialize CTMRG environments of peps tensors A with trivial tensors. """

    config = A[(0,0)].config

    tc = ((0,) * config.sym.NSYM,)
    Dc = (1,)

    env= CtmEnv(A)

    for ms in A.sites():
        B = A[ms].copy()
        B = B.unfuse_legs(axes=(0,1))
        env[ms] = Local_CTMEnv()
        env[ms].tl = ones(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        env[ms].tr = ones(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        env[ms].bl = ones(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        env[ms].br = ones(config=config, s=(1, -1), t=(tc, tc), D=(Dc, Dc))
        legs = [env[ms].tl.get_legs(1).conj(),
                B.get_legs(0).conj(),
                B.get_legs(0),
                env[ms].tr.get_legs(0).conj()]
        env[ms].t = ones(config=config, legs=legs)
        legs = [env[ms].br.get_legs(1).conj(),
                B.get_legs(2).conj(),
                B.get_legs(2),
                env[ms].bl.get_legs(0).conj()]
        env[ms].b = ones(config=config, legs=legs)
        legs = [env[ms].bl.get_legs(1).conj(),
                B.get_legs(1).conj(),
                B.get_legs(1),
                env[ms].tl.get_legs(0).conj()]
        env[ms].l = ones(config=config, legs=legs)
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
    vR = CTMenv.env2mps(index=state.Ny-1, index_type='r') # right boundary of indexed column through CTM environment tensors

    for ny in range(state.Ny - 1, -1, -1):

        Os = transfer_mpo(state, index=ny, index_type='column') # converts ny colum of PEPS to MPO
        vL = CTMenv.env2mps(index=ny, index_type='l') # left boundary of indexed column through CTM environment tensors

        env = mps.Env(vL, Os, vR).setup_(to = 'first')

        for nx in range(0, state.Nx):
            dpt = Os[nx].copy()
            loc_projectors = [match_ancilla_1s(pr, dpt.A) for pr in projectors]
            prob = []
            norm_prob = env.measure(bd=(nx - 1, nx))
            for proj in loc_projectors:
                dpt_pr = dpt.copy()
                dpt_pr.A = tensordot(dpt_pr.A, proj, axes=(4, 1))
                Os[nx] = dpt_pr
                env.update_env_(nx, to='last')
                prob.append(env.measure(bd=(nx, nx+1)) / norm_prob)

            assert abs(sum(prob) - 1) < 1e-12
            rand = rands[count]
            ind = sum(apr < rand for apr in accumulate(prob))
            out[(nx, ny)] = ind
            dpt.A = tensordot(dpt.A, loc_projectors[ind], axes=(4, 1))
            Os[nx] = dpt               # updated with the new collapse
            env.update_env_(nx, to='last')
            count += 1

        if opts_svd is None:
            opts_svd = {'D_total': max(vL.get_bond_dimensions())}

        vRnew = mps.zipper(Os, vR, opts_svd=opts_svd)
        if opts_var is None:
            opts_var = {}

        mps.compression_(vRnew, (Os, vR), method='1site', **opts_var)
        vR = vRnew
    return out


def _clear_operator_input(op, sites):
    op_dict = op.copy() if isinstance(op, dict) else {site: op for site in sites}
    for k, v in op_dict.items():
        if isinstance(v, dict):
            op_dict[k] = {(i,): vi for i, vi in v.items()}
        elif isinstance(v, dict):
            op_dict[k] = {(): v}
        else: # is iterable
            op_dict[k] = {(i,): vi for i, vi in enumerate(v)}
    return op_dict


def measure_2site(state, CTMenv, op1, op2, opts_svd, opts_var=None):
    """
    Calculate all 2-point correlations <o1 o2> in a finite peps.

    Takes CTM emvironments and operators.

    o1 and o2 are given as dict[tuple[int, int], dict[int, operators]],
    mapping sites with list of operators at each site.
    """
    out = {}
    if opts_var is None:
        opts_var =  {'max_sweeps': 2}

    Nx, Ny = state.Nx, state.Ny
    sites = [(nx, ny) for ny in range(Ny-1, -1, -1) for nx in range(Nx)]
    op1dict = _clear_operator_input(op1, sites)
    op2dict = _clear_operator_input(op2, sites)

    for nx1, ny1 in sites:
        for nz1, o1 in op1dict[nx1, ny1].items():
            vR = CTMenv.env2mps(index=ny1, index_type='r')
            vL = CTMenv.env2mps(index=ny1, index_type='l')
            Os = transfer_mpo(state, index=ny1, index_type='column')
            env = mps.Env(vL, Os, vR).setup_(to='first').setup_(to='last')
            norm_env = env.measure(bd=(-1, 0))

            if ny1 > 0:
                vRnext = mps.zipper(Os, vR, opts_svd=opts_svd)
                mps.compression_(vRnext, (Os, vR), method='1site', normalize=False, **opts_var)

            # first calculate on-site correlations
            Osnx1A = Os[nx1].A
            for nz2, o2 in op2dict[nx1, ny1].items():
                loc_o = match_ancilla_1s(o1 @ o2, Osnx1A)
                Os[nx1].A = tensordot(Osnx1A, loc_o, axes=(4, 1))
                env.update_env_(nx1, to='last')
                out[(nx1, ny1) + nz1, (nx1, ny1) + nz2] = env.measure(bd=(nx1, nx1+1)) / norm_env

            loc_o1 = match_ancilla_1s(o1, Osnx1A)
            Os[nx1].A = tensordot(Osnx1A, loc_o1, axes=(4, 1))
            env.setup_(to='last')

            if ny1 > 0:
                vRo1next = mps.zipper(Os, vR, opts_svd=opts_svd)
                mps.compression_(vRo1next, (Os, vR), method='1site', normalize=False, **opts_var)

            # calculate correlations along the row
            for nx2 in range(nx1 + 1, Nx):
                Osnx2A = Os[nx2].A
                for nz2, o2 in op2dict[nx2, ny1].items():
                    loc_o2 = match_ancilla_1s(o2, Osnx2A)
                    Os[nx2].A = tensordot(Osnx2A, loc_o2, axes=(4, 1))
                    env.update_env_(nx2, to='first')
                    out[(nx1, ny1) + nz1, (nx2, ny1) + nz2] = env.measure(bd=(nx2-1, nx2)) / norm_env

            # and all subsequent rows
            for ny2 in range(ny1-1, -1, -1):
                vR = vRnext
                vRo1 = vRo1next
                vL = CTMenv.env2mps(index=ny2, index_type='l')
                Os = transfer_mpo(state, index=ny2, index_type='column')
                env = mps.Env(vL, Os, vR).setup_(to='first')
                norm_env = env.measure(bd=(-1, 0))

                if ny2 > 0:
                    vRnext = mps.zipper(Os, vR, opts_svd=opts_svd)
                    mps.compression_(vRnext, (Os, vR), method='1site', normalize=False, **opts_var)
                    vRo1next = mps.zipper(Os, vRo1, opts_svd=opts_svd)
                    mps.compression_(vRo1next, (Os, vRo1), method='1site', normalize=False, **opts_var)

                env = mps.Env(vL, Os, vRo1).setup_(to='first').setup_(to='last')
                for nx2 in range(state.Nx):
                    Osnx2A = Os[nx2].A
                    for nz2, o2 in op2dict[nx2, ny2].items():
                        loc_o2 = match_ancilla_1s(o2, Osnx2A)
                        Os[nx2].A = tensordot(Osnx2A, loc_o2, axes=(4, 1))
                        env.update_env_(nx2, to='first')
                        out[(nx1, ny1) + nz1, (nx2, ny2) + nz2] = env.measure(bd=(nx2-1, nx2)) / norm_env
    return out


def measure_1site(state, CTMenv, op):
    """
    Calculate all 1-point expectation values <o> in a finite peps.

    Takes CTM emvironments and operators.

    o1 are given as dict[tuple[int, int], dict[int, operators]],
    mapping sites with list of operators at each site.
    """
    out = {}

    Nx, Ny = state.Nx, state.Ny
    sites = [(nx, ny) for ny in range(Ny-1, -1, -1) for nx in range(Nx)]
    opdict = _clear_operator_input(op, sites)

    for ny in range(Ny-1, -1, -1):
        vR = CTMenv.env2mps(index=ny, index_type='r')
        vL = CTMenv.env2mps(index=ny, index_type='l')
        Os = transfer_mpo(state, index=ny, index_type='column')
        env = mps.Env(vL, Os, vR).setup_(to='first').setup_(to='last')
        norm_env = env.measure()
        for nx in range(Nx):
            if (nx, ny) in opdict:
                Osnx1A = Os[nx].A
                for nz, o in opdict[nx, ny].items():
                    loc_o = match_ancilla_1s(o, Osnx1A)
                    Os[nx].A = tensordot(Osnx1A, loc_o, axes=(4, 1))
                    env.update_env_(nx, to='first')
                    out[(nx, ny) + nz] = env.measure(bd=(nx-1, nx)) / norm_env
    return out
