from .... import YastnError, tensordot, eye
from ...import mps
from .. import DoublePepsTensor
from ._env_auxlliary import *

# horizontal bd; here Nl, Nw = 3, 2
#
# (-2,-2)==(-2,-1)==(-2,0)==(-2,1)==(-2,2)==(-2,3)
#    ||      ||       ||      ||      ||      ||
# (-1,-2)==(-1,-1)==(-1,0)==(-1,1)==(-1,2)==(-1,3)
#    ||      ||       ||      ||      ||      ||
# (0, -2)==(0, -1)=== GA  ++  GB ===(0, 2)==(0, 3)
#    ||      ||       ||      ||      ||      ||
# (1, -2)==(1, -1)==(1, 0)==(1, 1)==(1, 2)==(1, 3)
#    ||      ||       ||      ||      ||      ||
# (2, -2)==(2, -1)==(2, 0)==(2, 1)==(2, 2)==(2, 3)

# vertical bd; here Nl, Nw = 3, 2
#
# (-2,-2)==(-2,-1)==(-2,0)==(-2,1)==(-2,2)
#    ||       ||      ||      ||      ||
# (-1,-2)==(-1,-1)==(-1,0)==(-1,1)==(-1,2)
#    ||       ||      ||      ||      ||
# (0, -2)==(0, -1)=== GA ===(0, 1)==(0, 2)
#    ||       ||      ++      ||      ||
# (1, -2)==(1, -1)=== GB ===(1, 1)==(1, 2)
#    ||       ||      ||      ||      ||
# (2, -2)==(2, -1)==(2, 0)==(2, 1)==(2, 2)
#    ||       ||      ||      ||      ||
# (3, -2)==(3, -1)==(3, 0)==(3, 1)==(3, 2)

_hair_dirn = {'t': hair_t, 'l': hair_l, 'b': hair_b, 'r': hair_r}
#_axis_dirn ={'t': 2, 'l': 3, 'b': 0, 'r': 1}
_axis_dirn = {'t': (1, 0), 'l': (1, 1), 'b': (0, 0), 'r': (0, 1)}

class EnvApproximate:
    def __init__(self, psi, which='65', opts_svd=None, opts_var=None, update_sweeps=None):
        if which not in ('43', '43h', '65', '65h', '87', '87h'):
            raise YastnError(f" Type of EnvApprox {which} not recognized.")
        self._which = which
        self.psi = psi
        if which in ('43', '43h'):
            self.Nl, self.Nw = 2, 1
        if which in ('65', '65h'):
            self.Nl, self.Nw = 3, 2
        if which in ('87', '87h'):
            self.Nl, self.Nw = 4, 3
        self._hairs = 'h' in which
        self.opts_var = {'max_sweeps': 2, } if opts_var is None else opts_var
        self.opts_svd = opts_svd
        self.update_sweeps = update_sweeps
        self._envs = {}

    @property
    def which(self):
        return self._which

    def __getitem__(self, key):
        return self._envs[key]

    def __setitem__(self, key, value):
        self._envs[key] = value

    def bond_metric(self, bd, QA, QB):
        """ Calculates bond metric. """
        dirn = bd.dirn
        dx = (0, 1) if dirn == 'h' else (1, 0)
        assert self.psi.nn_site(bd.site0, dx) == bd.site1

        try:
            if self.update_sweeps:
                self.update_env(bd)
            else:
                self.initialize_env(bd)
        except (KeyError, YastnError):
            self.initialize_env(bd)

        tmpo = self.transfer_mpo(bd, n=0, QA=QA, QB=QB)
        env = mps.Env(self[bd, -1], [tmpo, self[bd, 1]])
        Nl = self.Nl
        for n in range(Nl):
            env.update_env_(n, to='last')
            env.update_env_(2 * Nl - n - 1, to='first')
        g = tensordot(env.F[Nl-1, Nl], env.F[Nl, Nl-1], axes=((0, 2), (2, 0)))
        return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))

    def initialize_env(self, bd):
        if bd.dirn == "h":
            self[bd, self.Nw + 1] = self.boundary_mps(bd, 't')
            for nx in range(-self.Nw, 0):  # [bd, 1]
                tmpo = self.transfer_mpo(bd, n=nx)
                self[bd, -nx] = mps.zipper(tmpo, self[bd, -nx+1], opts_svd=self.opts_svd)
                mps.compression_(self[bd, -nx], (tmpo, self[bd, -nx+1]), **self.opts_var)

            self[bd, -self.Nw - 1] = self.boundary_mps(bd, 'b').conj()
            for nx in range(self.Nw, 0, -1):  # [bd, -1]
                tmpo = self.transfer_mpo(bd, n=nx).H
                self[bd, -nx] = mps.zipper(tmpo, self[bd, -nx-1], opts_svd=self.opts_svd)
                mps.compression_(self[bd, -nx], (tmpo, self[bd, -nx-1]), **self.opts_var)

        else:  # bd.dirn == "v":
            self[bd, self.Nw + 1] = self.boundary_mps(bd, 'r')
            for ny in range(self.Nw, 0, -1):  # [bd, 1]
                tmpo = self.transfer_mpo(bd, n=ny)
                self[bd, ny] = mps.zipper(tmpo, self[bd, ny+1], opts_svd=self.opts_svd)
                mps.compression_(self[bd, ny], (tmpo, self[bd, ny+1]), **self.opts_var)

            self[bd, -self.Nw - 1] = self.boundary_mps(bd, 'l').conj()
            for ny in range(-self.Nw, 0):  # [bd, -1]
                tmpo = self.transfer_mpo(bd, n=ny).H
                self[bd, ny] = mps.zipper(tmpo, self[bd, ny-1], opts_svd=self.opts_svd)
                mps.compression_(self[bd, ny], (tmpo, self[bd, ny-1]), **self.opts_var)

    def update_env(self, bd):
        if bd.dirn == "h":
            self[bd, self.Nw + 1] = self.boundary_mps(bd, 't')
            for nx in range(-self.Nw, 0):  # [bd, 1]
                tmpo = self.transfer_mpo(bd, n=nx)
                mps.compression_(self[bd, -nx], (tmpo, self[bd, -nx+1]), max_sweeps=self.update_sweeps)

            self[bd, -self.Nw - 1] = self.boundary_mps(bd, 'b').conj()
            for nx in range(self.Nw, 0, -1):  # [bd, -1]
                tmpo = self.transfer_mpo(bd, n=nx).H
                mps.compression_(self[bd, -nx], (tmpo, self[bd, -nx-1]), max_sweeps=self.update_sweeps)

        else:  # bd.dirn == "v":
            self[bd, self.Nw + 1] = self.boundary_mps(bd, 'r')
            for ny in range(self.Nw, 0, -1):  # [bd, 1]
                tmpo = self.transfer_mpo(bd, n=ny)
                mps.compression_(self[bd, ny], (tmpo, self[bd, ny+1]), max_sweeps=self.update_sweeps)

            self[bd, -self.Nw - 1] = self.boundary_mps(bd, 'l').conj()
            for ny in range(-self.Nw, 0):  # [bd, -1]
                tmpo = self.transfer_mpo(bd, n=ny).H
                mps.compression_(self[bd, ny], (tmpo, self[bd, ny-1]), max_sweeps=self.update_sweeps)

    def transfer_mpo(self, bd, n, QA=None, QB=None):
        H = mps.Mpo(N = 2 * self.Nl)

        if bd.dirn == "h":
            nx = n
            d = {(nx, ny): self.psi.nn_site(bd.site0, d=(nx, ny))
                for ny in range(-self.Nl + 1 - self._hairs, self.Nl + 1 + self._hairs)}
            tensors_from_psi(d, self.psi)
            if nx == 0:
                d[0, 0], d[0, 1] = QA, QB

            ny = -self.Nl + 1
            hl = hair_l(d[nx, ny - 1]) if self._hairs else None
            H.A[H.first] = edge_l(d[nx, ny], hl=hl).add_leg(s=-1, axis=0)
            for site in H.sweep(to='last', dl=1, df=1):
                ny += 1
                top = d[nx, ny].unfuse_legs(axes=(0, 1)) if d[nx, ny].ndim == 3 else d[nx, ny]
                H.A[site] = DoublePepsTensor(top=top, btm=top, transpose=(1, 2, 3, 0))
            ny += 1
            hr = hair_r(d[nx, ny + 1]) if self._hairs else None
            H.A[H.last] = edge_r(d[nx, ny], hr=hr).add_leg(s=1).transpose(axes=(1, 2, 3, 0))

        else:  # bd.dirn == "v":
            ny = n
            d = {(nx, ny): self.psi.nn_site(bd.site0, d=(nx, ny))
                for nx in range(-self.Nl + 1 - self._hairs, self.Nl + 1 + self._hairs)}
            tensors_from_psi(d, self.psi)
            if ny == 0:
                d[0, 0], d[1, 0] = QA, QB

            nx = -self.Nl + 1
            ht = hair_t(d[nx - 1, ny]) if self._hairs else None
            H.A[H.first] = edge_t(d[nx, ny], ht=ht).add_leg(s=-1, axis=0)
            for site in H.sweep(to='last', dl=1, df=1):
                nx += 1
                top = d[nx, ny].unfuse_legs(axes=(0, 1)) if d[nx, ny].ndim == 3 else d[nx, ny]
                H.A[site] = DoublePepsTensor(top=top, btm=top)
            nx += 1
            hb = hair_b(d[nx + 1, ny]) if self._hairs else None
            H.A[H.last] = edge_b(d[nx, ny], hb=hb).add_leg(s=1).transpose(axes=(1, 2, 3, 0))
        return H

    def boundary_mps(self, bd, dirn='r'):
        if dirn in 'lr':
            ny = self.Nw + 1 if dirn == 'r' else -self.Nw - 1
            nns = [(nx, ny) for nx in range(-self.Nl + 1, self.Nl + 1)]
        else: # dirn in 'tb':
            nx = self.Nw + 1 if dirn == 'b' else -self.Nw - 1
            nns = [(nx, ny) for ny in range(-self.Nl + 1, self.Nl + 1)]
        d = {nxy: self.psi.nn_site(bd.site0, d=nxy) for nxy in nns}
        tensors_from_psi(d, self.psi)
        if self._hairs:
            hair_dirn = _hair_dirn[dirn]
            return mps.product_mps([hair_dirn(d[nxy]).fuse_legs(axes=[(0, 1)]).conj() for nxy in nns])

        vectors, config = [], self.psi.config
        axis = _axis_dirn[dirn]
        for nxy in nns:
            leg = d[nxy].get_legs(axes=axis[0])  # peps tensor are now [t l] [b r] s
            leg = leg.unfuse_leg()  # we need to unfuse
            leg = leg[axis[1]]
            vectors.append(eye(config, legs=[leg, leg.conj()], isdiag=False).fuse_legs(axes=[(0, 1)]))
        return mps.product_mps(vectors)
