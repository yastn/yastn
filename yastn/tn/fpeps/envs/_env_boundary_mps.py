from ... import mps
from ._env_auxlliary import identity_tm_boundary


class EnvBoundaryMps:
    r""" Geometric information about the lattice provided to ctm tensors """
    def __init__(self, psi, opts_svd, setup='lr', opts_var=None):
        self.psi = psi
        self._env = {}

        li, ri = 0, psi.Ny-1
        ti, bi = 0, psi.Nx-1

        if 'l' in setup or 'r' in setup:
            tmpo = psi.transfer_mpo(n=ri, dirn='v')
            self._env['r', ri] = identity_tm_boundary(tmpo)
            tmpo = psi.transfer_mpo(n=li, dirn='v').H
            self._env['l', li] = identity_tm_boundary(tmpo)
        if 'b' in setup or 't' in setup:
            tmpo = psi.transfer_mpo(n=ti, dirn='h')
            self._env['t', ti] = identity_tm_boundary(tmpo)
            tmpo = psi.transfer_mpo(n=bi, dirn='h').H
            self._env['b', bi] = identity_tm_boundary(tmpo)

        self.info = {}

        if opts_var == None:
            opts_var = {'max_sweeps': 2, 'normalize': False,}

        if 'r' in setup:
            for ny in range(ri-1, li-1, -1):
                tmpo = psi.transfer_mpo(n=ny+1, dirn='v')
                phi0 = self._env['r', ny+1]
                self._env['r', ny], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
                mps.compression_(self._env['r', ny], (tmpo, phi0), **opts_var)
                self.info['r', ny] = {'discarded': discarded}

        if 'l' in setup:
            for ny in range(li+1, ri+1):
                tmpo = psi.transfer_mpo(n=ny-1, dirn='v').H
                phi0 = self._env['l', ny-1]
                self._env['l', ny], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
                mps.compression_(self._env['l', ny], (tmpo, phi0), **opts_var)
                self.info['l', ny] = {'discarded': discarded}

        if 't' in setup:
            for nx in range(ti+1, bi+1):
                tmpo = psi.transfer_mpo(n=nx-1, dirn='h')
                phi0 = self._env['t', nx-1]
                self._env['t', nx], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
                mps.compression_(self._env['t', nx], (tmpo, phi0), **opts_var)
                self.info['t', nx] = {'discarded': discarded}

        if 'b' in setup:
            for nx in range(bi-1, ti-1, -1):
                tmpo = psi.transfer_mpo(n=nx+1, dirn='h').H
                phi0 = self._env['b', nx+1]
                self._env['b', nx], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
                mps.compression_(self._env['b', nx], (tmpo, phi0), **opts_var)
                self.info['b', nx] = {'discarded': discarded}


    def env2mps(self, n, dirn):
        return self._env[dirn, n]
