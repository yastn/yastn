from ... import mps
from .._auxiliary import transfer_mpo
from .... import YastnError, ones, Leg, eye


class EnvBoundaryMps:
    r""" Geometric information about the lattice provided to ctm tensors """
    def __init__(self, psi, opts_svd, setup='lr', opts_var=None):
        self.psi = psi
        self._env = {}

        li, ri = 0, psi.Ny-1
        ti, bi = 0, psi.Nx-1

        if 'l' in setup or 'r' in setup:
            tmpo = transfer_mpo(psi, index=ri, index_type='column')
            self._env['r', ri] = identity_tm_boundary(tmpo)
            tmpo = transfer_mpo(psi, index=li, index_type='column').T.conj()
            self._env['l', li] = identity_tm_boundary(tmpo)
        if 'b' in setup or 't' in setup:
            tmpo = transfer_mpo(psi, index=ti, index_type='row')
            self._env['t', ti] = identity_tm_boundary(tmpo)
            tmpo = transfer_mpo(psi, index=bi, index_type='row').T.conj()
            self._env['b', bi] = identity_tm_boundary(tmpo)

        self.info = {}

        if opts_var == None:
            opts_var = {'max_sweeps': 2, 'normalize': False,}

        if 'r' in setup:
            for ny in range(ri-1, li-1, -1):
                tmpo = transfer_mpo(psi, index=ny+1, index_type='column')
                phi0 = self._env['r', ny+1]
                self._env['r', ny], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
                mps.compression_(self._env['r', ny], (tmpo, phi0), **opts_var)
                self.info['r', ny] = {'discarded': discarded}

        if 'l' in setup:
            for ny in range(li+1, ri+1):
                tmpo = transfer_mpo(psi, index=ny-1, index_type='column').T.conj()
                phi0 = self._env['l', ny-1]
                self._env['l', ny], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
                mps.compression_(self._env['l', ny], (tmpo, phi0), **opts_var)
                self.info['l', ny] = {'discarded': discarded}

        if 't' in setup:
            for nx in range(ti+1, bi+1):
                tmpo = transfer_mpo(psi, index=nx-1, index_type='row')
                phi0 = self._env['t', nx-1]
                self._env['t', nx], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
                mps.compression_(self._env['t', nx], (tmpo, phi0), **opts_var)
                self.info['t', nx] = {'discarded': discarded}

        if 'b' in setup:
            for nx in range(bi-1, ti-1, -1):
                tmpo = transfer_mpo(psi, index=nx+1, index_type='row').T.conj()
                phi0 = self._env['b', nx+1]
                self._env['b', nx], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
                mps.compression_(self._env['b', nx], (tmpo, phi0), **opts_var)
                self.info['b', nx] = {'discarded': discarded}


    def env2mps(self, index, index_type):
        return self._env[index_type, index]


def identity_tm_boundary(tmpo):
    """
    For transfer matrix MPO build of DoublePepsTensors,
    create MPS that contracts each DoublePepsTensor from the right.
    """
    phi = mps.Mps(N=tmpo.N)
    config = tmpo.config
    for n in phi.sweep(to='last'):
        legf = tmpo[n].get_legs(axes=3).conj()
        tmp = eye(config, legs=legf.unfuse_leg(), isdiag=False).fuse_legs(axes=[(0, 1)])
        phi[n] = tmp.add_leg(0, s=-1).add_leg(2, s=1)
    return phi
