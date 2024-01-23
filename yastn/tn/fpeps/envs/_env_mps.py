from ... import mps
from .._auxiliary import transfer_mpo
from .... import YastnError, ones, Leg


class MpsEnv():
    r""" Geometric information about the lattice provided to ctm tensors """
    def __init__(self, psi, opts_svd, setup='lr', opts_var=None):
        self.psi = psi

        self._env = {('r', psi.Ny-1):  trivial_mps_boundary(psi, psi.Ny-1, index_type='r'),
                ('l', 0):  trivial_mps_boundary(psi, 0, index_type='l')}

        self.info = {}

        if opts_var == None:
            opts_var = {'max_sweeps': 2,
                        'normalize': False,}

        for ny in range(psi.Ny-2, -1, -1):
            phi0 = self._env['r', ny + 1]
            Os = transfer_mpo(psi, index=ny + 1, index_type='column')

            phi, discarded = mps.zipper(Os, phi0, opts_svd, return_discarded=True)
            mps.compression_(phi, (Os, phi0), **opts_var)
            self._env['r', ny] = phi
            self.info['r', ny] = {'discarded': discarded}

        for ny in range(1, psi.Ny):
            phi0 = self._env['l', ny - 1]
            Os = transfer_mpo(psi, index=ny - 1, index_type='column', rotation=True)
            phi, discarded = mps.zipper(Os, phi0, opts_svd, return_discarded=True)
            mps.compression_(phi, (Os, phi0), **opts_var)
            self._env['l', ny] = phi
            self.info['l', ny] = {'discarded': discarded}

        for ny in range(psi.Ny):  # reverse left
            phi0 = self._env['l', ny]
            phi = mps.Mps(N=psi.Nx)
            for n in phi.sweep():
                phi[n] = phi0[psi.Nx - n - 1].transpose(axes=(2, 1, 0))
            self._env['l', ny] = phi.conj()


    def env2mps(self, index, index_type):
        return self._env[index_type, index]



def trivial_mps_boundary(psi, index, index_type='r'):
    """ sets trivial identity boundary of finite psi """
    if index_type in 'lr':
        phi = mps.Mps(N=psi.Nx)
    else:
        raise YastnError("only l and r boundaries are supported here. ")

    config = psi[0, 0].config
    tc = (0,) * config.sym.NSYM

    ll = Leg(config, s=-1, t=(tc,), D=(1,))
    lr = ll.conj()

    il = 3 if index_type == 'r' else 1

    for n in phi.sweep():
        B = psi[n, index]
        if B.ndim == 3:
            B = B.unfuse_legs(axes=(0, 1))
        lk = B.get_legs(axes=il)
        pos = n if index_type == 'r' else psi.Nx - n - 1
        phi[pos] =  ones(config=config, legs= [ll, lk.conj(), lk, lr]).fuse_legs(axes=(0, (1, 2), 3))
    return phi