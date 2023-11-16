from ...tn import mps
from yastn.tn.fpeps import Lattice
from ._auxiliary import transfer_mpo
from ... import YastnError, ones, Leg


class MpsEnv(Lattice):
    r""" Geometric information about the lattice provided to ctm tensors """
    def __init__(self, peps, opts_svd, setup='lr', opts_compression=None):
        super().__init__(lattice=peps.lattice, dims=peps.dims, boundary=peps.boundary)

        self._env = {('r', peps.Ny-1):  trivial_mps_boundary(peps, peps.Ny-1, index_type='r'),
                ('l', 0):  trivial_mps_boundary(peps, 0, index_type='l')}

        self.info = {}

        if opts_compression == None:
            opts_compression = {'overlap_tol': 1e-5,
                                'Schmidt_tol': 1e-5,
                                'max_sweeps': 20,
                                'normalize': False,
                                'opts_svd': opts_svd}

        for ny in range(peps.Ny-2, -1, -1):
            psi0 = self._env['r', ny + 1]
            Os = transfer_mpo(peps, index=ny + 1, index_type='column')

            psi, discarded = mps.zipper(Os, psi0, opts_svd, return_discarded=True)
            mps.compression_(psi, (Os, psi0), **opts_compression)
            self._env['r', ny] = psi
            self.info['r', ny] = {'discarded': discarded}

        for ny in range(1, peps.Ny):
            psi0 = self._env['l', ny - 1]
            Os = transfer_mpo(peps, index=ny - 1, index_type='column', rotation=True)
            psi, discarded = mps.zipper(Os, psi0, opts_svd, return_discarded=True)
            mps.compression_(psi, (Os, psi0), **opts_compression)
            self._env['l', ny] = psi
            self.info['l', ny] = {'discarded': discarded}

        for ny in range(peps.Ny):  # reverse left
            psi0 = self._env['l', ny]
            psi = mps.Mps(N=peps.Nx)
            for n in psi.sweep():
                psi[n] = psi0[peps.Nx - n - 1].transpose(axes=(2, 1, 0))
            self._env['l', ny] = psi.conj()


    def env2mps(self, index, index_type):
        return self._env[index_type, index]



def trivial_mps_boundary(peps, index, index_type='r'):
    """ sets trivial identity boundary of finite peps """
    if index_type in 'lr':
        psi = mps.Mps(N=peps.Nx)
    else:
        raise YastnError("only l and r boundaries are supported here. ")

    config = peps[0, 0].config
    tc = (0,) * config.sym.NSYM

    ll = Leg(config, s=-1, t=(tc,), D=(1,))
    lr = ll.conj()

    il = 3 if index_type == 'r' else 1

    for n in psi.sweep():
        B = peps[n, index]
        if B.ndim == 3:
            B = B.unfuse_legs(axes=(0, 1))
        lk = B.get_legs(axes=il)
        pos = n if index_type == 'r' else peps.Nx - n - 1
        psi[pos] =  ones(config=config, legs= [ll, lk.conj(), lk, lr]).fuse_legs(axes=(0, (1, 2), 3))
    return psi