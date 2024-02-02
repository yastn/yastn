from .... import YastnError, tensordot
from ...import mps
from .. import DoublePepsTensor
from ._env_auxlliary import *


class EnvApproximate:
    def __init__(self, psi, which='65', opts_svd=None, opts_var=None):
        if which not in ('43', '43h', '65', '65h', '87', '87h'):
            raise YastnError(f" Type of EnvApprox {which} not recognized.")
        self.psi = psi
        if which in ('43', '43h') :
            self.Nl = 2
            self.Nw = 1
        if which in ('65', '65h') :
            self.Nl = 3
            self.Nw = 2
        if which in ('87', '87h') :
            self.Nl = 4
            self.Nw = 3
        self.include_hairs = 'h' in which
        self.which = which
        if opts_var is None:
            opts_var = {'max_sweeps': 2, 'normalize': False,}
        self.opts_var = opts_var
        self.opts_svd = opts_svd
        self.data = {}
        self.min_spectrum = []


    def bond_metric(self, bd, QA, QB):
        """ Calculates bond metric. """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1

            # (-2,-2)==(-2,-1)==(-2,0)==(-2,1)==(-2,2)==(-2,3)
            #    ||      ||       ||      ||      ||      ||
            # (-1,-2)==(-1,-1)==(-1,0)==(-1,1)==(-1,2)==(-1,3)
            #    ||      ||       ||      ||      ||      ||
            # (0, -2)==(0, -1)=== GA  ++  GB ===(0, 2)==(0, 3)
            #    ||      ||       ||      ||      ||      ||
            # (1, -2)==(1, -1)==(1, 0)==(1, 1)==(1, 2)==(1, 3)
            #    ||      ||       ||      ||      ||      ||
            # (2, -2)==(2, -1)==(2, 0)==(2, 1)==(2, 2)==(2, 3)

            d = {(nx, ny): self.psi.nn_site(bd.site0, d=(nx, ny))
                 for nx in range(-self.Nw, self.Nw+1)
                 for ny in range(-self.Nl+1, self.Nl+1)}
            [d.pop(k) for k in [(0, 0), (0, 1)]]
            if self.include_hairs:
                for ny in range(-self.Nl+1, self.Nl+1):
                    d[-self.Nw - 1, ny] = self.psi.nn_site(bd.site0, d=(-self.Nw - 1, ny))
                    d[ self.Nw + 1, ny] = self.psi.nn_site(bd.site0, d=( self.Nw + 1, ny))
                for nx in range(-self.Nw, self.Nw+1):
                    d[nx, -self.Nl] = self.psi.nn_site(bd.site0, d=(nx, -self.Nl))
                    d[nx,  self.Nl+1] = self.psi.nn_site(bd.site0, d=(nx, self.Nl+1))
            tensors_from_psi(d, self.psi)
            d[0, 0] = QA
            d[0, 1] = QB

            tmpo = self.transfer_mpo(d, n=-self.Nw, dirn='h')
            if self.include_hairs:
                phit0 = mps.product_mps([hair_t(d[-self.Nw - 1, ny]).fuse_legs(axes=[(0, 1)]).conj() for ny in range(-self.Nl+1, self.Nl+1)])
            else:
                phit0 = identity_tm_boundary(tmpo)
            for nx in range(-self.Nw, 0):
                phit = mps.zipper(tmpo, phit0, opts_svd=self.opts_svd)
                mps.compression_(phit, (tmpo, phit0), **self.opts_var)
                phit0 = phit
                if nx < -1:
                    tmpo = self.transfer_mpo(d, n=nx+1, dirn='h')

            tmpo = self.transfer_mpo(d, n=self.Nw, dirn='h').T.conj()
            if self.include_hairs:
                phib0 = mps.product_mps([hair_b(d[self.Nw + 1, ny]).fuse_legs(axes=[(0, 1)]) for ny in range(-self.Nl+1, self.Nl+1)])
            else:
                phib0 = identity_tm_boundary(tmpo)
            for nx in range(self.Nw, 0, -1):
                phib = mps.zipper(tmpo, phib0, opts_svd=self.opts_svd)
                mps.compression_(phib, (tmpo, phib0), **self.opts_var)
                phib0 = phib
                if nx > 1:
                    tmpo = self.transfer_mpo(d, n=nx-1, dirn='h').T.conj()

            tmpo = self.transfer_mpo(d, n=0, dirn='h')
            g = g_from_env3(phib, tmpo, phit)

        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            #   (-2,-2)==(-2,-1)==(-2,0)==(-2,1)==(-2,2)
            #      ||       ||      ||      ||      ||
            #   (-1,-2)==(-1,-1)==(-1,0)==(-1,1)==(-1,2)
            #      ||       ||      ||      ||      ||
            #   (0, -2)==(0, -1)=== GA ===(0, 1)==(0, 2)
            #      ||       ||      ++      ||      ||
            #   (1, -2)==(1, -1)=== GB ===(1, 1)==(1, 2)
            #      ||       ||      ||      ||      ||
            #   (2, -2)==(2, -1)==(2, 0)==(2, 1)==(2, 2)
            #      ||       ||      ||      ||      ||
            #   (3, -2)==(3, -1)==(3, 0)==(3, 1)==(3, 2)

            d = {(nx, ny): self.psi.nn_site(bd.site0, d=(nx, ny)) for nx in range(-self.Nl+1, self.Nl+1) for ny in range(-self.Nw, self.Nw+1)}
            [d.pop(k) for k in [(0, 0), (1, 0)]]
            if self.include_hairs:
                for nx in range(-self.Nl+1, self.Nl+1):
                    d[nx, -self.Nw - 1] = self.psi.nn_site(bd.site0, d=(nx, -self.Nw - 1))
                    d[nx,  self.Nw + 1] = self.psi.nn_site(bd.site0, d=(nx,  self.Nw + 1))
                for ny in range(-self.Nw, self.Nw+1):
                    d[-self.Nl,  ny] = self.psi.nn_site(bd.site0, d=(-self.Nl,  ny))
                    d[self.Nl+1, ny] = self.psi.nn_site(bd.site0, d=(self.Nl+1, ny))
            tensors_from_psi(d, self.psi)
            d[0, 0] = QA
            d[1, 0] = QB

            tmpo = self.transfer_mpo(d, n=-self.Nw, dirn='v')
            if self.include_hairs:
                phir0 = mps.product_mps([hair_r(d[nx, self.Nw + 1]).fuse_legs(axes=[(0, 1)]).conj() for nx in range(-self.Nl+1, self.Nl+1)])
            else:
                phir0 = identity_tm_boundary(tmpo)
            for ny in range(self.Nw, 0, -1):
                phir = mps.zipper(tmpo, phir0, opts_svd=self.opts_svd)
                mps.compression_(phir, (tmpo, phir0), **self.opts_var)
                phir0 = phir
                if ny > 1:
                    tmpo = self.transfer_mpo(d, n=ny-1, dirn='v')

            tmpo = self.transfer_mpo(d, n=self.Nw, dirn='v').T.conj()
            if self.include_hairs:
                phil0 = mps.product_mps([hair_l(d[nx, -self.Nw - 1]).fuse_legs(axes=[(0, 1)]) for nx in range(-self.Nl+1, self.Nl+1)])
            else:
                phil0 = identity_tm_boundary(tmpo)
            for ny in range(0, self.Nw):
                phil = mps.zipper(tmpo, phil0, opts_svd=self.opts_svd)
                mps.compression_(phil, (tmpo, phil0), **self.opts_var)
                phil0 = phil
                if ny < -1:
                    tmpo = self.transfer_mpo(d, n=ny+1, dirn='v').T.conj()

            tmpo = self.transfer_mpo(d, n=0, dirn='v')
            g = g_from_env3(phil, tmpo, phir)

        # make hermitian and fix negative elements.
        g = (g + g.T.conj()) / 2
        S, U = g.eigh(axes=(0, 1))
        smin, smax = min(S._data), max(S._data)
        self.min_spectrum.append(smin / smax)
        S._data[S._data < abs(smin)] = abs(smin)
        return U @ S @ U.T.conj()


    def transfer_mpo(self, d, n, dirn='v'):
        H = mps.Mpo(N = 2 * self.Nl)
        if dirn == 'h':
            nx, ny = n, -self.Nl + 1
            hl = hair_l(d[nx, ny - 1]) if self.include_hairs else None
            el = edge_l(d[nx, ny], hl=hl).add_leg(s=-1, axis=0)
            H.A[H.first] = el
            for site in H.sweep(to='last', dl=1, df=1):
                ny += 1
                top = d[nx, ny]
                if top.ndim == 3:
                    top = top.unfuse_legs(axes=(0, 1))
                H.A[site] = DoublePepsTensor(top=top, btm=top, transpose=(1, 2, 3, 0))
            ny += 1
            hr = hair_r(d[nx, ny + 1]) if self.include_hairs else None
            er = edge_r(d[nx, ny], hr=hr).add_leg(s=1).transpose(axes=(1, 2, 3, 0))
            H.A[H.last] = er
        else:  # dirn == 'v':
            nx, ny = -self.Nl + 1, n
            ht = hair_t(d[nx - 1, ny]) if self.include_hairs else None
            et = edge_t(d[nx, ny], ht=ht).add_leg(s=-1, axis=0)
            H.A[H.first] = et
            for site in H.sweep(to='last', dl=1, df=1):
                nx += 1
                top = d[nx, ny]
                if top.ndim == 3:
                    top = top.unfuse_legs(axes=(0, 1))
                H.A[site] = DoublePepsTensor(top=top, btm=top)
            nx += 1
            hb = hair_b(d[nx + 1, ny]) if self.include_hairs else None
            eb = edge_b(d[nx, ny], hb=hb).add_leg(s=1).transpose(axes=(1, 2, 3, 0))
            H.A[H.last] = eb
        return H


def g_from_env3(bra, tmpo, ket):
    Nl = len(tmpo) // 2
    env = mps.Env3(bra, tmpo, ket)
    for n in range(Nl):
        env.update_env_(n, to='last')
        env.update_env_(2 * Nl - n - 1, to='first')
    g = tensordot(env.F[Nl-1, Nl], env.F[Nl, Nl-1], axes=((0, 2), (2, 0)))
    return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))
