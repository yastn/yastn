from __future__ import annotations

from .._doublePepsTensor import DoublePepsTensor
from .._geometry import Site
from ... import mps
from ....tensor import YastnError
from ....operators import sign_canonical_order

def contract_window(bra, tms, ket, i0, i1, opts_svd, opts_var):
    """ Helper funcion performing mps contraction of < mps0 | mpo mpo ... | mps1 >. """
    vec = ket
    for ny in range(i0, i1):
        vec_next = mps.zipper(tms[ny], vec, opts_svd=opts_svd, normalize=False)
        mps.compression_(vec_next, (tms[ny], vec), method='1site', normalize=False, **opts_var)
        vec = vec_next

    return mps.vdot(bra, tms[i1], vec)

class EnvExciSMA:
    """ EnvWindowSMA class for expectation values within PEPS with CTM boundary. """

    # def __init__(self, env_ctm, *operators, exci_bra=None, exci_ket=None, sites_op=None, sites_ten=None):
    #     sites = sites_op + sites_ten
    #     xrange = (min(site[0] for site in sites), max(site[0] for site in sites) + 1)
    #     yrange = (min(site[1] for site in sites), max(site[1] for site in sites) + 1)
    #     self.site = sites
    #     self.sites_op = sites_op
    #     self.sites_ten = sites_ten
    def __init__(self, env_ctm, xrange, yrange):
        self.psi = env_ctm.psi
        self.env_ctm = env_ctm
        self.xrange = xrange
        self.yrange = yrange

        self.Nx = self.xrange[1] - self.xrange[0]
        self.Ny = self.yrange[1] - self.yrange[0]
        self.offset = 1  #  for mpo tensor position; corresponds to extra CTM boundary tensor

        if env_ctm.nn_site((xrange[0], yrange[0]), (0, 0)) is None or \
           env_ctm.nn_site((xrange[1] - 1, yrange[1] - 1), (0, 0)) is None:
           raise YastnError(f"Window range {xrange=}, {yrange=} does not fit within the lattice.")

    def sites(self):
        return [Site(nx, ny) for ny in range(*self.yrange) for nx in range(*self.xrange)]

    def __getitem__(self, ind) -> mps.MpsMpoOBC:
        """
        Boundary MPS build of CTM tensors, or a transfer matrix MPO.

        CTM corner and edge tensors are included at the ends of MPS and MPO, respectively.
        Leg convention is consistent with mps.vdot(b.conj(), h, t) and mps.vdot(r.conj(), v, l).

        Parameters
        ----------
        n: int
            row/column (depending on dirn) index of the MPO transfer matrix.
            Boundary MPSs match MPO transfer matrix of a given index.
            Indexing is consistent with PEPS indexing.

        dirn: str
            'h', 't', and 'b' refer to a horizontal direction (n specifies a row)
            'h' is a horizontal/row MPO transfer matrix; 't' and 'b' are top and bottom boundary MPSs.
            'v', 'l', and 'r' refer to a vertical direction (n specifies a column).
            'v' is a vertical/column MPO transfer matrix; 'r' and 'l' are right and left boundary MPSs.
        """
        if len(ind) == 2:
            n, dirn = ind
            exci_bra = exci_ket = site_bra = site_ket = None
        elif len(ind) == 6:
            n, dirn, exci_bra, exci_ket, site_bra, site_ket = ind
        else:
            raise ValueError(f"Unexpected number of indices in __getitem__: {len(ind)}")

        if dirn in 'rvl' and not self.yrange[0] <= n < self.yrange[1]:
            raise YastnError(f"{n=} not within {self.yrange=}")

        if dirn == 'r':
            psi = mps.Mps(self.Nx + 2)
            for ind, nx in enumerate(range(*self.xrange), start=1):
                psi[ind] = self.env_ctm[nx, n].r
            psi[0] = self.env_ctm[self.xrange[0], n].tr.add_leg(axis=0)
            psi[self.Nx + 1] = self.env_ctm[self.xrange[1] - 1, n].br.add_leg(axis=2)
            return psi

        if dirn == 'v':
            op = mps.Mpo(self.Nx + 2)
            for ind, nx in enumerate(range(*self.xrange), start=1):
                # op.A[ind] = self.psi[nx, n].transpose(axes=(0, 3, 2, 1))
                new_bra = exci_bra if (nx, n) == site_bra else self.psi[nx, n].bra
                new_ket = exci_ket if (nx, n) == site_ket else self.psi[nx, n].ket
                psi_new = DoublePepsTensor(bra=new_bra, ket=new_ket)
                op.A[ind] = psi_new.transpose(axes=(0, 3, 2, 1))
            op.A[0] = self.env_ctm[self.xrange[0], n].t.add_leg(axis=0).transpose(axes=(0, 3, 2, 1))
            op.A[self.Nx + 1] = self.env_ctm[self.xrange[1] - 1, n].b.add_leg(axis=3).transpose(axes=(1, 0, 3, 2))
            return op

        if dirn == 'l':
            psi = mps.Mps(self.Nx + 2)
            for ind, nx in enumerate(range(*self.xrange), start=1):
                psi[ind] = self.env_ctm[nx, n].l.transpose(axes=(2, 1, 0))
            psi[0] = self.env_ctm[self.xrange[0], n].tl.add_leg(axis=2).transpose(axes=(2, 1, 0))
            psi[self.Nx + 1] = self.env_ctm[self.xrange[1] - 1, n].bl.add_leg(axis=0).transpose(axes=(2, 1, 0))
            return psi

        if dirn in 'thb' and not self.xrange[0] <= n < self.xrange[1]:
            raise YastnError(f"{n=} not within {self.xrange=}")

        if dirn == 't':
            psi = mps.Mps(self.Ny + 2)
            for ind, ny in enumerate(range(*self.yrange), start=1):
                psi[ind] = self.env_ctm[n, ny].t
            psi[0] = self.env_ctm[n, self.yrange[0]].tl.add_leg(axis=0)
            psi[self.Ny + 1] = self.env_ctm[n, self.yrange[1] - 1].tr.add_leg(axis=2)
            return psi

        if dirn == 'h':
            op = mps.Mpo(self.Ny + 2)
            for ind, ny in enumerate(range(*self.yrange), start=1):
                # op.A[ind] = self.psi[n, ny].transpose(axes=(1, 2, 3, 0))
                new_bra = exci_bra if (n, ny) == site_bra else self.psi[n, ny].bra
                new_ket = exci_ket if (n, ny) == site_ket else self.psi[n, ny].ket
                new_psi = DoublePepsTensor(bra=new_bra, ket=new_ket)
                op.A[ind] = new_psi.transpose(axes=(1, 2, 3, 0))
            op.A[0] = self.env_ctm[n, self.yrange[0]].l.add_leg(axis=0)
            op.A[self.Ny + 1] = self.env_ctm[n, self.yrange[1] - 1].r.add_leg(axis=3).transpose(axes=(1, 2, 3, 0))
            return op

        if dirn == 'b':
            psi = mps.Mps(self.Ny + 2)
            for ind, ny in enumerate(range(*self.yrange), start=1):
                psi[ind] = self.env_ctm[n, ny].b.transpose(axes=(2, 1, 0))
            psi[0] = self.env_ctm[n, self.yrange[0]].bl.add_leg(axis=2).transpose(axes=(2, 1, 0))
            psi[self.Ny + 1] = self.env_ctm[n, self.yrange[1] - 1].br.add_leg(axis=0).transpose(axes=(2, 1, 0))
            return psi

        raise YastnError(f"{dirn=} not recognized. Should be 't', 'h' 'b', 'r', 'v', or 'l'.")


    def measure_exci(self, *operators, exci_bra=None, exci_ket=None, site_bra=None, site_ket=None, sites_op=None, dirn='tb', opts_svd=None, opts_var=None):

        sites = [site_bra, site_ket] + sites_op

        if sites is None or len(operators) != len(sites_op):
            raise YastnError("Number of operators plus excited tensors and sites should match.")

        sign = sign_canonical_order(*operators, sites=sites_op, f_ordered=self.psi.f_ordered)
        ops = {}
        for n, op in zip(sites_op, operators):
            ops[n] = ops[n] @ op if n in ops else op

        if opts_var is None:
            opts_var = {'max_sweeps': 2}
        if opts_svd is None:
            rr = self.yrange if dirn == 'lr' else self.xrange
            D_total = max(max(self[i, d].get_bond_dimensions()) for i in range(*rr) for d in dirn)
            opts_svd = {'D_total': D_total}

        if dirn == 'lr':
            i0, i1 = self.yrange[0], self.yrange[1] - 1
            bra = self[i1, 'r'].conj()
            # tms = {ny: self[ny, 'v'] for ny in range(*self.yrange)}
            tms = {}
            for ny in range(*self.yrange):
                t_bra = exci_bra if ny == site_bra[1] else None
                t_ket = exci_ket if ny == site_ket[1] else None
                tms[ny] = self[ny, 'v', t_bra, t_ket, site_bra, site_ket]
            ket = self[i0, 'l']
            dx = self.xrange[0] - self.offset
            tens = {(nx, ny): tm[nx - dx] for ny, tm in tms.items() for nx in range(*self.xrange)}
        else:
            i0, i1 = self.xrange[0], self.xrange[1] - 1
            bra = self[i1, 'b'].conj()
            # tms = {nx: self[nx, 'h'] for nx in range(*self.xrange)}
            tms = {}
            for nx in range(*self.xrange):
                t_bra = exci_bra if nx == site_bra[0] else None
                t_ket = exci_ket if nx == site_ket[0] else None
                tms[nx] = self[nx, 'h', t_bra, t_ket, site_bra, site_ket]
            ket = self[i0, 't']
            dy = self.yrange[0] - self.offset
            tens = {(nx, ny): tm[ny - dy] for nx, tm in tms.items() for ny in range(*self.yrange)}

        # val_no = contract_window(bra, tms, ket, i0, i1, opts_svd, opts_var)

        nx0, ny0 = self.xrange[0], self.yrange[0]
        for (nx, ny), op in ops.items():
            tens[nx, ny].set_operator_(op)
            tens[nx, ny].add_charge_swaps_(op.n, axes=('b0' if nx == nx0 else 'k1'))
            for ii in range(nx0 + 1, nx):
                tens[ii, ny].add_charge_swaps_(op.n, axes=['k1', 'k4', 'b3'])
            if nx > nx0:
                tens[nx0, ny].add_charge_swaps_(op.n, axes=['b0', 'k4', 'b3'])
            for jj in range(ny0, ny):
                tens[nx0, jj].add_charge_swaps_(op.n, axes=['b0', 'k2', 'k4'])

        val_op = contract_window(bra, tms, ket, i0, i1, opts_svd, opts_var)
        # return sign * val_op / val_no
        return sign * val_op


    def measure_exci_norm_tl(self, exci_bra=None, exci_ket=None, opts_svd=None, opts_var=None):
        if opts_var is None:
            opts_var = {'max_sweeps': 2}
        if opts_svd is None:
            D_total = max(max(self[nx, dirn].get_bond_dimensions()) for nx in range(*self.xrange) for dirn in 'tb')
            opts_svd = {'D_total': D_total}

        sites = self.sites()
        out = {}
        
        nx0 = sites[0][0]
        vecs = {nx0: self[nx0, 't']}

        # for nx in range(self.xrange[0], self.xrange[1] - 1):
        #     t_bra = exci_bra if nx == nx0 else None
        #     t_ket = None
        #     tm = self[nx, 'h', t_bra, t_ket, (nx0, ny0), None]            
        #     vecs[nx + 1] = mps.zipper(tm, vecs[nx], opts_svd=opts_svd)
        #     mps.compression_(vecs[nx + 1], (tm, vecs[nx]), method='1site', normalize=False, **opts_var)

        for iy0, ny0 in enumerate(range(*self.yrange), start=1):
            if iy0 > 1:
                break
            # t_bra = exci_bra
            vecc, tm, vec = self[nx0, 'b'].conj(), self[nx0, 'h', exci_bra, None, (nx0, ny0), None] , vecs[nx0]
            # vecc, tm, vec = self[nx0, 'b'].conj(), self[nx0, 'h'] , vecs[nx0]
            
            env = mps.Env(vecc, [tm, vec]).setup_(to='first').setup_(to='last')
            # calculate onsite correlations
            ket0 = tm[iy0].ket
            # tm[iy0] = DoublePepsTensor(bra=tm[iy0].bra, ket=exci_ket[nx0, ny0]).transpose(axes=(1, 2, 3, 0))
            tm[iy0] = DoublePepsTensor(bra=exci_bra, ket=exci_ket[nx0, ny0]).transpose(axes=(1, 2, 3, 0))

            env.update_env_(iy0, to='first')
            out[(nx0, ny0), (nx0, ny0)] = env.measure(bd=(iy0-1, iy0))
            
            tm[iy0] = DoublePepsTensor(bra=exci_bra, ket=ket0).transpose(axes=(1, 2, 3, 0))
            # # env.update_env_(ny0, to='first')
            env.setup_(to='last')

            if nx0 < self.xrange[1] - 1:
                vec_o0_next = mps.zipper(tm, vec, opts_svd=opts_svd, normalize=False)
                mps.compression_(vec_o0_next, (tm, vec), method='1site', normalize=False, **opts_var)

            for iy1, ny1 in enumerate(range(ny0 + 1, self.yrange[1]), start=ny0 - self.yrange[0] + 2):
                ket0 = tm[iy1].ket
                tm[iy1] = DoublePepsTensor(bra=tm[iy1].bra, ket=exci_ket[nx0, ny1]).transpose(axes=(1, 2, 3, 0))
                env.update_env_(iy1, to='first')
                out[(nx0, ny0), (nx0, ny1)] = env.measure(bd=(iy1-1, iy1))
                tm[iy1] = DoublePepsTensor(bra=tm[iy1].bra, ket=ket0).transpose(axes=(1, 2, 3, 0))

            # subsequent rows
            for nx1 in range(self.xrange[0]+1, self.xrange[1]):
                vecc, tm, vec_o0 = self[nx1, 'b'].conj(), self[nx1, 'h'], vec_o0_next

                if nx1 < self.xrange[1] - 1:
                    vec_o0_next = mps.zipper(tm, vec_o0, opts_svd=opts_svd, normalize=False)
                    mps.compression_(vec_o0_next, (tm, vec_o0), method='1site', normalize=False, **opts_var)

                env = mps.Env(vecc, [tm, vec_o0]).setup_(to='last').setup_(to='first')
                for iy1, ny1 in enumerate(range(*self.yrange), start=1):
                    ket0 = tm[iy1].ket
                    tm[iy1] = DoublePepsTensor(bra=tm[iy1].bra, ket=exci_ket[nx1, ny1]).transpose(axes=(1, 2, 3, 0))
                    env.update_env_(iy1, to='first')
                    out[(nx0, ny0), (nx1, ny1)] = env.measure(bd=(iy1-1, iy1))
                    tm[iy1] = DoublePepsTensor(bra=tm[iy1].bra, ket=ket0).transpose(axes=(1, 2, 3, 0))

        return out
