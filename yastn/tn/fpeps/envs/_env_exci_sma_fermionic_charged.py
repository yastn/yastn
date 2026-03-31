from __future__ import annotations

from ... import mps
from .._doublePepsTensor_charged import DoublePepsTensorCharged
from ._env_exci_sma import EnvExciSMA


class EnvExciSMAFermionicCharged(EnvExciSMA):
    """
    Experimental charged-sector fermionic variant of ``EnvExciSMA``.

    This mirrors the neutral fermionic implementation but swaps in
    ``DoublePepsTensorCharged`` whenever bra/ket layers are replaced by charged
    excitation tensors. The charged double-layer wrapper uses an explicit
    fused-layer contraction path, trading speed for safer fermionic sign
    handling.
    """

    @staticmethod
    def _replace_layers(template, bra=None, ket=None):
        return DoublePepsTensorCharged(
            bra=template.bra if bra is None else bra,
            ket=template.ket if ket is None else ket,
            transpose=template._t,
            op=template.op,
            swaps=template.swaps,
        )

    def __getitem__(self, ind):
        if len(ind) == 2:
            n, dirn = ind
            exci_bra = exci_ket = site_bra = site_ket = None
        elif len(ind) == 6:
            n, dirn, exci_bra, exci_ket, site_bra, site_ket = ind
        else:
            return super().__getitem__(ind)

        if dirn == "v":
            op = mps.Mpo(self.Nx + 2)
            for ind, nx in enumerate(range(*self.xrange), start=1):
                base = self.psi[nx, n]
                bra = exci_bra if (nx, n) == site_bra else base.bra
                ket = exci_ket if (nx, n) == site_ket else base.ket
                op.A[ind] = self._replace_layers(base, bra=bra, ket=ket).transpose(
                    axes=(0, 3, 2, 1)
                )
            op.A[0] = self.env_ctm[self.xrange[0], n].t.add_leg(axis=0).transpose(
                axes=(0, 3, 2, 1)
            )
            op.A[self.Nx + 1] = self.env_ctm[self.xrange[1] - 1, n].b.add_leg(
                axis=3
            ).transpose(axes=(1, 0, 3, 2))
            return op

        if dirn == "h":
            op = mps.Mpo(self.Ny + 2)
            for ind, ny in enumerate(range(*self.yrange), start=1):
                base = self.psi[n, ny]
                bra = exci_bra if (n, ny) == site_bra else base.bra
                ket = exci_ket if (n, ny) == site_ket else base.ket
                op.A[ind] = self._replace_layers(base, bra=bra, ket=ket).transpose(
                    axes=(1, 2, 3, 0)
                )
            op.A[0] = self.env_ctm[n, self.yrange[0]].l.add_leg(axis=0)
            op.A[self.Ny + 1] = self.env_ctm[n, self.yrange[1] - 1].r.add_leg(
                axis=3
            ).transpose(axes=(1, 2, 3, 0))
            return op

        if dirn == "hb":
            op = mps.Mpo(self.Ny + 2)
            for ind, ny in enumerate(range(*self.yrange), start=1):
                base = self.psi[n, ny]
                bra = exci_bra if (n, ny) == site_bra else base.bra
                ket = exci_ket if (n, ny) == site_ket else base.ket
                op.A[ind] = self._replace_layers(base, bra=bra, ket=ket).transpose(
                    axes=(1, 0, 3, 2)
                )
            op.A[0] = self.env_ctm[n, self.yrange[0]].l.add_leg(axis=0).transpose(
                axes=(0, 3, 2, 1)
            )
            op.A[self.Ny + 1] = self.env_ctm[n, self.yrange[1] - 1].r.add_leg(
                axis=3
            ).transpose(axes=(1, 0, 3, 2))
            return op

        return super().__getitem__(ind)

    def measure_exci_ops(self, *operators, exci_psi=None, sites_op=None, opts_svd=None, opts_var=None):
        if opts_var is None:
            opts_var = {"max_sweeps": 2}
        if opts_svd is None:
            D_total = max(
                max(self[nx, dirn].get_bond_dimensions())
                for nx in range(*self.xrange)
                for dirn in "tb"
            )
            opts_svd = {"D_total": D_total}

        ops = {}
        for n, op in zip(sites_op, operators):
            ops[n] = ops[n] @ op if n in ops else op

        sites = self.sites()
        nx0 = sites[0][0]
        nx1 = sites[-1][0]
        ny0 = sites[0][1]
        ny1 = sites[-1][1]
        out = {tuple(site): None for site in sites}
        nxc, nyc = (nx0 + nx1) // 2, (ny0 + ny1) // 2

        tms = {}
        tmbs = {}
        for nx in range(*self.xrange):
            tms[nx] = self[nx, "h"]
            tmbs[nx] = self[nx, "hb"]
        dy = self.yrange[0] - self.offset
        tens = {(nx, ny): tm[ny - dy] for nx, tm in tms.items() for ny in range(*self.yrange)}
        tenbs = {(nx, ny): tmb[ny - dy] for nx, tmb in tmbs.items() for ny in range(*self.yrange)}

        nx0, ny0 = self.xrange[0], self.yrange[0]
        for (nx, ny), op in ops.items():
            tens[nx, ny].set_operator_(op)
            tens[nx, ny].add_charge_swaps_(op.n, axes=("b0" if nx == nx0 else "k1"))
            for ii in range(nx0 + 1, nx):
                tens[ii, ny].add_charge_swaps_(op.n, axes=["k1", "k4", "b3"])
            if nx > nx0:
                tens[nx0, ny].add_charge_swaps_(op.n, axes=["b0", "k4", "b3"])
            for jj in range(ny0, ny):
                tens[nx0, jj].add_charge_swaps_(op.n, axes=["b0", "k2", "k4"])

            tenbs[nx, ny].set_operator_(op)
            tenbs[nx, ny].add_charge_swaps_(op.n, axes=("b0" if nx == nx0 else "k1"))
            for ii in range(nx0 + 1, nx):
                tenbs[ii, ny].add_charge_swaps_(op.n, axes=["k1", "k4", "b3"])
            if nx > nx0:
                tenbs[nx0, ny].add_charge_swaps_(op.n, axes=["b0", "k4", "b3"])
            for jj in range(ny0, ny):
                tenbs[nx0, jj].add_charge_swaps_(op.n, axes=["b0", "k2", "k4"])

        nx_op_min = min([site[0] for site in sites_op])
        nx_op_max = max([site[0] for site in sites_op])

        vecs = {nx: self[nx, "t"] for nx in range(self.xrange[0], nx_op_min + 1)}
        veccs = {nx: self[nx, "b"] for nx in range(nx_op_max, self.xrange[1])}

        for nx in range(nx_op_min + 1, self.xrange[1]):
            vecs[nx] = mps.zipper(tms[nx - 1], vecs[nx - 1], opts_svd=opts_svd, normalize=False)
            mps.compression_(
                vecs[nx],
                (tms[nx - 1], vecs[nx - 1]),
                method="1site",
                normalize=False,
                **opts_var,
            )

        for nx in range(nx_op_max - 1, self.xrange[0] - 1, -1):
            veccs[nx] = mps.zipper(tmbs[nx + 1], veccs[nx + 1], opts_svd=opts_svd, normalize=False)
            mps.compression_(
                veccs[nx],
                (tmbs[nx + 1], veccs[nx + 1]),
                method="1site",
                normalize=False,
                **opts_var,
            )

        out = {}
        for ix0, nx0 in enumerate(range(*self.xrange), start=1):
            for iy0, ny0 in enumerate(range(*self.yrange), start=1):
                vecc, tm, vec = veccs[nx0], tms[nx0], vecs[nx0]

                bra0 = tm[iy0].bra
                tm[iy0] = self._replace_layers(tm[iy0], bra=exci_psi[nx0, ny0])

                env = mps.Env(vecc.conj(), [tm, vec]).setup_(to="first").setup_(to="last")
                ket0 = tm[iy0].ket
                tm[iy0] = self._replace_layers(tm[iy0], ket=exci_psi[nx0, ny0])
                env.update_env_(iy0, to="first")
                out[(nx0, ny0), (nx0, ny0)] = env.measure(bd=(iy0 - 1, iy0))
                tm[iy0] = self._replace_layers(tm[iy0], ket=ket0)

                if nx0 < self.xrange[1] - 1:
                    vec_o0_next = mps.zipper(tm, vec, opts_svd=opts_svd, normalize=False)
                    mps.compression_(
                        vec_o0_next, (tm, vec), method="1site", normalize=False, **opts_var
                    )

                for iy1, ny1 in enumerate(range(ny0 + 1, self.yrange[1]), start=ny0 - self.yrange[0] + 2):
                    ket0 = tm[iy1].ket
                    tm[iy1] = self._replace_layers(tm[iy1], ket=exci_psi[nx0, ny1])
                    env.update_env_(iy1, to="first")
                    out[(nx0, ny0), (nx0, ny1)] = env.measure(bd=(iy1 - 1, iy1))
                    tm[iy1] = self._replace_layers(tm[iy1], ket=ket0)

                for nx1 in range(nx0 + 1, self.xrange[1]):
                    vecc, tm, vec_o0 = veccs[nx1], tms[nx1], vec_o0_next

                    if nx1 < self.xrange[1] - 1:
                        vec_o0_next = mps.zipper(tm, vec_o0, opts_svd=opts_svd, normalize=False)
                        mps.compression_(
                            vec_o0_next,
                            (tm, vec_o0),
                            method="1site",
                            normalize=False,
                            **opts_var,
                        )

                    env = mps.Env(vecc.conj(), [tm, vec_o0]).setup_(to="last").setup_(to="first")
                    for iy1, ny1 in enumerate(range(*self.yrange), start=1):
                        ket0 = tm[iy1].ket
                        tm[iy1] = self._replace_layers(tm[iy1], ket=exci_psi[nx1, ny1])
                        env.update_env_(iy1, to="first")
                        out[(nx0, ny0), (nx1, ny1)] = env.measure(bd=(iy1 - 1, iy1))
                        tm[iy1] = self._replace_layers(tm[iy1], ket=ket0)
                tms[nx0][iy0] = self._replace_layers(tms[nx0][iy0], bra=bra0)
        return out

    def measure_exci_norm(self, exci_bra=None, exci_ket=None, opts_svd=None, opts_var=None):
        if opts_var is None:
            opts_var = {"max_sweeps": 2}
        if opts_svd is None:
            D_total = max(
                max(self[nx, dirn].get_bond_dimensions())
                for nx in range(*self.xrange)
                for dirn in "tb"
            )
            opts_svd = {"D_total": D_total}

        sites = self.sites()
        nx0 = sites[0][0]
        nx1 = sites[-1][0]
        ny0 = sites[0][1]
        ny1 = sites[-1][1]
        out = {tuple(site): None for site in sites}
        nxc, nyc = (nx0 + nx1) // 2, (ny0 + ny1) // 2

        for iy0, ny0 in enumerate(range(*self.yrange), start=1):
            if iy0 > 1:
                break

            vecc = self[nxc, "b"].conj()
            tm = self[nxc, "h", exci_bra, None, (nxc, nyc), None]
            tmb = self[nxc, "hb", exci_bra, None, (nxc, nyc), None]
            vec = self[nxc, "t"]

            env = mps.Env(vecc, [tm, vec]).setup_(to="first").setup_(to="last")

            if nxc < self.xrange[1] - 1:
                vec_o0_next = mps.zipper(tm, vec, opts_svd=opts_svd, normalize=False)
                mps.compression_(vec_o0_next, (tm, vec), method="1site", normalize=False, **opts_var)

                vecc_o0_last = mps.zipper(tmb, vecc.conj(), opts_svd, normalize=False)
                mps.compression_(
                    vecc_o0_last, (tmb, vecc.conj()), method="1site", normalize=False, **opts_var
                )

            ket0 = tm[iy0].ket
            tm[iy0] = self._replace_layers(tm[iy0], ket=exci_ket[nxc, ny0])
            env.update_env_(iy0, to="first")
            out[(nxc, ny0)] = env.measure(bd=(iy0 - 1, iy0))

            tm[iy0] = self._replace_layers(tm[iy0], ket=ket0)

            for iy1, ny1 in enumerate(range(ny0 + 1, self.yrange[1]), start=ny0 - self.yrange[0] + 2):
                ket0 = tm[iy1].ket
                tm[iy1] = self._replace_layers(tm[iy1], ket=exci_ket[nxc, ny1])
                env.update_env_(iy1, to="first")
                out[(nxc, ny1)] = env.measure(bd=(iy1 - 1, iy1))
                tm[iy1] = self._replace_layers(tm[iy1], ket=ket0)

            for nx1 in range(nxc + 1, self.xrange[1]):
                vecc, tm, vec_o0 = self[nx1, "b"].conj(), self[nx1, "h"], vec_o0_next

                if nx1 < self.xrange[1] - 1:
                    vec_o0_next = mps.zipper(tm, vec_o0, opts_svd=opts_svd, normalize=False)
                    mps.compression_(
                        vec_o0_next,
                        (tm, vec_o0),
                        method="1site",
                        normalize=False,
                        **opts_var,
                    )

                env = mps.Env(vecc, [tm, vec_o0]).setup_(to="last").setup_(to="first")
                for iy1, ny1 in enumerate(range(*self.yrange), start=1):
                    ket0 = tm[iy1].ket
                    tm[iy1] = self._replace_layers(tm[iy1], ket=exci_ket[nx1, ny1])
                    env.update_env_(iy1, to="first")
                    out[(nx1, ny1)] = env.measure(bd=(iy1 - 1, iy1))
                    tm[iy1] = self._replace_layers(tm[iy1], ket=ket0)

            for nx1 in range(nxc - 1, self.xrange[0] - 1, -1):
                vecc_o0, tmb, tm, vec = vecc_o0_last, self[nx1, "hb"], self[nx1, "h"], self[nx1, "t"]

                if nx1 > self.xrange[0]:
                    vecc_o0_last = mps.zipper(tmb, vecc_o0, opts_svd=opts_svd, normalize=False)
                    mps.compression_(
                        vecc_o0_last,
                        (tmb, vecc_o0),
                        method="1site",
                        normalize=False,
                        **opts_var,
                    )

                env = mps.Env(vecc_o0.conj(), [tm, vec]).setup_(to="last").setup_(to="first")
                for iy1, ny1 in enumerate(range(*self.yrange), start=1):
                    ket0 = tm[iy1].ket
                    tm[iy1] = self._replace_layers(tm[iy1], ket=exci_ket[nx1, ny1])
                    env.update_env_(iy1, to="first")
                    out[(nx1, ny1)] = env.measure(bd=(iy1 - 1, iy1))
                    tm[iy1] = self._replace_layers(tm[iy1], ket=ket0)

        return out

    def measure_exci_norm_tl(self, exci_bra=None, exci_ket=None, opts_svd=None, opts_var=None):
        if opts_var is None:
            opts_var = {"max_sweeps": 2}
        if opts_svd is None:
            D_total = max(
                max(self[nx, dirn].get_bond_dimensions())
                for nx in range(*self.xrange)
                for dirn in "tb"
            )
            opts_svd = {"D_total": D_total}

        sites = self.sites()
        out = {}

        nx0 = sites[0][0]

        for iy0, ny0 in enumerate(range(*self.yrange), start=1):
            if iy0 > 1:
                break
            vecc = self[nx0, "b"].conj()
            tm = self[nx0, "h", exci_bra, None, (nx0, ny0), None]
            vec = self[nx0, "t"]

            env = mps.Env(vecc, [tm, vec]).setup_(to="first").setup_(to="last")
            ket0 = tm[iy0].ket
            tm[iy0] = self._replace_layers(tm[iy0], ket=exci_ket[nx0, ny0])

            env.update_env_(iy0, to="first")
            out[(nx0, ny0), (nx0, ny0)] = env.measure(bd=(iy0 - 1, iy0))

            tm[iy0] = self._replace_layers(tm[iy0], bra=exci_bra, ket=ket0)

            if nx0 < self.xrange[1] - 1:
                vec_o0_next = mps.zipper(tm, vec, opts_svd=opts_svd, normalize=False)
                mps.compression_(vec_o0_next, (tm, vec), method="1site", normalize=False, **opts_var)

            for iy1, ny1 in enumerate(range(ny0 + 1, self.yrange[1]), start=ny0 - self.yrange[0] + 2):
                ket0 = tm[iy1].ket
                tm[iy1] = self._replace_layers(tm[iy1], ket=exci_ket[nx0, ny1])
                env.update_env_(iy1, to="first")
                out[(nx0, ny0), (nx0, ny1)] = env.measure(bd=(iy1 - 1, iy1))
                tm[iy1] = self._replace_layers(tm[iy1], ket=ket0)

            for nx1 in range(self.xrange[0] + 1, self.xrange[1]):
                vecc, tm, vec_o0 = self[nx1, "b"].conj(), self[nx1, "h"], vec_o0_next

                if nx1 < self.xrange[1] - 1:
                    vec_o0_next = mps.zipper(tm, vec_o0, opts_svd=opts_svd, normalize=False)
                    mps.compression_(
                        vec_o0_next,
                        (tm, vec_o0),
                        method="1site",
                        normalize=False,
                        **opts_var,
                    )

                env = mps.Env(vecc, [tm, vec_o0]).setup_(to="last").setup_(to="first")
                for iy1, ny1 in enumerate(range(*self.yrange), start=1):
                    ket0 = tm[iy1].ket
                    tm[iy1] = self._replace_layers(tm[iy1], ket=exci_ket[nx1, ny1])
                    env.update_env_(iy1, to="first")
                    out[(nx0, ny0), (nx1, ny1)] = env.measure(bd=(iy1 - 1, iy1))
                    tm[iy1] = self._replace_layers(tm[iy1], ket=ket0)

        return out


EnvExciSMA = EnvExciSMAFermionicCharged
