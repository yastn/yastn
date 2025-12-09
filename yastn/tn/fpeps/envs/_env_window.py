# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

from tqdm import tqdm

from ._env_auxlliary import clear_operator_input, clear_projectors
from .._geometry import Site
from ... import mps
from ....tensor import YastnError


class EnvWindow:
    """ EnvWindow class for expectation values within PEPS with CTM boundary. """

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
        n, dirn = ind

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
                op.A[ind] = self.psi[nx, n].transpose(axes=(0, 3, 2, 1))
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
                op.A[ind] = self.psi[n, ny].transpose(axes=(1, 2, 3, 0))
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


    def measure_2site(self, O0, O1, opts_svd=None, opts_var=None, site0='corner'):
        if site0 == 'corner':
            return measure_2site_corner_columns(self, O0, O1, self.xrange, self.yrange, opts_svd, opts_var)
        if site0 == 'row':
            pairs = [(s0, s1) for s0 in self.sites() for s1 in self.sites()]
            return _measure_2site_row(self, O0, O1, self.xrange, self.yrange, pairs, opts_svd, opts_var)
        raise YastnError("site0 should be 'corner' or 'row'. ")


    def sample(self, projectors, number=1, opts_svd=None, opts_var=None, progressbar=False, return_probabilities=False, flatten_one=True) -> dict[Site, list]:
        """
        Sample random configurations from PEPS.
        See :meth:`yastn.tn.fpeps.EnvCTM.sample` for description.
        """
        if opts_var is None:
            opts_var = {'max_sweeps': 2}
        if opts_svd is None:
            D_total = max(max(self[ny, dirn].get_bond_dimensions()) for ny in range(*self.yrange) for dirn in 'lr')
            opts_svd = {'D_total': D_total}

        sites = self.sites()
        projs_sites = clear_projectors(sites, projectors, self.xrange, self.yrange)

        out = {site: [] for site in sites}
        probabilities = []
        rands = self.psi.config.backend.rand(self.Nx * self.Ny * number)  # in [0, 1]
        count = 0

        for _ in tqdm(range(number), desc="Sample...", disable=not progressbar):
            probability = 1.
            vec = self[self.yrange[0], 'l']
            for ny in range(*self.yrange):
                vecc = self[ny, 'r'].conj()
                tm = self[ny, 'v']
                env = mps.Env(vecc, [tm, vec]).setup_(to='first')
                for ix, nx in enumerate(range(*self.xrange), start=self.offset):
                    env.update_env_(ix - 1, to='last')
                    norm_prob = env.measure(bd=(ix - 1, ix)).real
                    acc_prob = 0
                    for k, proj in projs_sites[nx, ny].items():
                        tm[ix].set_operator_(proj)
                        env.update_env_(ix, to='first')
                        prob = env.measure(bd=(ix-1, ix)).real / norm_prob
                        acc_prob += prob
                        if rands[count] < acc_prob:
                            out[nx, ny].append(k)
                            tm[ix].set_operator_(proj / prob)
                            probability *= prob
                            break
                    count += 1
                if ny + 1 < self.yrange[1]:
                    vec_new = mps.zipper(tm, vec, opts_svd=opts_svd)
                    mps.compression_(vec_new, (tm, vec), method='1site', **opts_var)
                    vec = vec_new
            probabilities.append(probability)

        if number == 1 and flatten_one:
            out = {site: smp.pop() for site, smp in out.items()}

        if return_probabilities:
            return out, probabilities
        return out


def _measure_2site_row(self, O0, O1, xrange, yrange, pairs, opts_svd=None, opts_var=None):
    """
    Calculate all 2-point correlations <o1 o2> in a finite peps.

    o1 and o2 are given as dict[tuple[int, int], dict[int, operators]],
    mapping sites with list of operators at each site.
    """
    if opts_var is None:
        opts_var = {'max_sweeps': 2}
    if opts_svd is None:
        D_total = max(max(self[nx, dirn].get_bond_dimensions()) for nx in range(*xrange) for dirn in 'tb')
        opts_svd = {'D_total': D_total}

    O0dict = clear_operator_input(O0, self.sites())
    O1dict = clear_operator_input(O1, self.sites())
    out = {}
    # All O0 should have the same charge  # TODO

    vecs = {nx: self[nx, 't'].shallow_copy() for nx in range(*xrange)}
    for nx in range(xrange[0], xrange[1] - 1):
        mps.compression_(vecs[nx + 1], (self[nx, 'h'], vecs[nx]), method='1site', normalize=False, **opts_var)

    for nx0 in range(*xrange):
        for ny0 in range(*yrange):
            iy0 = ny0 - yrange[0] + self.offset
            for nz0, o0 in O0dict[nx0, ny0].items():

                vecc, tm, vec = self[nx0, 'b'].conj(), self[nx0, 'h'], vecs[nx0]
                env = mps.Env(vecc, [tm, vec]).setup_(to='first').setup_(to='last')
                norm_env = env.measure(bd=(0, 1))

                # calculate on-site correlations
                if ((nx0, ny0), (nx0, ny0)) in pairs:
                    for nz1, o1 in O1dict[nx0, ny0].items():
                        tm[iy0].set_operator_(o0 @ o1)
                        env.update_env_(iy0, to='first')
                        out[(nx0, ny0) + nz0, (nx0, ny0) + nz1] = env.measure(bd=(iy0-1, iy0)) / norm_env

                tm[iy0].set_operator_(o0)
                env.setup_(to='last')

                if nx0 < xrange[1] - 1:
                    vec_o0_next = mps.zipper(tm, vec, opts_svd=opts_svd)
                    mps.compression_(vec_o0_next, (tm, vec), method='1site', normalize=False, **opts_var)

                for ny1 in range(ny0 + 1, yrange[1]):
                    iy1 = ny1 - yrange[0] + self.offset

                    if ((nx0, ny0), (nx0, ny1)) in pairs:
                        for nz1, o1 in O1dict[nx0, ny1].items():
                            tm[iy1].set_operator_(o1)
                            env.update_env_(iy1, to='first')
                            out[(nx0, ny0) + nz0, (nx0, ny1) + nz1] = env.measure(bd=(iy1-1, iy1)) / norm_env

                # all subsequent rows
                for nx1 in range(xrange[0] + 1, xrange[1]):
                    vecc, tm, vec_o0, vec = self[nx1, 'b'].conj(), self[nx1, 'h'], vec_o0_next, vecs[nx1]
                    norm_env = mps.vdot(vecc, tm, vec)

                    if nx1 < xrange[1] - 1:
                        vec_o0_next = mps.zipper(tm, vec_o0, opts_svd=opts_svd)
                        mps.compression_(vec_o0_next, (tm, vec_o0), method='1site', normalize=False, **opts_var)

                    env = mps.Env(vecc, [tm, vec_o0]).setup_(to='last').setup_(to='first')
                    for ny1 in range(*yrange):
                        iy1 = ny1 - yrange[0] + self.offset
                        if ((nx0, ny0), (nx1, ny1)) in pairs:
                            for nz1, o1 in O1dict[nx1, ny1].items():
                                tm[iy1].set_operator_(o1)
                                env.update_env_(iy1, to='first')
                                out[(nx0, ny0) + nz0, (nx1, ny1) + nz1] = env.measure(bd=(iy1-1, iy1)) / norm_env
        return out


def measure_2site_corner_columns(self, O0, O1, xrange, yrange, opts_svd=None, opts_var=None):
    """
    Calculate all 2-point correlations <o1 o2> in a finite peps.

    o1 and o2 are given as dict[tuple[int, int], dict[int, operators]],
    mapping sites with list of operators at each site.
    """
    if opts_var is None:
        opts_var = {'max_sweeps': 2}
    if opts_svd is None:
        D_total = max(max(self[ny, dirn].get_bond_dimensions()) for ny in range(*yrange) for dirn in 'lr')
        opts_svd = {'D_total': D_total}

    sites = self.sites()
    O0dict = clear_operator_input(O0, sites)
    O1dict = clear_operator_input(O1, sites)
    out = {}
    O1n = [*O1dict[sites[0]].values()][0].n  # All O1 should have the same charge
    # All O0 should have the same charge  # TODO

    (nx0, ny0), ix0 = sites[0], 1
    vecc, tm, vec = self[ny0, 'r'].conj(), self[ny0, 'v'], self[ny0, 'l']

    if ny0 < yrange[1] - 1:
        vec_next = mps.zipper(tm, vec, opts_svd=opts_svd)
        mps.compression_(vec_next, (tm, vec), method='1site', normalize=False, **opts_var)

    env = mps.Env(vecc, [tm, vec]).setup_(to='first').setup_(to='last')
    norm_env = env.measure(bd=(0, 1))
    # calculate on-site correlations
    for nz1, o1 in O1dict[nx0, ny0].items():
        tm[ix0].set_operator_(O0 @ o1)
        env.update_env_(ix0, to='first')
        out[(nx0, ny0), (nx0, ny0) + nz1] = env.measure(bd=(ix0-1, ix0)) / norm_env

    tm[ix0].set_operator_(O0)
    env.setup_(to='last')

    if ny0 < yrange[1] - 1:
        vec_O0_next = mps.zipper(tm, vec, opts_svd=opts_svd)
        mps.compression_(vec_O0_next, (tm, vec), method='1site', normalize=False, **opts_var)

    for ix1, nx1 in enumerate(range(nx0+1, xrange[1]), start=nx0-xrange[0] + self.offset + 1):
        for nz1, o1 in O1dict[nx1, ny0].items():
            tm[ix1].set_operator_(o1)
            env.update_env_(ix1, to='first')
            out[(nx0, ny0), (nx1, ny0) + nz1] = env.measure(bd=(ix1-1, ix1)) / norm_env

    # all subsequent rows
    for ny1 in range(yrange[0]+1, yrange[1]):
        vecc, tm, vec_O0, vec = self[ny1, 'r'].conj(), self[ny1, 'v'], vec_O0_next, vec_next
        norm_env = mps.vdot(vecc, tm, vec)

        if ny1 < yrange[1] - 1:
            vec_next = mps.zipper(tm, vec, opts_svd=opts_svd)
            mps.compression_(vec_next, (tm, vec), method='1site', normalize=False, **opts_var)
            vec_O0_next = mps.zipper(tm, vec_O0, opts_svd=opts_svd)
            mps.compression_(vec_O0_next, (tm, vec_O0,), method='1site', normalize=False, **opts_var)

        env = mps.Env(vecc, [tm, vec_O0]).setup_(to='last').setup_(to='first')
        for ix1, nx1 in enumerate(range(*xrange), start=self.offset):
            for nz1, o1 in O1dict[nx1, ny0].items():
                tm[ix1].set_operator_(o1)
                env.update_env_(ix1, to='first')
                out[(nx0, ny0), (nx1, ny1) + nz1] = env.measure(bd=(ix1-1, ix1)) / norm_env
    return out


def measure_2site_all(peps_env, O, P, opts_svd, opts_var=None):
    """
    Calculate all 2-point correlations <O_i P_j> in a finite PEPS.

    Takes CTM environments and operators.

    Parameters
    ----------
    O, P: dict[tuple[int, int], dict[int, operators]],
        mapping sites with list of operators at each site.
    """
    out = {}
    if opts_var is None:
        opts_var =  {'max_sweeps': 2}

    psi = peps_env.psi
    Nx, Ny = psi.Nx, psi.Ny
    sites = [(nx, ny) for ny in range(Ny-1, -1, -1) for nx in range(Nx)]
    op1dict = clear_operator_input(O, sites)
    op2dict = clear_operator_input(P, sites)

    for nx1, ny1 in sites:
        # print( f"Correlations from {nx1} {ny1} ... ")
        for nz1, o1 in op1dict[nx1, ny1].items():
            vR = peps_env.boundary_mps(n=ny1, dirn='r')
            vL = peps_env.boundary_mps(n=ny1, dirn='l')
            Os = psi.transfer_mpo(n=ny1, dirn='v').T
            env = mps.Env(vL.conj(), [Os, vR]).setup_(to='first').setup_(to='last')
            norm_env = env.measure(bd=(-1, 0))

            if ny1 > 0:
                vRnext = mps.zipper(Os, vR, opts_svd=opts_svd)
                mps.compression_(vRnext, (Os, vR), method='1site', normalize=False, **opts_var)

            # first calculate on-site correlations
            for nz2, o2 in op2dict[nx1, ny1].items():
                Os[nx1].set_operator_(o1 @ o2)
                env.update_env_(nx1, to='last')
                out[(nx1, ny1) + nz1, (nx1, ny1) + nz2] = env.measure(bd=(nx1, nx1+1)) / norm_env

            Os[nx1].set_operator_(o1)
            env.setup_(to='last')

            if ny1 > 0:
                vRo1next = mps.zipper(Os, vR, opts_svd=opts_svd)
                mps.compression_(vRo1next, (Os, vR), method='1site', normalize=False, **opts_var)

            # calculate correlations along the row
            for nx2 in range(nx1 + 1, Nx):
                for nz2, o2 in op2dict[nx2, ny1].items():
                    Os[nx2].set_operator_(o2)
                    env.update_env_(nx2, to='first')
                    out[(nx1, ny1) + nz1, (nx2, ny1) + nz2] = env.measure(bd=(nx2-1, nx2)) / norm_env

            # and all subsequent rows
            for ny2 in range(ny1-1, -1, -1):
                vR = vRnext
                vRo1 = vRo1next
                vL = peps_env.boundary_mps(n=ny2, dirn='l')
                Os = psi.transfer_mpo(n=ny2, dirn='v').T
                env = mps.Env(vL.conj(), [Os, vR]).setup_(to='first')
                norm_env = env.measure(bd=(-1, 0))

                if ny2 > 0:
                    vRnext = mps.zipper(Os, vR, opts_svd=opts_svd)
                    mps.compression_(vRnext, (Os, vR), method='1site', normalize=False, **opts_var)
                    vRo1next = mps.zipper(Os, vRo1, opts_svd=opts_svd)
                    mps.compression_(vRo1next, (Os, vRo1), method='1site', normalize=False, **opts_var)

                env = mps.Env(vL.conj(), [Os, vRo1]).setup_(to='first').setup_(to='last')
                for nx2 in range(psi.Nx):
                    for nz2, o2 in op2dict[nx2, ny2].items():
                        Os[nx2].set_operator_(o2)
                        env.update_env_(nx2, to='first')
                        out[(nx1, ny1) + nz1, (nx2, ny2) + nz2] = env.measure(bd=(nx2-1, nx2)) / norm_env
    return out