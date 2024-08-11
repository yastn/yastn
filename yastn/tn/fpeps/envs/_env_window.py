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
from itertools import accumulate
from tqdm import tqdm
from ... import mps
from .... import YastnError, Tensor
from .._gates_auxiliary import apply_gate_onsite
from .._geometry import Site


class EnvWindow:
    """ EnvWindow class for expectation values within PEPS with CTM boundary. """

    def __init__(self, env_ctm, xrange, yrange):
        self.psi = env_ctm.psi
        self.env_ctm = env_ctm
        self.xrange = xrange
        self.yrange = yrange
        self.Nx = self.xrange[1] - self.xrange[0]
        self.Ny = self.yrange[1] - self.yrange[0]

    def __getitem__(self, ind) -> yastn.tn.mps.MpsMpoOBC:
        """
        Boundary MPS build of CTM tensors, or a transfer matrix MPO.

        CTM corner and edge tensors are included at the ends of MPS and MPO, respectively.

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
            Leg convention is consistent with vdot(t, h, b) and vdot(r, v, l).
        """
        n, dirn = ind

        if dirn in 'rvl' and not self.yrange[0] <= n < self.yrange[1]:
            raise YastnError(f"{n=} not within {self.yrange=}")

        if dirn == 'r':
            psi = mps.Mps(self.Nx + 2)
            for ind, nx in enumerate(range(*self.xrange), start=1):
                psi[ind] =self.env_ctm[nx, n].r.conj()
            psi[0] =self.env_ctm[self.xrange[0], n].tr.add_leg(axis=0).conj()
            psi[self.Nx + 1] =self.env_ctm[self.xrange[1]-1, n].br.add_leg(axis=2).conj()
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
            psi[self.Nx + 1] = self.env_ctm[self.xrange[1]-1, n].bl.add_leg(axis=0).transpose(axes=(2, 1, 0))
            return psi

        if dirn in 'thb' and not self.xrange[0] <= n < self.xrange[1]:
            raise YastnError(f"{n=} not within {self.xrange=}")

        if dirn == 't':
            psi = mps.Mps(self.Ny + 2)
            for ind, ny in enumerate(range(*self.yrange), start=1):
                psi[ind] =self.env_ctm[n, ny].t
            psi[0] =self.env_ctm[n, self.yrange[0]].tl.add_leg(axis=0)
            psi[self.Ny + 1] =self.env_ctm[n, self.yrange[1]-1].tr.add_leg(axis=2)
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
                psi[ind] =self.env_ctm[n, ny].b.transpose(axes=(2, 1, 0)).conj()
            psi[0] =self.env_ctm[n, self.yrange[0]].bl.add_leg(axis=2).transpose(axes=(2, 1, 0)).conj()
            psi[self.Ny + 1] =self.env_ctm[n, self.yrange[1]-1].br.add_leg(axis=0).transpose(axes=(2, 1, 0)).conj()
            return psi

        raise YastnError(f"{dirn=} not recognized. Should be 't', 'h' 'b', 'r', 'v', or 'l'.")


    def measure_2site(self, op1, op2, opts_svd, site1='first', site2='all', opts_var=None):
        """
        Calculate all 2-point correlations <o1 o2> in a finite peps.

        o1 and o2 are given as dict[tuple[int, int], dict[int, operators]],
        mapping sites with list of operators at each site.
        """
        out = {}
        if opts_var is None:
            opts_var =  {'max_sweeps': 2}

        sites = [(nx, ny) for ny in range(*self.yrange) for nx in range(*self.xrange)]
        # op1dict = _clear_operator_input(op1, sites)
        # op2dict = _clear_operator_input(op2, sites)

        nx1, ny1 = sites[0]
        ix1 = 1

        vR = self[ny1, 'r']
        vO = self[ny1, 'v']
        vL = self[ny1, 'l']

        vLnext = mps.zipper(vO, vL, opts_svd=opts_svd)
        mps.compression_(vLnext, (vO, vL), method='1site', normalize=False, **opts_var)

        env = mps.Env(vR, [vO, vL]).setup_(to='first').setup_(to='last')
        norm_env = env.measure(bd=(-1, 0))

        # first calculate on-site correlations
        vOnx1A = vO[ix1].top
        vO[ix1].top = apply_gate_onsite(vOnx1A, op1 @ op2)
        env.update_env_(ix1, to='last')
        out[(nx1, ny1), (nx1, ny1)] = env.measure(bd=(ix1, ix1+1)) / norm_env

        vO[nx1].top = apply_gate_onsite(vOnx1A, op1)

        if ny1 < self.yrange[1] - 1:
            vLo1next = mps.zipper(vO, vL, opts_svd=opts_svd)
            mps.compression_(vLo1next, (vO, vL), method='1site', normalize=False, **opts_var)

        env.setup_(to='last')
        for ix2, nx2 in enumerate(range(*self.xrange), start=1):
            vOnx2A = vO[ix2].top
            vO[ix2].top = apply_gate_onsite(vOnx2A, op2)
            env.update_env_(ix2, to='first')
            out[(nx1, ny1), (nx2, ny1)] = env.measure(bd=(ix2-1, ix2)) / norm_env

        # and all subsequent rows
        for ny2 in range(self.yrange[0]+1, self.yrange[1]):
            vR = self[ny2, 'r']
            vO = self[ny2, 'v']
            vL = vLnext
            vLo1 = vLo1next
            env = mps.Env(vR, [vO, vL]).setup_(to='first')
            norm_env = env.measure(bd=(-1, 0))

            if ny2 < self.yrange[1] - 1:
                vLnext = mps.zipper(vO, vL, opts_svd=opts_svd)
                mps.compression_(vLnext, (vO, vL), method='1site', normalize=False, **opts_var)
                vLo1next = mps.zipper(vO, vLo1, opts_svd=opts_svd)
                mps.compression_(vLo1next, (vO, vLo1), method='1site', normalize=False, **opts_var)

            env = mps.Env(vR, [vO, vLo1]).setup_(to='first').setup_(to='last')
            for ix2, nx2 in enumerate(range(*self.xrange), start=1):
                vOnx2A = vO[ix2].top
                vO[ix2].top = apply_gate_onsite(vOnx2A, op2)
                env.update_env_(ix2, to='first')
                out[(nx1, ny1), (nx2, ny2)] = env.measure(bd=(ix2-1, ix2)) / norm_env
        return out

    def sample(self, projectors, number=1, opts_svd=None, opts_var=None, progressbar=False, return_info=False) -> dict[Site, list]:
        """
        Sample random configurations from PEPS.
        See :meth:`yastn.tn.fpeps.EnvCTM.sample` for description.
        """
        if opts_var is None:
            opts_var = {'max_sweeps': 2}
        if opts_svd is None:
            D_total = max(max(self[ny, dirn].get_bond_dimensions()) for ny in range(*self.yrange) for dirn in 'lr')
            opts_svd = {'D_total': D_total}

        sites = [Site(nx, ny) for ny in range(*self.yrange) for nx in range(*self.xrange)]

        # spread projectors over sites
        if not isinstance(projectors, dict) or all(isinstance(x, Tensor) for x in projectors.values()):
            projectors = {site: projectors for site in sites}
        if set(sites) != set(projectors.keys()):
            raise YastnError(f"projectors not defined for some sites in xrange={self.xrange}, yrange={self.yrange}.")

        # change each list of projectors into keys and projectors
        projs_sites = {}
        for k, v in projectors.items():
            if isinstance(v, dict):
                projs_sites[k, 'k'] = list(v.keys())
                projs_sites[k, 'p'] = list(v.values())
            else:
                projs_sites[k, 'k'] = list(range(len(v)))
                projs_sites[k, 'p'] = v

        out = {site: [] for site in sites}
        rands = (self.psi.config.backend.rand(self.Nx * self.Ny * number) + 1) / 2  # in [0, 1]
        count = 0

        info = {'opts_svd': opts_svd,
                'error': 0.}

        for _ in tqdm(range(number), desc="Sample...", disable=(not progressbar)):
            vec = self[self.yrange[0], 'l']
            for ny in range(*self.yrange):
                vecc = self[ny, 'r']
                tm = self[ny, 'v'].copy()
                env = mps.Env(vecc, [tm, vec]).setup_(to='first')
                env.update_env_(0, to='last')
                for ix, nx in enumerate(range(*self.xrange), start=1):
                    top = tm[ix].top
                    norm_prob = env.measure(bd=(ix-1, ix)).item()
                    prob = []
                    for proj in projs_sites[(nx, ny), 'p']:
                        tm[ix].top = apply_gate_onsite(top, proj)
                        env.update_env_(ix, to='last')
                        prob.append(env.measure(bd=(ix, ix+1)).item() / norm_prob)
                    error = abs(min(0., *(x.real for x in prob))) + max(abs(x.imag) for x in prob)
                    if error > 0.:
                        prob = [max(x.real, error) for x in prob]
                        info['error'] = max(error, info['error'])
                    norm_prob = sum(prob)
                    prob = [x / norm_prob for x in prob]
                    ind = sum(apr < rands[count] for apr in accumulate(prob))
                    out[nx, ny].append(projs_sites[(nx, ny), 'k'][ind])
                    tm[ix].top = apply_gate_onsite(top, projs_sites[(nx, ny), 'p'][ind])
                    env.update_env_(ix, to='last')
                    count += 1
                if ny + 1 < self.yrange[1]:
                    vec_new = mps.zipper(tm, vec, opts_svd=opts_svd)
                    mps.compression_(vec_new, (tm, vec), method='1site', **opts_var)
                    vec = vec_new
        if return_info:
            out['info'] = info
        return out