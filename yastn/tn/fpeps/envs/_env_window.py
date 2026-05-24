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

from .._doublePepsTensor import DoublePepsTensor
from .._gates_auxiliary import clear_operator_input, clear_projectors
from .._geometry import Site
from ... import mps
from ....tensor import Tensor, YastnError, add as tensor_add, sign_canonical_order


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


def set_operator_(tm, n, op, horizontal=True):
    """ helper function to cover 2-layers and 1-layer peps """
    if isinstance(tm[n], DoublePepsTensor):
        tm[n].set_operator_(op)
    else:
        axes = (1, 2, 3, 0) if horizontal else (0, 3, 2, 1)
        tm[n] = op.transpose(axes=axes)

def restore_old_tensor_(tm, n, old_ten):
    """ helper function to cover 2-layers and 1-layer peps """
    if isinstance(tm[n], DoublePepsTensor):
        tm[n].del_operator_()
    else:
        tm[n] = old_ten

def add_charge_swaps_(tm, n, charge, axes):
    """ helper function to cover 2-layers and 1-layer peps; fermionic strings assume 2-layers """
    if isinstance(tm[n], DoublePepsTensor):
        tm[n].add_charge_swaps_(charge, axes=axes)

def del_charge_swaps_(tm, n):
    """ helper function to cover 2-layers and 1-layer peps; fermionic strings assume 2-layers """
    if isinstance(tm[n], DoublePepsTensor):
        tm[n].del_charge_swaps_()


def _sample(env, projectors, xrange, yrange, offset, number=1, dirn='v', opts_svd=None, opts_var=None, progressbar=False, return_probabilities=False, flatten_one=True):
    """
    Worker function for sampling; works for EnvBoundaryMPS and EnvWindow (derived from EnvCTM).
    See docstring of :meth:`EnvCTM.sample`.
    """
    if opts_var is None:
        opts_var = {'max_sweeps': 2}

    if opts_svd is None:
        rr = yrange if dirn == 'v' else xrange
        dd = 'lr' if dirn == 'v' else 'tb'
        D_total = max(max(env[n, d].get_bond_dimensions()) for n in range(*rr) for d in dd)
        opts_svd = {'D_total': D_total}

    sites = [Site(nx, ny) for nx in range(*xrange) for ny in range(*yrange)]
    projs_sites = clear_projectors(sites, projectors)

    out, probs = {site: [] for site in sites}, []

    for _ in tqdm(range(number), desc="Sample...", disable=not progressbar):

        if dirn == 'v':
            smpl, prob = _sample_one_columns(env, projs_sites, xrange, yrange, offset, opts_svd, opts_var)
        else:  # dirn == 'h':
            smpl, prob = _sample_one_rows(env, projs_sites, xrange, yrange, offset, opts_svd, opts_var)

        if number == 1 and flatten_one:
            out, probs = smpl, prob
        else:
            for site, v in smpl.items():
                out[site].append(v)
            probs.append(prob)

    return (out, probs) if return_probabilities else out


def _sample_one_columns(self, projs_sites, xrange, yrange, offset, opts_svd=None, opts_var=None):
    """ Worker function for a single sample wher boundary MPSs are vertical. """
    out, probability, count = {}, 1., 0
    #
    vec = self[yrange[0], 'l']
    rands = vec.config.backend.rand((xrange[1] - xrange[0]) * (yrange[1] - yrange[0]))  # in [0, 1]
    #
    for ny in range(*yrange):
        vecc = self[ny, 'r'].conj()
        tm = self[ny, 'v']
        env = mps.Env(vecc, [tm, vec])
        env.setup_(to=offset)
        #
        for ix, nx in enumerate(range(*xrange), start=offset):
            proj_sum = tensor_add(*projs_sites[nx, ny].values())
            set_operator_(tm, ix, proj_sum)
            norm_prob = env.measure(bd=(ix-1, ix+1)).real
            acc_prob = 0
            for k, proj in projs_sites[nx, ny].items():
                set_operator_(tm, ix, proj)
                prob = env.measure(bd=(ix-1, ix+1)).real / norm_prob
                acc_prob += prob
                if rands[count] <= acc_prob:
                    break
            out[nx, ny] = k
            probability *= prob
            set_operator_(tm, ix, proj / prob)
            if nx + 1 < xrange[1]:
                env.update_env_(ix, to='last')
            count += 1
        if ny + 1 < yrange[1]:
            vec_new = mps.zipper(tm, vec, opts_svd=opts_svd)
            mps.compression_(vec_new, (tm, vec), method='1site', **opts_var)
            vec = vec_new
    return out, probability


def _sample_one_rows(self, projs_sites, xrange, yrange, offset, opts_svd=None, opts_var=None):
    """ Worker function for a single sample wher boundary MPSs are horizontal. """
    out, probability, count = {}, 1., 0
    #
    vec = self[xrange[0], 't']
    rands = vec.config.backend.rand((xrange[1] - xrange[0]) * (yrange[1] - yrange[0]))  # in [0, 1]
    #
    for nx in range(*xrange):
        vecc = self[nx, 'b'].conj()
        tm = self[nx, 'h']
        env = mps.Env(vecc, [tm, vec]).setup_(to=offset)
        #
        for iy, ny in enumerate(range(*yrange), start=offset):
            proj_sum = tensor_add(*projs_sites[nx, ny].values())
            set_operator_(tm, iy, proj_sum)
            norm_prob = env.measure(bd=(iy-1, iy+1)).real
            acc_prob = 0
            for k, proj in projs_sites[nx, ny].items():
                set_operator_(tm, iy, proj)
                prob = env.measure(bd=(iy-1, iy+1)).real / norm_prob
                acc_prob += prob
                if rands[count] <= acc_prob:
                    break
            out[nx, ny] = k
            probability *= prob
            set_operator_(tm, iy, proj / prob)
            if ny + 1 < yrange[1]:
                env.update_env_(iy, to='last')
            count += 1
        if nx + 1 < xrange[1]:
            vec_new = mps.zipper(tm, vec, opts_svd=opts_svd)
            mps.compression_(vec_new, (tm, vec), method='1site', **opts_var)
            vec = vec_new
    return out, probability


def _measure_2site(env, O0, O1, xrange, yrange, offset, pairs="corner <=", dirn='v', opts_svd=None, opts_var=None):
    """
    Worker function for measure_2site; works for EnvBoundaryMPS and EnvWindow (derived from EnvCTM).
    See docstring of :meth:`EnvCTM.measure_2site`.
    """
    sites = [Site(nx, ny) for nx in range(*xrange) for ny in range(*yrange)]
    O0dict = clear_operator_input(O0, sites)
    O1dict = clear_operator_input(O1, sites)

    if isinstance(pairs, str):
        if 'corner' in pairs:
            s0 = Site(xrange[0], yrange[0])
            pairs_all = [(s0, s1) for s1 in sites]
        elif 'row' in pairs:
            row_sites = [Site(xrange[0], ny) for ny in range(*yrange)]
            pairs_all = [(s0, s1) for s0 in row_sites for s1 in sites]
        else:
            pairs_all = [(s0, s1) for s0 in sites for s1 in sites]

        pairs_select = []
        so = (lambda site: site) if dirn == 'h' else (lambda site: site[::-1])
        if "<" in pairs:
            pairs_select += [(s0, s1) for s0, s1 in pairs_all if so(s0) < so(s1)]
        if "=" in pairs:
            pairs_select += [(s0, s1) for s0, s1 in pairs_all if s0 == s1]
        pairs = pairs_select

    # All O0 should have the same charge; the same for O1; this is added for fermionic systems
    n0 = next(iter(O0dict[sites[0]].values())).n
    n1 = next(iter(O1dict[sites[0]].values())).n
    if env.psi.config.fermionic and \
       (any(x.n != n0 for d in O0dict.values() for x in d.values()) or \
        any(x.n != n1 for d in O1dict.values() for x in d.values())):
            raise YastnError("All O0 (O1) operators should have the same charge.")

    if opts_var is None:
        opts_var = {'max_sweeps': 2}

    if dirn == 'v':
        return _measure_2site_columns(env, O0dict, O1dict, xrange, yrange, offset, pairs, opts_svd, opts_var)
    else:  # dirn == 'h':
        return _measure_2site_rows(env, O0dict, O1dict, xrange, yrange, offset, pairs, opts_svd, opts_var)


def _measure_2site_rows(self, O0dict, O1dict, xrange, yrange, offset, pairs, opts_svd=None, opts_var=None):
    """
    Calculate all 2-point correlations <o1 o2> in a finite peps.

    o1 and o2 are given as dict[tuple[int, int], dict[int, operators]],
    mapping sites with list of operators at each site.
    """
    if opts_svd is None:
        D_total = max(max(self[nx, dirn].get_bond_dimensions()) for nx in range(*xrange) for dirn in 'tb')
        opts_svd = {'D_total': D_total}

    vecs = {nx: self[nx, 't'].shallow_copy() for nx in range(*xrange)}
    for nx in range(xrange[0], xrange[1] - 1):  # this is done to propagete the norm of boundary vectors
        mps.compression_(vecs[nx + 1], [self[nx, 'h'], vecs[nx]], method='1site', normalize=False, **opts_var)

    out = {}  # dict for results

    for nx0 in range(*xrange):
        for ny0 in range(*yrange):
            iy0 = ny0 - yrange[0] + offset
            for nz0, o0 in O0dict[nx0, ny0].items():

                vecc, tm, vec = self[nx0, 'b'].conj(), self[nx0, 'h'], vecs[nx0]
                env = mps.Env(vecc, [tm, vec])
                env.setup_(to=iy0)
                norm_env = env.measure(bd=(iy0-1, iy0+1))

                # calculate on-site correlations
                nx1, ny1 = nx0, ny0
                if ((nx0, ny0), (nx1, ny1)) in pairs:
                    for nz1, o1 in O1dict[nx0, ny0].items():
                        if not isinstance(tm[iy0], DoublePepsTensor):
                            raise YastnError("Cannot calculate same-site correlator in 1-layer Peps; remove '=' from pairs.")
                        set_operator_(tm, iy0, o0 @ o1, horizontal=True)
                        out[(nx0, ny0) + nz0, (nx1, ny1) + nz1] = env.measure(bd=(iy0-1, iy0+1)) / norm_env

                set_operator_(tm, iy0, o0, horizontal=True)
                add_charge_swaps_(tm, iy0, o0.n, axes=['k4', 'k2'])

                for ny1 in range(ny0 + 1, yrange[1]):
                    iy1 = ny1 - yrange[0] + offset
                    env.update_env_(iy1 - 1, to='last')
                    if ((nx0, ny0), (nx1, ny1)) in pairs:
                        old_tensor = tm[iy1]
                        add_charge_swaps_(tm, iy1, o0.n, axes='b0')
                        for nz1, o1 in O1dict[nx1, ny1].items():
                            set_operator_(tm, iy1, o1, horizontal=True)
                            out[(nx0, ny0) + nz0, (nx1, ny1) + nz1] = env.measure(bd=(iy1-1, iy1+1)) / norm_env
                        restore_old_tensor_(tm, iy1, old_tensor)
                        add_charge_swaps_(tm, iy1, o0.n, axes=['k4', 'k2'])

                iy_end = yrange[1] - 1 - yrange[0] + offset  # the last index in iy loops
                if nx0 < xrange[1] - 1:
                    add_charge_swaps_(tm, iy_end, o0.n, axes='b3')
                    vec_o0_next = mps.zipper(tm, vec, opts_svd=opts_svd)
                    mps.compression_(vec_o0_next, (tm, vec), method='1site', normalize=False, **opts_var)

                # all subsequent rows
                for nx1 in range(nx0 + 1, xrange[1]):
                    vecc, tm, vec_o0, vec = self[nx1, 'b'].conj(), self[nx1, 'h'], vec_o0_next, vecs[nx1]
                    norm_env = mps.vdot(vecc, tm, vec)

                    if nx1 < xrange[1] - 1:
                        add_charge_swaps_(tm, iy_end, o0.n, axes=['k3', 'b3'])  # iy1 follows from the end of previous loop
                        vec_o0_next = mps.zipper(tm, vec_o0, opts_svd=opts_svd)
                        mps.compression_(vec_o0_next, (tm, vec_o0), method='1site', normalize=False, **opts_var)
                        del_charge_swaps_(tm, iy_end)

                    add_charge_swaps_(tm, iy_end, o0.n, axes='k3')  # iy1 follows from the end of previous loop
                    env = mps.Env(vecc, [tm, vec_o0])
                    env.setup_(to=iy_end)

                    for ny1 in range(yrange[1] - 1, yrange[0] - 1, -1):
                        iy1 = ny1 - yrange[0] + offset
                        if iy1 < iy_end:
                            env.update_env_(iy1 + 1, to='first')

                        if ((nx0, ny0), (nx1, ny1)) in pairs:
                            old_tensor = tm[iy1]
                            add_charge_swaps_(tm, iy1, o0.n, axes=['k2', 'k4'])
                            for nz1, o1 in O1dict[nx1, ny1].items():
                                set_operator_(tm, iy1, o1, horizontal=True)
                                out[(nx0, ny0) + nz0, (nx1, ny1) + nz1] = env.measure(bd=(iy1-1, iy1+1)) / norm_env
                            restore_old_tensor_(tm, iy1, old_tensor)
                            add_charge_swaps_(tm, iy1, o0.n, axes='b0')
    return out


def _measure_2site_columns(self, O0dict, O1dict, xrange, yrange, offset, pairs, opts_svd=None, opts_var=None):
    """
    Calculate all 2-point correlations <o1 o2> in a finite peps.

    o1 and o2 are given as dict[tuple[int, int], dict[int, operators]],
    mapping sites with list of operators at each site.
    """
    if opts_svd is None:
        D_total = max(max(self[ny, dirn].get_bond_dimensions()) for ny in range(*yrange) for dirn in 'lr')
        opts_svd = {'D_total': D_total}

    vecs = {ny: self[ny, 'l'].shallow_copy() for ny in range(*yrange)}
    for ny in range(yrange[0], yrange[1] - 1):  # this is done to propagete the norm of boundary vectors
        mps.compression_(vecs[ny + 1], [self[ny, 'v'], vecs[ny]], method='1site', normalize=False, **opts_var)

    out = {}  # dict for results

    for ny0 in range(*yrange):
        for nx0 in range(*xrange):
            ix0 = nx0 - xrange[0] + offset
            for nz0, o0 in O0dict[nx0, ny0].items():

                vecc, tm, vec = self[ny0, 'r'].conj(), self[ny0, 'v'], vecs[ny0]
                env = mps.Env(vecc, [tm, vec])
                env.setup_(to=ix0)
                norm_env = env.measure(bd=(ix0-1, ix0+1))

                # calculate on-site correlations
                nx1, ny1 = nx0, ny0
                if ((nx0, ny0), (nx1, ny1)) in pairs:
                    for nz1, o1 in O1dict[nx0, ny0].items():
                        if not isinstance(tm[ix0], DoublePepsTensor):
                            raise YastnError("Cannot calculate same-site correlator in 1-layer Peps; remove '=' from pairs.")
                        set_operator_(tm, ix0, o0 @ o1, horizontal=False)
                        out[(nx0, ny0) + nz0, (nx1, ny1) + nz1] = env.measure(bd=(ix0-1, ix0+1)) / norm_env

                set_operator_(tm, ix0, o0, horizontal=False)
                add_charge_swaps_(tm, ix0, o0.n, axes=['k4', 'b3'])

                for nx1 in range(nx0 + 1, xrange[1]):
                    ix1 = nx1 - xrange[0] + offset
                    env.update_env_(ix1 - 1, to='last')
                    if ((nx0, ny0), (nx1, ny1)) in pairs:
                        old_tensor = tm[ix1]
                        add_charge_swaps_(tm, ix1, o0.n, axes='k1')
                        for nz1, o1 in O1dict[nx1, ny1].items():
                            set_operator_(tm, ix1, o1, horizontal=False)
                            out[(nx0, ny0) + nz0, (nx1, ny1) + nz1] = env.measure(bd=(ix1-1, ix1+1)) / norm_env
                        restore_old_tensor_(tm, ix1, old_tensor)
                        add_charge_swaps_(tm, ix1, o0.n, axes=['b4', 'b3'])

                ix_end = xrange[1] - 1 - xrange[0] + offset  # the last index in iy loops
                if ny0 < yrange[1] - 1:
                    add_charge_swaps_(tm, ix_end, o0.n, axes='k2')
                    vec_o0_next = mps.zipper(tm, vec, opts_svd=opts_svd)
                    mps.compression_(vec_o0_next, [tm, vec], method='1site', normalize=False, **opts_var)

                # all subsequent columns
                for ny1 in range(ny0 + 1, yrange[1]):
                    vecc, tm, vec_o0, vec = self[ny1, 'r'].conj(), self[ny1, 'v'], vec_o0_next, vecs[ny1]
                    norm_env = mps.vdot(vecc, tm, vec)

                    if ny1 < yrange[1] - 1:
                        add_charge_swaps_(tm, ix_end, o0.n, axes=['b2', 'k2'])  # iy1 follows from the end of previous loop
                        vec_o0_next = mps.zipper(tm, vec_o0, opts_svd=opts_svd)
                        mps.compression_(vec_o0_next, (tm, vec_o0), method='1site', normalize=False, **opts_var)
                        del_charge_swaps_(tm, ix_end)

                    add_charge_swaps_(tm, ix_end, o0.n, axes='b2')  # iy1 follows from the end of previous loop
                    env = mps.Env(vecc, [tm, vec_o0])
                    env.setup_(to=ix_end)

                    for nx1 in range(xrange[1]-1, xrange[0]-1, -1):
                        ix1 = nx1 - xrange[0] + offset
                        if ix1 < ix_end:
                            env.update_env_(ix1 + 1, to='first')

                        if ((nx0, ny0), (nx1, ny1)) in pairs:
                            old_tensor = tm[ix1]
                            add_charge_swaps_(tm, ix1, o0.n, axes=['b3', 'b4'])
                            for nz1, o1 in O1dict[nx1, ny1].items():
                                set_operator_(tm, ix1, o1, horizontal=False)
                                out[(nx0, ny0) + nz0, (nx1, ny1) + nz1] = env.measure(bd=(ix1-1, ix1+1)) / norm_env
                            restore_old_tensor_(tm, ix1, old_tensor)
                            add_charge_swaps_(tm, ix1, o0.n, axes='k1')

    return out




def _measure_nsite(env, *operators, sites=None, dirn='tb', opts_svd=None, opts_var=None) -> float:
    r"""
    Calculate expectation value of a product of local operators.

    dirn == 'lr' or 'tb'
    """
    if sites is None or len(operators) != len(sites):
        raise YastnError("Number of operators and sites should match.")

    # unpack operators if operators provided as a Lattice or dict
    operators = [op[site] if not isinstance(op, Tensor) else op for op, site in zip(operators, sites)]

    sign = sign_canonical_order(*operators, sites=sites, f_ordered=env.psi.f_ordered)
    ops = {}
    for n, op in zip(sites, operators):
        ops[n] = ops[n] @ op if n in ops else op

    if opts_var is None:
        opts_var = {'max_sweeps': 2}
    if opts_svd is None:
        rr = env.yrange if dirn == 'lr' else env.xrange
        D_total = max(max(env[i, d].get_bond_dimensions()) for i in range(*rr) for d in dirn)
        opts_svd = {'D_total': D_total}

    if dirn == 'lr':
        i0, i1 = env.yrange[0], env.yrange[1] - 1
        bra = env[i1, 'r'].conj()
        tms = {ny: env[ny, 'v'] for ny in range(*env.yrange)}
        ket = env[i0, 'l']
        dx = env.xrange[0] - env.offset
        tens = {(nx, ny): tm[nx - dx] for ny, tm in tms.items() for nx in range(*env.xrange)}
    else:
        i0, i1 = env.xrange[0], env.xrange[1] - 1
        bra = env[i1, 'b'].conj()
        tms = {nx: env[nx, 'h'] for nx in range(*env.xrange)}
        ket = env[i0, 't']
        dy = env.yrange[0] - env.offset
        tens = {(nx, ny): tm[ny - dy] for nx, tm in tms.items() for ny in range(*env.yrange)}

    val_no = contract_window(bra, tms, ket, i0, i1, opts_svd, opts_var)

    nx0, ny0 = env.xrange[0], env.yrange[0]
    for (nx, ny), op in ops.items():
        set_operator_(tens, (nx, ny), op)
        add_charge_swaps_(tens, (nx, ny), op.n, axes=('b0' if nx == nx0 else 'k1'))
        for ii in range(nx0 + 1, nx):
            add_charge_swaps_(tens, (ii, ny), op.n, axes=['k1', 'k4', 'b3'])
        if nx > nx0:
            add_charge_swaps_(tens, (nx0, ny), op.n, axes=['b0', 'k4', 'b3'])
        for jj in range(ny0, ny):
            add_charge_swaps_(tens, (nx0, jj), op.n, axes=['b0', 'k2', 'k4'])

    val_op = contract_window(bra, tms, ket, i0, i1, opts_svd, opts_var)
    return sign * val_op / val_no


def contract_window(bra, tms, ket, i0, i1, opts_svd, opts_var):
    """ Helper funcion performing mps contraction of < mps0 | mpo mpo ... | mps1 >. """
    vec = ket
    for ny in range(i0, i1):
        vec_next = mps.zipper(tms[ny], vec, opts_svd=opts_svd)
        mps.compression_(vec_next, (tms[ny], vec), method='1site', normalize=False, **opts_var)
        vec = vec_next
    return mps.vdot(bra, tms[i1], vec)
