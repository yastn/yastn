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
from itertools import accumulate
from .... import Tensor, YastnError
from ... import mps
from ._env_auxlliary import identity_tm_boundary
from .._gates_auxiliary import apply_gate_onsite


class EnvBoundaryMps:
    r"""
    Geometric information about the lattice provided to ctm tensors
    """

    def __init__(self, psi, opts_svd, setup='l', opts_var=None):
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

    def boundary_mps(self, n, dirn):
        return self._env[dirn, n]


    def measure_1site(peps_env, op):
        """
        Calculate all 1-point expectation values <o> in a finite peps.

        Takes CTM emvironments and operators.

        o1 are given as dict[tuple[int, int], dict[int, operators]],
        mapping sites with list of operators at each site.
        """
        out = {}

        psi = peps_env.psi
        Nx, Ny = psi.Nx, psi.Ny
        sites = [(nx, ny) for ny in range(Ny-1, -1, -1) for nx in range(Nx)]
        opdict = _clear_operator_input(op, sites)

        for ny in range(Ny-1, -1, -1):
            vR = peps_env.boundary_mps(n=ny, dirn='r')
            vL = peps_env.boundary_mps(n=ny, dirn='l')
            Os = psi.transfer_mpo(n=ny, dirn='v')
            env = mps.Env(vL, [Os, vR]).setup_(to='first').setup_(to='last')
            norm_env = env.measure()
            for nx in range(Nx):
                if (nx, ny) in opdict:
                    Osnx1A = Os[nx].top
                    for nz, o in opdict[nx, ny].items():
                        Os[nx].top = apply_gate_onsite(Osnx1A, o)
                        env.update_env_(nx, to='first')
                        out[(nx, ny) + nz] = env.measure(bd=(nx-1, nx)) / norm_env
        return out


    def measure_2site(peps_env, op1, op2, opts_svd, opts_var=None):
        """
        Calculate all 2-point correlations <o1 o2> in a finite peps.

        Takes CTM emvironments and operators.

        o1 and o2 are given as dict[tuple[int, int], dict[int, operators]],
        mapping sites with list of operators at each site.
        """
        out = {}
        if opts_var is None:
            opts_var =  {'max_sweeps': 2}

        psi = peps_env.psi
        Nx, Ny = psi.Nx, psi.Ny
        sites = [(nx, ny) for ny in range(Ny-1, -1, -1) for nx in range(Nx)]
        op1dict = _clear_operator_input(op1, sites)
        op2dict = _clear_operator_input(op2, sites)

        for nx1, ny1 in sites:
            # print( f"Correlations from {nx1} {ny1} ... ")
            for nz1, o1 in op1dict[nx1, ny1].items():
                vR = peps_env.boundary_mps(n=ny1, dirn='r')
                vL = peps_env.boundary_mps(n=ny1, dirn='l')
                Os = psi.transfer_mpo(n=ny1, dirn='v')
                env = mps.Env(vL, [Os, vR]).setup_(to='first').setup_(to='last')
                norm_env = env.measure(bd=(-1, 0))

                if ny1 > 0:
                    vRnext = mps.zipper(Os, vR, opts_svd=opts_svd)
                    mps.compression_(vRnext, (Os, vR), method='1site', normalize=False, **opts_var)

                # first calculate on-site correlations
                Osnx1A = Os[nx1].top
                for nz2, o2 in op2dict[nx1, ny1].items():
                    Os[nx1].top = apply_gate_onsite(Osnx1A, o1 @ o2)
                    env.update_env_(nx1, to='last')
                    out[(nx1, ny1) + nz1, (nx1, ny1) + nz2] = env.measure(bd=(nx1, nx1+1)) / norm_env

                Os[nx1].top = apply_gate_onsite(Osnx1A, o1)
                env.setup_(to='last')

                if ny1 > 0:
                    vRo1next = mps.zipper(Os, vR, opts_svd=opts_svd)
                    mps.compression_(vRo1next, (Os, vR), method='1site', normalize=False, **opts_var)

                # calculate correlations along the row
                for nx2 in range(nx1 + 1, Nx):
                    Osnx2A = Os[nx2].top
                    for nz2, o2 in op2dict[nx2, ny1].items():
                        Os[nx2].top = apply_gate_onsite(Osnx2A, o2)
                        env.update_env_(nx2, to='first')
                        out[(nx1, ny1) + nz1, (nx2, ny1) + nz2] = env.measure(bd=(nx2-1, nx2)) / norm_env

                # and all subsequent rows
                for ny2 in range(ny1-1, -1, -1):
                    vR = vRnext
                    vRo1 = vRo1next
                    vL = peps_env.boundary_mps(n=ny2, dirn='l')
                    Os = psi.transfer_mpo(n=ny2, dirn='v')
                    env = mps.Env(vL, [Os, vR]).setup_(to='first')
                    norm_env = env.measure(bd=(-1, 0))

                    if ny2 > 0:
                        vRnext = mps.zipper(Os, vR, opts_svd=opts_svd)
                        mps.compression_(vRnext, (Os, vR), method='1site', normalize=False, **opts_var)
                        vRo1next = mps.zipper(Os, vRo1, opts_svd=opts_svd)
                        mps.compression_(vRo1next, (Os, vRo1), method='1site', normalize=False, **opts_var)

                    env = mps.Env(vL, [Os, vRo1]).setup_(to='first').setup_(to='last')
                    for nx2 in range(psi.Nx):
                        Osnx2A = Os[nx2].top
                        for nz2, o2 in op2dict[nx2, ny2].items():
                            Os[nx2].top = apply_gate_onsite(Osnx2A, o2)
                            env.update_env_(nx2, to='first')
                            out[(nx1, ny1) + nz1, (nx2, ny2) + nz2] = env.measure(bd=(nx2-1, nx2)) / norm_env
        return out

    def sample(peps_env, projectors, opts_svd=None, opts_var=None):
        """
        Sample a random configuration from a finite peps.

        Takes  CTM emvironments and a complete list of projectors to sample from.
        """
        psi = peps_env.psi
        config = psi[0, 0].config
        rands = (config.backend.rand(psi.Nx * psi.Ny) + 1) / 2

        out = {}
        count = 0
        vR = peps_env.boundary_mps(n=psi.Ny-1, dirn='r') # right boundary of indexed column through CTM environment tensors

        for ny in range(psi.Ny - 1, -1, -1):

            Os = psi.transfer_mpo(n=ny, dirn='v') # converts ny colum of PEPS to MPO
            vL = peps_env.boundary_mps(n=ny, dirn='l') # left boundary of indexed column through CTM environment tensors

            env = mps.Env(vL, [Os, vR]).setup_(to = 'first')

            for nx in range(0, psi.Nx):
                dpt = Os[nx].copy()
                loc_projectors = projectors[nx, ny]
                prob = []
                norm_prob = env.measure(bd=(nx - 1, nx))
                for proj in loc_projectors:
                    dpt_pr = dpt.copy()
                    dpt_pr.top = apply_gate_onsite(dpt_pr.top, proj)
                    Os[nx] = dpt_pr
                    env.update_env_(nx, to='last')
                    prob.append(env.measure(bd=(nx, nx+1)) / norm_prob)

                assert abs(sum(prob) - 1) < 1e-12
                rand = rands[count]
                ind = sum(apr < rand for apr in accumulate(prob))
                out[(nx, ny)] = ind
                dpt.top = apply_gate_onsite(dpt.top, loc_projectors[ind])
                Os[nx] = dpt  # updated with the new collapse
                env.update_env_(nx, to='last')
                count += 1

            if opts_svd is None:
                opts_svd = {'D_total': max(vL.get_bond_dimensions())}

            vRnew = mps.zipper(Os, vR, opts_svd=opts_svd)
            if opts_var is None:
                opts_var = {}

            mps.compression_(vRnew, (Os, vR), method='1site', **opts_var)
            vR = vRnew
        return out

    def sample_MC_(proj_env, st0, st1, st2, psi, projectors, opts_svd, opts_var, trial="local"):
        """
        MC steps in a finite peps. Makes two steps
        while sweeping finite lattice back and forth.

        Takes emvironments and a complete list of projectors to sample from.

        proj_env, st1, st2 are updated in place
        """

        if trial == "local":
            _sample_MC_column = _sample_MC_column_local
        elif trial == "uniform":
            _sample_MC_column = _sample_MC_column_uniform
        else:
            raise YastnError(f"{trial=} not supported.")

        Nx, Ny = psi.Nx, psi.Ny
        config = psi[0, 0].config
        # pre-draw uniformly distributed random numbers as iterator;
        rands = iter((config.backend.rand(2 * Nx * Ny) + 1) / 2)  # in [0, 1]

        # sweep though the lattice
        accept = 0
        for ny in range(Ny-1, -1, -1):
            vR, Os, _, astep = _sample_MC_column(ny, proj_env, st0, st1, psi, projectors, rands)
            accept += astep
            if ny > 0:
                vRnew = mps.zipper(Os, vR, opts_svd=opts_svd)
                mps.compression_(vRnew, (Os, vR), method='1site', **opts_var)
                proj_env._env['r', ny-1] = vRnew
                proj_env._env.pop(('l', ny))

        for ny in range(Ny):
            _, Os, vL, astep = _sample_MC_column(ny, proj_env, st1, st2, psi, projectors, rands)
            accept += astep
            if ny < Ny - 1:
                OsT = Os.H
                vLnew = mps.zipper(OsT, vL, opts_svd=opts_svd)
                mps.compression_(vLnew, (OsT, vL), method='1site', **opts_var)
                proj_env._env['l', ny+1] = vLnew
                proj_env._env.pop(('r', ny))

        return accept / (2 * Nx * Ny)  # acceptance rate


def _clear_operator_input(op, sites):
    op_dict = op.copy() if isinstance(op, dict) else {site: op for site in sites}
    for k, v in op_dict.items():
        if isinstance(v, dict):
            op_dict[k] = {(i,): vi for i, vi in v.items()}
        elif isinstance(v, Tensor):
            op_dict[k] = {(): v}
        else: # is iterable
            op_dict[k] = {(i,): vi for i, vi in enumerate(v)}
    return op_dict


def _sample_MC_column_local(ny, proj_env, st0, st1, psi, projectors, rands):
    # update is proposed based on local probabilies
    vR = proj_env.boundary_mps(n=ny, dirn='r')
    Os = proj_env.psi.transfer_mpo(n=ny, dirn='v')
    vL = proj_env.boundary_mps(n=ny, dirn='l')
    env = mps.Env(vL, [Os, vR]).setup_(to='first')
    for nx in range(psi.Nx):
        amp = env.hole(nx).tensordot(psi[nx, ny], axes=((0, 1), (0, 1)))
        prob = [abs(amp.vdot(pr, conj=(0, 0))) ** 2 for pr in projectors[nx, ny]]
        sumprob = sum(prob)
        prob = [x / sumprob for x in prob]
        rand = next(rands)
        ind = sum(x < rand for x in accumulate(prob))
        st1[nx, ny] = ind
        proj_env.psi[nx, ny] = (psi[nx, ny] @ projectors[nx, ny][ind])
        Os[nx] = proj_env.psi[nx, ny]
        env.update_env_(nx, to='last')
    accept = psi.Nx
    return vR, Os, vL, accept


def _sample_MC_column_uniform(ny, proj_env, st0, st1, psi, projectors, rands):
    # update is proposed from uniform local distribution
    config = proj_env.psi[0, 0].config
    accept = 0
    vR = proj_env.boundary_mps(n=ny, dirn='r')
    Os = proj_env.psi.transfer_mpo(n=ny, dirn='v')
    vL = proj_env.boundary_mps(n=ny, dirn='l')
    env = mps.Env(vL, [Os, vR]).setup_(to='first')
    for nx in range(psi.Nx):
        A = psi[nx, ny]
        ind0 = st0[nx, ny]

        ind1 = config.backend.randint(0, len(projectors[nx, ny]))

        pr0 = projectors[nx, ny][ind0]
        pr1 = projectors[nx, ny][ind1]

        A0 = (A @ pr0).unfuse_legs(axes=(0, 1))
        A1 = (A @ pr1).unfuse_legs(axes=(0, 1))

        Os[nx] = A1
        env.update_env_(nx, to='last')
        prob_new = abs(env.measure(bd=(nx, nx+1))) ** 2

        Os[nx] = A0
        env.update_env_(nx, to='last')
        prob_old = abs(env.measure(bd=(nx, nx+1))) ** 2

        if next(rands) < prob_new / prob_old:  # accept
            accept += 1
            st1[nx, ny] = ind1
            proj_env.psi[nx, ny] = A1
            Os[nx] = A1
            env.update_env_(nx, to='last')
        else:  # reject
            st1[nx, ny] = ind0
    return vR, Os, vL, accept
