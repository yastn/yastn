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
from tqdm import tqdm
from .... import Tensor, YastnError, tensordot
from ... import mps
from ._env_auxlliary import identity_tm_boundary
from ._env_measure import _measure_nsite
from .._peps import Peps, Peps2Layers


class EnvBoundaryMPS(Peps):
    r"""
    Boundary MPS class for finite PEPS contraction.
    """

    def __init__(self, psi, opts_svd, setup='l', opts_var=None):
        super().__init__(psi.geometry)
        self.psi = psi
        self._env = {}
        self.offset = 0

        li, ri = 0, psi.Ny-1
        ti, bi = 0, psi.Nx-1

        if 'l' in setup or 'r' in setup:
            tmpo = psi.transfer_mpo(n=li, dirn='v')
            self._env[li, 'l'] = identity_tm_boundary(tmpo)
            tmpo = psi.transfer_mpo(n=ri, dirn='v').T
            self._env[ri, 'r'] = identity_tm_boundary(tmpo)
        if 'b' in setup or 't' in setup:
            tmpo = psi.transfer_mpo(n=ti, dirn='h')
            self._env[ti, 't'] = identity_tm_boundary(tmpo)
            tmpo = psi.transfer_mpo(n=bi, dirn='h').T
            self._env[bi, 'b'] = identity_tm_boundary(tmpo)

        self.info = {}

        if opts_var == None:
            opts_var = {'max_sweeps': 2, 'normalize': False,}

        if 'r' in setup:
            for ny in range(ri-1, li-1, -1):
                tmpo = psi.transfer_mpo(n=ny+1, dirn='v').T
                phi0 = self._env[ny+1, 'r']
                self._env[ny, 'r'], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
                mps.compression_(self._env[ny, 'r'], (tmpo, phi0), **opts_var)
                self.info[ny, 'r'] = {'discarded': discarded}

        if 'l' in setup:
            for ny in range(li+1, ri+1):
                tmpo = psi.transfer_mpo(n=ny-1, dirn='v')
                phi0 = self._env[ny-1, 'l']
                self._env[ny, 'l'], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
                mps.compression_(self._env[ny, 'l'], (tmpo, phi0), **opts_var)
                self.info[ny, 'l'] = {'discarded': discarded}

        if 't' in setup:
            for nx in range(ti+1, bi+1):
                tmpo = psi.transfer_mpo(n=nx-1, dirn='h')
                phi0 = self._env[nx-1, 't']
                self._env[nx, 't'], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
                mps.compression_(self._env[nx, 't'], (tmpo, phi0), **opts_var)
                self.info[nx, 't'] = {'discarded': discarded}

        if 'b' in setup:
            for nx in range(bi-1, ti-1, -1):
                tmpo = psi.transfer_mpo(n=nx+1, dirn='h').T
                phi0 = self._env[nx+1, 'b']
                self._env[nx, 'b'], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
                mps.compression_(self._env[nx, 'b'], (tmpo, phi0), **opts_var)
                self.info[nx, 'b'] = {'discarded': discarded}

    def save_to_dict(self) -> dict:
        r"""
        Serialize EnvBoundaryMPS into a dictionary.
        """
        psi = self.psi
        if isinstance(psi, Peps2Layers):
            psi = psi.ket

        d = {'class': 'EnvBoundaryMPS', 'psi': psi.save_to_dict()}
        d['env'] = {k: v.save_to_dict() for k, v in self._env.items()}
        d['info'] = {k: v.copy() for k, v in self.info.items()}
        return d

    def boundary_mps(self, n, dirn):
        return self._env[n, dirn]

    def __getitem__(self, ind):
        if ind[1] == 'v' or ind[1] == 'h':
            return self.psi.transfer_mpo(n=ind[0], dirn=ind[1])
        return self._env[ind]

    def measure_1site(peps_env, O, site=None):
        """
        Calculate all 1-point expectation values <O_j> in a finite PEPS.

        Takes CTM environments and operators.

        Parameters
        ----------
        O: dict[tuple[int, int], dict[int, operators]]
            mapping sites with list of operators at each site.
        """
        #
        psi = peps_env.psi
        if site is None:
            out = {}
            Nx, Ny = psi.Nx, psi.Ny
            sites = [(nx, ny) for ny in range(Ny-1, -1, -1) for nx in range(Nx)]
            opdict = _clear_operator_input(O, sites)

            for ny in range(Ny):
                bra = peps_env.boundary_mps(n=ny, dirn='r')
                tm = psi.transfer_mpo(n=ny, dirn='v')
                ket = peps_env.boundary_mps(n=ny, dirn='l')
                env = mps.Env(bra.conj(), [tm, ket]).setup_(to='first').setup_(to='last')
                norm_env = env.measure()
                for nx in range(Nx):
                    if (nx, ny) in opdict:
                        for nz, op in opdict[nx, ny].items():
                            tm[nx].set_operator_(op)
                            env.update_env_(nx, to='first')
                            out[(nx, ny) + nz] = env.measure(bd=(nx - 1, nx)) / norm_env
            return out
        # else:
        nx, ny = site
        bra = peps_env.boundary_mps(n=ny, dirn='r')
        tm = psi.transfer_mpo(n=ny, dirn='v')
        ket = peps_env.boundary_mps(n=ny, dirn='l')
        env = mps.Env(bra.conj(), [tm, ket]).setup_(to='first').setup_(to='last')
        norm_env = env.measure()
        if isinstance(O, dict):
            out = {}
            for k, op in O.items():
                tm[nx].set_operator_(op)
                env.update_env_(nx, to='first')
                out[k] = env.measure(bd=(nx - 1, nx)) / norm_env
            return out
        tm[nx].set_operator_(O)
        env.update_env_(nx, to='first')
        return env.measure(bd=(nx - 1, nx)) / norm_env
  

    def measure_nn(peps_env, OP):
        """
        Calculate 2-point expectation values on <O_i P_j> in a finite on NN bonds.

        ----------
        OP: dict[Bond, dict[int, operators]]
            mapping bond with list of two-point operators.
        """
        out = {}

        psi = peps_env.psi

        OPv, OPh = {}, {}
        for bond, ops in OP.items():
            dirn, l_ordered = peps_env.nn_bond_type(bond)
            assert l_ordered
            if dirn == 'h':
                nx = bond[0][0]
                if nx not in OPh:
                    OPh[nx] = {}
                OPh[nx][bond] = ops
            else:
                ny = bond[0][1]
                if ny not in OPv:
                    OPv[ny] = {}
                OPv[ny][bond] = ops

        for ny, bond_ops in OPv.items():
            bra = peps_env.boundary_mps(n=ny, dirn='r')
            tm = psi.transfer_mpo(n=ny, dirn='v')
            ket = peps_env.boundary_mps(n=ny, dirn='l')
            env = mps.Env(bra.conj(), [tm, ket]).setup_(to='first').setup_(to='last')
            norm_env = env.measure()
            for bond, ops in sorted(bond_ops.items()):
                s0, s1 = bond
                nx = s0[0]
                for nz, (op1, op2) in ops.items():
                    tm[nx].set_operator_(op1)
                    tm[nx+1].set_operator_(op2)
                    env.update_env_(nx+1, to='first')
                    env.update_env_(nx, to='first')
                    tm[nx].del_operator_()
                    tm[nx+1].del_operator_()
                    out[s0, s1, nz] = env.measure(bd=(nx - 1, nx)) / norm_env

        for nx, bond_ops in OPh.items():
            bra = peps_env.boundary_mps(n=nx, dirn='b')
            tm = psi.transfer_mpo(n=nx, dirn='h')
            ket = peps_env.boundary_mps(n=nx, dirn='t')
            env = mps.Env(bra.conj(), [tm, ket]).setup_(to='first').setup_(to='last')
            norm_env = env.measure()
            for bond, ops in sorted(bond_ops.items()):
                s0, s1 = bond
                ny = s0[1]
                for nz, (op1, op2) in ops.items():
                    tm[ny].set_operator_(op1)
                    tm[ny+1].set_operator_(op2)
                    env.update_env_(ny+1, to='first')
                    env.update_env_(ny, to='first')
                    tm[ny].del_operator_()
                    tm[ny+1].del_operator_()
                    out[s0, s1, nz] = env.measure(bd=(ny - 1, ny)) / norm_env

        return out

    def measure_nsite(self, *operators, sites=None) -> float:
        r"""
        Calculate expectation value of a product of local operators.

        Fermionic strings are incorporated for fermionic operators by employing :meth:`yastn.swap_gate`.

        Parameters
        ----------
        operators: Sequence[yastn.Tensor]
            List of local operators to calculate <O0_s0 O1_s1 ...>.

        sites: Sequence[int]
            A list of sites [s0, s1, ...] matching corresponding operators.
        """
        self.xrange = (0, self.psi.Nx) # (min(site[0] for site in sites), max(site[0] for site in sites) + 1)
        self.yrange = (min(site[1] for site in sites), max(site[1] for site in sites) + 1)
        dirn = 'lr'
        return _measure_nsite(self, *operators, sites=sites, dirn=dirn)


    def measure_2site(peps_env, O, P, opts_svd, opts_var=None):
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
        op1dict = _clear_operator_input(O, sites)
        op2dict = _clear_operator_input(P, sites)

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

    def sample(peps_env, projectors, number=1, opts_svd=None, opts_var=None, progressbar=False, return_probabilities=False):
        """
        Sample a random configuration from a finite PEPS.

        Takes  CTM environments and a complete list of projectors to sample from.
        """
        psi = peps_env.psi
        config = psi[0, 0].config
        rands = (config.backend.rand(psi.Nx * psi.Ny * number) + 1) / 2

        # change each list of projectors into keys and projectors
        projs_sites = {}
        for k, v in projectors.items():
            projs_sites[k] = dict(v) if isinstance(v, dict) else dict(enumerate(v))
            for k, pr in projs_sites[k].items():
                if pr.ndim == 1:  # vectors need conjugation
                    if abs(pr.norm() - 1) > 1e-10:
                        raise YastnError("Local states to project on should be normalized.")
                    projs_sites[k] = tensordot(pr, pr.conj(), axes=((), ()))
                elif pr.ndim == 2:
                    if (pr.n != pr.config.sym.zero()) or abs(pr @ pr - pr).norm() > 1e-10:
                        raise YastnError("Matrix projectors should be projectors, P @ P == P.")
                else:
                    raise YastnError("Projectors should consist of vectors (ndim=1) or matrices (ndim=2).")

        out = {site: [] for site in peps_env.sites()}
        probabilities = []
        count = 0

        for _ in tqdm(range(number), desc="Sample...", disable=not progressbar):
            probability = 1.

            vR = peps_env.boundary_mps(n=psi.Ny-1, dirn='r') # right boundary of indexed column through CTM environment tensors
            for ny in range(psi.Ny - 1, -1, -1):
                Os = psi.transfer_mpo(n=ny, dirn='v').T  # converts ny column of PEPS to MPO
                vL = peps_env.boundary_mps(n=ny, dirn='l')  # left boundary of indexed column through CTM environment tensors
                env = mps.Env(vL.conj(), [Os, vR]).setup_(to = 'first')
                for nx in range(0, psi.Nx):
                    acc_prob = 0
                    norm_prob = env.measure(bd=(nx - 1, nx)).real
                    for k, pr in projs_sites[(nx, ny)].items():
                        Os[nx].set_operator_(pr)
                        env.update_env_(nx, to='last')
                        prob = env.measure(bd=(nx, nx+1)).real / norm_prob
                        acc_prob += prob 
                        if rands[count] < acc_prob:
                            probability *= prob
                            out[nx, ny].append(k)
                            Os[nx].set_operator_(pr / prob)
                            break
                    env.update_env_(nx, to='last')
                    count += 1

                if opts_svd is None:
                    opts_svd = {'D_total': max(vL.get_bond_dimensions())}

                vRnew = mps.zipper(Os, vR, opts_svd=opts_svd)
                if opts_var is None:
                    opts_var = {}

                mps.compression_(vRnew, (Os, vR), method='1site', **opts_var)
                vR = vRnew
            probabilities.append(probability)

        if return_probabilities:
            return out, probabilities
        return out


    def sample2(peps_env, projectors, opts_svd=None, opts_var=None):
        """
        Sample a random configuration from a finite PEPS.

        Takes  CTM environments and a complete list of projectors to sample from.
        """
        psi = peps_env.psi
        config = psi[0, 0].config
        rands = (config.backend.rand(psi.Nx * psi.Ny) + 1) / 2

        # change each list of projectors into keys and projectors
        projs_sites = {}
        for k, v in projectors.items():
            if isinstance(v, dict):
                projs_sites[k, 'k'] = list(v.keys())
                projs_sites[k, 'p'] = list(v.values())
            else:
                projs_sites[k, 'k'] = list(range(len(v)))
                projs_sites[k, 'p'] = v

            for j, pr in enumerate(projs_sites[k, 'p']):
                if pr.ndim == 1:  # vectors need conjugation
                    if abs(pr.norm() - 1) > 1e-10:
                        raise YastnError("Local states to project on should be normalized.")
                    projs_sites[k, 'p'][j] = tensordot(pr, pr.conj(), axes=((), ()))
                elif pr.ndim == 2:
                    if (pr.n != pr.config.sym.zero()) or abs(pr @ pr - pr).norm() > 1e-10:
                        raise YastnError("Matrix projectors should be projectors, P @ P == P.")
                else:
                    raise YastnError("Projectors should consist of vectors (ndim=1) or matrices (ndim=2).")

        out = {}
        probability = 1.0
        count = 0
        vR = peps_env.boundary_mps(n=psi.Ny-1, dirn='r') # right boundary of indexed column through CTM environment tensors

        for ny in range(psi.Ny - 1, -1, -1):

            Os = psi.transfer_mpo(n=ny, dirn='v').T  # converts ny column of PEPS to MPO
            vL = peps_env.boundary_mps(n=ny, dirn='l')  # left boundary of indexed column through CTM environment tensors

            env = mps.Env(vL.conj(), [Os, vR]).setup_(to = 'first')

            for nx in range(0, psi.Nx):
                prob = []
                norm_prob = env.measure(bd=(nx - 1, nx)).real
                for pr in projs_sites[(nx, ny), 'p']:
                    Os[nx].set_operator_(pr)
                    env.update_env_(nx, to='last')
                    prob.append(env.measure(bd=(nx, nx+1)).real / norm_prob)

                assert abs(sum(prob) - 1) < 1e-12
                rand = rands[count]
                ind = sum(apr < rand for apr in accumulate(prob))
                out[nx, ny] = projs_sites[(nx, ny), 'k'][ind]
                probability *= prob[ind]
                Os[nx].set_operator_(projs_sites[(nx, ny), 'p'][ind] / prob[ind]) # updated with the new collapse
                env.update_env_(nx, to='last')
                count += 1

            if opts_svd is None:
                opts_svd = {'D_total': max(vL.get_bond_dimensions())}

            vRnew = mps.zipper(Os, vR, opts_svd=opts_svd)
            if opts_var is None:
                opts_var = {}

            mps.compression_(vRnew, (Os, vR), method='1site', **opts_var)
            vR = vRnew
        
        out['probability'] = probability
        return out

    def sample_MC_(proj_env, st0, st1, st2, psi, projectors, opts_svd, opts_var, trial="local"):
        """
        Monte Carlo steps in a finite peps. Makes two steps
        while sweeping finite lattice back and forth.

        Takes environments and a complete list of projectors to sample from.

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
                proj_env._env[ny-1, 'r'] = vRnew
                proj_env._env.pop((ny, 'l'))

        for ny in range(Ny):
            _, Os, vL, astep = _sample_MC_column(ny, proj_env, st1, st2, psi, projectors, rands)
            accept += astep
            if ny < Ny - 1:
                OsT = Os.T
                vLnew = mps.zipper(OsT, vL, opts_svd=opts_svd)
                mps.compression_(vLnew, (OsT, vL), method='1site', **opts_var)
                proj_env._env[ny+1, 'l'] = vLnew
                proj_env._env.pop((ny, 'r'))

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
    Os = proj_env.psi.transfer_mpo(n=ny, dirn='v').T
    vL = proj_env.boundary_mps(n=ny, dirn='l')
    env = mps.Env(vL.conj(), [Os, vR]).setup_(to='first')
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
    Os = proj_env.psi.transfer_mpo(n=ny, dirn='v').T
    vL = proj_env.boundary_mps(n=ny, dirn='l')
    env = mps.Env(vL.conj(), [Os, vR]).setup_(to='first')
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
