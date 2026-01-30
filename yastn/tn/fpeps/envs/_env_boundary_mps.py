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

from ._env_contractions import identity_boundary
from ._env_window import _measure_nsite, _measure_2site, _sample
from .._gates_auxiliary import clear_operator_input
from .._peps import PEPS_CLASSES, Peps2Layers
from ... import mps
from ....tensor import YastnError


class EnvBoundaryMPS():
    r"""
    Boundary MPS class for finite PEPS contraction.
    """

    def __init__(self, psi, opts_svd, setup='r', opts_var=None):
        r"""
        Calculate boundary MPSs for finite PEPS.

        Consequative MPS follows from contracting transfer matrix with previous MPS.
        This employs :meth:`yastn.tn.mps.zipper` followed by :meth:`yastn.tn.mps.compression_` for refinement.

        Parameters
        ----------
        psi: fpeps.Peps
            Finite PEPS to be contracted.

        opts_svd: dict
            Passed to :meth:`yastn.tn.mps.zipper` and :meth:`yastn.tn.mps.compression_` (if ``method="2site"`` is used in ``opts_var``)
            Controls bond dimensions of boundary MPSs.

        setup: str
            String containing directions from which the square lattice is contracted, consisting of characters "l", "r", "t", "b".
            E.g., setup="lr" would calculate boundary MPSs from the left and from the right sites of the lattice. The default is "l".

        opts_var: dict
            Options passed to :meth:`yastn.tn.mps.compression_`. The default is ``None`` which sets opts_var={max_sweeps: 2, normalization: False}.
        """
        self.geometry = psi.geometry
        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "f_ordered", "nn_bond_dirn"]:
            setattr(self, name, getattr(self.geometry, name))

        self.psi = psi
        self._env = {}

        self.offset = 0

        ti, bi = 0, psi.Nx-1
        li, ri = 0, psi.Ny-1

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
                self._update_boundary_(ny, 'r', opts_svd, opts_var)

        if 'l' in setup:
            for ny in range(li+1, ri+1):
                self._update_boundary_(ny, 'l', opts_svd, opts_var)

        if 't' in setup:
            for nx in range(ti+1, bi+1):
                self._update_boundary_(nx, 't', opts_svd, opts_var)
            if self.boundary == 'cylinder':  # attempt to converge iteratively
                psi0 = self._env[bi, 't']
                for _ in range(100):
                    for nx in range(ti, bi+1):
                        self._update_boundary_(nx, 't', opts_svd, opts_var)
                    if (self._env[bi, 't'] - psi0).norm() < 1e-6:
                        break
                    psi0 = self._env[bi, 't']

        if 'b' in setup:
            for nx in range(bi-1, ti-1, -1):
                self._update_boundary_(nx, 'b', opts_svd, opts_var)
            if self.boundary == 'cylinder':  # attempt to converge iteratively
                psi0 = self._env[ti, 'b']
                for _ in range(100):
                    for nx in range(bi, ti-1, -1):
                        self._update_boundary_(nx, 'b', opts_svd, opts_var)
                    if (self._env[ti, 'b'] - psi0).norm() < 1e-6:
                        break
                    psi0 = self._env[ti, 'b']

    def _update_boundary_(self, n, dirn, opts_svd, opts_var):
        if dirn == 'r':
            tmpo = self.psi.transfer_mpo(n=n+1, dirn='v').T
            phi0 = self._env[n+1, 'r']
        if dirn == 'l':
            tmpo = self.psi.transfer_mpo(n=n-1, dirn='v')
            phi0 = self._env[n-1, 'l']
        if dirn == 't':
            nx = (n - 1) % self.Nx
            tmpo = self.psi.transfer_mpo(n=nx, dirn='h')
            phi0 = self._env[nx, 't']
        if dirn == 'b':
            nx = (n + 1) % self.Nx
            tmpo = self.psi.transfer_mpo(n=nx, dirn='h').T
            phi0 = self._env[nx, 'b']
        self._env[n, dirn], discarded = mps.zipper(tmpo, phi0, opts_svd, return_discarded=True)
        mps.compression_(self._env[n, dirn], (tmpo, phi0), **opts_var, opts_svd=opts_svd)
        self.info[n, dirn] = {'discarded': discarded}

    def to_dict(self, level=2):
        r"""
        Serialize EnvBoundaryMPS to a dictionary.
        Complementary function is :meth:`yastn.EnvBoundaryMPS.from_dict` or a general :meth:`yastn.from_dict`.
        See :meth:`yastn.Tensor.to_dict` for further description.
        """
        return {'type': type(self).__name__,
                'dict_ver': 1,
                'psi': self.psi.to_dict(level=level),
                'env': {k: v.to_dict(level=level) for k, v in self._env.items()},
                'info': {k: v.copy() for k, v in self.info.items()}
                }

    @classmethod
    def from_dict(cls, d, config=None):
        r"""
        De-serializes EnvBoundaryMPS from the dictionary ``d``.
        See :meth:`yastn.Tensor.from_dict` for further description.
        """
        if 'dict_ver' not in d:
            psi = PEPS_CLASSES['Peps'].from_dict(d['psi'], config)
        elif d['dict_ver'] == 1:
            psi = PEPS_CLASSES[d['psi']['type']].from_dict(d['psi'], config=config)
        env = EnvBoundaryMPS(psi, opts_svd={}, setup='')
        for k, v in d['env'].items():
            env._env[k] = mps.MpsMpoOBC.from_dict(v, config)
        for k, v in d['info'].items():
            env.info[k] = v.copy()
        return env

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
        n, dirn = ind
        if dirn in 'tbh':
            n = n % self.Nx
        if dirn in ['h', 'v']:
            return self.psi.transfer_mpo(n=n, dirn=dirn)
        return self._env[n, dirn]

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
            opdict = clear_operator_input(O, sites)

            for ny in range(Ny):
                bra = peps_env.boundary_mps(n=ny, dirn='r')
                tm = psi.transfer_mpo(n=ny, dirn='v')
                ket = peps_env.boundary_mps(n=ny, dirn='l')
                env = mps.Env(bra.conj(), [tm, ket]).setup_(to='first').setup_(to='last')
                norm_env = env.measure()
                for nx in range(Nx):
                    if (nx, ny) in opdict:
                        for nz, op in opdict[nx, ny].items():
                            if op.ndim == 2:
                                tm[nx].set_operator_(op)
                            else:  # for a single-layer Peps, replace with new peps tensor
                                tm[nx] = op.transpose(axes=(0, 3, 2, 1))
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
                if op.ndim == 2:
                    tm[nx].set_operator_(op)
                else:  # for a single-layer Peps, replace with new peps tensor
                    tm[nx] = op.transpose(axes=(0, 3, 2, 1))
                env.update_env_(nx, to='first')
                out[k] = env.measure(bd=(nx - 1, nx)) / norm_env
            return out
        if O.ndim == 2:
            tm[nx].set_operator_(O)
        else:  # for a single-layer Peps, replace with new peps tensor
            tm[nx] = O.transpose(axes=(0, 3, 2, 1))
        env.update_env_(nx, to='first')
        return env.measure(bd=(nx - 1, nx)) / norm_env

    def measure_nn(peps_env, OP, P=None):
        """
        Calculate 2-point expectation values on <O_i P_j> in a finite on NN bonds.

        ----------
        OP: dict[Bond, dict[int, Tensor]]
            mapping bond with list of two-point operators.

        P: Tensor
            For the same Tensor O and P on all bonds, can provide both as Tensors.
            The default is None, where OP is a dict.
        """
        out = {}
        psi = peps_env.psi

        if psi.boundary != "obc":
            raise YastnError("EnvBoundaryMPS.measure_nn currently supports only open boundary conditions.")

        if P is not None:
            OP = {bond: (OP, P) for bond in psi.bonds()}
        OP = {k: {(): v} if not isinstance(v, dict) else v for k, v in OP.items()}

        OPv, OPh = {}, {}
        for bond, ops in OP.items():
            dirn = peps_env.nn_bond_dirn(*bond)
            assert dirn in ('lr', 'tb')
            if dirn == 'lr':
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
                    tm[nx + 1].set_operator_(op2)
                    env.update_env_(nx + 1, to='first')
                    env.update_env_(nx, to='first')
                    tm[nx].del_operator_()
                    tm[nx + 1].del_operator_()
                    out[(s0, s1) + nz] = env.measure(bd=(nx - 1, nx)) / norm_env

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
                    tm[ny + 1].set_operator_(op2)
                    env.update_env_(ny + 1, to='first')
                    env.update_env_(ny, to='first')
                    tm[ny].del_operator_()
                    tm[ny + 1].del_operator_()
                    out[(s0, s1) + nz] = env.measure(bd=(ny - 1, ny)) / norm_env

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

    def measure_2site(self, O, P, xrange=None, yrange=None, pairs='corner <=', dirn='v', opts_svd=None, opts_var=None):
        r"""
        Calculate expectation values :math:`\langle \textrm{O}_i \textrm{P}_j \rangle`
        of local operators :code:`O` and :code:`P` for pairs of lattice sites :math:`i, j`.

        Parameters
        ----------
        O, P: yastn.Tensor
            one-site operators. It is possible to provide a dict of :class:`yastn.tn.fpeps.Lattice` object
            mapping operators to sites.
            For each site, it is possible to provide a list or dict of operators, where the expectation value is calculated
            for each combination of those operators

        xrange: None | tuple[int, int]
            range of rows forming a window, [r0, r1); r0 included, r1 excluded.
            For None, takes a single unit cell of the lattice.

        yrange: tuple[int, int]
            range of columns forming a window.
            For None, takes a single unit cell of the lattice.

        pairs: str | list[tuple[tule[int, int], tuple[int, int]]]
            Limits the pairs of sites to calculate the expectation values.
            If 'corner' in pairs, O is limited to top-left corner of the lattice
            If 'row' in pairs, O is limited to top row of the lattice

        dirn: str
            'h' or 'v', where the boundary MPSs used for truncation are, respectively, horizontal or vertical.
            The default is 'v'.

        opts_svd: dict
            Options passed to :meth:`yastn.linalg.svd` used to truncate virtual spaces of boundary MPSs used in sampling.
            The default is ``None``, in which case take ``D_total`` as the largest dimension from CTM environment.

        opts_svd: dict
            Options passed to :meth:`yastn.tn.mps.compression_` used in the refining of boundary MPSs.
            The default is ``None``, in which case make 2 variational sweeps.
        """
        if xrange is None:
            xrange = [0, self.Nx]
        if yrange is None:
            yrange = [0, self.Ny]
        offset = yrange[0] if dirn == 'h' else xrange[0]
        return _measure_2site(self, O, P, xrange, yrange, offset=offset, pairs=pairs, dirn=dirn, opts_svd=opts_svd, opts_var=opts_var)

    def sample(env, projectors, xrange=None, yrange=None, dirn='v', number=1, opts_svd=None, opts_var=None, progressbar=False, return_probabilities=False, flatten_one=True, **kwargs):
        r"""
        Sample random configurations from PEPS.
        Output a dictionary linking sites with lists of sampled projectors` keys for each site.
        Projectors should be summing up to identity -- this is not checked.

        Parameters
        ----------
        projectors: Dict[Any, yast.Tensor] | Sequence[yast.Tensor] | Dict[Site, Dict[Any, yast.Tensor]]
            Projectors to sample from. We can provide a dict(key: projector), where the sampled results will be given as keys,
            and the same set of projectors is used at each site. For a list of projectors, the keys follow from enumeration.
            Finally, we can provide a dictionary between each site and sets of projectors.

        xrange: None | tuple[int, int]
            range of rows forming a window, [r0, r1); r0 included, r1 excluded.
            For None, takes a single unit cell of the lattice, which is the default.

        yrange: None | tuple[int, int]
            range of columns forming a window.
            For None, takes a single unit cell of the lattice, which is the default.

        dirn: str
            'h' or 'v', where the boundary MPSs used for truncation are, respectively, horizontal or vertical.
            The default is 'v'.

        number: int
            Number of independent samples.

        progressbar: bool
            Whether to display progressbar. The default is ``False``.

        return_probabilities: bool
            Whether to return a tuple (samples, probabilities). The default is ``False``, where a dict samples is returned.

        flatten_one: bool
            Whether, for number==1, pop one-element lists for each lattice site to return samples={site: ind, } instead of {site: [ind]}.
            The default is ``True``.
        """
        if xrange is None:
            xrange = [0, env.Nx]
        if yrange is None:
            yrange = [0, env.Ny]
        offset = yrange[0] if dirn == 'h' else xrange[0]
        return _sample(env, projectors, xrange, yrange, offset=offset, dirn=dirn,
                number=number, opts_svd=opts_svd, opts_var=opts_var,
                progressbar=progressbar, return_probabilities=return_probabilities, flatten_one=flatten_one)

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
        rands = iter(config.backend.rand(2 * Nx * Ny))  # in [0, 1]

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


def _sample_MC_column_local(ny, proj_env, st0, st1, psi, projectors, rands):
    # update is proposed based on local probabilies
    vR = proj_env.boundary_mps(n=ny, dirn='r')
    Os = proj_env.psi.transfer_mpo(n=ny, dirn='v').T
    vL = proj_env.boundary_mps(n=ny, dirn='l')
    env = mps.Env(vL.conj(), [Os, vR]).setup_(to='first')
    for nx in range(psi.Nx):
        amp = env.hole(nx).tensordot(psi[nx, ny], axes=((0, 1, 2, 3), (0, 1, 2, 3)))
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

        A0 = (A @ pr0)
        A1 = (A @ pr1)

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


def identity_tm_boundary(tmpo):
    """
    For transfer matrix MPO build of DoublePepsTensors,
    create MPS that contracts each DoublePepsTensor from the right.
    """
    phi = mps.Mps(N=tmpo.N)
    config = tmpo.config
    for n in phi.sweep(to='last'):
        legf = tmpo[n].get_legs(axes=3).conj()
        tmp = identity_boundary(config, legf)
        phi[n] = tmp.add_leg(0, s=-1).add_leg(2, s=1)
    return phi
