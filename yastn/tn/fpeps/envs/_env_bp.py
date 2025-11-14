# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
from itertools import pairwise
from typing import NamedTuple
from warnings import warn

from tqdm import tqdm

from ._env_auxlliary import *
from ._env_auxlliary import clear_projectors
from ._env_boundary_mps import _clear_operator_input
from ._env_dataclasses import EnvBP_local
from .._evolution import BipartiteBondMetric, BondMetric
from .._gates_auxiliary import fkron, gate_fix_swap_gate, match_ancilla
from .._geometry import Bond, Site, Lattice
from .._peps import Peps2Layers, DoublePepsTensor, PEPS_CLASSES
from ....initialize import eye
from ....tensor import YastnError, tensordot, vdot, ncon, Tensor


class BP_out(NamedTuple):
    sweeps: int = 0
    max_diff: float = None
    converged: bool = False


class EnvBP():

    def __init__(self, psi, init='eye', tol_positive=1e-12, which="BP"):
        r"""
        Environment used in belief propagation contraction scheme.

        Parameters
        ----------
        psi: yastn.tn.Peps
            PEPS lattice to be contracted using BP.
            If ``psi`` has physical legs, a double-layer PEPS with no physical legs is formed.

        init: str
            None, 'eye'. Initialization scheme, see :meth:`yastn.tn.fpeps.EnvBP.reset_`.

        which: str
            Type of environment from 'BP', 'NN+BP', 'NNN+BP'
        """
        self.geometry = psi.geometry
        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "f_ordered", "nn_bond_dirn"]:
            setattr(self, name, getattr(self.geometry, name))

        self.psi = Peps2Layers(psi) if psi.has_physical() else psi
        self.env = Lattice(self.geometry, objects={site: EnvBP_local() for site in self.sites()})
        self.tol_positive = tol_positive
        self._set_which(which)

        if init not in (None, 'eye'):
            raise YastnError(f"EnvBP {init=} not recognized. Should be 'eye' or None.")
        if init is not None:
            self.reset_(init=init)

    def _get_which(self):
        return self._which

    def _set_which(self, which):
        if which not in ('NNN+BP', 'NN+BP', 'BP'):
            raise YastnError(f" Type of EnvBP bond_metric {which=} not recognized.")
        self._which = which

    which = property(fget=_get_which, fset=_set_which)

    @property
    def config(self):
        return self.psi.config

    def __getitem__(self, site):
        return self.env[site]

    def __setitem__(self, site, obj):
        self.env[site] = obj

    def copy(self) -> EnvBP:
        env = EnvBP(self.psi, init=None)
        env.env = self.env.copy()
        return env

    def clone(self) -> EnvBP:
        env = EnvBP(self.psi, init=None)
        env.env = self.env.clone()
        return env

    def shallow_copy(self) -> EnvBP:
        env = EnvBP(self.psi, init=None)
        env.env = self.env.shallow_copy()
        return env

    def to_dict(self, level=2):
        r"""
        Serialize EnvBP to a dictionary.
        Complementary function is :meth:`yastn.EnvBP.from_dict` or a general :meth:`yastn.from_dict`.
        See :meth:`yastn.Tensor.to_dict` for further description.
        """
        return {'type': type(self).__name__,
                'dict_ver': 1,
                'psi': self.psi.to_dict(level=level),
                'env': self.env.to_dict(level=level)}

    @classmethod
    def from_dict(cls, d, config=None):
        r"""
        De-serializes EnvBP from the dictionary ``d``.
        See :meth:`yastn.Tensor.from_dict` for further description.
        """
        if 'dict_ver' not in d:
            psi = PEPS_CLASSES["Peps"].from_dict(d['psi'], config)
            env = EnvBP(psi, init=None)
            for site in env.sites():
                for dirn, v in d['data'][site].items():
                    setattr(env[site], dirn, Tensor.from_dict(v, config))
            return env

        if d['dict_ver'] == 1:
            if cls.__name__ != d['type']:
                raise YastnError(f"{cls.__name__} does not match d['type'] == {d['type']}")
            psi = PEPS_CLASSES[d['psi']['type']].from_dict(d['psi'], config=config)
            env = cls(psi, init=None)
            env.env = Lattice.from_dict(d['env'], config=config)
            return env

    def save_to_dict(self) -> dict:
        r"""
        Serialize EnvBP into a dictionary.

        !!! This method is deprecated; use to_dict() instead !!!
        """
        warn('This method is deprecated; use to_dict() instead.', DeprecationWarning, stacklevel=2)

        psi = self.psi
        if isinstance(psi, Peps2Layers):
            psi = psi.ket

        d = {'class': 'EnvBP',
             'psi': psi.save_to_dict(),
             'data': {}}

        for site in self.sites():
            d_local = {dirn: getattr(self[site], dirn).save_to_dict()
                       for dirn in ['t', 'l', 'b', 'r']}
            d['data'][site] = d_local
        return d

    def reset_(self, init='eye'):
        r"""
        Initialize BP environment.

        Parameters
        ----------
        init: str
            For 'eye' starts with identity environments.
        """
        config = self.psi.config
        for site in self.sites():
            legs = self.psi[site].get_legs()
            for ind, dirn in enumerate('tlbr'):
                if self.nn_site(site, d=dirn) is None or init == 'eye':
                    tmp = eye(config, legs=legs[ind].unfuse_leg(), isdiag=False)
                tmp = tmp / tmp.norm()
                setattr(self[site], dirn, tmp)

    def measure_1site(self, O, site=None) -> dict:
        r"""
        Calculate local expectation values within BP environment.

        Returns a number if ``site`` is provided.
        If ``None``, returns a dictionary {site: value} for all unique lattice sites.

        Parameters
        ----------
        env: EnvBP
            Class containing BP environment tensors along with lattice structure data.

        O: Tensor
            Single-site operator
        """

        if site is None:
            opdict = _clear_operator_input(O, self.sites())
        else:
            opdict = {site: {(): O}}

        out = {}
        for site, ops in opdict.items():

            lenv = self[site]
            ten = self.psi[site]

            if isinstance(ten, DoublePepsTensor):
                Atlbr = ncon([ten.ket, lenv.t, lenv.l, lenv.b, lenv.r], [(1, 2, 3, 4, -4), (-0, 1), (-1, 2), (-2, 3), (-3, 4)])
                val_no = vdot(ten.bra, Atlbr)

                for nz, op in ops.items():
                    op = match_ancilla(ten.ket, op)
                    Atmp = tensordot(Atlbr, op, axes=(4, 1))
                    val_op = vdot(ten.bra, Atmp)
                    out[site + nz] = val_op / val_no
            else:
                pass

        if len(out) == 1:
            return out[site + nz]
        return out

    def measure_nn(self, O, P, bond=None) -> dict:
        r"""
        Calculate nearest-neighbor expectation values within BP environment.

        Return a number if the nearest-neighbor ``bond`` is provided.
        If ``None``, returns a dictionary {bond: value} for all unique lattice bonds.

        Parameters
        ----------
        O, P: yastn.Tensor
            Calculate <O_s0 P_s1>.
            P is applied first, which might matter for fermionic operators.

        bond: yastn.tn.fpeps.Bond | tuple[tuple[int, int], tuple[int, int]]
            Bond of the form (s0, s1). Sites s0 and s1 should be nearest-neighbors on the lattice.
        """

        if bond is None:
            if isinstance(O, dict):
                Odict = _clear_operator_input(O, self.sites())
                Pdict = _clear_operator_input(P, self.sites())
                out = {}
                for (s0, s1) in self.bonds():
                    for nz0, op0 in Odict[s0].items():
                        for nz1, op1 in Pdict[s1].items():
                            out[s0 + nz0, s1 + nz1] = self.measure_nn(op0, op1, bond=(s0, s1))
                return out
            else:
                return {bond: self.measure_nn(O, P, bond) for bond in self.bonds()}

        if O.ndim == 2 and P.ndim == 2:
            O, P = fkron(O, P, sites=(0, 1), merge=False)

        dirn = self.nn_bond_dirn(*bond)
        if O.ndim == 3 and P.ndim == 3:
            O, P = gate_fix_swap_gate(O, P, dirn, self.f_ordered(*bond))

        s0, s1 = bond if dirn in ('lr', 'tb') else bond[::-1]
        G0, G1 = (O, P) if dirn in ('lr', 'tb') else (P, O)
        env0, env1 = self[s0], self[s1]
        ten0, ten1 = self.psi[s0], self.psi[s1]

        if dirn in ('lr', 'rl'):
            tmp0 = hair_l(ten0.bra, ht=env0.t, hl=env0.l, hb=env0.b, A_ket=ten0.ket)
            tmp1 = hair_r(ten1.bra, ht=env1.t, hr=env1.r, hb=env1.b, A_ket=ten1.ket)
            val_no = vdot(tmp0, tmp1, conj=(0, 0))

            ten0 = ten0.apply_gate_on_ket(G0, dirn='l')  # if G0.ndim <= 3 else G0
            ten1 = ten1.apply_gate_on_ket(G1, dirn='r')  # if G1.ndim <= 3 else G1

            tmp0 = hair_l(ten0.bra, ht=env0.t, hl=env0.l, hb=env0.b, A_ket=ten0.ket)
            tmp1 = hair_r(ten1.bra, ht=env1.t, hr=env1.r, hb=env1.b, A_ket=ten1.ket)
            val_op = vdot(tmp0, tmp1, conj=(0, 0))
        else:  # dirn in ('tb', 'bt'):
            tmp0 = hair_t(ten0.bra, ht=env0.t, hl=env0.l, hr=env0.r, A_ket=ten0.ket)
            tmp1 = hair_b(ten1.bra, hl=env1.l, hr=env1.r, hb=env1.b, A_ket=ten1.ket)
            val_no = vdot(tmp0, tmp1, conj=(0, 0))

            ten0 = ten0.apply_gate_on_ket(G0, dirn='t')  # if G0.ndim <= 3 else G0
            ten1 = ten1.apply_gate_on_ket(G1, dirn='b')  # if G1.ndim <= 3 else G1

            tmp0 = hair_t(ten0.bra, ht=env0.t, hl=env0.l, hr=env0.r, A_ket=ten0.ket)
            tmp1 = hair_b(ten1.bra, hl=env1.l, hr=env1.r, hb=env1.b, A_ket=ten1.ket)
            val_op = vdot(tmp0, tmp1, conj=(0, 0))

        return val_op / val_no

    def update_(self) -> float:
        r"""
        Perform one step of BP update. Environment tensors are updated in place.

        Returns
        -------
        diff: maximal difference between belief tensors befor and after the update.
        """
        #
        env_tmp = None  # EnvBP(self.psi, init=None)  # empty environments
        diffs  = [self.update_bond_(bond, env_tmp=env_tmp) for bond in self.bonds('h')]
        diffs += [self.update_bond_(bond[::-1], env_tmp=env_tmp) for bond in self.bonds('h')[::-1]]
        diffs += [self.update_bond_(bond, env_tmp=env_tmp) for bond in self.bonds('v')]
        diffs += [self.update_bond_(bond[::-1], env_tmp=env_tmp) for bond in self.bonds('v')[::-1]]
        #
        # update_storage_(self, env_tmp)
        return max(diffs)

    def update_bond_(env, bond, env_tmp=None):
        #
        if env_tmp is None:
            env_tmp = env  # update env in-place

        bond = Bond(*bond)
        dirn = env.nn_bond_dirn(*bond)
        s0, s1 = bond
        ten0, env0 = env.psi[s0], env[s0]

        if dirn == 'lr':
            new_l = hair_l(ten0.bra, ht=env0.t, hl=env0.l, hb=env0.b, A_ket=ten0.ket)
            new_l = regularize_belief(new_l, env.tol_positive)
            diff = diff_beliefs(env[s1].l, new_l)
            env_tmp[s1].l = new_l
        if dirn == 'rl':
            new_r = hair_r(ten0.bra, ht=env0.t, hb=env0.b, hr=env0.r, A_ket=ten0.ket)
            new_r = regularize_belief(new_r, env.tol_positive)
            diff = diff_beliefs(env[s1].r, new_r)
            env_tmp[s1].r = new_r
        if dirn == 'tb':
            new_t = hair_t(ten0.bra, ht=env0.t, hl=env0.l, hr=env0.r, A_ket=ten0.ket)
            new_t = regularize_belief(new_t, env.tol_positive)
            diff = diff_beliefs(env[s1].t, new_t)
            env_tmp[s1].t = new_t
        if dirn == 'bt':
            new_b = hair_b(ten0.bra, hl=env0.l, hb=env0.b, hr=env0.r, A_ket=ten0.ket)
            new_b = regularize_belief(new_b, env.tol_positive)
            diff = diff_beliefs(env[s1].b, new_b)
            env_tmp[s1].b = new_b
        return diff

    def bond_metric(self, Q0, Q1, s0, s1, dirn):
        r"""
        Calculates bond metric within BP environment.

        ::

            If which == 'BP':

                     t       t
                     ║       ║
                l════Q0══  ══Q1═══r
                     ║       ║
                     b       b


            If which == 'NN+BP':

                      t        t
                      ║        ║
                l══(-1 +0)══(-1 +1)══r
                      ║        ║
                l═════Q0══   ══Q1════r
                      ║        ║
                l══(+1 +0)══(+1 +1)══l
                      ║        ║
                      b        b


            If which == 'NNN+BP':

                      t       t        t       t
                      ║       ║        ║       ║
                l══(-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)══r
                      ║       ║        ║       ║
                l══(+0 -1)════Q0══   ══Q1═══(+0 +2)══r
                      ║       ║        ║       ║
                l══(+1 -1)=(+1 +0)══(+1 +1)=(+1 +2)══r
                      ║       ║        ║       ║
                      b       b        b       b

        """
        if dirn in ("h", "lr") and self.which == "BP":
            assert self.psi.nn_site(s0, (0, 1)) == s1
            vecl = hair_l(Q0, hl=self[s0].l, ht=self[s0].t, hb=self[s0].b)
            vecr = hair_r(Q1, hr=self[s1].r, ht=self[s1].t, hb=self[s1].b).T
            return BipartiteBondMetric(gL=vecl, gR=vecr)  # (rr' rr,  ll ll')

        if dirn in ("v", "tb") and self.which == "BP":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            vect = hair_t(Q0, hl=self[s0].l, ht=self[s0].t, hr=self[s0].r)
            vecb = hair_b(Q1, hr=self[s1].r, hb=self[s1].b, hl=self[s1].l).T
            return BipartiteBondMetric(gL=vect, gR=vecb)  # (bb' bb,  tt tt')

        if dirn in ("h", "lr") and self.which == "NN+BP":
            assert self.psi.nn_site(s0, (0, 1)) == s1

            m = {d: self.psi.nn_site(s0, d=d) for d in [(-1,0), (0,-1), (1,0), (1,1), (0,2), (-1,1)]}
            mm = dict(m)  # for testing for None
            tensors_from_psi(m, self.psi)
            m = {k: (v.ket if isinstance(v, DoublePepsTensor) else v) for k, v in m.items()}

            sm = mm[0, -1]
            env_hl = hair_l(m[0, -1]) if sm is None else hair_l(m[0, -1], ht=self[sm].t, hl=self[sm].l, hb=self[sm].b)
            sm = mm[0, 2]
            env_hr = hair_r(m[0,  2]) if sm is None else hair_r(m[0,  2], ht=self[sm].t, hb=self[sm].b, hr=self[sm].r)
            env_l = edge_l(Q0, hl=env_hl)  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(Q1, hr=env_hr)  # [tr tr'] [ll ll'] [br br']

            sm = mm[-1, 0]
            ctl = cor_tl(m[-1, 0]) if sm is None else cor_tl(m[-1, 0], ht=self[sm].t, hl=self[sm].l)
            sm = mm[-1, 1]
            ctr = cor_tr(m[-1, 1]) if sm is None else cor_tr(m[-1, 1], ht=self[sm].t, hr=self[sm].r)
            sm = mm[ 1, 1]
            cbr = cor_br(m[ 1, 1]) if sm is None else cor_br(m[ 1, 1], hb=self[sm].b, hr=self[sm].r)
            sm = mm[ 1, 0]
            cbl = cor_bl(m[ 1, 0]) if sm is None else cor_bl(m[ 1, 0], hb=self[sm].b, hl=self[sm].l)

            g = tensordot((cbr @ cbl) @ env_l, (ctl @ ctr) @ env_r, axes=((0, 2), (2, 0)))  # [rr rr'] [ll ll']
            return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))

        if dirn in ("v", "tb") and self.which == "NN+BP":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            m = {d: self.psi.nn_site(s0, d=d) for d in [(-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,1)]}
            mm = dict(m)  # for testing for None
            tensors_from_psi(m, self.psi)
            m = {k: (v.ket if isinstance(v, DoublePepsTensor) else v) for k, v in m.items()}

            sm = mm[-1, 0]
            env_ht = hair_t(m[-1, 0]) if sm is None else hair_t(m[-1, 0], ht=self[sm].t, hl=self[sm].l, hr=self[sm].r)
            sm = mm[2, 0]
            env_hb = hair_b(m[ 2, 0]) if sm is None else hair_b(m[ 2, 0], hl=self[sm].l, hb=self[sm].b, hr=self[sm].r)
            env_t = edge_t(Q0, ht=env_ht)  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(Q1, hb=env_hb)  # [rb rb'] [tt tt'] [lb lb']

            sm = mm[1, -1]
            cbl = cor_bl(m[1, -1]) if sm is None else cor_bl(m[1, -1], hb=self[sm].b, hl=self[sm].l)
            sm = mm[0, -1]
            ctl = cor_tl(m[0, -1]) if sm is None else cor_tl(m[0, -1], ht=self[sm].t, hl=self[sm].l)
            sm = mm[0,  1]
            ctr = cor_tr(m[0,  1]) if sm is None else cor_tr(m[0,  1], ht=self[sm].t, hr=self[sm].r)
            sm = mm[1,  1]
            cbr = cor_br(m[1,  1]) if sm is None else cor_br(m[1,  1], hb=self[sm].b, hr=self[sm].r)

            g = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
            return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))

        if dirn in ("h", "lr") and self.which == "NNN+BP":
            assert self.psi.nn_site(s0, (0, 1)) == s1
            sts = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (1,2), (0,2), (-1,2), (-1,1), (-1,0)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            mm = dict(m)  # for testing for None
            tensors_from_psi(m, self.psi)
            m = {k: (v.ket if isinstance(v, DoublePepsTensor) else v) for k, v in m.items()}

            sm = mm[0, -1]
            ell = edge_l(m[0, -1]) if sm is None else edge_l(m[0, -1], hl=self[sm].l)
            sm = mm[-1, -1]
            clt = cor_tl(m[-1, -1]) if sm is None else cor_tl(m[-1, -1], ht=self[sm].t, hl=self[sm].l)
            sm = mm[-1, 0]
            elt = edge_t(m[-1, 0]) if sm is None else edge_t(m[-1, 0], ht=self[sm].t)
            sm = mm[1, 0]
            elb = edge_b(m[1, 0]) if sm is None else edge_b(m[1, 0], hb=self[sm].b)
            sm = mm[1, -1]
            clb = cor_bl(m[1, -1]) if sm is None else cor_bl(m[1, -1], hb=self[sm].b, hl=self[sm].l)
            vecl = append_vec_tl(Q0, Q0, ell @ (clt @ elt))
            vecl = tensordot(elb @ clb, vecl, axes=((2, 1), (0, 1)))

            sm = mm[0, 2]
            err = edge_r(m[0, 2]) if sm is None else edge_r(m[0, 2], hr=self[sm].r)
            sm = mm[1, 2]
            crb = cor_br(m[1, 2]) if sm is None else cor_br(m[1, 2], hb=self[sm].b, hr=self[sm].r)
            sm = mm[1, 1]
            erb = edge_b(m[1, 1]) if sm is None else edge_b(m[1, 1], hb=self[sm].b)
            sm = mm[-1, 1]
            ert = edge_t(m[-1, 1]) if sm is None else edge_t(m[-1, 1], ht=self[sm].t)
            sm = mm[-1, 2]
            crt = cor_tr(m[-1, 2]) if sm is None else cor_tr(m[-1, 2], hr=self[sm].r, ht=self[sm].t)
            vecr = append_vec_br(Q1, Q1, err @ (crb @ erb))
            vecr = tensordot(ert @ crt, vecr, axes=((2, 1), (0, 1)))
            g = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [rr rr'] [ll ll']
            return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))

        if dirn in ("v", "tb") and self.which == "NNN+BP":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            sts = [(-1,-1), (0,-1), (1,-1), (2,-1), (2,0), (2,1), (1,1), (0,1), (-1,1), (-1,0)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            mm = dict(m)  # for testing for None
            tensors_from_psi(m, self.psi)
            m = {k: (v.ket if isinstance(v, DoublePepsTensor) else v) for k, v in m.items()}

            sm = mm[0, -1]
            etl = edge_l(m[0, -1]) if sm is None else edge_l(m[0, -1], hl=self[sm].l)
            sm = mm[-1, -1]
            ctl = cor_tl(m[-1, -1]) if sm is None else cor_tl(m[-1, -1], hl=self[sm].l, ht=self[sm].t)
            sm = mm[-1, 0]
            ett = edge_t(m[-1, 0]) if sm is None else edge_t(m[-1, 0], ht=self[sm].t)
            sm = mm[-1, 1]
            ctr = cor_tr(m[-1, 1]) if sm is None else cor_tr(m[-1, 1], hr=self[sm].r, ht=self[sm].t)
            sm = mm[0, 1]
            etr = edge_r(m[0, 1]) if sm is None else edge_r(m[0, 1], hr=self[sm].r)
            vect = append_vec_tl(Q0, Q0, etl @ (ctl @ ett))
            vect = tensordot(vect, ctr @ etr, axes=((2, 3), (0, 1)))

            sm = mm[1, 1]
            ebr = edge_r(m[1, 1]) if sm is None else edge_r(m[1, 1], hr=self[sm].r)
            sm = mm[2, 1]
            cbr = cor_br(m[2, 1]) if sm is None else cor_br(m[2, 1], hr=self[sm].r, hb=self[sm].b)
            sm = mm[2, 0]
            ebb = edge_b(m[2, 0]) if sm is None else edge_b(m[2, 0], hb=self[sm].b)
            sm = mm[2, -1]
            cbl = cor_bl(m[2, -1]) if sm is None else cor_bl(m[2, -1], hb=self[sm].b, hl=self[sm].l)
            sm = mm[1, -1]
            ebl = edge_l(m[1, -1]) if sm is None else edge_l(m[1, -1], hl=self[sm].l)
            vecb = append_vec_br(Q1, Q1, ebr @ (cbr @ ebb))
            vecb = tensordot(vecb, cbl @ ebl, axes=((2, 3), (0, 1)))
            g = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
            return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))

    def pre_truncation_(env, sites):
        for s0, s1 in pairwise(sites[-1::-1]):
            env.update_bond_((s0, s1))
        for s0, s1 in pairwise(sites):
            env.update_bond_((s0, s1))

    def post_truncation_(env, bond, max_sweeps=1):
        env.update_bond_(bond)
        env.update_bond_(bond[::-1])
        if max_sweeps > 0:
            env.iterate_(max_sweeps=max_sweeps)

    def iterate_(env, max_sweeps=1, iterator_step=None, diff_tol=None):
        r"""
        Perform BP updates :meth:`yastn.tn.fpeps.EnvBP.update_` until convergence.
        Convergence can be measured based on maximal difference between old and new tensors.

        Outputs iterator if ``iterator_step`` is given, which allows
        inspecting ``env``, e.g., calculating expectation values,
        outside of ``iterate_`` function after every ``iterator_step`` sweeps.

        Parameters
        ----------
        max_sweeps: int
            Maximal number of sweeps.

        iterator_step: int
            If int, ``iterate_`` returns a generator that would yield output after every iterator_step sweeps.
            The default is ``None``, in which case  ``iterate_`` sweeps are performed immediately.

        diff_tol: float
            Convergence tolerance for the change of belief tensors in one iteration.
            The default is None, in which case convergence is not checked and it is up to user to implement
            convergence check.

        Returns
        -------
        Generator if iterator_step is not ``None``.

        BP_out(NamedTuple)
            NamedTuple including fields:

                * ``sweeps`` number of performed lbp updates.
                * ``max_diff`` maximal difference between old and new belief tensors.
                * ``converged`` whether convergence based on ``diff_tol`` has been reached.
        """
        tmp = _iterate_(env, max_sweeps, iterator_step, diff_tol)
        return tmp if iterator_step else next(tmp)

    def sample(self, projectors, number=1, xrange=None, yrange=None, progressbar=False, return_probabilities=False, flatten_one=True, **kwargs) -> dict[Site, list]:
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

        number: int
            Number of independent samples.

        xrange: tuple[int, int]
            range of rows to sample from, [r0, r1); r0 included, r1 excluded.

        yrange: tuple[int, int]
            range of columns to sample from.

        progressbar: bool
            Whether to display progressbar. The default is ``False``.

        return_probabilities: bool
            Whether to return a tuple (samples, probabilities). The default is ``False``, where a dict samples is returned.

        flatten_one: bool
            Whether, for number==1, pop one-element lists for each lattice site to return samples={site: ind, } instead of {site: [ind]}.
            The default is ``True``.
        """
        if xrange is None:
            xrange = [0, self.Nx]
        if yrange is None:
            yrange = [0, self.Ny]

        if self.nn_site((xrange[0], yrange[0]), (0, 0)) is None or \
           self.nn_site((xrange[1] - 1, yrange[1] - 1), (0, 0)) is None:
           raise YastnError(f"Window range {xrange=}, {yrange=} does not fit within the lattice.")

        sites = [Site(nx, ny) for ny in range(*yrange) for nx in range(*xrange)]
        projs_sites = clear_projectors(sites, projectors, xrange, yrange)

        out = {site: [] for site in sites}
        probabilities = []
        rands = (self.psi.config.backend.rand(self.Nx * self.Ny * number) + 1) / 2  # in [0, 1]
        count = 0

        for _ in tqdm(range(number), desc="Sample...", disable=not progressbar):
            probability = 1.
            env = {site: EnvBP_local() for site in sites}
            for (nx, ny) in sites:
                nx0, ny0 = nx % self.Nx, ny % self.Ny
                env[nx, ny].t = self[nx0, ny0].t.copy()
                env[nx, ny].l = self[nx0, ny0].l.copy()
                env[nx, ny].b = self[nx0, ny0].b.copy()
                env[nx, ny].r = self[nx0, ny0].r.copy()

            for ny in range(*yrange):
                for nx in range(*xrange):
                    nx0, ny0 = nx % self.Nx, ny % self.Ny
                    lenv = env[nx, ny]
                    ten = self.psi[nx0, ny0]
                    Atlbr = ncon([ten.ket, lenv.t, lenv.l, lenv.b, lenv.r], [(1, 2, 3, 4, -4), (-0, 1), (-1, 2), (-2, 3), (-3, 4)])
                    norm_prob = vdot(ten.bra, Atlbr)
                    acc_prob = 0
                    for k, proj in projs_sites[(nx, ny)].items():
                        proj = match_ancilla(ten.ket, proj)
                        Atmp = tensordot(Atlbr, proj, axes=(4, 1))
                        prob = vdot(ten.bra, Atmp) / norm_prob
                        acc_prob += prob
                        if rands[count] < acc_prob:
                            out[nx, ny].append(k)
                            ketp = tensordot(ten.ket, proj, axes=(4, 1)) / prob
                            if nx + 1 < xrange[1]:
                                new_t = hair_t(ten.bra, ht=lenv.t, hl=lenv.l, hr=lenv.r, A_ket=ketp)
                                new_t = regularize_belief(new_t, self.tol_positive)
                                env[nx + 1, ny].t = new_t
                            if ny + 1 < yrange[1]:
                                new_l = hair_l(ten.bra, ht=lenv.t, hl=lenv.l, hb=lenv.b, A_ket=ketp)
                                new_l = regularize_belief(new_l, self.tol_positive)
                                env[nx, ny + 1].l = new_l
                            probability *= prob
                            break
                    count += 1
            probabilities.append(probability)

        if number == 1 and flatten_one:
            out = {site: smp.pop() for site, smp in out.items()}

        if return_probabilities:
            return out, probabilities
        return out


def _iterate_(env, max_sweeps, iterator_step, diff_tol):
    """ Generator for ctmrg_(). """
    converged = None
    for sweep in range(1, max_sweeps + 1):
        max_diff = env.update_()

        if diff_tol is not None and max_diff < diff_tol:
            converged = True
            break

        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield BP_out(sweeps=sweep, max_diff=max_diff, converged=converged)
    yield BP_out(sweeps=sweep, max_diff=max_diff, converged=converged)


def regularize_belief(mat, tol):
    """ Make matrix mat hermitian and positive, truncating eigenvalues at a given relative tolerance. """
    mat = mat + mat.H
    S, U = mat.eigh_with_truncation(axes=(0, 1), tol=tol)
    S = S / S.norm()
    return U @ S @ U.H


def diff_beliefs(old, new):
    try:
        return (old - new).norm()
    except YastnError:
        return 1.
