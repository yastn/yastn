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
from dataclasses import dataclass
from typing import NamedTuple, Union
from .... import Tensor, ones, eye, YastnError, tensordot, vdot, ncon
from .._peps import Peps, Peps2Layers, DoublePepsTensor
from .._gates_auxiliary import apply_gate_onsite, gate_product_operator, gate_fix_order, match_ancilla
from .._geometry import Bond
from ._env_auxlliary import *
# from ._env_ctm import update_old_env_


@dataclass()
class EnvLBP_local():
    r"""
    Dataclass for LBP environment tensors at a single Peps site on square lattice.

    Contains fields ``t``,  ``l``, ``b``, ``r``
    """
    t  : Union[Tensor, None] = None  # top
    l  : Union[Tensor, None] = None  # left
    b  : Union[Tensor, None] = None  # bottom
    r  : Union[Tensor, None] = None  # right


class LBP_out(NamedTuple):
    sweeps : int = 0
    max_diff : float = None
    converged : bool = False


class EnvLBP(Peps):

    def __init__(self, psi, init='eye', tol_positive=1e-12, which="LBP"):
        r"""
        Environment used in LBP

        Parameters
        ----------
        psi: yastn.tn.Peps
            PEPS lattice to be contracted using LBP.
            If ``psi`` has physical legs, a double-layer PEPS with no physical legs is formed.

        init: str
            None, 'eye'. Initialization scheme, see :meth:`yastn.tn.fpeps.EnvLBP.reset_`.
        """
        super().__init__(psi.geometry)
        self.psi = Peps2Layers(psi) if psi.has_physical() else psi
        self.tol_positive = tol_positive

        if which not in ('NN+LBP', 'LBP'):
            raise YastnError(f" Type of EnvLBP bond_metric {which=} not recognized.")

        self.which = which
        if init not in (None, 'eye'):
            raise YastnError(f"EnvCTM {init=} not recognized. Should be 'eye' or None.")
        for site in self.sites():
            self[site] = EnvLBP_local()
        if init is not None:
            self.reset_(init=init)

    @property
    def config(self):
        return self.psi.config

    def reset_(self, init='eye'):
        r"""
        Initialize LBP environment.

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
        Calculate local expectation values within CTM environment.

        Returns a number if ``site`` is provided.
        If ``None``, returns a dictionary {site: value} for all unique lattice sites.

        Parameters
        ----------
        env: EnvCtm
            Class containing LBP environment tensors along with lattice structure data.

        O: Tensor
            Single-site operator
        """
        if site is None:
            return {site: self.measure_1site(O, site) for site in self.sites()}

        lenv = self[site]
        ten = self.psi[site]

        if isinstance(ten, DoublePepsTensor):
            Aket = ten.ket.unfuse_legs(axes=(0, 1))  # t l b r s
            Abra = ten.bra.unfuse_legs(axes=(0, 1))  # t l b r s
            Aket = ncon([Aket, lenv.t, lenv.l, lenv.b, lenv.r], [(1, 2, 3, 4, -4), (-0, 1), (-1, 2), (-2, 3), (-3, 4)])
            val_no = vdot(Abra, Aket)
            op = match_ancilla(ten.ket, O)
            Aket = tensordot(Aket, op, axes=(4, 1))
            val_op = vdot(Abra, Aket)
        else:
            pass
        return val_op / val_no

    def measure_nn(self, O, P, bond=None) -> dict:
        r"""
        Calculate nearest-neighbor expectation values within LBP environment.

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
             return {bond: self.measure_nn(O, P, bond) for bond in self.bonds()}

        bond = Bond(*bond)
        dirn, l_ordered = self.nn_bond_type(bond)
        f_ordered = self.f_ordered(*bond)
        s0, s1 = bond if l_ordered else bond[::-1]
        env0, env1 = self[s0], self[s1]
        ten0, ten1 = self.psi[s0], self.psi[s1]

        if O.ndim == 2 and P.ndim == 2:
            G0, G1 = gate_product_operator(O, P, l_ordered, f_ordered)
        elif O.ndim == 3 and P.ndim == 3:
            G0, G1 = gate_fix_order(O, P, l_ordered, f_ordered)

        if dirn == 'h':
            tmp0 = hair_l(ten0.bra, ht=env0.t, hl=env0.l, hb=env0.b, Aket=ten0.ket)
            tmp1 = hair_r(ten1.bra, ht=env1.t, hr=env1.r, hb=env1.b, Aket=ten1.ket)
            val_no = vdot(tmp0, tmp1, conj=(0, 0))

            if O.ndim <= 3:
                Aket0 = apply_gate_onsite(ten0.ket, G0, dirn='l')
            # else:
            #     ten0 = O
            if P.ndim <= 3:
                Aket1 = apply_gate_onsite(ten1.ket, G1, dirn='r')
            # else:
            #     ten1 = P

            tmp0 = hair_l(ten0.bra, ht=env0.t, hl=env0.l, hb=env0.b, Aket=Aket0)
            tmp1 = hair_r(ten1.bra, ht=env1.t, hr=env1.r, hb=env1.b, Aket=Aket1)
            val_op = vdot(tmp0, tmp1, conj=(0, 0))
        else:  # dirn == 'v':
            tmp0 = hair_t(ten0.bra, ht=env0.t, hl=env0.l, hr=env0.r, Aket=ten0.ket)
            tmp1 = hair_b(ten1.bra, hl=env1.l, hr=env1.r, hb=env1.b, Aket=ten1.ket)
            val_no = vdot(tmp0, tmp1, conj=(0, 0))

            if O.ndim <= 3:
                Aket0 = apply_gate_onsite(ten0.ket, G0, dirn='t')
            # else:
            #     ten0 = O

            if P.ndim <= 3:
                Aket1 = apply_gate_onsite(ten1.ket, G1, dirn='b')
            # else:
            #     ten1 = P

            tmp0 = hair_t(ten0.bra, ht=env0.t, hl=env0.l, hr=env0.r, Aket=Aket0)
            tmp1 = hair_b(ten1.bra, hl=env1.l, hr=env1.r, hb=env1.b, Aket=Aket1)
            val_op = vdot(tmp0, tmp1, conj=(0, 0))

        return val_op / val_no

    def update_(self) -> float:
        r"""
        Perform one step of LBP update. Environment tensors are updated in place.

        Returns
        -------
        diff: maximal difference between belief tensors befor and after the update.
        """
        #
        env_tmp = None  # EnvLBP(self.psi, init=None)  # empty environments
        diffs  = [self.update_bond_(bond, env_tmp=env_tmp) for bond in self.bonds('h')]
        diffs += [self.update_bond_(bond[::-1], env_tmp=env_tmp) for bond in self.bonds('h')[::-1]]
        diffs += [self.update_bond_(bond, env_tmp=env_tmp) for bond in self.bonds('v')]
        diffs += [self.update_bond_(bond[::-1], env_tmp=env_tmp) for bond in self.bonds('v')[::-1]]
        #
        # update_old_env_(self, env_tmp)
        return max(diffs)

    def update_bond_(env, bond, env_tmp=None):
        #
        if env_tmp is None:
            env_tmp = env  # update env in-place

        bond = Bond(*bond)
        dirn, l_ordered = env.nn_bond_type(bond)
        s0, s1 = bond
        ten0, env0 = env.psi[s0], env[s0]

        if dirn == 'h' and l_ordered:
            new_l = hair_l(ten0.bra, ht=env0.t, hl=env0.l, hb=env0.b, Aket=ten0.ket)
            new_l = regularize_belief(new_l, env.tol_positive)
            diff = diff_beliefs(env[s1].l, new_l)
            env_tmp[s1].l = new_l
        if dirn == 'h' and not l_ordered:
            new_r = hair_r(ten0.bra, ht=env0.t, hb=env0.b, hr=env0.r, Aket=ten0.ket)
            new_r = regularize_belief(new_r, env.tol_positive)
            diff = diff_beliefs(env[s1].r, new_r)
            env_tmp[s1].r = new_r
        if dirn == 'v' and l_ordered:
            new_t = hair_t(ten0.bra, ht=env0.t, hl=env0.l, hr=env0.r, Aket=ten0.ket)
            new_t = regularize_belief(new_t, env.tol_positive)
            diff = diff_beliefs(env[s1].t, new_t)
            env_tmp[s1].t = new_t
        if dirn == 'v' and not l_ordered:
            new_b = hair_b(ten0.bra, hl=env0.l, hb=env0.b, hr=env0.r, Aket=ten0.ket)
            new_b = regularize_belief(new_b, env.tol_positive)
            diff = diff_beliefs(env[s1].b, new_b)
            env_tmp[s1].b = new_b
        return diff

    def bond_metric(self, Q0, Q1, s0, s1, dirn):
        r"""
        Calculates Full-Update metric tensor.

        ::

            If dirn == 'h':

                     t       t
                     ║       ║
                l════Q0══  ══Q1═══r
                     ║       ║
                     b       b


            If dirn == 'v':

                     t
                     ║
                l═══0Q0═══r
                     ╳
                l═══1Q1═══r
                     ║
                     b
        """
        if dirn == "h" and self.which == "LBP":
            assert self.psi.nn_site(s0, (0, 1)) == s1
            vecl = hair_l(Q0, hl=self[s0].l, ht=self[s0].t, hb=self[s0].b)
            vecr = hair_r(Q1, hr=self[s1].r, ht=self[s1].t, hb=self[s1].b).T
            return (vecl, vecr)  # (rr' rr,  ll ll')

        if dirn == "v" and self.which == "LBP":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            vect = hair_t(Q0, hl=self[s0].l, ht=self[s0].t, hr=self[s0].r)
            vecb = hair_b(Q1, hr=self[s1].r, hb=self[s1].b, hl=self[s1].l).T
            return (vect, vecb)  # (bb' bb,  tt tt')

        if dirn == "h" and self.which == "NN+LBP":
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
            return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))

        if dirn == "v" and self.which == "NN+LBP":
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
            return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))

    def post_evolution_(env, bond, **kwargs):
        env.update_bond_(bond)
        env.update_bond_(bond[::-1])

    def lbp_(env, max_sweeps=1, iterator_step=None, diff_tol=None):
        r"""
        Perform LBP updates :meth:`yastn.tn.fpeps.EnvLBP.update_` until convergence.
        Convergence can be measured based on maximal difference between old and new tensors.

        Outputs iterator if ``iterator_step`` is given, which allows
        inspecting ``env``, e.g., calculating expectation values,
        outside of ``lbp_`` function after every ``iterator_step`` sweeps.

        Parameters
        ----------
        max_sweeps: int
            Maximal number of sweeps.

        iterator_step: int
            If int, ``ctmrg_`` returns a generator that would yield output after every iterator_step sweeps.
            The default is ``None``, in which case  ``ctmrg_`` sweeps are performed immediately.

        diff_tol: float
            Convergence tolerance for the change of belief tensors in one iteration.
            The default is None, in which case convergence is not checked and it is up to user to implement
            convergence check.

        Returns
        -------
        Generator if iterator_step is not ``None``.

        LBP_out(NamedTuple)
            NamedTuple including fields:

                * ``sweeps`` number of performed lbp updates.
                * ``max_diff`` maximal difference between old and new belief tensors.
                * ``converged`` whether convergence based on ``diff_tol`` has been reached.
        """
        tmp = _lbp_(env, max_sweeps, iterator_step, diff_tol)
        return tmp if iterator_step else next(tmp)


def _lbp_(env, max_sweeps, iterator_step, diff_tol):
    """ Generator for ctmrg_(). """
    converged = None
    for sweep in range(1, max_sweeps + 1):
        max_diff = env.update_()

        if diff_tol is not None and max_diff < diff_tol:
            converged = True
            break

        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield LBP_out(sweeps=sweep, max_diff=max_diff, converged=converged)
    yield LBP_out(sweeps=sweep, max_diff=max_diff, converged=converged)


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
