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
from typing import NamedTuple, Union, Callable
import logging
from .... import Tensor, ones, eye, YastnError, tensordot, vdot, ncon
from .._peps import Peps, Peps2Layers, DoublePepsTensor
from .._gates_auxiliary import apply_gate_onsite, gate_product_operator, gate_fix_order, match_ancilla
from .._geometry import Bond
from ._env_auxlliary import *
from ._env_ctm import update_old_env_


@dataclass()
class EnvLBP_local():
    r"""
    Dataclass for CTM environment tensors associated with Peps lattice site.

    Contains fields ``tl``, ``t``, ``tr``, ``r``, ``br``, ``b``, ``bl``, ``l``
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
    def __init__(self, psi, init='eye'):
        r"""
        Environment used in LBP

        Parameters
        ----------
        psi: yastn.tn.Peps
            PEPS lattice to be contracted using CTM.
            If ``psi`` has physical legs, a double-layer PEPS with no physical legs is formed.

        init: str
            None, 'eye'. Initialization scheme, see :meth:`yastn.tn.fpeps.EnvCTM.reset_`.

        """
        super().__init__(psi.geometry)
        self.psi = Peps2Layers(psi) if psi.has_physical() else psi
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
            Class containing CTM environment tensors along with lattice structure data.

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
        Calculate nearest-neighbor expectation values within CTM environment.

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


    def update_(self):
        r"""
        Perform one step of CTMRG update. Environment tensors are updated in place.

        The function performs a CTMRG update for a square lattice using the corner transfer matrix
        renormalization group (CTMRG) algorithm. The update is performed in two steps: a horizontal move
        and a vertical move. The projectors for each move are calculated first, and then the tensors in
        the CTM environment are updated using the projectors. The boundary conditions of the lattice
        determine whether trivial projectors are needed for the move.

        Returns
        -------
        proj: Peps structure loaded with CTM projectors related to all lattice site.
        """
        #
        env_tmp = EnvLBP(self.psi, init=None)  # empty environments
        diffs  = [self.update_bond(bond, env_tmp=env_tmp) for bond in self.bonds()]
        diffs += [self.update_bond(bond[::-1], env_tmp=env_tmp) for bond in self.bonds()]

        update_old_env_(self, env_tmp)
        return max(diffs)


    def update_bond(env, bond, env_tmp=None):
        if env_tmp is None:
            env_tmp = env

        bond = Bond(*bond)
        dirn, l_ordered = env.nn_bond_type(bond)
        s0, s1 = bond
        ten0 = env.psi[s0]
        env0 = env[s0]

        if dirn == 'h' and l_ordered:
            new_l = hair_l(ten0.bra, ht=env0.t, hl=env0.l, hb=env0.b, Aket=ten0.ket)
            new_l = new_l / new_l.norm()
            diff = (env[s1].l - new_l).norm()
            env_tmp[s1].l = new_l
        elif dirn == 'h' and not l_ordered:
            new_r = hair_r(ten0.bra, ht=env0.t, hb=env0.b, hr=env0.r, Aket=ten0.ket)
            new_r = new_r / new_r.norm()
            diff = (env[s1].r - new_r).norm()
            env_tmp[s1].r = new_r
        elif dirn == 'v' and l_ordered:
            new_t = hair_t(ten0.bra, ht=env0.t, hl=env0.l, hr=env0.r, Aket=ten0.ket)
            new_t = new_t / new_t.norm()
            diff = (env[s1].t - new_t).norm()
            env_tmp[s1].t = new_t
        else: # dirn == 'v' and not l_ordered:
            new_b = hair_b(ten0.bra, hl=env0.l, hb=env0.b, hr=env0.r, Aket=ten0.ket)
            new_b = new_b / new_b.norm()
            diff = (env[s1].b - new_b).norm()
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
        env0, env1 = self[s0], self[s1]
        if dirn == "h":
            assert self.psi.nn_site(s0, (0, 1)) == s1
            vecl = hair_l(Q0, hl=env0.l, ht=env0.t, hb=env0.b).T
            vecr = hair_r(Q1, hr=env1.r, ht=env1.t, hb=env1.b).T
            g = (vecl, vecr)  # ([ll ll'] [rr rr'])
        else: # dirn == "v":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            vect = hair_t(Q0, hl=env0.l, ht=env0.t, hr=env0.r).T
            vecb = hair_r(Q1, hr=env1.r, hb=env1.b, hl=env1.l).T
            g = (vecb, vect)   # ([bb bb'] [tt tt'])
        return g


    def lbp_(env, max_sweeps=1, iterator_step=None, diff_tol=None):
        r"""
        Perform CTMRG updates :meth:`yastn.tn.fpeps.EnvCTM.update_` until convergence.
        Convergence can be measured based on singular values of CTM environment corner tensors.

        Outputs iterator if ``iterator_step`` is given, which allows
        inspecting ``env``, e.g., calculating expectation values,
        outside of ``ctmrg_`` function after every ``iterator_step`` sweeps.

        Parameters
        ----------
        max_sweeps: int
            Maximal number of sweeps.

        iterator_step: int
            If int, ``ctmrg_`` returns a generator that would yield output after every iterator_step sweeps.
            The default is ``None``, in which case  ``ctmrg_`` sweeps are performed immediately.

        diff_tol: float
            Convergence tolerance for the change of singular values of all corners in a single update.
            The default is None, in which case convergence is not checked and it is up to user to implement
            convergence check.

        Returns
        -------
        Generator if iterator_step is not ``None``.

        LBP_out(NamedTuple)
            NamedTuple including fields:

                * ``sweeps`` number of performed ctmrg updates.
                * ``max_dsv`` norm of singular values change in the worst corner in the last sweep.
                * ``converged`` whether convergence based on ``corner_tol`` has been reached.
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
