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
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Sequence
import logging
from .... import Tensor, rand, ones, eye, YastnError, Leg, tensordot, qr, truncation_mask, vdot, decompress_from_1d
from ....operators import sign_canonical_order
from ... import mps
from .._peps import Peps, Peps2Layers
from .._gates_auxiliary import apply_gate_onsite, gate_product_operator, gate_fix_order
from .._geometry import Bond, Site
from ._env_auxlliary import *
from ._env_window import EnvWindow
from ._env_measure import _measure_nsite
from ._env_boundary_mps import _clear_operator_input

logger = logging.Logger('ctmrg')

@dataclass()
class EnvCTM_local():
    r"""
    Dataclass for CTM environment tensors associated with Peps lattice site.

    Contains fields ``tl``, ``t``, ``tr``, ``r``, ``br``, ``b``, ``bl``, ``l``
    """
    tl : Union[Tensor, None] = None  # top-left
    t  : Union[Tensor, None] = None  # top
    tr : Union[Tensor, None] = None  # top-right
    r  : Union[Tensor, None] = None  # right
    br : Union[Tensor, None] = None  # bottom-right
    b  : Union[Tensor, None] = None  # bottom
    bl : Union[Tensor, None] = None  # bottom-left
    l  : Union[Tensor, None] = None  # left


@dataclass()
class EnvCTM_projectors():
    r""" Dataclass for CTM projectors associated with Peps lattice site. """
    hlt : Union[Tensor, None] = None  # horizontal left top
    hlb : Union[Tensor, None] = None  # horizontal left bottom
    hrt : Union[Tensor, None] = None  # horizontal right top
    hrb : Union[Tensor, None] = None  # horizontal right bottom
    vtl : Union[Tensor, None] = None  # vertical top left
    vtr : Union[Tensor, None] = None  # vertical top right
    vbl : Union[Tensor, None] = None  # vertical bottom left
    vbr : Union[Tensor, None] = None  # vertical bottom right


class CTMRG_out(NamedTuple):
    sweeps : int = 0
    max_dsv : float = None
    converged : bool = False
    max_D : int = 1


class EnvCTM(Peps):
    def __init__(self, psi, init='rand', leg=None):
        r"""
        Environment used in Corner Transfer Matrix Renormalization Group algorithm.

        Note:
            Index convention for environment tensors::

                C--1 0--T--2 0--C
                |       |       |
                0       1       1
                2               0
                |               |
                T--1         1--T
                |               |
                0               2
                1       1       0
                |       |       |
                C--0 2--T--0 1--C

            * enlarged corners: anti-clockwise

        Parameters
        ----------
        psi: yastn.tn.Peps
            PEPS lattice to be contracted using CTM.
            If ``psi`` has physical legs, a double-layer PEPS with no physical legs is formed.

        init: str
            None, 'eye' or 'rand'. Initialization scheme, see :meth:`yastn.tn.fpeps.EnvCTM.reset_`.

        leg: Optional[yastn.Leg]
            Passed to :meth:`yastn.tn.fpeps.EnvCTM.reset_` to further customize initialization.
        """
        super().__init__(psi.geometry)
        self.psi = Peps2Layers(psi) if psi.has_physical() else psi
        if init not in (None, 'rand', 'eye', 'dl'):
            raise YastnError(f"EnvCTM {init=} not recognized. Should be 'rand', 'eye', 'dl', None.")
        for site in self.sites():
            self[site] = EnvCTM_local()
        if init is not None:
            self.reset_(init=init, leg=leg)

    @property
    def config(self):
        return self.psi.config

    def max_D(self):
        return max(max(max(getattr(self[site], dirn).get_shape()) for dirn in ['tl', 'tr', 'bl', 'br'])
                   for site in self.sites())

    # Cloning/Copying/Detaching(view)
    #
    def copy(self) -> EnvCTM:
        env = EnvCTM(self.psi, init=None)
        for site in env.sites():
            for dirn in ['tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r']:
                setattr(env[site], dirn, getattr(self[site], dirn).copy())
        return env

    def clone(self) -> EnvCTM:
        r"""
        Return a clone of the environment preserving the autograd - resulting clone is a part
        of the computational graph. Data of cloned environment tensors is indepedent
        from the originals.
        """
        env = EnvCTM(self.psi, init=None)
        for site in env.sites():
            for dirn in ['tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r']:
                setattr(env[site], dirn, getattr(self[site], dirn).clone())
        return env

    def detach(self) -> EnvCTM:
        r"""
        Return a detached view of the environment - resulting environment is **not** a part
        of the computational graph. Data of detached environment tensors is shared
        with the originals.
        """
        env = EnvCTM(self.psi, init=None)
        for site in env.sites():
            for dirn in ['tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r']:
                setattr(env[site], dirn, getattr(self[site], dirn).detach())
        return env

    def detach_(self):
        r"""
        Detach all environment tensors from the computational graph.
        Data of environment tensors in detached environment is a `view` of the original data.
        """
        for site in self.sites():
            for dirn in ["tl", "tr", "bl", "br", "t", "l", "b", "r"]:
                try:
                    getattr(self[site], dirn)._data.detach_()
                except RuntimeError:
                    setattr(self[site], dirn, getattr(self[site], dirn).detach())


    def shallow_copy(self) -> EnvCTM:
        env = EnvCTM(self.psi, init=None)
        for site in env.sites():
            for dirn in ['tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r']:
                setattr(env[site], dirn, getattr(self[site], dirn))
        return env

    def reset_(self, init='rand', leg=None, **kwargs):
        r"""
        Initialize CTMRG environment.

        Parameters
        ----------
        init: str
            ['eye', 'rand', 'dl']
            For 'eye' starts with identity environments of dimension 1.
            For 'rand' sets environments randomly.
            For 'dl' and Env of double-layer PEPS, trace on-site tensors to initialize environment.

        leg: None | yastn.Leg
            If not provided, random initialization has CTMRG bond dimension set to 1.
            Otherwise, the provided Leg is used to initialize CTMRG virtual legs.
        """
        config = self.psi.config
        leg0 = Leg(config, s=1, t=(config.sym.zero(),), D=(1,))

        if init == 'dl':
            self.init_env_from_onsite_(**kwargs)
        else:
            if leg is None:
                leg = leg0

            for site in self.sites():
                legs = self.psi[site].get_legs()

                for dirn in ('tl', 'tr', 'bl', 'br'):
                    if self.nn_site(site, d=dirn) is None or init == 'eye':
                        setattr(self[site], dirn, eye(config, legs=[leg0, leg0.conj()], isdiag=False))
                    else:
                        setattr(self[site], dirn, rand(config, legs=[leg, leg.conj()]))

                for ind, dirn in enumerate('tlbr'):
                    if self.nn_site(site, d=dirn) is None or init == 'eye':
                        tmp1 = identity_boundary(config, legs[ind].conj())
                        tmp0 = eye(config, legs=[leg0, leg0.conj()], isdiag=False)
                        tmp = tensordot(tmp0, tmp1, axes=((), ())).transpose(axes=(0, 2, 1))
                        setattr(self[site], dirn, tmp)
                    else:
                        setattr(self[site], dirn, rand(config, legs=[leg, legs[ind].conj(), leg.conj()]))

    def init_env_from_onsite_(self, normalize: Union[str,Callable]='inf'):
        r"""
        Initialize CTMRG environment by tracing on-site double-layer tensors A.

        For double-layer PEPS, the top-left corner is initialized as

            C_(bb',rr')= \sum_{ll',tt',s} A_tlbrs A*_t'l'b'r's

        with other corners initialized analogously. The half-row/-column tensors T are
        also initialized by tracing. For top half-column tensors

            T_(ll',bb',rr') = \sum_{tt',s} A_tlbrs A*_t'l'b'r's

        and analogously for the remaining T tensors.

        Args:
            normalize: Normalization of initial environment tensors or custom normalization function
                with signature f(Tensor)->Tensor.
                For 'inf' (default) normalizes magnitude of largest element to 1, i.e. L-infinity norm.
        """
        assert isinstance(self.psi, Peps2Layers), "Initialization by traced double-layer on-site tensors requires double-layer PEPS"
        for site in self.sites():
            for dirn, edge_f in (('t', edge_t), ('l', edge_l), ('b', edge_b), ('r', edge_r),
                                 ('tl', cor_tl), ('tr', cor_tr), ('bl', cor_bl), ('br', cor_br)):
                shifted_site = self.nn_site(site, dirn)

                if shifted_site is not None:
                    A_bra = self.psi.bra[shifted_site]
                    A_ket = self.psi.ket[shifted_site]
                else:
                    A_bra = A_ket = trivial_peps_tensor(self.config)
                T = edge_f(A_bra=A_bra, A_ket=A_ket)
                T = T / T.norm(p=normalize) if isinstance(normalize, str) else normalize(T)
                setattr(self[site], dirn, T)

    def expand_outward_(self):
        """
        Enlarges the environment by one layer of PEPS tensors. No truncation is performed.

        It can be used to build initial "dl" (or "nn") environment from initial "eye" environment.
        """
        env_tmp = EnvCTM(self.psi, init=None)  # empty environments
        #
        for site in self.sites():
            tl = self.nn_site(site, d='tl')
            if tl is not None:
                tmp = self[tl].l @ self[tl].tl @ self[tl].t
                tmp = tensordot(tmp, self.psi[tl], axes=((2, 1), (0, 1)))
                tmp = tmp.fuse_legs(axes=((0, 2), (1, 3)))
                env_tmp[site].tl = tmp / tmp.norm(p='inf')
            else:
                env_tmp[site].tl = self[site].tl
            #
            bl = self.nn_site(site, d='bl')
            if bl is not None:
                tmp = self[bl].b @ self[bl].bl @ self[bl].l
                tmp = tensordot(tmp, self.psi[bl], axes=((2, 1), (1, 2)))
                tmp = tmp.fuse_legs(axes=((0, 3), (1, 2)))
                env_tmp[site].bl = tmp / tmp.norm(p='inf')
            else:
                env_tmp[site].bl = self[site].bl
            #
            tr = self.nn_site(site, d='tr')
            if tr is not None:
                tmp = self[tr].t @ self[tr].tr @ self[tr].r
                tmp = tensordot(tmp, self.psi[tr], axes=((1, 2), (0, 3)))
                tmp = tmp.fuse_legs(axes=((0, 2), (1, 3)))
                env_tmp[site].tr = tmp / tmp.norm(p='inf')
            else:
                env_tmp[site].tr = self[site].tr
            #
            br = self.nn_site(site, d='br')
            if br is not None:
                tmp = self[br].r @ self[br].br @ self[br].b
                tmp = tensordot(tmp, self.psi[br], axes=((2, 1), (2, 3)))
                tmp = tmp.fuse_legs(axes=((0, 2), (1, 3)))
                env_tmp[site].br = tmp / tmp.norm(p='inf')
            else:
                env_tmp[site].br = self[site].br
            #
            # trivial_projectors_(proj, dir, env)
            #
            l = self.nn_site(site, d='l')
            if l is not None:
                l0, _, l2 = self[l].l.get_legs()
                l_t, _, l_b, _ = self.psi[l].get_legs()
                c_tl = env_tmp[site].tl.get_legs(axes=0)
                c_bl = env_tmp[site].bl.get_legs(axes=1)

                if tl:
                    proj_hlt = eye(self.config, legs=[c_tl, c_tl.conj()], isdiag=False)  # proj[l].hlt
                    proj_hlt = proj_hlt.unfuse_legs(axes=0)
                else:
                    proj_hlt = eye(self.config, legs=[l_t.conj(), c_tl.conj()], isdiag=False)  # proj[l].hlt
                    proj_hlt = proj_hlt.add_leg(axis=0, leg=l2.conj())
                if bl:
                    proj_hlb = eye(self.config, legs=[c_bl, c_bl.conj()], isdiag=False)  # proj[l].hlb
                    proj_hlb = proj_hlb.unfuse_legs(axes=0)
                else:
                    proj_hlb = eye(self.config, legs=[l_b.conj(), c_bl.conj()], isdiag=False)  # proj[l].hlb
                    proj_hlb = proj_hlb.add_leg(axis=0, leg=l0.conj())

                tmp = self[l].l @ proj_hlt
                tmp = tensordot(self.psi[l], tmp, axes=((0, 1), (2, 1)))
                tmp = tensordot(proj_hlb, tmp, axes=((0, 1), (2, 0)))
                env_tmp[site].l = tmp / tmp.norm(p='inf')
            else:
                env_tmp[site].l = self[site].l
            #
            r = self.nn_site(site, d='r')
            if r is not None:
                l0, _, l2 = self[r].r.get_legs()
                r_t, _, r_b, _ = self.psi[r].get_legs()
                c_br = env_tmp[site].br.get_legs(axes=0)
                c_tr = env_tmp[site].tr.get_legs(axes=1)

                if br:
                    proj_hrb = eye(self.config, legs=(c_br, c_br.conj()), isdiag=False)  # proj[r].hrb
                    proj_hrb = proj_hrb.unfuse_legs(axes=0)
                else:
                    proj_hrb = eye(self.config, legs=(r_b.conj(), c_br.conj()), isdiag=False)  # proj[r].hrb
                    proj_hrb = proj_hrb.add_leg(axis=0, leg=l2.conj())
                if tr:
                    proj_hrt = eye(self.config, legs=(c_tr, c_tr.conj()), isdiag=False)  # proj[r].hrt
                    proj_hrt = proj_hrt.unfuse_legs(axes=0)
                else:
                    proj_hrt = eye(self.config, legs=(r_t.conj(), c_tr.conj()), isdiag=False)  # proj[r].hrt
                    proj_hrt = proj_hrt.add_leg(axis=0, leg=l0.conj())

                tmp = self[r].r @ proj_hrb
                tmp = tensordot(self.psi[r], tmp, axes=((2, 3), (2, 1)))
                tmp = tensordot(proj_hrt, tmp, axes=((0, 1), (2, 0)))
                env_tmp[site].r = tmp / tmp.norm(p='inf')
            else:
                env_tmp[site].r = self[site].r
            #
            t = self.nn_site(site, d='t')
            if t is not None:
                l0, _, l2 = self[t].t.get_legs()
                _, t_l, _, t_r = self.psi[t].get_legs()
                c_tr = env_tmp[site].tr.get_legs(axes=0)
                c_tl = env_tmp[site].tl.get_legs(axes=1)

                if tr:
                    proj_vtr = eye(self.config, legs=[c_tr, c_tr.conj()], isdiag=False)  # proj[t].vtr
                    proj_vtr = proj_vtr.unfuse_legs(axes=0)
                else:
                    proj_vtr = eye(self.config, legs=[t_r.conj(), c_tr.conj()], isdiag=False)  # proj[t].vtr
                    proj_vtr = proj_vtr.add_leg(axis=0, leg=l2.conj())

                if tl:
                    proj_vtl = eye(self.config, legs=[c_tl, c_tl.conj()], isdiag=False)  # proj[t].vtl
                    proj_vtl = proj_vtl.unfuse_legs(axes=0)
                else:
                    proj_vtl = eye(self.config, legs=[t_l.conj(), c_tl.conj()], isdiag=False)  # proj[t].vtl
                    proj_vtl = proj_vtl.add_leg(axis=0, leg=l0.conj())

                tmp = tensordot(proj_vtl, self[t].t, axes=(0, 0))
                tmp = tensordot(tmp, self.psi[t], axes=((2, 0), (0, 1)))
                tmp = tensordot(tmp, proj_vtr, axes=((1, 3), (0, 1)))
                env_tmp[site].t = tmp / tmp.norm(p='inf')
            else:
                env_tmp[site].t = self[site].t
            #
            b = self.nn_site(site, d='b')
            if b is not None:
                l0, _, l2 = self[b].b.get_legs()
                _, b_l, _, b_r = self.psi[b].get_legs()
                c_bl = env_tmp[site].bl.get_legs(axes=0)
                c_br = env_tmp[site].br.get_legs(axes=1)

                if bl:
                    proj_vbl = eye(self.config, legs=[c_bl, c_bl.conj()], isdiag=False)  # proj[b].vbl
                    proj_vbl = proj_vbl.unfuse_legs(axes=0)
                else:
                    proj_vbl = eye(self.config, legs=[b_l.conj(), c_bl.conj()], isdiag=False)  # proj[b].vbl
                    proj_vbl = proj_vbl.add_leg(axis=0, leg=l2.conj())

                if br:
                    proj_vbr = eye(self.config, legs=[c_br, c_br.conj()], isdiag=False)  # proj[b].vbr
                    proj_vbr = proj_vbr.unfuse_legs(axes=0)
                else:
                    proj_vbr = eye(self.config, legs=[b_r.conj(), c_br.conj()], isdiag=False)  # proj[b].vbr
                    proj_vbr = proj_vbr.add_leg(axis=0, leg=l0.conj())

                tmp = tensordot(proj_vbr, self[b].b, axes=(0, 0))
                tmp = tensordot(tmp, self.psi[b], axes=((2, 0), (2, 3)))
                tmp = tensordot(tmp, proj_vbl, axes=((1, 3), (0, 1)))
                env_tmp[site].b = tmp / tmp.norm(p='inf')
            else:
                env_tmp[site].b = self[site].b
        #
        # modify existing environment in place
        update_old_env_(self, env_tmp)


    def boundary_mps(self, n, dirn):
        r""" Convert environmental tensors of Ctm to an MPS """
        if dirn == 'b':
            H = mps.Mps(N=self.Ny)
            for ny in range(self.Ny):
                H.A[ny] = self[n, ny].b.transpose(axes=(2, 1, 0))
        elif dirn == 'r':
            H = mps.Mps(N=self.Nx)
            for nx in range(self.Nx):
                H.A[nx] = self[nx, n].r
        elif dirn == 't':
            H = mps.Mps(N=self.Ny)
            for ny in range(self.Ny):
                H.A[ny] = self[n, ny].t
        elif dirn == 'l':
            H = mps.Mps(N=self.Nx)
            for nx in range(self.Nx):
                H.A[nx] = self[nx, n].l.transpose(axes=(2, 1, 0))
        return H

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

        # if site is None:
        #     return {site: self.measure_1site(O, site) for site in self.sites()}

        if site is None:
            opdict = _clear_operator_input(O, self.sites())
            return_one = False
        else:
            return_one = True
            opdict = {site: {(): O}}

        out = {}
        for site, ops in opdict.items():
            lenv = self[site]
            ten = self.psi[site]
            vect = (lenv.l @ lenv.tl) @ (lenv.t @ lenv.tr)
            vecb = (lenv.r @ lenv.br) @ (lenv.b @ lenv.bl)

            tmp = tensordot(vect, ten, axes=((2, 1), (0, 1)))
            val_no = tensordot(vecb, tmp, axes=((0, 1, 2, 3), (1, 3, 2, 0))).to_number()

            for nz, op in ops.items():
                if op.ndim == 2:
                    ten.set_operator_(op)
                else:  # for a single-layer Peps, replace with new peps tensor
                    ten = op
                tmp = tensordot(vect, ten, axes=((2, 1), (0, 1)))
                val_op = tensordot(vecb, tmp, axes=((0, 1, 2, 3), (1, 3, 2, 0))).to_number()
                out[site + nz] = val_op / val_no

        if return_one and not isinstance(O, dict):
            return out[site + nz]
        return out

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

        # if bond is None:
        #      return {bond: self.measure_nn(O, P, bond) for bond in self.bonds()}

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
        # else:
        #     raise YastnError("Both operators O and P should have the same ndim==2, or ndim=3.")

        if dirn == 'h':
            vecl = (env0.bl @ env0.l) @ (env0.tl @ env0.t)
            vecr = (env1.tr @ env1.r) @ (env1.br @ env1.b)

            tmp0 = tensordot(ten0, vecl, axes=((0, 1), (2, 1)))
            tmp0 = tensordot(env0.b, tmp0, axes=((1, 2), (0, 2)))
            tmp1 = tensordot(vecr, ten1, axes=((2, 1), (2, 3)))
            tmp1 = tensordot(tmp1, env1.t, axes=((2, 0), (1, 2)))
            val_no = vdot(tmp0, tmp1, conj=(0, 0))

            if O.ndim <= 3:
                ten0.ket = apply_gate_onsite(ten0.ket, G0, dirn='l')
            else:
                ten0 = O
            if P.ndim <= 3:
                ten1.ket = apply_gate_onsite(ten1.ket, G1, dirn='r')
            else:
                ten1 = P

            tmp0 = tensordot(ten0, vecl, axes=((0, 1), (2, 1)))
            tmp0 = tensordot(env0.b, tmp0, axes=((1, 2), (0, 2)))
            tmp1 = tensordot(vecr, ten1, axes=((2, 1), (2, 3)))
            tmp1 = tensordot(tmp1, env1.t, axes=((2, 0), (1, 2)))
            val_op = vdot(tmp0, tmp1, conj=(0, 0))
        else:  # dirn == 'v':
            vect = (env0.l @ env0.tl) @ (env0.t @ env0.tr)
            vecb = (env1.r @ env1.br) @ (env1.b @ env1.bl)

            tmp0 = tensordot(vect, ten0, axes=((2, 1), (0, 1)))
            tmp0 = tensordot(tmp0, env0.r, axes=((1, 3), (0, 1)))
            tmp1 = tensordot(ten1, vecb, axes=((2, 3), (2, 1)))
            tmp1 = tensordot(env1.l, tmp1, axes=((0, 1), (3, 1)))
            val_no = vdot(tmp0, tmp1, conj=(0, 0))

            if O.ndim <= 3:
                ten0.ket = apply_gate_onsite(ten0.ket, G0, dirn='t')
            else:
                ten0 = O

            if P.ndim <= 3:
                ten1.ket = apply_gate_onsite(ten1.ket, G1, dirn='b')
            else:
                ten1 = P

            tmp0 = tensordot(vect, ten0, axes=((2, 1), (0, 1)))
            tmp0 = tensordot(tmp0, env0.r, axes=((1, 3), (0, 1)))
            tmp1 = tensordot(ten1, vecb, axes=((2, 3), (2, 1)))
            tmp1 = tensordot(env1.l, tmp1, axes=((0, 1), (3, 1)))
            val_op = vdot(tmp0, tmp1, conj=(0, 0))

        return val_op / val_no

    def measure_2x2(self, *operators, sites=None):
        r"""
        Calculate expectation value of a product of local operators
        in a :math:`2 \times 2` window within the CTM environment.
        Perform exact contraction of the window.

        Parameters
        ----------
        operators: Sequence[yastn.Tensor]
            List of local operators to calculate <O0_s0 O1_s1 ...>.

        sites: Sequence[tuple[int, int]]
            A list of sites [s0, s1, ...] matching corresponding operators.
        """
        if sites is None or len(operators) != len(sites):
            raise YastnError("Number of operators and sites should match.")

        sign = sign_canonical_order(*operators, sites=sites, f_ordered=self.f_ordered)
        ops = {}
        for n, op in zip(sites, operators):
            ops[n] = ops[n] @ op if n in ops else op

        minx = min(site[0] for site in sites)  # tl corner
        miny = min(site[1] for site in sites)

        maxx = max(site[0] for site in sites)  # br corner
        maxy = max(site[1] for site in sites)

        if minx == maxx and self.nn_site((minx, miny), 'b') is None:
            minx -= 1  # for a finite system
        if miny == maxy and self.nn_site((minx, miny), 'r') is None:
            miny -= 1  # for a finite system

        tl = Site(minx, miny)
        tr = self.nn_site(tl, 'r')
        br = self.nn_site(tl, 'br')
        bl = self.nn_site(tl, 'b')
        window = [tl, tr, br, bl]

        if any(site not in window for site in sites):
            raise YastnError("Sites do not form a 2x2 window.")

        ten_tl = self.psi[tl]
        ten_tr = self.psi[tr]
        ten_br = self.psi[br]
        ten_bl = self.psi[bl]

        vec_tl = self[tl].l @ (self[tl].tl @ self[tl].t)
        vec_tr = self[tr].t @ (self[tr].tr @ self[tr].r)
        vec_br = self[br].r @ (self[br].br @ self[br].b)
        vec_bl = self[bl].b @ (self[bl].bl @ self[bl].l)

        cor_tl = tensordot(vec_tl, ten_tl, axes=((2, 1), (0, 1)))
        cor_tl = cor_tl.fuse_legs(axes=((0, 2), (1, 3)))
        cor_tr = tensordot(vec_tr, ten_tr, axes=((1, 2), (0, 3)))
        cor_tr = cor_tr.fuse_legs(axes=((0, 2), (1, 3)))
        cor_br = tensordot(vec_br, ten_br, axes=((2, 1), (2, 3)))
        cor_br = cor_br.fuse_legs(axes=((0, 2), (1, 3)))
        cor_bl = tensordot(vec_bl, ten_bl, axes=((2, 1), (1, 2)))
        cor_bl = cor_bl.fuse_legs(axes=((0, 3), (1, 2)))

        val_no = vdot(cor_tl @ cor_tr, tensordot(cor_bl, cor_br, axes=(0, 1)), conj=(0, 0))

        if tl in ops:
            ten_tl.set_operator_(ops[tl])
        if bl in ops:
            ten_bl.set_operator_(ops[bl])
            ten_bl.add_charge_swaps_(ops[bl].n, axes='k1')
            ten_tl.add_charge_swaps_(ops[bl].n, axes=['b3', 'k4'])
        if tr in ops:
            ten_tr.set_operator_(ops[tr])
            ten_tr.add_charge_swaps_(ops[tr].n, axes='b0')
            ten_tl.add_charge_swaps_(ops[tr].n, axes=['k2', 'k4'])
        if br in ops:
            ten_br.set_operator_(ops[br])
            ten_br.add_charge_swaps_(ops[br].n, axes='k1')
            ten_tr.add_charge_swaps_(ops[br].n, axes=['b3', 'b0', 'k4'])
            ten_tl.add_charge_swaps_(ops[br].n, axes=['k2', 'k4'])

        if ten_tl.has_operator_or_swap():
            cor_tl = tensordot(vec_tl, ten_tl, axes=((2, 1), (0, 1)))
            cor_tl = cor_tl.fuse_legs(axes=((0, 2), (1, 3)))
        if ten_bl.has_operator_or_swap():
            cor_bl = tensordot(vec_bl, ten_bl, axes=((2, 1), (1, 2)))
            cor_bl = cor_bl.fuse_legs(axes=((0, 3), (1, 2)))
        if ten_tr.has_operator_or_swap():
            cor_tr = tensordot(vec_tr, ten_tr, axes=((1, 2), (0, 3)))
            cor_tr = cor_tr.fuse_legs(axes=((0, 2), (1, 3)))
        if ten_br.has_operator_or_swap():
            cor_br = tensordot(vec_br, ten_br, axes=((2, 1), (2, 3)))
            cor_br = cor_br.fuse_legs(axes=((0, 2), (1, 3)))

        val_op = vdot(cor_tl @ cor_tr, tensordot(cor_bl, cor_br, axes=(0, 1)), conj=(0, 0))
        return sign * val_op / val_no


    def measure_line(self, *operators, sites=None):
        r"""
        Calculate expectation value of a product of local opertors
        along a horizontal or vertical line within CTM environment.
        Perform exact contraction of a width-one window.

        Parameters
        ----------
        operators: Sequence[yastn.Tensor]
            List of local operators to calculate <O0_s0 O1_s1 ...>.

        sites: Sequence[tuple[int, int]]
            List of sites that should match operators.
        """
        if sites is None or len(operators) != len(sites):
            raise YastnError("Number of operators and sites should match.")

        sign = sign_canonical_order(*operators, sites=sites, f_ordered=self.f_ordered)
        ops = {}
        for n, op in zip(sites, operators):
            ops[n] = ops[n] @ op if n in ops else op

        xs = sorted(set(site[0] for site in sites))
        ys = sorted(set(site[1] for site in sites))
        if len(xs) > 1 and len(ys) > 1:
            raise YastnError("Sites should form a horizontal or vertical line.")

        env_win = EnvWindow(self, (xs[0], xs[-1] + 1), (ys[0], ys[-1] + 1))
        horizontal = (len(xs) == 1)
        if horizontal:
            vr = env_win[xs[0], 't']
            tm = env_win[xs[0], 'h']
            vl = env_win[xs[0], 'b'].conj()
            axes_op = 'b0'
            axes_string = ('b0', 'k2', 'k4')
        else:  # vertical
            vr = env_win[ys[0], 'l']
            tm = env_win[ys[0], 'v']
            vl = env_win[ys[0], 'r'].conj()
            axes_op = 'k1'
            axes_string = ('k1', 'k4', 'b3')

        val_no = mps.vdot(vl, tm, vr)

        for site, op in ops.items():
            ind = site[0] - xs[0] + site[1] - ys[0] + 1
            if op.ndim == 2:
                tm[ind].set_operator_(op)
                tm[ind].add_charge_swaps_(op.n, axes=axes_op)
                for ii in range(1, ind):
                    tm[ii].add_charge_swaps_(op.n, axes=axes_string)
            else:
                axes = (1, 2, 3, 0) if horizontal else (0, 3, 2, 1)
                tm[ind] = op.transpose(axes=axes)

        val_op = mps.vdot(vl, tm, vr)
        return sign * val_op / val_no


    def measure_nsite(self, *operators, sites=None) -> float:
        r"""
        Calculate expectation value of a product of local operators.
        Perform approximate contraction of a windows of PEPS sites
        within CTM environment using boundary MPS.
        The size of the window is taken to include provided sites.

        Parameters
        ----------
        operators: Sequence[yastn.Tensor]
            List of local operators to calculate <O0_s0 O1_s1 ...>.

        sites: Sequence[int]
            A list of sites [s0, s1, ...] matching corresponding operators.
        """
        xrange = (min(site[0] for site in sites), max(site[0] for site in sites) + 1)
        yrange = (min(site[1] for site in sites), max(site[1] for site in sites) + 1)
        env_win = EnvWindow(self, xrange, yrange)
        dirn = 'lr' if (xrange[1] - xrange[0]) >= (yrange[1] - yrange[0]) else 'tb'
        return _measure_nsite(env_win, *operators, sites=sites, dirn=dirn)


    def measure_2site(self, O, P, xrange, yrange, opts_svd=None, opts_var=None, bonds='<') -> dict[Site, list]:
        r"""
        Calculate 2-point correlations <O P> between top-left corner of the window, and all sites in the window.

        wip: other combinations of 2-sites and fermionically-nontrivial operators will be coverad latter.

        Parameters
        ----------
        O, P: yastn.Tensor
            one-site operators

        xrange: tuple[int, int]
            range of rows forming a window, [r0, r1); r0 included, r1 excluded.

        yrange: tuple[int, int]
            range of columns forming a window.

        opts_svd: dict
            Options passed to :meth:`yastn.linalg.svd` used to truncate virtual spaces of boundary MPSs used in sampling.
            The default is ``None``, in which case take ``D_total`` as the largest dimension from CTM environment.

        opts_svd: dict
            Options passed to :meth:`yastn.tn.mps.compression_` used in the refining of boundary MPSs.
            The default is ``None``, in which case make 2 variational sweeps.

        bonds: tuple[int, int] | Sequence[tuple[int, int]] | str
            Which 2-site correlators to calculate.
            For a single bond, tuple[int, int], return float. Otherwise, return dict[bond, float].
            It is possible to provide a string to build a list of bonds as:

            * '<' for all i < j.
            * '=' for all i == j.
            * '>' for all i > j.
            * 'a' for all i, j; equivalent to "<=>".

            The default is '<'.
        """
        env_win = EnvWindow(self, xrange, yrange)
        return env_win.measure_2site(O, P, opts_svd=opts_svd, opts_var=opts_var)


    def sample(self, xrange, yrange, projectors, number=1, opts_svd=None, opts_var=None, progressbar=False, return_info=False) -> dict[Site, list]:
        r"""
        Sample random configurations from PEPS. Output a dictionary linking sites with lists of sampled projectors` keys for each site.

        It does not check whether projectors sum up to identity -- probabilities of provided projectors get normalized to one.
        If negative probabilities are observed (signaling contraction errors), ``error = max(abs(negatives))``,
        and all probabilities below that error level are fixed to error (before consecutive renormalization of probabilities to one).

        Parameters
        ----------
        xrange: tuple[int, int]
            range of rows to sample from, [r0, r1); r0 included, r1 excluded.

        yrange: tuple[int, int]
            range of columns to sample from.

        projectors: Dict[Any, yast.Tensor] | Sequence[yast.Tensor] | Dict[Site, Dict[Any, yast.Tensor]]
            Projectors to sample from. We can provide a dict(key: projector), where the sampled results will be given as keys,
            and the same set of projectors is used at each site. For a list of projectors, the keys follow from enumeration.
            Finally, we can provide a dictionary between each site and sets of projectors.

        number: int
            Number of drawn samples.

        opts_svd: dict
            Options passed to :meth:`yastn.linalg.svd` used to truncate virtual spaces of boundary MPSs used in sampling.
            The default is ``None``, in which case take ``D_total`` as the largest dimension from CTM environment.

        opts_var: dict
            Options passed to :meth:`yastn.tn.mps.compression_` used in the refining of boundary MPSs.
            The default is ``None``, in which case make 2 variational sweeps.

        progressbar: bool
            Whether to display progressbar. The default is ``False``.

        return_info: bool
            Whether to include in the outputted dictionary a field ``info`` with dictionary
            that contains information about the amplitude of contraction errors
            (largest negative probability), D_total, etc. The default is ``False``.
        """
        env_win = EnvWindow(self, xrange, yrange)
        return env_win.sample(projectors, number, opts_svd, opts_var, progressbar, return_info)

    def post_evolution_(env, bond, **kwargs):
        pass

    def update_(env, opts_svd, method='2site', **kwargs):
        r"""
        Perform one step of CTMRG update. Environment tensors are updated in place.

        The function performs a CTMRG update for a square lattice using the corner transfer matrix
        renormalization group (CTMRG) algorithm. The update is performed in two steps: a horizontal move
        and a vertical move. The projectors for each move are calculated first, and then the tensors in
        the CTM environment are updated using the projectors. The boundary conditions of the lattice
        determine whether trivial projectors are needed for the move.

        Parameters
        ----------
        opts_svd: dict
            A dictionary of options to pass to SVD truncation algorithm.
            This sets EnvCTM bond dimension.

        method: str
            '2site' or '1site'. The default is '2site'.
            '2site' uses the standard 4x4 enlarged corners, allowing to enlarge EnvCTM bond dimension.
            '1site' uses smaller 4x2 corners. It is significantly faster, but is less stable and
            does not allow to grow EnvCTM bond dimension.

        checkpoint_move: bool
            Whether to use (reentrant) checkpointing for the move. The default is ``False``

        Returns
        -------
        proj: Peps structure loaded with CTM projectors related to all lattice site.
        """
        if all(s not in opts_svd for s in ('tol', 'tol_block')):
            opts_svd['tol'] = 1e-14
        if method not in ('1site', '2site', '2site+'):
            raise YastnError(f"CTM update {method=} not recognized. Should be '1site', '2site' or '2site+')")
        checkpoint_move= kwargs.get('checkpoint_move',False)

        #
        # Empty structure for projectors
        proj = Peps(env.geometry)
        for site in proj.sites(): proj[site] = EnvCTM_projectors()

        def _compress_env(env):
            shallow= {
                'psi': {site: env.psi.bra[site] for site in env.sites()} if isinstance(env.psi,Peps2Layers) \
                    else {site: env.psi[site] for site in env.sites()},
                'env': tuple( env_t for site in env.sites() for k,env_t in env[site].__dict__.items() )}
            dtypes= set(tuple( t.yastn_dtype for t in shallow['psi'].values()) + tuple(t.yastn_dtype for t in shallow['env']))
            assert len(dtypes)<2, f"CTM update: all tensors of state and environment should have the same dtype, got {dtypes}"
            unrolled= {'psi': {site: t.compress_to_1d() for site,t in shallow['psi'].items()},
                'env': tuple(t.compress_to_1d() for t in shallow['env'])}
            meta= {'psi': {site: t_and_meta[1] for site,t_and_meta in unrolled['psi'].items()}, 'env': tuple(meta for t,meta in unrolled['env'])}
            data= tuple( t for t,m in unrolled['psi'].values())+tuple( t for t,m in unrolled['env'])
            return data, meta

        def _compress_proj(proj, empty_proj):
            data, meta= tuple(zip( *(t.compress_to_1d() if not (t is None) else empty_proj.compress_to_1d() \
                for site in proj.sites() for t in proj[site].__dict__.values()) ))
            return data, meta

        def _decompress_env(data,meta):
            loc_bra= Peps(env.geometry, {site: decompress_from_1d(t,t_meta) for site,t,t_meta in zip(env.sites(),data[:len(env.sites())],meta['psi'].values())})
            loc_env = EnvCTM( Peps2Layers(loc_bra) if isinstance(env.psi,Peps2Layers) else loc_bra, init=None)

            # assign backend tensors
            #
            data_env= data[len(env.sites()):]
            for i,site in enumerate(loc_env.sites()):
                for env_t,t,t_meta in zip(loc_env[site].__dict__.keys(),data_env[i*8:(i+1)*8],meta['env'][i*8:(i+1)*8]):
                    setattr(loc_env[site],env_t,decompress_from_1d(t,t_meta))
            return loc_env

        #
        # get projectors and compute updated env tensors
        # TODO currently supports only <psi|psi> for double-layer peps
        for d in ['lr', 'tb']:

            if checkpoint_move:
                outputs_meta= {}

                # extract raw parametric tensors as a tuple
                inputs_t, inputs_meta= _compress_env(env)

                def f_update_core_2dir(move_d,loc_im,*inputs_t):
                    loc_env= _decompress_env(inputs_t,loc_im)
                    env_tmp, proj_tmp= _update_core_2dir(loc_env, move_d, opts_svd, method=method, **kwargs)

                    update_old_env_(loc_env, env_tmp)

                    # return backend tensors - only environment and projectors
                    #
                    out_env_data, out_env_meta= _compress_env(loc_env)
                    out_proj_data, out_proj_meta= _compress_proj(proj_tmp, Tensor(config=next(iter(out_env_meta['psi'].values()))['config']))

                    outputs_meta['env']= out_env_meta['env']
                    outputs_meta['proj']= out_proj_meta

                    return out_env_data[len(loc_env.sites()):] + out_proj_data

                if env.config.backend.BACKEND_ID == "torch":
                    if checkpoint_move=='reentrant':
                        use_reentrant= True
                    elif checkpoint_move=='nonreentrant':
                        use_reentrant= False
                    checkpoint_F= env.config.backend.checkpoint
                    outputs= checkpoint_F(f_update_core_2dir,d,inputs_meta,*inputs_t,\
                                      **{'use_reentrant': use_reentrant, 'debug': False})
                else:
                    raise RuntimeError(f"CTM update: checkpointing not supported for backend {env.config.BACKEND_ID}")

                # update tensors of env and proj
                for i,site in enumerate(env.sites()):
                    for env_t,t,t_meta in zip(env[site].__dict__.keys(),outputs[i*8:(i+1)*8],outputs_meta['env'][i*8:(i+1)*8]):
                        setattr(env[site],env_t,decompress_from_1d(t,t_meta))

                for i,site in enumerate(proj.sites()):
                    for proj_t,t,t_meta in zip(proj[site].__dict__.keys(),outputs[8*len(env.sites()):][i*8:(i+1)*8],outputs_meta['proj'][i*8:(i+1)*8]):
                        setattr(proj[site],proj_t, decompress_from_1d(t,t_meta) if t_meta['struct'].size>0 else None)

            else:
                env_tmp, proj_tmp= _update_core_2dir(env, d, opts_svd, method=method, **kwargs)
                update_old_env_(env, env_tmp)
                store_projectors_(proj, proj_tmp)
        return proj


    def bond_metric(self, Q0, Q1, s0, s1, dirn):
        r"""
        Calculates Full-Update metric tensor.

        ::

            If dirn == 'h':

                tl═══t═══════t═══tr
                ║    ║       ║    ║
                l════Q0══  ══Q1═══r
                ║    ║       ║    ║
                bl═══b═══════b═══br


            If dirn == 'v':

                tl═══t═══tr
                ║    ║    ║
                l═══0Q0═══r
                ║    ╳    ║
                l═══1Q1═══r
                ║    ║    ║
                bl═══b═══br
        """
        env0, env1 = self[s0], self[s1]
        if dirn == "h":
            assert self.psi.nn_site(s0, (0, 1)) == s1
            vecl = append_vec_tl(Q0, Q0, env0.l @ (env0.tl @ env0.t))
            vecl = tensordot(env0.b @ env0.bl, vecl, axes=((2, 1), (0, 1)))
            vecr = append_vec_br(Q1, Q1, env1.r  @ (env1.br @ env1.b))
            vecr = tensordot(env1.t @ env1.tr, vecr, axes=((2, 1), (0, 1)))
            g = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            vect = append_vec_tl(Q0, Q0, env0.l @ (env0.tl @ env0.t))
            vect = tensordot(vect, env0.tr @ env0.r, axes=((2, 3), (0, 1)))
            vecb = append_vec_br(Q1, Q1, env1.r @ (env1.br @ env1.b))
            vecb = tensordot(vecb, env1.bl @ env1.l, axes=((2, 3), (0, 1)))
            g = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']

        g = g / g.trace(axes=(0, 1)).to_number()
        return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))

    def save_to_dict(self) -> dict:
        r"""
        Serialize EnvCTM into a dictionary.
        """
        psi = self.psi
        if isinstance(psi, Peps2Layers):
            psi = psi.ket

        d = {'class': 'EnvCTM',
             'psi': psi.save_to_dict(),
             'data': {}}
        for site in self.sites():
            d_local = {dirn: getattr(self[site], dirn).save_to_dict()
                       for dirn in ['tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r']}
            d['data'][site] = d_local
        return d

    def check_corner_bond_dimension(env, disp=False):

        dict_bond_dimension = {}
        dict_symmetric_sector = {}
        for site in env.sites():
            if disp:
                print(site)
            corners = [env[site].tl, env[site].bl, env[site].br, env[site].tr]
            corners_id = ["tl", "bl", "br", "tr"]
            for ii in range(4):
                dict_symmetric_sector[site, corners_id[ii]] = []
                dict_bond_dimension[site, corners_id[ii]] = []
                if disp:
                    print(corners_id[ii])
                for leg in range (0, 2):
                    temp_t = []
                    temp_D = []
                    for it in range(len(corners[ii].get_legs()[leg].t)):
                        temp_t.append(corners[ii].get_legs()[leg].t[it])
                        temp_D.append(corners[ii].get_legs()[leg].D[it])
                    if disp:
                        print(temp_t)
                        print(temp_D)
                    dict_symmetric_sector[site, corners_id[ii]].append(temp_t)
                    dict_bond_dimension[site, corners_id[ii]].append(temp_D)
        return [dict_bond_dimension, dict_symmetric_sector]

    def initialize_ctm_with_old_ctm(env, psi, env_old):
        for site in psi.sites():
            env[site].tl = env_old[site].tl
            env[site].tr = env_old[site].tr
            env[site].bl = env_old[site].bl
            env[site].br = env_old[site].br
            env[site].l = env_old[site].l
            env[site].r = env_old[site].r
            env[site].b = env_old[site].b
            env[site].t = env_old[site].t

    def ctmrg_(env, opts_svd=None, method='2site', max_sweeps=1, iterator_step=None, corner_tol=None, truncation_f : Callable=None, **kwargs):
        r"""
        Perform CTMRG updates :meth:`yastn.tn.fpeps.EnvCTM.update_` until convergence.
        Convergence can be measured based on singular values of CTM environment corner tensors.

        Outputs iterator if ``iterator_step`` is given, which allows
        inspecting ``env``, e.g., calculating expectation values,
        outside of ``ctmrg_`` function after every ``iterator_step`` sweeps.

        Parameters
        ----------
        opts_svd: dict
            A dictionary of options to pass to SVD truncation algorithm.
            This sets EnvCTM bond dimension.

        method: str
            '2site' or '1site'. The default is '2site'.
            '2site' uses the standard 4x4 enlarged corners, allowing to enlarge EnvCTM bond dimension.
            '1site' uses smaller 4x2 corners. It is significantly faster, but is less stable and
            does not allow to grow EnvCTM bond dimension.

        max_sweeps: int
            Maximal number of sweeps.

        iterator_step: int
            If int, ``ctmrg_`` returns a generator that would yield output after every iterator_step sweeps.
            The default is ``None``, in which case  ``ctmrg_`` sweeps are performed immediately.

        corner_tol: float
            Convergence tolerance for the change of singular values of all corners in a single update.
            The default is None, in which case convergence is not checked and it is up to user to implement
            convergence check.

        truncation_f:
            Custom projector truncation function with signature ``truncation_f(S: Tensor)->Tensor``, consuming
            rank-1 tensor with singular values. If provided, truncation parameters passed to SVD decomposition
            are ignored.

        checkpoint_move: Union[str, bool]
            Whether to use checkpointing for the CTM updates. The default is ``False``.
            Otherwise, in case of PyTorch backend it can be set to 'reentrant' for reentrant checkpointing
            or 'nonreentrant' for non-reentrant checkpointing, see https://pytorch.org/docs/stable/checkpoint.html.

        use_qr: bool
            Whether to include intermediate QR decomposition while calculating projectors.
            The default is True.

        Returns
        -------
        Generator if iterator_step is not ``None``.

        CTMRG_out(NamedTuple)
            NamedTuple including fields:

                * ``sweeps`` number of performed ctmrg updates.
                * ``max_dsv`` norm of singular values change in the worst corner in the last sweep.
                * ``converged`` whether convergence based on ``corner_tol`` has been reached.
        """
        if "checkpoint_move" in kwargs:
            if env.config.backend.BACKEND_ID == "torch":
                assert kwargs["checkpoint_move"] in ['reentrant','nonreentrant',False], f"Invalid choice for {kwargs['checkpoint_move']}"
        kwargs["truncation_f"]= truncation_f
        tmp = _iterate_ctmrg_(env, opts_svd, method, max_sweeps, iterator_step, corner_tol, **kwargs)
        return tmp if iterator_step else next(tmp)

    iterate_ = ctmrg_  #  allow using EnvCtm.iterate_() instead of allow using EnvCtm.ctmrg_()


def ctm_conv_corner_spec(env : EnvCTM, history : Sequence[dict[tuple[Site,str],Tensor]]=[],
                         corner_tol : Union[None,float]=1.0e-8)->tuple[bool,float,Sequence[dict[tuple[Site,str],Tensor]]]:
    """
    Evaluate convergence of CTM by computing the difference of environment corner spectra between consecutive CTM steps.
    """
    history.append(calculate_corner_svd(env))
    max_dsv = max((history[-1][k] - history[-2][k]).norm().item() for k in history[-1]) if len(history)>1 else float('Nan')

    return (corner_tol is not None and max_dsv < corner_tol), max_dsv, history


def _iterate_ctmrg_(env, opts_svd, method, max_sweeps, iterator_step, corner_tol, **kwargs):
    """ Generator for ctmrg_(). """
    max_dsv, converged = None, False
    for sweep in range(1, max_sweeps + 1):
        env.update_(opts_svd=opts_svd, method=method, **kwargs)

        # use default CTM convergence check
        if corner_tol is not None:
            if sweep==1: history = []
            converged, max_dsv, history= ctm_conv_corner_spec(env.detach(), history, corner_tol)
            logging.info(f'Sweep = {sweep:03d}; max_diff_corner_singular_values = {max_dsv:0.2e}')

            if converged:
                break

        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, max_D=env.max_D(), converged=converged)
    yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, max_D=env.max_D(), converged=converged)


def calculate_corner_svd(env : dict[tuple[Site,str],Tensor]):
    """
    Return normalized SVD spectra, with largest singular value set to unity, of all corner tensors of environment.
    The corners are indexed by pair of Site and corner identifier.
    """
    corner_sv = {}
    for site in env.sites():
        corner_sv[site, 'tl'] = env[site].tl.svd(compute_uv=False)
        corner_sv[site, 'tr'] = env[site].tr.svd(compute_uv=False)
        corner_sv[site, 'bl'] = env[site].bl.svd(compute_uv=False)
        corner_sv[site, 'br'] = env[site].br.svd(compute_uv=False)
    for k, v in corner_sv.items():
        corner_sv[k] = v / v.norm(p='inf')
    return corner_sv


def _update_core_2dir(env, dir : str, opts_svd : dict, **kwargs):
        assert dir in ['lr', 'rl', 'tb', 'bt'], "Invalid directions"
        update_env_= update_env_horizontal_ if dir in ['lr', 'rl'] else update_env_vertical_
        method= kwargs.get('method', '2site')
        if method == '2site':
            update_proj_ = update_2site_projectors_ 
        elif method == '1site': 
            update_proj_ = update_1site_projectors_ 
        elif method == '2site+': 
            update_proj_ = update_extended_2site_projectors_ 
        #
        # Empty structure for projectors
        proj = Peps(env.geometry)
        for site in proj.sites():
            proj[site] = EnvCTM_projectors()
        #
        # projectors
        for site in env.sites():
            update_proj_(proj, site, dir, env, opts_svd, **kwargs)
        trivial_projectors_(proj, dir, env)  # fill (trivial) projectors on edges
        #
        # update move
        env_tmp = EnvCTM(env.psi, init=None)  # empty environments
        for site in env.sites():
            update_env_(env_tmp, site, env, proj, dir)

        return env_tmp, proj


    # TODO
    # executes single directional, including comp. of the projectors
    # def _update_core_1dir():


def update_2site_projectors_(proj, site, dirn, env, opts_svd, **kwargs):
    r"""
    Calculate new projectors for CTM moves from 4x4 extended corners.
    """
    psi = env.psi
    sites = [psi.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))]
    if None in sites:
        return

    use_qr = kwargs.get("use_qr", True)

    tl, tr, bl, br = sites

    cor_tl = env[tl].l @ env[tl].tl @ env[tl].t
    cor_tl = tensordot(cor_tl, psi[tl], axes=((2, 1), (0, 1)))
    cor_tl = cor_tl.fuse_legs(axes=((0, 2), (1, 3)))

    cor_bl = env[bl].b @ env[bl].bl @ env[bl].l
    cor_bl = tensordot(cor_bl, psi[bl], axes=((2, 1), (1, 2)))
    cor_bl = cor_bl.fuse_legs(axes=((0, 3), (1, 2)))

    cor_tr = env[tr].t @ env[tr].tr @ env[tr].r
    cor_tr = tensordot(cor_tr, psi[tr], axes=((1, 2), (0, 3)))
    cor_tr = cor_tr.fuse_legs(axes=((0, 2), (1, 3)))

    cor_br = env[br].r @ env[br].br @ env[br].b
    cor_br = tensordot(cor_br, psi[br], axes=((2, 1), (2, 3)))
    cor_br = cor_br.fuse_legs(axes=((0, 2), (1, 3)))

    if ('l' in dirn) or ('r' in dirn):
        cor_tt = cor_tl @ cor_tr  # b(left) b(right)
        cor_bb = cor_br @ cor_bl  # t(right) t(left)

    if 'r' in dirn:
        _, r_t = qr(cor_tt, axes=(0, 1)) if use_qr else (None, cor_tt)
        _, r_b = qr(cor_bb, axes=(1, 0)) if use_qr else (None, cor_bb.T)
        proj[tr].hrb, proj[br].hrt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)

    if 'l' in dirn:
        _, r_t = qr(cor_tt, axes=(1, 0)) if use_qr else (None, cor_tt.T)
        _, r_b = qr(cor_bb, axes=(0, 1)) if use_qr else (None, cor_bb)
        proj[tl].hlb, proj[bl].hlt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)

    if ('t' in dirn) or ('b' in dirn):
        cor_ll = cor_bl @ cor_tl  # l(bottom) l(top)
        cor_rr = cor_tr @ cor_br  # r(top) r(bottom)

    if 't' in dirn:
        _, r_l = qr(cor_ll, axes=(0, 1)) if use_qr else (None, cor_ll)
        _, r_r = qr(cor_rr, axes=(1, 0)) if use_qr else (None, cor_rr.T)
        proj[tl].vtr, proj[tr].vtl = proj_corners(r_l, r_r, opts_svd=opts_svd, **kwargs)

    if 'b' in dirn:
        _, r_l = qr(cor_ll, axes=(1, 0)) if use_qr else (None, cor_ll.T)
        _, r_r = qr(cor_rr, axes=(0, 1)) if use_qr else (None, cor_rr)
        proj[bl].vbr, proj[br].vbl = proj_corners(r_l, r_r, opts_svd=opts_svd, **kwargs)




def update_extended_2site_projectors_(proj, site, dirn, env, opts_svd, **kwargs):
    r"""
    Calculate new projectors for CTM moves from 4x4 extended corners.
    """
    psi = env.psi
    sites = [psi.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))]
    if None in sites:
        return

    use_qr = kwargs.get("use_qr", True)

    tl, tr, bl, br = sites

    if ('l' in dirn) or ('r' in dirn):
        sl = psi[tl].get_shape(axes=2)
        ltl = env.nn_site(tl, d='l')
        lbl = env.nn_site(bl, d='l')
        if sl == 1 and ltl and lbl:
            cor_ltl = env[ltl].l @ env[ltl].tl @ env[ltl].t
            cor_ltl = tensordot(cor_ltl, psi[ltl], axes=((2, 1), (0, 1)))
            cor_tl = tensordot(cor_ltl, env[tl].t, axes=(1, 0)) 
            cor_tl = tensordot(cor_tl, psi[tl], axes=((3, 2), (0, 1)))
            cor_tl = cor_tl.fuse_legs(axes=((0, 1, 3), (2, 4)))

            cor_lbl = env[lbl].b @ env[lbl].bl @ env[lbl].l
            cor_lbl = tensordot(cor_lbl, psi[lbl], axes=((2, 1), (1, 2)))
            cor_bl = env[bl].b @ cor_lbl
            cor_bl = tensordot(cor_bl, psi[bl], axes=((4, 1), (1, 2)))
            cor_bl = cor_bl.fuse_legs(axes=((0, 4), (1, 2, 3)))
        else:
            cor_tl = env[tl].l @ env[tl].tl @ env[tl].t
            cor_tl = tensordot(cor_tl, psi[tl], axes=((2, 1), (0, 1)))
            cor_tl = cor_tl.fuse_legs(axes=((0, 2), (1, 3)))

            cor_bl = env[bl].b @ env[bl].bl @ env[bl].l
            cor_bl = tensordot(cor_bl, psi[bl], axes=((2, 1), (1, 2)))
            cor_bl = cor_bl.fuse_legs(axes=((0, 3), (1, 2)))


        sr = psi[tr].get_shape(axes=2)
        rtr = env.nn_site(tr, d='r')
        rbr = env.nn_site(br, d='r')
        if sr == 1 and rtr and rbr:
            cor_rtr = env[rtr].t @ env[rtr].tr @ env[rtr].r
            cor_rtr = tensordot(cor_rtr, psi[rtr], axes=((1, 2), (0, 3)))
            cor_tr = env[tr].t @ cor_rtr
            cor_tr = tensordot(cor_tr, psi[tr], axes=((1, 3), (0, 3)))
            cor_tr = cor_tr.fuse_legs(axes=((0, 3), (1, 2, 4)))

            cor_rbr = env[rbr].r @ env[rbr].br @ env[rbr].b
            cor_rbr = tensordot(cor_rbr, psi[rbr], axes=((2, 1), (2, 3)))
            cor_br = tensordot(cor_rbr, env[br].b, axes=(1, 0))
            cor_br = tensordot(cor_br, psi[br], axes=((3, 2), (2, 3)))
            cor_br = cor_br.fuse_legs(axes=((0, 1, 3), (2, 4)))
        else:
            cor_tr = env[tr].t @ env[tr].tr @ env[tr].r
            cor_tr = tensordot(cor_tr, psi[tr], axes=((1, 2), (0, 3)))
            cor_tr = cor_tr.fuse_legs(axes=((0, 2), (1, 3)))

            cor_br = env[br].r @ env[br].br @ env[br].b
            cor_br = tensordot(cor_br, psi[br], axes=((2, 1), (2, 3)))
            cor_br = cor_br.fuse_legs(axes=((0, 2), (1, 3)))

        cor_tt = cor_tl @ cor_tr  # b(left) b(right)
        cor_bb = cor_br @ cor_bl  # t(right) t(left)

    if 'r' in dirn:
        _, r_t = qr(cor_tt, axes=(0, 1)) if use_qr else (None, cor_tt)
        _, r_b = qr(cor_bb, axes=(1, 0)) if use_qr else (None, cor_bb.T)
        proj[tr].hrb, proj[br].hrt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)

    if 'l' in dirn:
        _, r_t = qr(cor_tt, axes=(1, 0)) if use_qr else (None, cor_tt.T)
        _, r_b = qr(cor_bb, axes=(0, 1)) if use_qr else (None, cor_bb)
        proj[tl].hlb, proj[bl].hlt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)

    if ('t' in dirn) or ('b' in dirn):

        st = psi[tl].get_shape(axes=3)
        ttl = env.nn_site(tl, d='t')
        ttr = env.nn_site(tr, d='t')
        
        if st == 1 and ttl and ttr:
            cor_ttl = env[ttl].l @ env[ttl].tl @ env[ttl].t
            cor_ttl = tensordot(cor_ttl, psi[ttl], axes=((2, 1), (0, 1)))
            cor_tl = env[tl].l @ cor_ttl
            cor_tl = tensordot(cor_tl, psi[tl], axes=((3, 1), (0, 1)))
            cor_tl = cor_ltl.fuse_legs(axes=((0, 3), (1, 2, 4)))

            cor_ttr = env[ttr].t @ env[ttr].tr @ env[ttr].r
            cor_ttr = tensordot(cor_ttr, psi[ttr], axes=((1, 2), (0, 3)))
            cor_tr = tensordot(cor_ttr, env[tr].r, axes=(1, 0))
            cor_tr = tensordot(cor_tr, psi[tr], axes=((2, 3), (0, 3)))
            cor_tr = cor_tr.fuse_legs(axes=((0, 1, 3), (2, 4)))
        else:
            cor_tl = env[tl].l @ env[tl].tl @ env[tl].t
            cor_tl = tensordot(cor_tl, psi[tl], axes=((2, 1), (0, 1)))
            cor_tl = cor_tl.fuse_legs(axes=((0, 2), (1, 3)))

            cor_tr = env[tr].t @ env[tr].tr @ env[tr].r
            cor_tr = tensordot(cor_tr, psi[tr], axes=((1, 2), (0, 3)))
            cor_tr = cor_tr.fuse_legs(axes=((0, 2), (1, 3)))

        sb = psi[bl].get_shape(axes=3)
        bbl = env.nn_site(bl, d='b')
        bbr = env.nn_site(br, d='b')
        if sb == 1 and bbl and bbr:
            cor_bbl = env[bbl].b @ env[bbl].bl @ env[bbl].l
            cor_bbl = tensordot(cor_bbl, psi[bbl], axes=((2, 1), (1, 2)))
            cor_bl = tensordot(cor_bbl, env[bl].l, axes=(1, 0))
            cor_bl = tensordot(cor_bl, psi[bl], axes=((3, 1), (1, 2))) 
            cor_bl = cor_bl.fuse_legs(axes=((0, 1, 4), (2, 3)))

            cor_bbr = env[bbr].r @ env[bbr].br @ env[bbr].b
            cor_bbr = tensordot(cor_bbr, psi[bbr], axes=((2, 1), (2, 3)))
            cor_br = env[br].r @ cor_bbr 
            cor_br = tensordot(cor_br, psi[br], axes=((3, 1), (2, 3)))
            cor_br = cor_br.fuse_legs(axes=((0, 3), (1, 2, 4)))
        else:
            cor_bl = env[bl].b @ env[bl].bl @ env[bl].l
            cor_bl = tensordot(cor_bl, psi[bl], axes=((2, 1), (1, 2)))
            cor_bl = cor_bl.fuse_legs(axes=((0, 3), (1, 2)))

            cor_br = env[br].r @ env[br].br @ env[br].b
            cor_br = tensordot(cor_br, psi[br], axes=((2, 1), (2, 3)))
            cor_br = cor_br.fuse_legs(axes=((0, 2), (1, 3)))

        cor_tt = cor_tl @ cor_tr  # b(left) b(right)
        cor_bb = cor_br @ cor_bl  # t(right) t(left)

        cor_ll = cor_bl @ cor_tl  # l(bottom) l(top)
        cor_rr = cor_tr @ cor_br  # r(top) r(bottom)

    if 't' in dirn:
        _, r_l = qr(cor_ll, axes=(0, 1)) if use_qr else (None, cor_ll)
        _, r_r = qr(cor_rr, axes=(1, 0)) if use_qr else (None, cor_rr.T)
        proj[tl].vtr, proj[tr].vtl = proj_corners(r_l, r_r, opts_svd=opts_svd, **kwargs)

    if 'b' in dirn:
        _, r_l = qr(cor_ll, axes=(1, 0)) if use_qr else (None, cor_ll.T)
        _, r_r = qr(cor_rr, axes=(0, 1)) if use_qr else (None, cor_rr)
        proj[bl].vbr, proj[br].vbl = proj_corners(r_l, r_r, opts_svd=opts_svd, **kwargs)


def update_1site_projectors_(proj, site, dirn, env, opts_svd, **kwargs):
    r"""
    Calculate new projectors for CTM moves from 4x2 extended corners.
    """
    psi = env.psi
    sites = [psi.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))]
    if None in sites:
        return

    tl, tr, bl, br = sites

    if ('l' in dirn) or ('r' in dirn):
        cor_tl = (env[bl].tl @ env[bl].t).fuse_legs(axes=((0, 1), 2))
        cor_tr = (env[br].t @ env[br].tr).fuse_legs(axes=(0, (2, 1)))
        cor_br = (env[tr].br @ env[tr].b).fuse_legs(axes=((0, 1), 2))
        cor_bl = (env[tl].b @ env[tl].bl).fuse_legs(axes=(0, (2, 1)))
        r_tl, r_tr = regularize_1site_corners(cor_tl, cor_tr)
        r_br, r_bl = regularize_1site_corners(cor_br, cor_bl)

    if 'r' in dirn:
        proj[tr].hrb, proj[br].hrt = proj_corners(r_tr, r_br, opts_svd=opts_svd, **kwargs)

    if 'l' in dirn:
        proj[tl].hlb, proj[bl].hlt = proj_corners(r_tl, r_bl, opts_svd=opts_svd, **kwargs)

    if ('t' in dirn) or ('b' in dirn):
        cor_bl = (env[br].bl @ env[br].l).fuse_legs(axes=((0, 1), 2))
        cor_tl = (env[tr].l @ env[tr].tl).fuse_legs(axes=(0, (2, 1)))
        cor_tr = (env[tl].tr @ env[tl].r).fuse_legs(axes=((0, 1), 2))
        cor_br = (env[bl].r @ env[bl].br).fuse_legs(axes=(0, (2, 1)))
        r_bl, r_tl = regularize_1site_corners(cor_bl, cor_tl)
        r_tr, r_br = regularize_1site_corners(cor_tr, cor_br)

    if 't' in dirn:
        proj[tl].vtr, proj[tr].vtl = proj_corners(r_tl, r_tr, opts_svd=opts_svd, **kwargs)

    if 'b' in dirn:
        proj[bl].vbr, proj[br].vbl = proj_corners(r_bl, r_br, opts_svd=opts_svd, **kwargs)


def regularize_1site_corners(cor_0, cor_1):
    Q_0, R_0 = qr(cor_0, axes=(0, 1))
    Q_1, R_1 = qr(cor_1, axes=(1, 0))
    R01 = tensordot(R_0, R_1, axes=(1, 1))
    U_0, S, U_1 = R01.svd(axes=(0, 1), fix_signs=True)
    S = S.sqrt()
    r_0 = tensordot((U_0 @ S), Q_0, axes=(0, 1))
    r_1 = tensordot((S @ U_1), Q_1, axes=(1, 1))
    return r_0, r_1

def proj_corners(r0, r1, opts_svd, **kwargs):
    r""" Projectors in between r0 @ r1.T corners. """
    rr = tensordot(r0, r1, axes=(1, 1))
    fix_signs= opts_svd.get('fix_signs',True)
    truncation_f= kwargs.get('truncation_f',None)
    if truncation_f is None:
        u, s, v = rr.svd(axes=(0, 1), sU=r0.s[1], fix_signs=fix_signs, **kwargs)
        Smask = truncation_mask(s, **opts_svd)
        u, s, v = Smask.apply_mask(u, s, v, axes=(-1, 0, 0))
    else:
        u, s, v = rr.svd_with_truncation(axes=(0, 1), sU=r0.s[1], mask_f=truncation_f, **kwargs)

    rs = s.rsqrt()
    p0 = tensordot(r1, (rs @ v).conj(), axes=(0, 1)).unfuse_legs(axes=0)
    p1 = tensordot(r0, (u @ rs).conj(), axes=(0, 0)).unfuse_legs(axes=0)
    return p0, p1

_trivial = (('hlt', 'r', 'l', 'tl', 2, 0, 0),
            ('hlb', 'r', 'l', 'bl', 0, 2, 1),
            ('hrt', 'l', 'r', 'tr', 0, 0, 1),
            ('hrb', 'l', 'r', 'br', 2, 2, 0),
            ('vtl', 'b', 't', 'tl', 0, 1, 1),
            ('vtr', 'b', 't', 'tr', 2, 3, 0),
            ('vbl', 't', 'b', 'bl', 2, 1, 0),
            ('vbr', 't', 'b', 'br', 0, 3, 1))


def trivial_projectors_(proj, dirn, env):
    r"""
    Adds trivial projectors if not present at the edges of the lattice with open boundary conditions.
    """
    config = env.psi.config
    for site in env.sites():
        for s0, s1, s2, s3, a0, a1, a2 in _trivial:
            if s2 in dirn and getattr(proj[site], s0) is None:
                site_nn = env.nn_site(site, d=s1)
                if site_nn is not None:
                    l0 = getattr(env[site], s2).get_legs(a0).conj()
                    l1 = env.psi[site].get_legs(a1).conj()
                    l2 = getattr(env[site_nn], s3).get_legs(a2).conj()
                    setattr(proj[site], s0, ones(config, legs=(l0, l1, l2)))

# Given projectors, compress enlarged corners and half-row/column tensors to obtain updated environment tensors
#
def update_env_directional_(env_tmp : EnvCTM, env : EnvCTM, site : Site, proj : EnvCTM_projectors, dir : str):
    r"""
    Peform single directional update of CTMRG for environment tensors of``site`` in direction ``dir`` using provided projectors.
    The updated environment tensors are stored in ``env_tmp``.
    """
    assert dir in ['l','r','t','b'], "Invalid direction"
    if dir in ['l','r']:
        update_env_horizontal_(env_tmp, site, env, proj, dir)
    else:
        update_env_vertical_(env_tmp, site, env, proj, dir)


def update_env_horizontal_(env_tmp, site, env, proj, dir : str = 'lr'):
    r"""
    Horizontal move of CTM step. Compute updated environment tensors given projectors for ``site``
    in left (``dir='l'``), right ``dir='r'``, or both directions (``dir='lr'``).
    Updated environment tensors are stored in ``env_tmp``.
    """
    assert 'l' in dir or 'r' in dir, "Invalid directions"
    psi = env.psi

    if 'l' in dir:
        l = psi.nn_site(site, d='l')
        if l is not None:
            tmp = env[l].l @ proj[l].hlt
            tmp = tensordot(psi[l], tmp, axes=((0, 1), (2, 1)))
            tmp = tensordot(proj[l].hlb, tmp, axes=((0, 1), (2, 0)))
            env_tmp[site].l = tmp / tmp.norm(p='inf')

        tl = psi.nn_site(site, d='tl')
        if tl is not None:
            tmp = tensordot(proj[tl].hlb, env[l].tl @ env[l].t, axes=((0, 1), (0, 1)))
            env_tmp[site].tl = tmp / tmp.norm(p='inf')

        bl = psi.nn_site(site, d='bl')
        if bl is not None:
            tmp = tensordot(env[l].b, env[l].bl @ proj[bl].hlt, axes=((2, 1), (0, 1)))
            env_tmp[site].bl = tmp / tmp.norm(p='inf')

    if 'r' in dir:
        r = psi.nn_site(site, d='r')
        if r is not None:
            tmp = env[r].r @ proj[r].hrb
            tmp = tensordot(psi[r], tmp, axes=((2, 3), (2, 1)))
            tmp = tensordot(proj[r].hrt, tmp, axes=((0, 1), (2, 0)))
            env_tmp[site].r = tmp / tmp.norm(p='inf')

        tr = psi.nn_site(site, d='tr')
        if tr is not None:
            tmp = tensordot(env[r].t, env[r].tr @ proj[tr].hrb, axes=((2, 1), (0, 1)))
            env_tmp[site].tr = tmp / tmp.norm(p='inf')

        br = psi.nn_site(site, d='br')
        if br is not None:
            tmp = tensordot(proj[br].hrt, env[r].br @ env[r].b, axes=((0, 1), (0, 1)))
            env_tmp[site].br = tmp / tmp.norm(p='inf')


def update_env_vertical_(env_tmp, site, env, proj, dir : str = 'tb'):
    r"""
    Vertical move of CTM step. Compute updated environment tensors given projectors for ``site``
    in top (``dir='t'``), bottom ``dir='b'``, or both directions (``dir='tb'``).
    Updated environment tensors are stored in ``env_tmp``.
    """
    assert 't' in dir or 'b' in dir, "Invalid directions"
    psi = env.psi

    if 't' in dir:
        t = psi.nn_site(site, d='t')
        if t is not None:
            tmp = tensordot(proj[t].vtl, env[t].t, axes=(0, 0))
            tmp = tensordot(tmp, psi[t], axes=((2, 0), (0, 1)))
            tmp = tensordot(tmp, proj[t].vtr, axes=((1, 3), (0, 1)))
            env_tmp[site].t = tmp / tmp.norm(p='inf')

        tl = psi.nn_site(site, d='tl')
        if tl is not None:
            tmp = tensordot(env[t].l, env[t].tl @ proj[tl].vtr, axes=((2, 1), (0, 1)))
            env_tmp[site].tl = tmp / tmp.norm(p='inf')

        tr = psi.nn_site(site, d='tr')
        if tr is not None:
            tmp = tensordot(proj[tr].vtl, env[t].tr @ env[t].r, axes=((0, 1), (0, 1)))
            env_tmp[site].tr =  tmp / tmp.norm(p='inf')

    if 'b' in dir:
        b = psi.nn_site(site, d='b')
        if b is not None:
            tmp = tensordot(proj[b].vbr, env[b].b, axes=(0, 0))
            tmp = tensordot(tmp, psi[b], axes=((2, 0), (2, 3)))
            tmp = tensordot(tmp, proj[b].vbl, axes=((1, 3), (0, 1)))
            env_tmp[site].b = tmp / tmp.norm(p='inf')

        bl = psi.nn_site(site, d='bl')
        if bl is not None:
            tmp = tensordot(proj[bl].vbr, env[b].bl @ env[b].l, axes=((0, 1), (0, 1)))
            env_tmp[site].bl = tmp / tmp.norm(p='inf')

        br = psi.nn_site(site, d='br')
        if br is not None:
            tmp = tensordot(env[b].r, env[b].br @ proj[br].vbl, axes=((2, 1), (0, 1)))
            env_tmp[site].br = tmp / tmp.norm(p='inf')


def update_old_env_(env, env_tmp):
    r"""
    Update tensors in env with the ones from env_tmp that are not None.
    """
    for site in env.sites():
        for k, v in env_tmp[site].__dict__.items():
            if v is not None:
                setattr(env[site], k, v)


def store_projectors_(proj : Peps, proj_other : Peps):
    r"""
    Update projectors in ``proj`` with projectors stored in ``proj_other`` (ignoring unassigned projectors i.e. ``None``).

    Args:
        proj : PEPS with set of projectors ``EnvCTM_projectors`` assigned to each site.
        proj_other : PEPS with set of projectors ``EnvCTM_projectors`` assigned to each site.
    """
    for site in proj.sites():
        for k, v in proj_other[site].__dict__.items():
            if v is not None:
                setattr(proj[site], k, v)