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
from .... import Tensor, rand, ones, eye, YastnError, Leg, tensordot, qr, truncation_mask, vdot
from ... import mps
from .._peps import Peps, Peps2Layers
from .._gates_auxiliary import apply_gate_onsite, gate_product_operator, gate_fix_order
from .._geometry import Bond, Site
from ._env_auxlliary import *
from ._env_window import EnvWindow

logger = logging.Logger('ctmrg')

@dataclass()
class EnvCTM_local():
    r"""
    Dataclass for CTM environment tensors associated with Peps lattice site.

    Contains fields 'tl', 't', 'tr', 'r', 'br', 'b', 'bl', 'l'
    """
    tl = None # top-left
    t = None  # top
    tr = None # top-right
    r = None  # right
    br = None # bottom-right
    b = None  # bottom
    bl = None # bottom-left
    l = None  # left


@dataclass()
class EnvCTM_projectors():
    r""" Dataclass for CTM projectors associated with Peps lattice site. """
    hlt : any = None  # horizontal left top
    hlb : any = None  # horizontal left bottom
    hrt : any = None  # horizontal right top
    hrb : any = None  # horizontal right bottom
    vtl : any = None  # vertical top left
    vtr : any = None  # vertical top right
    vbl : any = None  # vertical bottom left
    vbr : any = None  # vertical bottom right


class CTMRG_out(NamedTuple):
    sweeps : int = 0
    max_dsv : float = None
    converged : bool = False


class EnvCTM(Peps):
    def __init__(self, psi, init='rand', leg=None):
        r"""
        Environment used in Corner Transfer Matrix Renormalization algorithm.

        Note:
            Index convention for environment tensor
                
                * enlarged corners: anti-clockwise

        Parameters
        ----------
        psi: yastn.tn.Peps
            Peps lattice to be contracted using CTM.
            If :code:`psi` has physical legs, a double-layer PEPS with no physical legs is formed.

        init: str
            None, 'eye' or 'rand'. Initialization scheme, see :meth:`yastn.tn.fpeps.EnvCTM.reset_`.

        leg: yastn.Leg | None
            Passed to :meth:`yastn.tn.fpeps.EnvCTM.reset_`.
        """
        super().__init__(psi.geometry)
        self.psi = Peps2Layers(psi) if psi.has_physical() else psi
        if init not in (None, 'rand', 'eye', 'dl'):
            raise YastnError(f"EnvCTM {init=} not recognized. Should be 'rand', 'eye', 'dl', None.")
        for site in self.sites():
            self[site] = EnvCTM_local()
        if init is not None:
            self.reset_(init=init, leg=leg)

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
                getattr(self[site], dirn)._data.detach_()


    def reset_(self, init='rand', leg=None, **kwargs):
        r"""
        Initialize random CTMRG environments of peps tensors.

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
        elif init == 'nn':
            pass
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

    def init_env_from_onsite_(self, normalize : Union[str,Callable]='inf'):
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
            for dirn, cor_f in ( ('tl', cor_tl), ('tr', cor_tr), ('bl', cor_bl), ('br', cor_br)):
                shifted_site= self.nn_site(site,dirn)
                C= cor_f(self.psi.bra[shifted_site], A_ket=self.psi.ket[shifted_site])
                C= C/C.norm(p=normalize) if isinstance(normalize,str) else normalize(C)
                setattr(self[site], dirn, C)

            for dirn, edge_f in ( ('t',edge_t), ('l',edge_l), ('b',edge_b), ('r',edge_r) ):
                shifted_site= self.nn_site(site,dirn)
                T= edge_f(self.psi.bra[shifted_site], A_ket=self.psi.ket[shifted_site])
                T= T/T.norm(p=normalize) if isinstance(normalize,str) else normalize(T)
                setattr(self[site], dirn, T)


    def boundary_mps(self, n, dirn):
        r""" Convert environmental tensors of Ctm to an MPS """
        if dirn == 'b':
            H = mps.Mps(N=self.Ny)
            for ny in range(self.Ny):
                H.A[ny] = self[n, ny].b.transpose(axes=(2, 1, 0))
            H = H.conj()
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
            H = H.conj()
        return H

    def measure_1site(self, op, site=None) -> dict:
        r"""
        Calculate local expectation values within CTM environment.

        Return a number if site is provided.
        If None, returns a dictionary {site: value} for all unique lattice sites.

        Parameters
        ----------
        env: class CtmEnv
            class containing ctm environment tensors along with lattice structure data

        op: single site operator
        """
        if site is None:
            return {site: self.measure_1site(op, site) for site in self.sites()}

        lenv = self[site]
        ten = self.psi[site]
        vect = (lenv.l @ lenv.tl) @ (lenv.t @ lenv.tr)
        vecb = (lenv.r @ lenv.br) @ (lenv.b @ lenv.bl)

        tmp = ten._attach_01(vect)
        val_no = tensordot(vecb, tmp, axes=((0, 1, 2, 3), (2, 3, 1, 0))).to_number()

        ten.set_operator_(op)
        tmp = ten._attach_01(vect)
        val_op = tensordot(vecb, tmp, axes=((0, 1, 2, 3), (2, 3, 1, 0))).to_number()

        return val_op / val_no

    def measure_nn(self, O0, O1, bond=None) -> dict:
        r"""
        Calculate nearest-neighbor expectation values within CTM environment.

        Return a number if the nearest-neighbor bond is provided.
        If None, returns a dictionary {bond: value} for all unique lattice bonds.

        Parameters
        ----------
        O0, O1: yastn.Tensor
            Calculate <O0_s0 O1_s1>.
            O1 is applied first, which might matter for fermionic operators.

        bond: yastn.tn.fpeps.Bond | tuple[tuple[int, int], tuple[int, int]]
            Bond of the form (s0, s1). Sites s0 and s1 should be nearest-neighbors on the lattice.
        """

        if bond is None:
             return {bond: self.measure_nn(O0, O1, bond) for bond in self.bonds()}

        bond = Bond(*bond)
        dirn, l_ordered = self.nn_bond_type(bond)
        f_ordered = self.f_ordered(bond)
        s0, s1 = bond if l_ordered else bond[::-1]
        env0, env1 = self[s0], self[s1]
        ten0, ten1 = self.psi[s0], self.psi[s1]

        if O0.ndim == 2 and O1.ndim == 2:
            G0, G1 = gate_product_operator(O0, O1, l_ordered, f_ordered)
        elif O0.ndim == 3 and O1.ndim == 3:
            G0, G1 = gate_fix_order(O0, O1, l_ordered, f_ordered)
        else:
            raise YastnError("Both operators O0 and O1 should have the same ndim==2, or ndim=3.")

        if dirn == 'h':
            vecl = (env0.bl @ env0.l) @ (env0.tl @ env0.t)
            vecr = (env1.tr @ env1.r) @ (env1.br @ env1.b)

            tmp0 = ten0._attach_01(vecl)
            tmp0 = tensordot(env0.b, tmp0, axes=((2, 1), (0, 1)))
            tmp1 = ten1._attach_23(vecr)
            tmp1 = tensordot(env1.t, tmp1, axes=((2, 1), (0, 1)))
            val_no = tensordot(tmp0, tmp1, axes=((0, 1, 2), (1, 0, 2))).to_number()

            ten0.ket = apply_gate_onsite(ten0.ket, G0, dirn='l')
            ten1.ket = apply_gate_onsite(ten1.ket, G1, dirn='r')

            tmp0 = ten0._attach_01(vecl)
            tmp0 = tensordot(env0.b, tmp0, axes=((2, 1), (0, 1)))
            tmp1 = ten1._attach_23(vecr)
            tmp1 = tensordot(env1.t, tmp1, axes=((2, 1), (0, 1)))
            val_op = tensordot(tmp0, tmp1, axes=((0, 1, 2), (1, 0, 2))).to_number()
        else:  # dirn == 'v':
            vect = (env0.l @ env0.tl) @ (env0.t @ env0.tr)
            vecb = (env1.r @ env1.br) @ (env1.b @ env1.bl)

            tmp0 = ten0._attach_01(vect)
            tmp0 = tensordot(tmp0, env0.r, axes=((2, 3), (0, 1)))
            tmp1 = ten1._attach_23(vecb)
            tmp1 = tensordot(tmp1, env1.l, axes=((2, 3), (0, 1)))
            val_no = tensordot(tmp0, tmp1, axes=((0, 1, 2), (2, 1, 0))).to_number()

            ten0.ket = apply_gate_onsite(ten0.ket, G0, dirn='t')
            ten1.ket = apply_gate_onsite(ten1.ket, G1, dirn='b')

            tmp0 = ten0._attach_01(vect)
            tmp0 = tensordot(tmp0, env0.r, axes=((2, 3), (0, 1)))
            tmp1 = ten1._attach_23(vecb)
            tmp1 = tensordot(tmp1, env1.l, axes=((2, 3), (0, 1)))
            val_op = tensordot(tmp0, tmp1, axes=((0, 1, 2), (2, 1, 0))).to_number()

        return val_op / val_no

    def measure_2x2(self, *operators, sites=None):
        """
        Calculate expectation value of a product of local operators
        in a 2x2 window within the CTM environment.

        At the moment, it works only for bosonic operators (fermionic are todo).

        Parameters
        ----------
        operators: Sequence[yastn.Tensor]
            List of local operators to calculate <O0_s0 O1_s1 ...>.

        sites: Sequence[tuple[int, int]]
            A list of sites [s0, s1, ...] matching corresponding operators.
        """
        if len(operators) != len(sites):
            raise YastnError("Number of operators and sites should match.")
        ops = dict(zip(sites, operators))
        if len(sites) != len(ops):
            raise YastnError("Sites should not repeat.")

        mx = min(site[0] for site in sites)  # tl corner
        my = min(site[1] for site in sites)

        tl = Site(mx, my)
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

        cor_tl = ten_tl._attach_01(vec_tl)
        cor_tl = cor_tl.fuse_legs(axes=((0, 1), (2, 3)))
        cor_tr = ten_tr._attach_30(vec_tr)
        cor_tr = cor_tr.fuse_legs(axes=((0, 1), (2, 3)))
        cor_br = ten_br._attach_23(vec_br)
        cor_br = cor_br.fuse_legs(axes=((0, 1), (2, 3)))
        cor_bl = ten_bl._attach_12(vec_bl)
        cor_bl = cor_bl.fuse_legs(axes=((0, 1), (2, 3)))

        val_no = vdot(cor_tl @ cor_tr @ cor_br, cor_bl.T, conj=(0, 0))

        if tl in ops:
            ten_tl.set_operator_(ops[tl])
            cor_tl = ten_tl._attach_01(vec_tl)
            cor_tl = cor_tl.fuse_legs(axes=((0, 1), (2, 3)))
        if tr in ops:
            ten_tr.set_operator_(ops[tr])
            cor_tr = ten_tr._attach_30(vec_tr)
            cor_tr = cor_tr.fuse_legs(axes=((0, 1), (2, 3)))
        if br in ops:
            ten_br.set_operator_(ops[br])
            cor_br = ten_br._attach_23(vec_br)
            cor_br = cor_br.fuse_legs(axes=((0, 1), (2, 3)))
        if bl in ops:
            ten_bl.set_operator_(ops[bl])
            cor_bl = ten_bl._attach_12(vec_bl)
            cor_bl = cor_bl.fuse_legs(axes=((0, 1), (2, 3)))

        val_op = vdot(cor_tl @ cor_tr @ cor_br, cor_bl.T, conj=(0, 0))

        return val_op / val_no


    def measure_line(self, *operators, sites=None):
        """
        Calculate expectation value of a product of local opertors
        along a horizontal or vertical line within CTM environment.

        At the moment works only for bosonic operators (fermionic are to do).

        Parameters
        ----------
        operators: Sequence[yastn.Tensor]
            List of local operators to calculate <O0_s0 O1_s1 ...>.

        sites: Sequence[tuple[int, int]]
            List of sites that should match operators.
        """
        if len(operators) != len(sites):
            raise YastnError("Number of operators and sites should match.")
        ops = dict(zip(sites, operators))
        if len(sites) != len(ops):
            raise YastnError("Sites should not repeat.")

        xs = sorted(set(site[0] for site in sites))
        ys = sorted(set(site[1] for site in sites))
        if len(xs) > 1 and len(ys) > 1:
            raise YastnError("Sites should form a horizontal or vertical line.")

        env_win = EnvWindow(self, (xs[0], xs[-1] + 1), (ys[0], ys[-1] + 1))
        if len(xs) == 1: # horizontal
            vr = env_win[xs[0], 't']
            tm = env_win[xs[0], 'h']
            vl = env_win[xs[0], 'b']
        else:  # len(ys) == 1:  # vertical
            vr = env_win[ys[0], 'l']
            tm = env_win[ys[0], 'v']
            vl = env_win[ys[0], 'r']

        val_no = mps.vdot(vl, tm, vr)

        for site, op in ops.items():
            ind = site[0] - xs[0] + site[1] - ys[0] + 1
            tm[ind].set_operator_(op)

        val_op = mps.vdot(vl, tm, vr)
        return val_op / val_no


    def measure_2site(self, O0, O1, xrange, yrange, opts_svd=None, opts_var=None) -> dict[Site, list]:
        """
        Calculate 2-point correlations <o1 o2> between top-left corner of the window, and all sites in the window.

        wip: other combinations of 2-sites and fermionically-nontrivial operators will be coverad latter.

        Parameters
        ----------
        O1, O2: yastn.Tensor
            one-site operators

        xrange: tuple[int, int]
            range of rows forming a window, [r0, r1); r0 included, r1 excluded.

        yrange: tuple[int, int]
            range of columns forming a window.

        opts_svd: dict
            Options passed to :meth:`yastn.linalg.svd` used to truncate virtual spaces of boundary MPSs used in sampling.
            The default is None, in which case take :code:`D_total` as the largest dimension from CTM environment.

        opts_svd: dict
            Options passed to :meth:`yastn.tn.mps.compression_` used in the refining of boundary MPSs.
            The default is None, in which case make 2 variational sweeps.
        """
        env_win = EnvWindow(self, xrange, yrange)
        return env_win.measure_2site(O0, O1, opts_svd=opts_svd, opts_var=opts_var)


    def sample(self, xrange, yrange, projectors, number=1, opts_svd=None, opts_var=None, progressbar=False, return_info=False) -> dict[Site, list]:
        """
        Sample random configurations from PEPS. Output a dictionary linking sites with lists of sampled projectors` keys for each site.

        It does not check whether projectors sum up to identity -- probabilities of provided projectors get normalized to one.
        If negative probabilities are observed (signaling contraction errors), error = max(abs(negatives)),
        and all probabilities below that error level are fixed to error (before consecutive renormalization of probabilities to one).

        Parameters
        ----------
        xrange: tuple[int, int]
            range of rows to sample from, [r0, r1); r0 included, r1 excluded.

        trange: tuple[int, int]
            range of columns to sample from.

        projectors: Dict[Any, yast.Tensor] | Sequence[yast.Tensor] | Dict[Site, Dict[Any, yast.Tensor]]
            Projectors to sample from. We can provide a dict(key: projector), where the sampled results will be given as keys,
            and the same set of projectors is used at each site. For a list of projectors, the keys follow from enumeration.
            Finally, we can provide a dictionary between each site and sets of projectors.

        number: int
            Number of drawn samples.

        opts_svd: dict
            Options passed to :meth:`yastn.linalg.svd` used to truncate virtual spaces of boundary MPSs used in sampling.
            The default is None, in which case take :code:`D_total` as the largest dimension from CTM environment.

        opts_svd: dict
            Options passed to :meth:`yastn.tn.mps.compression_` used in the refining of boundary MPSs.
            The default is None, in which case make 2 variational sweeps.

        progressbar: bool
            Whether to display progressbar. The default is False.

        return_info: bool
            Whether to include in the outputted dictionary a field :code:`info` with dictionary
            that contains information about the amplitude of contraction errors
            (largest negative probability), D_total, etc. The default is False.
        """
        env_win = EnvWindow(self, xrange, yrange)
        return env_win.sample(projectors, number, opts_svd, opts_var, progressbar, return_info)


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
            A dictionary of options to pass to the SVD algorithm.

        method: str
            '2site' or '1site'. The default is '2site'.
            '2site' uses the standard 4x4 enlarged corners, allowing to enlarge EnvCTM bond dimension.
            '1site' uses smaller 4x2 corners. It is significantly faster, but is less stable and
            does not allow to grow EnvCTM bond dimension.

        Returns
        -------
        proj: Peps structure loaded with CTM projectors related to all lattice site.
        """
        if all(s not in opts_svd for s in ('tol', 'tol_block')):
            opts_svd['tol'] = 1e-14
        if method not in ('1site', '2site'):
            raise YastnError(f"CTM update {method=} not recognized. Should be '1site' or '2site'")
        update_proj_ = update_2site_projectors_ if method == '2site' else update_1site_projectors_
        #
        # Empty structure for projectors
        proj = Peps(env.geometry)
        for site in proj.sites():
            proj[site] = EnvCTM_projectors()
        #
        # horizontal projectors
        for site in env.sites():
            update_proj_(proj, site, 'lr', env, opts_svd, **kwargs)
        trivial_projectors_(proj, 'lr', env)  # fill None's
        #
        # horizontal move
        env_tmp = EnvCTM(env.psi, init=None)  # empty environments
        for site in env.sites():
            update_env_horizontal_(env_tmp, site, env, proj)
        update_old_env_(env, env_tmp)
        #
        # vertical projectors
        for site in env.sites():
            update_proj_(proj, site, 'tb', env, opts_svd, **kwargs)
        trivial_projectors_(proj, 'tb', env)
        #
        # vertical move
        env_tmp = EnvCTM(env.psi, init=None)
        for site in env.sites():
            update_env_vertical_(env_tmp, site, env, proj)
        update_old_env_(env, env_tmp)
        #
        return proj

    def bond_metric(self, Q0, Q1, s0, s1, dirn):
        r"""
        Calculates Full-Update metric tensor.

        ::

            If dirn == 'h':
                tl == tt  ==  tt == tr
                |     |        |     |
                ll == GA-+  +-GB == rr
                |     |        |     |
                bl == bb  ==  bb == br

            If dirn == 'v':
                tl == tt == tr
                |     |      |
                ll == GA == rr
                |     ++     |
                ll == GB == rr
                |     |      |
                bl == bb == br
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
        """
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
        Convergence is based on singular values of CTM environment corner tensors.

        Outputs iterator if :code:`iterator_step` is given, which allows
        inspecting :code:`env`, e.g., calculating expectation values,
        outside of :code:`ctmrg_` function after every :code:`iterator_step` sweeps.

        Parameters
        ----------
        opts_svd: dict
            A dictionary of options to pass to the SVD algorithm.

        method: str
            '2site' or '1site'. The default is '2site'.
            '2site' uses the standard 4x4 enlarged corners, allowing to enlarge EnvCTM bond dimension.
            '1site' uses smaller 4x2 corners. It is significantly faster, but is less stable and
            does not allow to grow EnvCTM bond dimension.

        max_sweeps: int
            Maximal number of sweeps.

        iterator_step: int
            If int, :code:`ctmrg_` returns a generator that would yield output after every iterator_step sweeps.
            Default is None, in which case  :code:`ctmrg_` sweeps are performed immediately.

        corner_tol: float
            Convergence tolerance for the change of singular values of all corners in a single update.
            The default is None, in which case convergence is not checked and it is up to user to implement
            convergence check.

        truncation_f:
            Custom projector truncation function with signature ``truncation_f(S: Tensor)->Tensor``, consuming
            rank-1 tensor with singular values. If provided, truncation parameters passed to SVD decomposition
            are ignored.

        Returns
        -------
        Generator if iterator_step is not None.

        CTMRG_out(NamedTuple)
            NamedTuple including fields:

                * :code:`sweeps` number of performed ctmrg updates.
                * :code:`max_dsv` norm of singular values change in the worst corner in the last sweep.
                * :code:`converged` whether convergence based on conrer_tol has been reached.
        """
        kwargs["truncation_f"]= truncation_f
        tmp = _ctmrg_(env, opts_svd, method, max_sweeps, iterator_step, corner_tol, **kwargs)
        return tmp if iterator_step else next(tmp)


def ctm_conv_corner_spec(env : EnvCTM, history : Sequence[dict[tuple[Site,str],Tensor]]=[], 
                         corner_tol : Union[None,float]=1.0e-8)->tuple[bool,float,Sequence[dict[tuple[Site,str],Tensor]]]:
    """ 
    Evaluate convergence of CTM by computing the difference of environment corner spectra between consecutive CTM steps.
    """
    history.append(calculate_corner_svd(env))
    max_dsv = max((history[-1][k] - history[-2][k]).norm().item() for k in history[-1]) if len(history)>1 else float('Nan')

    return (corner_tol is not None and max_dsv < corner_tol), max_dsv, history


def _ctmrg_(env, opts_svd, method, max_sweeps, iterator_step, corner_tol, **kwargs):
    """ Generator for ctmrg_(). """        
    max_dsv, converged= None, False
    for sweep in range(1, max_sweeps + 1):
        env.update_(opts_svd=opts_svd, method=method, **kwargs)
        
        # use default CTM convergence check
        if corner_tol is not None:
            if sweep==1: history=[]
            converged, max_dsv, history= ctm_conv_corner_spec(env.detach(), history, corner_tol)
            logging.info(f'Sweep = {sweep:03d};  max_diff_corner_singular_values = {max_dsv:0.2e}')

            if converged:
                break

        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, converged=converged)
    yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, converged=converged)


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


def update_2site_projectors_(proj, site, dirn, env, opts_svd, **kwargs):
    r"""
    Calculate new projectors for CTM moves from 4x4 extended corners.
    """
    psi = env.psi
    sites = [psi.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))]
    if None in sites:
        return

    tl, tr, bl, br = sites

    cor_tl = psi[tl]._attach_01(env[tl].l @ env[tl].tl @ env[tl].t)
    cor_tl = cor_tl.fuse_legs(axes=((0, 1), (2, 3))) # b r
    cor_bl = psi[bl]._attach_12(env[bl].b @ env[bl].bl @ env[bl].l)
    cor_bl = cor_bl.fuse_legs(axes=((0, 1), (2, 3))) # t r
    cor_tr = psi[tr]._attach_30(env[tr].t @ env[tr].tr @ env[tr].r)
    cor_tr = cor_tr.fuse_legs(axes=((0, 1), (2, 3))) # l b
    cor_br = psi[br]._attach_23(env[br].r @ env[br].br @ env[br].b)
    cor_br = cor_br.fuse_legs(axes=((0, 1), (2, 3))) # t l

    if ('l' in dirn) or ('r' in dirn):
        cor_tt = cor_tl @ cor_tr # b(left) b(right)
        cor_bb = cor_br @ cor_bl # t(right) t(left)

    use_qr= kwargs.get("use_qr",True)
    if 'r' in dirn:
        _, r_t = qr(cor_tt, axes=(0, 1)) if use_qr else None, cor_tt
        _, r_b = qr(cor_bb, axes=(1, 0)) if use_qr else None, cor_bb.T
        proj[tr].hrb, proj[br].hrt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)

    if 'l' in dirn:
        _, r_t = qr(cor_tt, axes=(1, 0)) if use_qr else None, cor_tt.T
        _, r_b = qr(cor_bb, axes=(0, 1)) if use_qr else None, cor_bb
        proj[tl].hlb, proj[bl].hlt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)

    if ('t' in dirn) or ('b' in dirn):
        cor_ll = cor_bl @ cor_tl # l(bottom) l(top)
        cor_rr = cor_tr @ cor_br # r(top) r(bottom)

    if 't' in dirn:
        _, r_l = qr(cor_ll, axes=(0, 1)) if use_qr else None, cor_ll
        _, r_r = qr(cor_rr, axes=(1, 0)) if use_qr else None, cor_rr.T
        proj[tl].vtr, proj[tr].vtl = proj_corners(r_l, r_r, opts_svd=opts_svd, **kwargs)

    if 'b' in dirn:
        _, r_l = qr(cor_ll, axes=(1, 0)) if use_qr else None, cor_ll.T
        _, r_r = qr(cor_rr, axes=(0, 1)) if use_qr else None, cor_rr
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

def proj_corners(r0, r1, opts_svd, svd_method='full', D_block=None, **kwargs):
    r""" Projectors in between r0 @ r1.T corners. """
    rr = tensordot(r0, r1, axes=(1, 1))

    fix_signs= opts_svd.get('fix_signs',True)
    truncation_f= kwargs.get('truncation_f',None)
    if truncation_f is None:
        if svd_method == 'full':
            u, s, v = rr.svd(axes=(0, 1), sU=r0.s[1], fix_signs=fix_signs)
        elif svd_method == 'arnoldi':
            if D_block is not None:
                u, s, v = rr.svd_arnoldi(axes=(0, 1), sU=r0.s[1], fix_signs=fix_signs, D_block=D_block)
            else:
                raise YastnError(f"D_block is not specified for svd_arnoldi.")
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


def update_env_horizontal_(env_tmp, site, env, proj):
    r"""
    Horizontal move of CTM step. Calculate environment tensors for given site.
    """
    psi = env.psi

    l = psi.nn_site(site, d='l')
    if l is not None:
        tmp = env[l].l @ proj[l].hlt
        tmp = psi[l]._attach_01(tmp)
        tmp = tensordot(proj[l].hlb, tmp, axes=((0, 1), (0, 1))).transpose(axes=(0, 2, 1))
        env_tmp[site].l = tmp / tmp.norm(p='inf')

    r = psi.nn_site(site, d='r')
    if r is not None:
        tmp = env[r].r @ proj[r].hrb
        tmp = psi[r]._attach_23(tmp)
        tmp = tensordot(proj[r].hrt, tmp, axes=((0, 1), (0, 1))).transpose(axes=(0, 2, 1))
        env_tmp[site].r = tmp / tmp.norm(p='inf')

    tl = psi.nn_site(site, d='tl')
    if tl is not None:
        tmp = tensordot(proj[tl].hlb, env[l].tl @ env[l].t, axes=((0, 1), (0, 1)))
        env_tmp[site].tl = tmp / tmp.norm(p='inf')

    tr = psi.nn_site(site, d='tr')
    if tr is not None:
        tmp = tensordot(env[r].t, env[r].tr @ proj[tr].hrb, axes=((2, 1), (0, 1)))
        env_tmp[site].tr = tmp / tmp.norm(p='inf')

    bl = psi.nn_site(site, d='bl')
    if bl is not None:
        tmp = tensordot(env[l].b, env[l].bl @ proj[bl].hlt, axes=((2, 1), (0, 1)))
        env_tmp[site].bl = tmp / tmp.norm(p='inf')

    br = psi.nn_site(site, d='br')
    if br is not None:
        tmp = tensordot(proj[br].hrt, env[r].br @ env[r].b, axes=((0, 1), (0, 1)))
        env_tmp[site].br = tmp / tmp.norm(p='inf')


def update_env_vertical_(env_tmp, site, env, proj):
    r"""
    Vertical move of CTM step. Calculate environment tensors for given site.
    """
    psi = env.psi

    t = psi.nn_site(site, d='t')
    if t is not None:
        tmp = proj[t].vtl.transpose(axes=(2, 1, 0)) @ env[t].t
        tmp = psi[t]._attach_01(tmp)
        tmp = tensordot(tmp, proj[t].vtr, axes=((2, 3), (0, 1)))
        env_tmp[site].t = tmp / tmp.norm(p='inf')

    b = psi.nn_site(site, d='b')
    if b is not None:
        tmp = proj[b].vbr.transpose(axes=(2, 1, 0)) @ env[b].b
        tmp = psi[b]._attach_23(tmp)
        tmp = tensordot(tmp, proj[b].vbl, axes=((2, 3), (0, 1)))
        env_tmp[site].b = tmp / tmp.norm(p='inf')

    tl = psi.nn_site(site, d='tl')
    if tl is not None:
        tmp = tensordot(env[t].l, env[t].tl @ proj[tl].vtr, axes=((2, 1), (0, 1)))
        env_tmp[site].tl = tmp / tmp.norm(p='inf')

    tr = psi.nn_site(site, d='tr')
    if tr is not None:
        tmp = tensordot(proj[tr].vtl, env[t].tr @ env[t].r, axes=((0, 1), (0, 1)))
        env_tmp[site].tr =  tmp / tmp.norm(p='inf')

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
