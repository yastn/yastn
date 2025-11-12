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
import logging
from typing import Callable, Sequence

from ._env_auxlliary import *
from ._env_ctm import EnvCTM, update_storage_, CTMRG_out, _partial_svd_predict_spec
from ._env_dataclasses import EnvCTM_c4v_local, EnvCTM_c4v_projectors
from .._geometry import RectangularUnitcell, Lattice
from .._peps import Peps, Peps2Layers, DoublePepsTensor, PEPS_CLASSES
from ....initialize import eye, split_data_and_meta, combine_data_and_meta
from ....tensor import Leg, YastnError, tensordot, truncation_mask, truncation_mask_multiplets

logger = logging.Logger('ctmrg')

class EnvFlip:
    """Read-only view: tensors are flipped on access."""
    __slots__ = ("_base",)

    def __init__(self, base: EnvCTM_c4v_local):
        self._base = base

    # attribute access
    def __getattr__(self, dirn):
        return getattr(self._base, dirn).flip_signature()

    def __repr__(self):
        return f"EnvFlip(base={self._base!r})"

class EnvCTM_c4v(EnvCTM):
    def __init__(self, psi, init='eye', ket=None):
        r"""
        Environment used in Corner Transfer Matrix Renormalization Group algorithm for C4v symmetric
        single-site iPEPS. Here, the on-site tensor is assumed to be C4v-symmetric, i.e. transform
        covariantly under rotation by 90 degrees and reflection across the x and y axes.

        Formulation with internal symmetries requires a choice of signature, which necessarily breaks explicit
        single-site character. We need at least two different tensors to represent the state, e.g.::

                (+)         (-)
            (+)--A--(+) (-)--B--(-)
                (+)         (-)
                (-)         (+)
            (-)--B--(-) (+)--A--(+)
                (-)         (+)

        The tensor B is a function of tensor A as B = A.flip_signature()

        There is just one unique C and one unique T tensor making up the environment, the
        C,T tensors for A- and B-sublattices are related by same signature transformation.
        Here, we chose top-left corner and top transfer tensor of sublattice A.

        Index convention for environment tensors follows from on-site tensors::

            C_A--(+),  (-)--T_A--(-)
             |               |
            (+)             (-)

        Parameters
        ----------
        psi: yastn.tn.Peps
            PEPS lattice to be contracted using CTM.
            If ``psi`` has physical legs, a double-layer PEPS with no physical legs is formed.

        init: str
            None, 'eye' or 'dl'. Initialization scheme, see :meth:`yastn.tn.fpeps.EnvCTM.reset_`.

        ket: Optional[yastn.tn.Peps]
            If provided, and ``psi`` has physical legs, forms a double-layer PEPS <psi | ket>.
        """
        self.geometry = psi.geometry
        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "f_ordered", "nn_bond_dirn"]:
            setattr(self, name, getattr(self.geometry, name))

        self.psi = Peps2Layers(bra=psi, ket=ket) if psi.has_physical() else psi
        self.env = Lattice(self.geometry, objects={site: EnvCTM_c4v_local() for site in self.sites()})
        self.proj = Lattice(self.geometry, objects={site: EnvCTM_c4v_projectors() for site in self.sites()})

        if init not in (None, 'eye', 'dl'):
            raise YastnError(f"{type(self).__name__} {init=} not recognized. Should be 'rand', 'eye', 'dl', or None.")
        if init is not None:
            self.reset_(init=init)

    def __getitem__(self, site):
        if (site[0] + site[1])%2 == 1:
            return EnvFlip(self.env[site])
        else:
            return self.env[site]

    def max_D(self):
        m_D = 0
        for site in self.sites():
            if getattr(self[site], 'tl') is not None:
                m_D = max(max(getattr(self[site], 'tl').get_shape()), m_D)
        return m_D

    def reset_(self, init='eye'):
        r"""
        Initialize C4v-symmetric CTMRG environment::

            C--T--C => C---T--T'--T--C => C--T-- & --T'-- <=>
            T--A--T    T---A--B---A--T    T--A--   --B---     C'--T'--
            C--T--C    T'--B--A---B--T    |  |       |        |   |
                       T---A--B---A--T
                       C---T--T'--T--C

        Ther are two different T tensors - one for A-sublattice and one for B-sublattice.
        They are related by adjoint * complex conjugation (i.e. :meth:`flip_signature`)

        Parameters
        ----------
        init: str
            ['eye', 'dl']
            For 'eye' starts with identity environments of dimension 1.
            For 'dl' and Env of double-layer PEPS, trace on-site tensors to initialize environment.
        """
        assert init in ['eye', 'dl'], "Invalid initialization type. Should be 'eye' or 'dl'."

        if init == 'eye':
            config = self.psi.config
            leg0 = Leg(config, s=1, t=(config.sym.zero(),), D=(1,))

            self[0,0].tl = eye(config, legs=[leg0, leg0.conj()], isdiag=False)
            legs = self.psi[0,0].get_legs()
            tmp1 = identity_boundary(config, legs[0].conj())
            tmp0 = eye(config, legs=[leg0, leg0.conj()], isdiag=False)
            tmp = tensordot(tmp0, tmp1, axes=((), ())).transpose(axes=(0, 2, 1))
            self[0,0].t = tmp

            self[0,0].tl= self[0,0].tl.flip_charges(axes=1)
            self[0,0].t= self[0,0].t.flip_charges(axes=0)
        elif init == 'dl':
            # create underlying two-site bipartite PEPS
            g= RectangularUnitcell(pattern=[[0,1],[1,0]])
            bp= Peps(geometry=g, \
                    tensors={ g.sites()[0]: self.psi.ket[0,0], g.sites()[1]: self.psi.ket[0,0].conj() }, )
            env_bp= EnvCTM(bp, init='eye')
            env_bp.expand_outward_()
            # env_bp.init_env_from_onsite_()

            self[0,0].t= env_bp[0,0].t.drop_leg_history(axes=(0,2)).switch_signature(axes=0)
            self[0,0].tl= env_bp[0,0].tl.drop_leg_history(axes=(0,1)).switch_signature(axes=1)


    def update_(env, opts_svd, method='default', **kwargs):
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
            'sl'

        checkpoint_move: bool
            Whether to use (reentrant) checkpointing for the move. The default is ``False``

        Returns
        -------
            proj: Peps structure loaded with CTM projectors related to all lattice site.
        """
        if all(s not in opts_svd for s in ('tol', 'tol_block')):
            opts_svd['tol'] = 1e-14
        if method not in ('default',):
            raise YastnError(f"CTM update {method=} not recognized. Should be 'default' or ...")
        # if 'policy' in kwargs:
        #     opts_svd['policy'] = kwargs.get('policy')
        checkpoint_move= kwargs.pop('checkpoint_move',False)

        if checkpoint_move:
            def f_update_core_2dir(loc_im,*inputs_t):
                loc_env= EnvCTM_c4v.from_dict(combine_data_and_meta(inputs_t, loc_im))

                _update_core_dir(loc_env, "default", opts_svd, method=method, **kwargs)
                out_dict = loc_env.to_dict(level=0)
                out_data, out_meta = split_data_and_meta(out_dict)

                return out_data, out_meta

            if env.config.backend.BACKEND_ID == "torch":
                env_dict = env.to_dict(level=0)
                inputs_t, inputs_meta = split_data_and_meta(env_dict)
                if checkpoint_move=='reentrant':
                    use_reentrant= True
                elif checkpoint_move=='nonreentrant':
                    use_reentrant= False
                checkpoint_F= env.config.backend.checkpoint
                out_data, out_meta = checkpoint_F(f_update_core_2dir, inputs_meta,*inputs_t,\
                                    **{'use_reentrant': use_reentrant, 'debug': False})
            else:
                raise RuntimeError(f"CTM update: checkpointing not supported for backend {env.config.BACKEND_ID}")
            # reconstruct env from output tensors
            out_env_dict = combine_data_and_meta(out_data, out_meta)
            env.update_from_dict_(out_env_dict)
        else:
            _update_core_dir(env, "default", opts_svd, method=method, **kwargs)



    def get_env_bipartite(self):
        g= RectangularUnitcell(pattern=[[0,1],[1,0]])
        bp= Peps(geometry=g, \
            tensors={ g.sites()[0]: self.psi.ket[0,0], g.sites()[1]: self.psi.ket[0,0].flip_signature() }, )
        env_bp= EnvCTM(bp, init=None)
        s0= g.sites()[0]
        env_bp[s0].tr= env_bp[s0].br= env_bp[s0].bl= env_bp[s0].tl= self[0,0].tl
        env_bp[s0].l= env_bp[s0].b= env_bp[s0].r= env_bp[s0].t= self[0,0].t
        s1= g.sites()[1]
        env_bp[s1].tr= env_bp[s1].br= env_bp[s1].bl= env_bp[s1].tl= self[0,0].tl.flip_signature()
        env_bp[s1].l= env_bp[s1].b= env_bp[s1].r= env_bp[s1].t= self[0,0].t.flip_signature()
        return env_bp


    def save_to_dict(self) -> dict:
        r"""
        Serialize EnvCTM into a dictionary.
        """
        psi = self.psi
        if isinstance(psi, Peps2Layers):
            psi = psi.ket

        d = {'class': 'EnvCTM_C4v',
             'psi': psi.save_to_dict(),
             'data': {}}
        for site in self.sites():
            d_local = {dirn: getattr(self[site], dirn).save_to_dict()
                       for dirn in ['tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r'] if getattr(self[site], dirn)}
            d['data'][site] = d_local
        return d

    def ctmrg_(env, opts_svd, method='default', max_sweeps=1, iterator_step=1, corner_tol=None, truncation_f: Callable=None,  **kwargs):
        if "checkpoint_move" in kwargs:
            if env.config.backend.BACKEND_ID == "torch":
                assert kwargs["checkpoint_move"] in ['reentrant','nonreentrant',False], f"Invalid choice for {kwargs['checkpoint_move']}"
        # BUG: fails when uncomment the following line
        # kwargs["truncation_f"]= truncation_f
        tmp = _iterate_ctmrg_(env, opts_svd, method, max_sweeps, iterator_step, corner_tol, **kwargs)
        return tmp if iterator_step else next(tmp)

def _iterate_ctmrg_(env, opts_svd, method, max_sweeps, iterator_step, corner_tol, **kwargs):
    """ Generator for ctmrg_(). """
    max_dsv, converged = None, False
    proj_history = None
    for sweep in range(1, max_sweeps + 1):
        env.update_(opts_svd=opts_svd, method=method, proj_history=proj_history, **kwargs)
        current_proj= env.proj
        # Here, we have access to all projectors obtained in the previous CTM step
        # For partial SVD solvers, we need
        # 1. estimate of how many singular triples to solve for in each block, both blocks kept in truncation
        #    and blocks discarded in truncation
        # 2. perform truncation, typically restricting only total number of singular triples
        #
        policy= opts_svd.get('policy','fullrank')
        if policy not in ['fullrank', "qr"]:
            if proj_history is None:
                # Empty structure for projectors
                proj_history = Peps(env.geometry)
                for site in proj_history.sites(): proj_history[site] = EnvCTM_c4v_projectors()
            for site in current_proj.sites():
                if current_proj[site].vtl is not None:
                    proj_history[site].vtl = current_proj[site].vtl.get_legs(-1)
                if current_proj[site].vtr is not None:
                    proj_history[site].vtr = current_proj[site].vtr.get_legs(0)

        # Default CTM convergence check
        if corner_tol is not None:
            if sweep==1: history = []
            converged, max_dsv, history= ctm_c4v_conv_corner_spec(env.detach(), history, corner_tol)
            logging.info(f'Sweep = {sweep:03d}; max_diff_corner_singular_values = {max_dsv}')
            if converged: break

        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, max_D=env.max_D(), converged=converged)
    yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, max_D=env.max_D(), converged=converged)


def leg_charge_conv_check(env : EnvCTM_c4v, history : Sequence[Leg] = None, conv_len=3):
    r"""
    CTM convergence check targeting distribution of charges only (ignoring corner spectra).

    Returns
    -------
        converged : bool
            If charge sectors stay constant for more than ``conv_len`` CTM steps, return ``True``.
        history : Sequence[Leg]
            Past charge sectors of corner tensor
    """
    tD = env[(0,0)].tl.get_legs(axes=0).tD
    converged = True
    # number of past env interations to check against
    # TODO make adjustable
    conv_len = 3
    history.append(tD)
    if len(history) < conv_len:
        return False, history
    for i in range(1, conv_len+1):
        if tD != history[-i]:
            converged = False
            break
    return converged, history

def _update_core_dir(env, dir : str, opts_svd : dict, **kwargs):
    assert dir in ['default'], "Invalid directions"
    method= kwargs.get('method','default')
    policy= opts_svd.get('policy','fullrank')
    psh = kwargs.pop("proj_history", None)
    # Inherit _partial_svd_predict_spec from EnvCTM
    svd_predict_spec= lambda s0,p0,s1,p1: opts_svd.get('D_block', float('inf')) if psh is None else \
        _partial_svd_predict_spec(getattr(psh[s0],p0), getattr(psh[s1],p1), opts_svd.get('sU', 1))

    #
    # Empty structure for projectors
    #
    s0 = env.psi.sites()[0]

    # 1) get tl enlarged corner and projector from ED/SVD
    # (+) 0--tl--1 0--t--2 (-)
    #                 1
    cor_tl_2x1 = env[s0].tl @ env[s0].t
    # (-) 0--t--2 0--tl--1 0--t--2->3(-)
    #        1                1->2
    cor_tl = env[s0].t @ cor_tl_2x1
    # tl--t---1 (-)
    # t---A--3 (fusion of + and -)
    # 0   2
    cor_tl = tensordot(cor_tl, env.psi[s0], axes=((2, 1), (0, 1)))
    cor_tl = cor_tl.fuse_legs(axes=((0, 2), (1, 3)))

    # Note: U(1)-symm corner is not hermitian. Instead blocks related by conj of charges are hermitian conjugates,
    #       i.e. (2,-2) and (-2,2) blocks are hermitian conjugates.
    R= cor_tl_2x1.flip_signature().fuse_legs(axes=((0, 1), 2)) if policy in ['qr'] else cor_tl
    opts_svd["D_block"]= svd_predict_spec(s0, "vtl", s0, "vtr")
    opts_svd["sU"]= 1
    env.proj[s0].vtl, s, env.proj[s0].vtr= proj_sym_corner(R, opts_svd, **kwargs)

    # 2) update move corner
    P = env.proj[s0].vtl
    env_tmp = EnvCTM(env.psi, init=None)  # empty environments
    if policy in ['symeig']:
        assert (cor_tl-cor_tl.H)<1e-12,"enlarged corner is not hermitian"
        env_tmp[s0].tl = s/s.norm(p='inf')
    elif policy in ["qr"]:
        S= P.flip_signature().tensordot( cor_tl @ P.flip_signature(), (0, 0))
        S= S.flip_charges()
        env_tmp[s0].tl= (S/S.norm(p='inf'))
    else:
        S= ((env.proj[s0].vtr.conj() @ P) @ s)
        env_tmp[s0].tl= (S/S.norm(p='inf'))

    # 3) update move half-row/-column tensor. Here, P is to act on B-sublattice T tensor
    #
    #   Note:
    #   flip_signature() is equivalent to conj().conj_blocks(), which changes the total charge from +n to -n
    #   flip_charges(axes) is equivalent to switch_signature(axes), which leaves the total charge unchanged
    #
    P= P.unfuse_legs(axes=0)
    # 1<-2--P--0    0--T--2->3
    #        --1->0    1->2
    tmp = tensordot(P, env[s0].t.flip_signature(), axes=(0, 0)) # Pass from T_A to T_B
    #  0<-1--P-----T--3->1  0--P--2
    #        |     2           |
    #        |      0          |
    #         --0 1--A--3   1--
    #                2=>1
    _b_sublattice= env.psi.bra[s0].flip_signature() # transform A with signature [1,1,1,1,1] into B with [-1,-1,-1,-1,-1]
    tmp = tensordot(tmp, DoublePepsTensor(bra=_b_sublattice, ket=_b_sublattice), axes=((0, 2), (1, 0)))
    tmp = tensordot(tmp, P, axes=((1, 3), (0, 1)))
    tmp = tmp.flip_charges(axes=(0,2)) #tmp.switch_signature(axes=(0,2))

    # tmp= 0.5*(tmp + tmp.transpose(axes=(2,1,0)))
    env_tmp[s0].t = tmp / tmp.norm(p='inf')
    #
    update_storage_(env, env_tmp)


def proj_sym_corner(rr, opts_svd, **kwargs):
    r""" Projector on largest (by magnitude) eigenvalues of (hermitian) symmetric corner. """
    policy = opts_svd.get('policy', 'symeig')
    truncation_f= kwargs.get('truncation_f',\
        lambda x : truncation_mask_multiplets(x,keep_multiplets=True, \
            D_total=opts_svd['D_total'], D_block=opts_svd['D_block'], tol=opts_svd['tol'], \
            eps_multiplet=opts_svd['eps_multiplet'], hermitian=True, ) )

    if policy in ['symeig']:
        # TODO U1-c4v-symmetric corner is not Hermitian
        raise YastnError("Policy 'symeig' is not supported for c4v-symmetric corner projector.")
        # TODO fix_signs ?
        # _kwargs= dict(kwargs)
        # for k in ["method", "use_qr",]: del _kwargs[k]
        # s,u= rr.eigh_with_truncation(axes=(0,1), sU=rr.s[1], which='LM', mask_f= truncation_f, **opts_svd, **_kwargs)
        # v= None
    elif policy in ['fullrank', 'randomized', 'block_arnoldi', 'block_propack']:
        # sU = ? r0.s[1]
        if truncation_f is None:
            u, s, v = rr.svd(axes=(0, 1), **opts_svd)
            Smask = truncation_mask(s, **opts_svd)
            u, s, v = Smask.apply_mask(u, s, v, axes=(-1, 0, 0))
        else:
            u, s, v = rr.svd_with_truncation(axes=(0, 1), mask_f=truncation_f, **opts_svd)
    elif policy in ['qr']:
        u, s= rr.qr(axes=(0, 1), sQ=1, Qaxis=-1, Raxis=0)
        v= None
    else:
        raise YastnError(f"Unsupported policy {policy} for c4v-symmetric corner projector.")

    return u, s, v

def ctm_c4v_conv_corner_spec(env, history=[], corner_tol=1.0e-8):
    """
    Evaluate convergence of CTM by computing the difference of environment corner spectra between consecutive CTM steps.
    """
    history.append(calculate_c4v_corner_svd(env))
    def spec_diff(x,y):
        if x is not None and y is not None:
            return (x - y).norm().item()
        elif x is None and y is None:
            return 0
        else:
            return float('Inf')
    max_dsv = max(spec_diff(history[-1][k], history[-2][k]) for k in history[-1]) if len(history)>1 else float('Nan')
    history[-1]['max_dsv'] = max_dsv

    return (corner_tol is not None and max_dsv < corner_tol), max_dsv, history


def calculate_c4v_corner_svd(env):
    """
    Return normalized SVD spectra, with largest singular value set to unity, of all corner tensors of environment.
    The corners are indexed by pair of Site and corner identifier.
    """
    _get_spec= lambda x: x.svd(compute_uv=False) if not (x is None) and not x.isdiag else x

    corner_sv = {}
    corner_sv[(0, 0), 'tl'] = _get_spec(env[0,0].tl)
    for k, v in corner_sv.items():
        if not corner_sv[k] is None:
            corner_sv[k] = v / v.norm(p='inf')
    return corner_sv