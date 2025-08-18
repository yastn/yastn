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
import logging
from .... import Tensor, YastnError, tensordot, truncation_mask, decompress_from_1d, \
    truncation_mask_multiplets
from .._peps import Peps, Peps2Layers, DoublePepsTensor
from .._geometry import RectangularUnitcell
from ._env_auxlliary import *
from ._env_ctm import EnvCTM, decompress_env_1d, EnvCTM_projectors, store_projectors_, update_old_env_

logger = logging.Logger('ctmrg')


class EnvCTM_c4v(EnvCTM):
    def __init__(self, psi, init='eye', leg=None):
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

        leg: Optional[yastn.Leg]
            Passed to :meth:`yastn.tn.fpeps.EnvCTM.reset_` to further customize initialization.
        """
        super().__init__(psi, init=None)
        if init not in (None, 'rand', 'eye', 'dl'):
            raise YastnError(f"EnvCTM_c4v {init=} not recognized. Should be 'rand', 'eye', 'dl', None.")
        if init is not None:
            self.reset_(init=init, leg=leg)

    # Cloning/Copying/Detaching(view)
    #
    def copy(self) -> EnvCTM:
        env = EnvCTM_c4v(self.psi, init=None)
        for site in env.sites():
            for dirn in ['tl', 't']:
                setattr(env[site], dirn, getattr(self[site], dirn).copy())
        return env

    def shallow_copy(self) -> EnvCTM_c4v:
        env = EnvCTM_c4v(self.psi, init=None)
        for site in env.sites():
            for dirn in ['tl', 't']:
                setattr(env[site], dirn, getattr(self[site], dirn))
        return env

    def clone(self) -> EnvCTM_c4v:
        r"""
        Return a clone of the environment preserving the autograd - resulting clone is a part
        of the computational graph. Data of cloned environment tensors is indepedent
        from the originals.
        """
        env = EnvCTM_c4v(self.psi, init=None)
        for site in env.sites():
            for dirn in ['tl', 't']:
                setattr(env[site], dirn, getattr(self[site], dirn).clone())
        return env

    def detach(self) -> EnvCTM_c4v:
        r"""
        Return a detached view of the environment - resulting environment is **not** a part
        of the computational graph. Data of detached environment tensors is shared
        with the originals.
        """
        env = EnvCTM_c4v(self.psi, init=None)
        for site in env.sites():
            for dirn in ['tl', 't']:
                setattr(env[site], dirn, getattr(self[site], dirn).detach())
        return env

    def detach_(self):
        r"""
        Detach all environment tensors from the computational graph.
        Data of environment tensors in detached environment is a `view` of the original data.
        """
        for site in self.sites():
            for dirn in ['tl', 't']:
                if getattr(self[site], dirn) is None:
                    continue
                try:
                    getattr(self[site], dirn)._data.detach_()
                except RuntimeError:
                    setattr(self[site], dirn, getattr(self[site], dirn).detach())

    # def compress_env_c4v_1d(env):
    #     r"""
    #     Compress environment to data tensors and (hashable) metadata, see :func:`yastn.tensor.compress_to_1d`.

    #     Parameters
    #     ----------
    #     env : EnvCTM_c4v
    #         Environment instance to be transformed.

    #     Returns
    #     -------
    #     (tuple[Tensor] , dict)
    #         A pair where the first element is a tuple of raw data tensors (of type derived from backend)
    #         and the second is a dict with corresponding metadata.
    #     """
    #     shallow= {
    #         'psi': {site: env.psi.bra[site] for site in env.sites()} if isinstance(env.psi,Peps2Layers) \
    #             else {site: env.psi[site] for site in env.sites()},
    #         'env': tuple( env_t for site in env.sites() for k,env_t in env[site].__dict__.items() if env_t is not None)}
    #     dtypes= set(tuple( t.yastn_dtype for t in shallow['psi'].values()) + tuple(t.yastn_dtype for t in shallow['env']))
    #     assert len(dtypes)<2, f"CTM update: all tensors of state and environment should have the same dtype, got {dtypes}"
    #     unrolled= {'psi': {site: t.compress_to_1d() for site,t in shallow['psi'].items()},
    #         'env': tuple(t.compress_to_1d() for t in shallow['env'])}
    #     meta= {'psi': {site: t_and_meta[1] for site,t_and_meta in unrolled['psi'].items()}, 'env': tuple(meta for t,meta in unrolled['env']),
    #            '2layer': isinstance(env.psi, Peps2Layers), 'geometry': env.geometry, 'sites': env.sites()}
    #     data= tuple( t for t,m in unrolled['psi'].values())+tuple( t for t,m in unrolled['env'])
    #     return data, meta

    def reset_(self, init='eye', leg=None, **kwargs):
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

        leg: None | yastn.Leg
            If not provided, random initialization has CTMRG bond dimension set to 1.
            Otherwise, the provided Leg is used to initialize CTMRG virtual legs.
        """
        assert init in ['eye', 'dl'], "Invalid initialization type. Should be 'eye' or 'dl'."

        if init == 'eye':
            super().reset_(init='eye', leg=leg, **kwargs)
            self[0,0].tl= self[0,0].tl.flip_charges(axes=1)
            self[0,0].t= self[0,0].t.flip_charges(axes=0)
        elif init == 'dl':
            # create underlying two-site bipartite PEPS
            g= RectangularUnitcell(pattern=[[0,1],[1,0]])
            bp= Peps(geometry=g, \
                    tensors={ g.sites()[0]: self.psi.ket[0,0], g.sites()[1]: self.psi.ket[0,0].conj() }, )
            env_bp= EnvCTM(bp, init='eye', leg=leg)
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
        if 'policy' in kwargs:
            opts_svd['policy'] = kwargs.get('policy')
        checkpoint_move= kwargs.pop('checkpoint_move',False)

        #
        # Empty structure for projectors
        proj = Peps(env.geometry)
        for site in proj.sites(): proj[site] = EnvCTM_projectors()

        def _compress_proj(proj, empty_proj):
            data, meta= tuple(zip( *(t.compress_to_1d() if not (t is None) else empty_proj.compress_to_1d() \
                for site in proj.sites() for t in proj[site].__dict__.values()) ))
            return data, meta

        #
        # get projectors and compute updated env tensors
        # TODO currently supports only <psi|psi> for double-layer peps


        if checkpoint_move:
            outputs_meta= {}

            # extract raw parametric tensors as a tuple
            inputs_t, inputs_meta= env.compress_env_1d()

            def f_update_core_2dir(move_d,loc_im,*inputs_t):
                loc_env= decompress_env_c4v_1d(inputs_t,loc_im)

                env_tmp, proj_tmp= _update_core_dir(loc_env, "default", opts_svd, method=method, **kwargs)
                update_old_env_(loc_env, env_tmp)

                # return backend tensors - only environment and projectors
                #
                out_env_data, out_env_meta= loc_env.compress_env_1d()
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
                outputs= checkpoint_F(f_update_core_2dir,None,inputs_meta,*inputs_t,\
                                    **{'use_reentrant': use_reentrant, 'debug': False})
            else:
                raise RuntimeError(f"CTM update: checkpointing not supported for backend {env.config.BACKEND_ID}")

            # update tensors of env and proj
            for i,site in enumerate(env.sites()):
                for env_t,t,t_meta in zip(env[site].__dict__.keys(),outputs[i*8:(i+1)*8],outputs_meta['env'][i*8:(i+1)*8]):
                    setattr(env[site],env_t,decompress_from_1d(t,t_meta) if t is not None else None)

            for i,site in enumerate(proj.sites()):
                for proj_t,t,t_meta in zip(proj[site].__dict__.keys(),outputs[8*len(env.sites()):][i*8:(i+1)*8],outputs_meta['proj'][i*8:(i+1)*8]):
                    setattr(proj[site],proj_t,decompress_from_1d(t,t_meta) if t_meta['struct'].size>0 else None)

        else:
            env_tmp, proj_tmp= _update_core_dir(env, "default", opts_svd, method=method, **kwargs)
            update_old_env_(env, env_tmp)
            store_projectors_(proj, proj_tmp)
        return proj


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
                       for dirn in ['tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r']}
            d['data'][site] = d_local
        return d


def decompress_env_c4v_1d(data,meta)->EnvCTM_c4v:
    """
    Reconstruct the environment from its compressed form.

    Parameters
    ----------
    data : Sequence[Tensor]
        Collection of 1D data tensors for both environment and underlying PEPS.
    meta : dict
        Holds metadata of original environment (and PEPS).

    Returns
    -------
    EnvCTM
    """
    loc_env= decompress_env_1d(data,meta)
    res= EnvCTM_c4v(psi=loc_env.psi, init=None)
    for site in loc_env.sites():
        for dirn in ['tl','t']:
            if getattr(loc_env[site], dirn) is not None:
                setattr(res[site], dirn, getattr(loc_env[site], dirn))
    return res


def _update_core_dir(env, dir : str, opts_svd : dict, **kwargs):
        assert dir in ['default'], "Invalid directions"
        method= kwargs.get('method','default')
        policy= opts_svd.get('policy','fullrank')

        #
        # Empty structure for projectors
        proj = Peps(env.geometry)
        for site in proj.sites():
            proj[site] = EnvCTM_projectors()
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
        proj[s0].vtl, s, proj[s0].vtr= proj_sym_corner(R, opts_svd, sU=1, **kwargs)

        # 2) update move corner
        P= proj[s0].vtl
        env_tmp = EnvCTM(env.psi, init=None)  # empty environments
        if policy in ['symeig']:
            assert (cor_tl-cor_tl.H)<1e-12,"enlarged corner is not hermitian"
            env_tmp[s0].tl = s/s.norm(p='inf')
        elif policy in ["qr"]:
            S= P.flip_signature().tensordot( cor_tl @ P.flip_signature(), (0, 0))
            S= S.flip_charges()
            env_tmp[s0].tl= (S/S.norm(p='inf'))
        else:
            S= ((proj[s0].vtr.conj() @ P) @ s)
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

        return env_tmp, proj


def proj_sym_corner(rr, opts_svd, **kwargs):
    r""" Projector on largest (by magnitude) eigenvalues of (hermitian) symmetric corner. """
    policy = opts_svd.get('policy', 'symeig')
    fix_signs= opts_svd.get('fix_signs',True)
    truncation_f= kwargs.get('truncation_f',\
        lambda x : truncation_mask_multiplets(x,keep_multiplets=True, \
            D_total=opts_svd['D_total'], tol=opts_svd['tol'], \
            eps_multiplet=opts_svd['eps_multiplet'], hermitian=True, ) )

    if policy in ['symeig']:
        # TODO fix_signs ?
        _kwargs= dict(kwargs)
        for k in ["method", "use_qr",]: del _kwargs[k]
        s,u= rr.eigh_with_truncation(axes=(0,1), sU=rr.s[1], which='LM', mask_f= truncation_f, **opts_svd, **_kwargs)
        v= None
    elif policy in ['fullrank', 'lowrank', 'arnoldi', 'krylov']:
        # sU = ? r0.s[1]
        if truncation_f is None:
            u, s, v = rr.svd(axes=(0, 1), fix_signs=fix_signs, **kwargs)
            Smask = truncation_mask(s, **opts_svd)
            u, s, v = Smask.apply_mask(u, s, v, axes=(-1, 0, 0))
        else:
            u, s, v = rr.svd_with_truncation(axes=(0, 1), mask_f=truncation_f, **kwargs)
    elif policy in ['qr']:
        u, s= rr.qr(axes=(0, 1), sQ=1, Qaxis=-1, Raxis=0)
        v= None

    return u, s, v