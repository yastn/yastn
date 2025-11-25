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

from ._env_ctm import EnvCTM, update_storage_, _partial_svd_predict_spec
from ._env_dataclasses import EnvCTM_c4v_local, EnvCTM_c4v_projectors
from .._geometry import Lattice
from .._peps import Peps2Layers
from ....tensor import Leg, YastnError, tensordot, truncation_mask_multiplets

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

class PsiFlip:
    """Read-only view: tensors are flipped on access."""
    __slots__ = ("_base",)

    def __init__(self, psi):
        self._base = psi

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __getitem__(self, site):
        if (site[0] + site[1]) % 2 == 1:
            return self._base[site].flip_signature()
        return self._base[site]


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

        if not isinstance(psi, PsiFlip):
            psi = PsiFlip(psi)
        if ket and not isinstance(ket, PsiFlip):
            ket = PsiFlip(ket)
        self.psi = Peps2Layers(bra=psi, ket=ket) if psi.has_physical() else psi
        self.env = Lattice(self.geometry, objects={site: EnvCTM_c4v_local() for site in self.sites()})
        self.proj = Lattice(self.geometry, objects={site: EnvCTM_c4v_projectors() for site in self.sites()})

        if init not in (None, 'eye', 'dl'):
            raise YastnError(f"{type(self).__name__} {init=} not recognized. Should be 'rand', 'eye', 'dl', or None.")
        if init is not None:
            self.reset_(init=init)

    def __getitem__(self, site):
        if (site[0] + site[1]) % 2 == 1:
            return EnvFlip(self.env[site])
        else:
            return self.env[site]


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
        super().reset_(init=init)
        if init == 'eye':
            for site in self.sites():
                self[site].t = self[site].t.flip_charges(axes=0)
                self[site].tl = self[site].tl.flip_charges(axes=1)

    def iterate_(env, opts_svd=None, method='2site', max_sweeps=1, iterator=False, corner_tol=None, truncation_f: Callable = None, **kwargs):
        return super().iterate_(opts_svd=opts_svd, moves='d', method=method, max_sweeps=max_sweeps, iterator=iterator, corner_tol=corner_tol, truncation_f=truncation_f, **kwargs)
        # move = 'd' has len(move) == 1, as iterate_ will for loop over the string move

    ctmrg_ = iterate_

    def update_(env, opts_svd, method='2site', **kwargs):
        kwargs['moves'] = 'd'
        return super().update_(opts_svd=opts_svd, method=method, **kwargs)

    def _update_core_(env, move : str, opts_svd : dict, **kwargs):
        assert move in ['d'], "Invalid move"
        policy = opts_svd.get('policy', 'fullrank')
        #
        s0 = env.psi.sites()[0]
        #
        if policy in ['fullrank', "qr"] or env.proj[s0].vtl is None or env.proj[s0].vtr is None:
            svd_predict_spec = lambda s0, p0, s1, p1: opts_svd.get('D_block', float('inf'))
        else:
            psh = Lattice(env.geometry)
            psh[s0] = EnvCTM_c4v_projectors(vtl=env.proj[s0].vtl.get_legs(-1),
                                            vtr=env.proj[s0].vtr.get_legs(0))
            svd_predict_spec = lambda s0, p0, s1, p1: _partial_svd_predict_spec(getattr(psh[s0], p0), getattr(psh[s1], p1), opts_svd.get('sU', 1))
        #
        # 1) get tl enlarged corner and projector from ED/SVD
        #
        # (+) 0--tl--1 0--t--2 (-)
        #                 1
        #
        cor_tl_2x1 = env[s0].tl @ env[s0].t
        #
        # (-) 0--t--2 0--tl--1 0--t--2->3(-)
        #        1                1->2
        #
        cor_tl = env[s0].t @ cor_tl_2x1
        #
        # tl--t---1 (-)
        # t---A--3 (fusion of + and -)
        # 0   2
        cor_tl = tensordot(cor_tl, env.psi[s0], axes=((2, 1), (0, 1)))
        cor_tl = cor_tl.fuse_legs(axes=((0, 2), (1, 3)))
        #
        # Note: U(1)-symm corner is not hermitian. Instead blocks related by conj of charges are hermitian conjugates,
        #       i.e. (2, -2) and (-2, 2) blocks are hermitian conjugates.
        #
        if policy == 'qr':
            R = cor_tl_2x1.flip_signature().fuse_legs(axes=((0, 1), 2))
        else:
            R = cor_tl
        opts_svd["D_block"] = svd_predict_spec(s0, "vtl", s0, "vtr")
        opts_svd["sU"] = 1
        env.proj[s0].vtl, s, env.proj[s0].vtr = proj_sym_corner(R, opts_svd, **kwargs)
        #
        # 2) update move corner
        #
        P = env.proj[s0].vtl
        env_tmp = EnvCTM(env.psi, init=None)  # empty environments
        if policy in ['symeig']:
            assert (cor_tl - cor_tl.H) < 1e-12, "Enlarged corner is not hermitian"
            env_tmp[s0].tl = s / s.norm(p='inf')
        elif policy in ["qr"]:
            S = P.flip_signature().tensordot(cor_tl @ P.flip_signature(), (0, 0))
            S = S.flip_charges()
            env_tmp[s0].tl= S / S.norm(p='inf')
        else:
            S = (env.proj[s0].vtr.conj() @ P) @ s
            env_tmp[s0].tl = S / S.norm(p='inf')
        #
        # 3) update move half-row/-column tensor. Here, P is to act on B-sublattice T tensor
        #
        #   Note:
        #   flip_signature() is equivalent to conj().conj_blocks(), which changes the total charge from +n to -n
        #   flip_charges(axes) is equivalent to switch_signature(axes), which leaves the total charge unchanged
        #
        P = P.unfuse_legs(axes=0)
        #
        # 1<-2--P--0    0--T--2->3
        #        --1->0    1->2
        #
        tmp = tensordot(P, env[s0].t.flip_signature(), axes=(0, 0)) # Pass from T_A to T_B
        #  0<-1--P-----T--3->1  0--P--2
        #        |     2           |
        #        |      0          |
        #         --0 1--A--3   1--
        #                2=>1
        s1 = env.nn_site(s0, d='r')
        tmp = tensordot(tmp, env.psi[s1], axes=((0, 2), (1, 0)))
        tmp = tensordot(tmp, P, axes=((1, 3), (0, 1)))
        tmp = tmp.flip_charges(axes=(0, 2))  # tmp.switch_signature(axes=(0,2))
        #
        # tmp= 0.5*(tmp + tmp.transpose(axes=(2,1,0)))
        env_tmp[s0].t = tmp / tmp.norm(p='inf')
        #
        update_storage_(env, env_tmp)


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


def proj_sym_corner(rr, opts_svd, **kwargs):
    r""" Projector on largest (by magnitude) eigenvalues of (hermitian) symmetric corner. """
    policy = opts_svd.get('policy', 'symeig')
    default_truncation_f = lambda x : truncation_mask_multiplets(x,
                                                                 keep_multiplets=True,
                                                                 D_total=opts_svd['D_total'],
                                                                 D_block=opts_svd['D_block'],
                                                                 tol=opts_svd['tol'],
                                                                 eps_multiplet=opts_svd['eps_multiplet'],
                                                                 hermitian=True)
    truncation_f = kwargs.get('truncation_f', default_truncation_f)


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
        # u, s, v = rr.svd_with_truncation(axes=(0, 1), mask_f=truncation_f, **opts_svd)  # BUG in one of the tests
        u, s, v = rr.svd_with_truncation(axes=(0, 1), mask_f=default_truncation_f, **opts_svd)

    elif policy in ['qr']:
        u, s = rr.qr(axes=(0, 1), sQ=1, Qaxis=-1, Raxis=0)
        v = None
    else:
        raise YastnError(f"Unsupported policy {policy} for c4v-symmetric corner projector.")

    return u, s, v
