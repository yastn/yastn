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

from ._env_ctm import EnvCTM, proj_corners
from ._env_contractions import *
from ._env_dataclasses import EnvCTM_c4v_local, EnvCTM_c4v_projectors
from .._geometry import Lattice
from .._peps import Peps2Layers
from ....tensor import Leg, YastnError, tensordot

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

    _default_corner_signature = (1,1)

    def __init__(self, psi, init='eye', bra=None):
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

        bra: Optional[yastn.tn.Peps]
            If provided, and ``psi`` has physical legs, forms a double-layer PEPS <bra | psi>.
        """
        self.geometry = psi.geometry
        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "f_ordered", "nn_bond_dirn"]:
            setattr(self, name, getattr(self.geometry, name))

        if not isinstance(psi, PsiFlip):
            psi = PsiFlip(psi)
        if bra and not isinstance(bra, PsiFlip):
            bra = PsiFlip(bra)
        self.psi = Peps2Layers(ket=psi, bra=bra) if psi.has_physical() else psi
        self.env = Lattice(self.geometry, objects={site: EnvCTM_c4v_local() for site in self.sites()})
        self.proj = Lattice(self.geometry, objects={site: EnvCTM_c4v_projectors() for site in self.sites()})

        if init not in (None, 'eye', 'dl'):
            raise YastnError(f"{type(self).__name__} {init=} not recognized. Should be 'rand', 'eye', 'dl', or None.")
        if init is not None:
            self.reset_(init=init)

        self.profiling_mode = None

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

    def iterate_(env, opts_svd=None, method='2x2', max_sweeps=1, iterator=False, corner_tol=None, truncation_f: Callable = None, **kwargs):
        return super().iterate_(opts_svd=opts_svd, moves='d', method=method, max_sweeps=max_sweeps, iterator=iterator, corner_tol=corner_tol, truncation_f=truncation_f, **kwargs)
        # move = 'd' has len(move) == 1, as iterate_ will for loop over the string move

    ctmrg_ = iterate_

    def update_(env, opts_svd, method='2x2', **kwargs):
        kwargs['moves'] = 'd'
        return super().update_(opts_svd=opts_svd, method=method, **kwargs)

    def _update_core_(env, move: str, opts_svd: dict, method: str, **kwargs):
        assert move in ['d'], "Invalid move"
        if '2x2' in method:
            env._update_2x2_(opts_svd, **kwargs)
        elif '1x2' in method or '2x1' in method:
            svd_proj = ('svd' in method)
            env._update_1x2_(svd_proj=svd_proj, **kwargs)
        else:
            raise YastnError(f"Unsupported {method=} for c4v-symmetric corner projector.")

    def _update_1x2_(env, svd_proj=False, **kwargs):
        #
        s0 = env.psi.sites()[0]
        #
        if not svd_proj:
            cor_tl_2x1 = env[s0].tl @ env[s0].t
            Q, _ = cor_tl_2x1.qr(axes=((0, 1), 2), sQ=1)
            #Q, _, _ = cor_tl_2x1.svd(axes=((0, 1), 2), sU=1)

            env.proj[s0].vtl = Q
            #
            cor_tl = env[s0].t @ cor_tl_2x1
            cor_tl = tensordot(cor_tl, env.psi[s0], axes=((2, 1), (0, 1)))
            new_tl = tensordot(cor_tl, Q, axes=((1, 3), (0, 1)))
            new_tl = tensordot(Q, new_tl, axes=((0, 1), (0, 1)))
            #
            tmp = tensordot(Q, env[s0].t, axes=(0, 0))
            tmp = tensordot(tmp, env.psi[s0], axes=((0, 2), (1, 0)))
            tmp = tensordot(tmp, Q, axes=((1, 3), (0, 1)))
            new_t = tmp.flip_signature()
        else:  # WIP
            r1 = env[s0].tl @ env[s0].t
            cor_tl = env[s0].t @ r1
            cor_tl = tensordot(cor_tl, env.psi[s0], axes=((2, 1), (0, 1)))

            r0 = env[s0].t @ env[s0].tl
            r0 = r0.fuse_legs(axes=(0, (2, 1))).flip_signature()
            r1 = r1.fuse_legs(axes=(2, (0, 1)))
            p0, p1 = proj_corners(r0, r1, opts_svd={}, cutoff=1e-10, **kwargs)  # TODO: cutoff set by hand; can we make it inverse-free?
            p0 = p0.unfuse_legs(axes=0)
            p1 = p1.unfuse_legs(axes=0)
            p0 = p0.flip_signature()

            new_tl = tensordot(cor_tl.flip_signature(), p0, axes=((1, 3), (0, 1)))
            new_tl = tensordot(p1, new_tl, axes=((0, 1), (0, 1)))
            #
            s1 = env.nn_site(s0, d='r')
            tmp = tensordot(p1, env[s1].t, axes=(0, 0))
            tmp = tensordot(tmp, env.psi[s1], axes=((0, 2), (1, 0)))
            new_t = tensordot(tmp, p0, axes=((1, 3), (0, 1)))
            new_t = new_t.flip_charges(axes=(0, 2))
        #
        env[s0].tl = new_tl / new_tl.norm(p='inf')
        env[s0].t = new_t / new_t.norm(p='inf')

    def _update_2x2_(env, opts_svd, **kwargs):
        #
        s0 = env.psi.sites()[0]
        #
        policy = opts_svd.get('policy', 'fullrank')
        if policy != 'fullrank' and env.proj[s0].vtl is not None and env.proj[s0].vtr is not None:
            opts_svd["k_block"] = env._partial_svd_predict_spec(env.proj[s0].vtl.get_legs(-1), env.proj[s0].vtr.get_legs(0), sU=1)
        elif "k_block" not in opts_svd:
            opts_svd["k_block"] = float('inf')
        #
        #
        cor_tl = env[s0].t @ (env[s0].tl @ env[s0].t)
        cor_tl = tensordot(cor_tl, env.psi[s0], axes=((2, 1), (0, 1)))
        #
        U, S, V = cor_tl.svd_with_truncation(axes=((0, 2), (1, 3)), sU=1, **opts_svd)
        env.proj[s0].vtl, env.proj[s0].vtr = U, V
        #
        new_tl = tensordot(V.conj(), U, axes=((1, 2), (0, 1))) @ S
        new_tl = new_tl / new_tl.norm(p='inf')
        #
        s1 = env.nn_site(s0, d='r')
        tmp = tensordot(U, env[s1].t, axes=(0, 0))
        tmp = tensordot(tmp, env.psi[s1], axes=((0, 2), (1, 0)))
        tmp = tensordot(tmp, U, axes=((1, 3), (0, 1)))
        tmp = tmp.flip_charges(axes=(0, 2))
        new_t = tmp / tmp.norm(p='inf')
        #
        env[s0].tl = new_tl
        env[s0].t = new_t


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
    tD = env[(0, 0)].tl.get_legs(axes=0).tD
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
