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
import sys
from typing import NamedTuple, Callable, Sequence
from warnings import warn

from ._env_contractions import identity_boundary, corner2x2, append_vec_tl, append_vec_br
from ._env_dataclasses import EnvCTM_local, EnvCTM_projectors
from .._evolution import BondMetric
from .._geometry import Site, Lattice
from .._peps import PEPS_CLASSES, Peps2Layers
from ... import mps
from ....initialize import rand, rand_like, ones, eye
from ....tensor import Tensor, YastnError, Leg, tensordot, qr, ncon, truncation_mask
from ...._split_combine_dict import split_data_and_meta, combine_data_and_meta

logger = logging.getLogger(__name__)

class CTMRG_out(NamedTuple):
    sweeps: int = 0
    max_dsv: float = None
    converged: bool = False
    max_D: int = 1


class EnvCTM():

    _default_corner_signature = (1, -1)

    def __init__(self, psi, init='rand', leg=None, bra=None):
        r"""
        Environment used in Corner Transfer Matrix Renormalization Group algorithm.

        Note:
            Index convention for environment tensors::

                C---1 0---T---2 0---C
                |         |         |
                0         1         1
                2                   0
                |                   |
                T---1           1---T
                |                   |
                0                   2
                1         1         0
                |         |         |
                C---0 2---T---0 1---C

            * enlarged corners: anti-clockwise

        Parameters
        ----------
        psi: yastn.tn.Peps
            PEPS lattice to be contracted using CTM.
            If ``psi`` has physical legs, a double-layer PEPS with no physical legs is formed.

        init: str | None
            None, 'eye', 'rand', or 'dl'. Initialization scheme, see :meth:`yastn.tn.fpeps.EnvCTM.reset_`.

        leg: Optional[yastn.Leg]
            Passed to :meth:`yastn.tn.fpeps.EnvCTM.reset_` to further customize initialization.

        bra: Optional[yastn.tn.Peps]
            If provided, and ``psi`` has physical legs, forms a double-layer PEPS <bra | ket>.
        """
        self.geometry = psi.geometry
        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "f_ordered", "nn_bond_dirn"]:
            setattr(self, name, getattr(self.geometry, name))

        self.psi = Peps2Layers(ket=psi, bra=bra) if psi.has_physical() else psi
        self.env = Lattice(self.geometry, objects={site: EnvCTM_local() for site in self.sites()})
        self.proj = Lattice(self.geometry, objects={site: EnvCTM_projectors() for site in self.sites()})
        # Recycled subspace-iteration bases. Keys are ``(site, pair)``, where
        # pair is one of ``hlb``, ``hrb``, ``vtr``, or ``vbr``.
        self.X = {}
        self.Y = {}
        self._si_age = {}

        if init not in (None, 'rand', 'eye', 'dl'):
            raise YastnError(f"{type(self).__name__} {init=} not recognized. Should be 'rand', 'eye', 'dl', or None.")
        if init is not None:
            self.reset_(init=init, leg=leg)

        self.profiling_mode = None

    def __repr__(self) -> str:
        return f"EnvCTM(envs={super().__repr__()},\nproj={self.proj})"

    @property
    def config(self):
        return self.psi.config

    def __getitem__(self, site):
        return self.env[site]

    def __setitem__(self, site, obj):
        self.env[site] = obj

    def max_D(self):
        """
        Bond dimension of largest sector in the environment.
        """
        m_D = 0
        for site in self.sites():
            for dirn in self[site].fields(among=['tl', 'tr', 'bl', 'br']):
                if getattr(self[site], dirn) is not None:
                    m_D = max(max(getattr(self[site], dirn).get_shape()), m_D)
        return m_D

    def effective_chi(self):
        r"""
        :return: returns the effective bond dimension of the environment
        :rtype: int

        The effective bond dimension is defined as maximum of sum of sector dimensions
        among all environment indices/legs.
        """
        max_chi= max( [ sum(l.D) for site in self.sites() for dirn in ['tl', 'tr', 'bl', 'br',] \
                       for l in getattr(self[site], dirn).get_legs() ] )
        return max_chi

    # Cloning/Copying/Detaching(view)
    #
    def copy(self) -> EnvCTM:
        r"""
        Return a clone of the environment preserving the autograd - resulting clone is a part
        of the computational graph. Data of cloned environment tensors is indepedent
        from the originals.
        """
        cls = type(self)
        env = cls(self.psi, init=None)
        env.env = self.env.copy()
        env.proj = self.proj.copy()
        env.X = {k: v.copy() for k, v in self.X.items()}
        env.Y = {k: v.copy() for k, v in self.Y.items()}
        env._si_age = self._si_age.copy()
        return env

    def shallow_copy(self) -> EnvCTM:
        cls = type(self)
        env = cls(self.psi, init=None)
        env.env = self.env.shallow_copy()
        env.proj = self.proj.shallow_copy()
        env.X = self.X.copy()
        env.Y = self.Y.copy()
        env._si_age = self._si_age.copy()
        return env

    def to(self, device: str=None, dtype: str=None, **kwargs) -> EnvCTM:
        r"""
        Return a clone of the environment on specified device and/or dtype.
        Resulting environment is a part of the computational graph.
        Data of environment tensors in the new environment is indepedent
        from the originals.
        """
        #TODO Ket ?
        env = type(self)(psi=self.psi.bra.to(device=device, dtype=dtype, **kwargs), init=None)
        env.env = self.env.to(device=device, dtype=dtype, **kwargs)
        env.proj = self.proj.to(device=device, dtype=dtype, **kwargs)
        env.X = {k: v.to(device=device, dtype=dtype, **kwargs) for k, v in self.X.items()}
        env.Y = {k: v.to(device=device, dtype=dtype, **kwargs) for k, v in self.Y.items()}
        env._si_age = self._si_age.copy()
        return env

    def clone(self) -> EnvCTM:
        r"""
        Return a clone of the environment preserving the autograd - resulting clone is a part
        of the computational graph. Data of cloned environment tensors is indepedent
        from the originals.
        """
        cls = type(self)
        env = cls(self.psi.clone(), init=None)
        env.env = self.env.clone()
        env.proj = self.proj.clone()
        env.X = {k: v.clone() for k, v in self.X.items()}
        env.Y = {k: v.clone() for k, v in self.Y.items()}
        env._si_age = self._si_age.copy()
        return env

    def detach(self) -> EnvCTM:
        r"""
        Return a detached view of the environment - resulting environment is **not** a part
        of the computational graph. Data of detached environment tensors is shared
        with the originals.
        """
        cls = type(self)
        env = cls(self.psi, init=None)
        env.env = self.env.detach()
        env.proj = self.proj.detach()
        env.X = {k: v.detach() for k, v in self.X.items()}
        env.Y = {k: v.detach() for k, v in self.Y.items()}
        env._si_age = self._si_age.copy()
        return env

    def detach_(self):
        r"""
        Detach all environment tensors from the computational graph.
        Data of environment tensors in detached environment is a `view` of the original data.
        """
        self.env.detach_()
        self.proj.detach_()
        self.X = {k: v.detach() for k, v in self.X.items()}
        self.Y = {k: v.detach() for k, v in self.Y.items()}

    def to_dict(self, level=2):
        r"""
        Serialize EnvCTM to a dictionary.
        Complementary function is :meth:`yastn.EnvCTM.from_dict` or a general :meth:`yastn.from_dict`.
        See :meth:`yastn.Tensor.to_dict` for further description.
        """
        return {'type': type(self).__name__,
                'dict_ver': 1,
                'psi': self.psi.to_dict(level=level),
                'env': self.env.to_dict(level=level),
                'proj': self.proj.to_dict(level=level),
                'si_X': [dict(site=tuple(site), pair=pair, tensor=t.to_dict(level=level))
                         for (site, pair), t in self.X.items()],
                'si_Y': [dict(site=tuple(site), pair=pair, tensor=t.to_dict(level=level))
                         for (site, pair), t in self.Y.items()],
                'si_age': [dict(site=tuple(site), pair=pair, age=age)
                           for (site, pair), age in self._si_age.items()]}

    @classmethod
    def from_dict(cls, d, config=None):
        r"""
        De-serializes EnvCTM from the dictionary ``d``.
        See :meth:`yastn.Tensor.from_dict` for further description.
        """
        if 'dict_ver' not in d:
            psi = PEPS_CLASSES["Peps"].from_dict(d['psi'], config)
            env = EnvCTM(psi, init=None)
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
            env.proj = Lattice.from_dict(d['proj'], config=config)
            env.X = {(Site(*x['site']), x['pair']): Tensor.from_dict(x['tensor'], config=config)
                     for x in d.get('si_X', ())}
            env.Y = {(Site(*x['site']), x['pair']): Tensor.from_dict(x['tensor'], config=config)
                     for x in d.get('si_Y', ())}
            env._si_age = {(Site(*x['site']), x['pair']): x['age']
                           for x in d.get('si_age', ())}
            return env

    def update_from_dict_(self, d):
        psi = PEPS_CLASSES[d['psi']['type']].from_dict(d['psi'])
        tmp = type(self)(psi, init=None)
        self.psi = tmp.psi
        self.env = Lattice.from_dict(d['env'])
        self.proj = Lattice.from_dict(d['proj'])
        self.X = {(Site(*x['site']), x['pair']): Tensor.from_dict(x['tensor'])
                  for x in d.get('si_X', ())}
        self.Y = {(Site(*x['site']), x['pair']): Tensor.from_dict(x['tensor'])
                  for x in d.get('si_Y', ())}
        self._si_age = {(Site(*x['site']), x['pair']): x['age']
                        for x in d.get('si_age', ())}

    def save_to_dict(self) -> dict:
        r"""
        Serialize EnvCTM into a dictionary.

        !!! This method is deprecated; use to_dict() instead !!!
        """
        warn('This method is deprecated; use to_dict() instead.', DeprecationWarning, stacklevel=2)

        psi = self.psi
        if isinstance(psi, Peps2Layers):
            psi = psi.ket

        d = {'class': type(self).__name__,
             'psi': psi.save_to_dict(),
             'data': {}}
        for site in self.sites():
            d_local = {dirn: getattr(self[site], dirn).save_to_dict()
                       for dirn in self[site].fields()}
            d['data'][site] = d_local
        return d

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
            Leg signature is fixed to the default values.
        """
        normalize = kwargs.get('normalize', 'inf')
        self.X.clear()
        self.Y.clear()
        self._si_age.clear()

        if init == 'dl':
            self.reset_(init='eye')
            self.expand_outward_()
            for site in self.sites():
                for dirn in self[site].fields():
                    T = getattr(self[site], dirn)
                    T = T / T.norm(p=normalize) if isinstance(normalize, str) else normalize(T)
                    setattr(self[site], dirn, T)
            return

        cs = self._default_corner_signature
        leg_one_0 = Leg(self.config, s=cs[0], t=(self.config.sym.zero(),), D=(1,))
        leg_one_1 = Leg(self.config, s=cs[1], t=(self.config.sym.zero(),), D=(1,))

        leg_0 = leg_one_0 if leg is None else (leg if leg.s == cs[0] else leg.conj())
        leg_1 = leg_one_1 if leg is None else (leg if leg.s == cs[1] else leg.conj())

        li = {'t': 0, 'l': 1, 'b': 2, 'r': 3}

        for site in self.sites():
            legs = self.psi[site].get_legs()

            for dirn in self[site].fields(among=['tl', 'tr', 'bl', 'br']):
                shifted_site = self.nn_site(site, d=dirn)
                if init == 'eye' or shifted_site is None:
                    T = eye(self.config, legs=[leg_one_0, leg_one_1], isdiag=False)
                elif init == 'rand':
                    T = rand(self.config, legs=[leg_0, leg_1])
                T = T / T.norm(p=normalize) if isinstance(normalize, str) else normalize(T)
                setattr(self[site], dirn, T)

            for dirn in self[site].fields(among=['t', 'l', 'b', 'r']):
                shifted_site = self.nn_site(site, d=dirn)
                if init == 'eye' or shifted_site is None:
                    tmp1 = identity_boundary(self.config, legs[li[dirn]].conj())
                    tmp0 = eye(self.config, legs=[leg_one_1.conj(), leg_one_0.conj()], isdiag=False)
                    T = tensordot(tmp0, tmp1, axes=((), ())).transpose(axes=(0, 2, 1))
                elif init == 'rand':
                    T = rand(self.config, legs=[leg_1.conj(), legs[li[dirn]].conj(), leg_0.conj()])
                T = T / T.norm(p=normalize) if isinstance(normalize, str) else normalize(T)
                setattr(self[site], dirn, T)


    def expand_outward_(self):
        """
        Enlarges the environment by one layer of PEPS tensors. No truncation is performed.

        It can be used to build initial "dl" (or "nn") environment from initial "eye" environment.
        """
        cls = type(self)
        env_tmp = cls(self.psi, init=None)  # empty environments
        #
        for site in self.sites():
            tl = self.nn_site(site, d='tl')
            if 'tl' in self[site].fields() and tl is not None:
                tmp = self[tl].l @ self[tl].tl @ self[tl].t
                tmp = tensordot(tmp, self.psi[tl], axes=((2, 1), (0, 1)))
                tmp = tmp.fuse_legs(axes=((0, 2), (1, 3)))
                env_tmp[site].tl = tmp / tmp.norm(p='inf')
            #
            bl = self.nn_site(site, d='bl')
            if 'bl' in self[site].fields() and bl is not None:
                tmp = self[bl].b @ self[bl].bl @ self[bl].l
                tmp = tensordot(tmp, self.psi[bl], axes=((2, 1), (1, 2)))
                tmp = tmp.fuse_legs(axes=((0, 3), (1, 2)))
                env_tmp[site].bl = tmp / tmp.norm(p='inf')
            #
            tr = self.nn_site(site, d='tr')
            if 'tr' in self[site].fields() and tr is not None:
                tmp = self[tr].t @ self[tr].tr @ self[tr].r
                tmp = tensordot(tmp, self.psi[tr], axes=((1, 2), (0, 3)))
                tmp = tmp.fuse_legs(axes=((0, 2), (1, 3)))
                env_tmp[site].tr = tmp / tmp.norm(p='inf')
            #
            br = self.nn_site(site, d='br')
            if 'br' in self[site].fields() and br is not None:
                tmp = self[br].r @ self[br].br @ self[br].b
                tmp = tensordot(tmp, self.psi[br], axes=((2, 1), (2, 3)))
                tmp = tmp.fuse_legs(axes=((0, 2), (1, 3)))
                env_tmp[site].br = tmp / tmp.norm(p='inf')

            l = self.nn_site(site, d='l')
            if 'l' in self[site].fields() and l is not None:
                l0, _, l2 = self[l].l.get_legs()
                l_t, _, l_b, _ = self.psi[l].get_legs()

                tmp0 = eye(self.config, legs=[l2.conj(), l2], isdiag=False)
                tmp1 = eye(self.config, legs=[l_t.conj(), l_t], isdiag=False)
                proj_hlt = ncon([tmp0, tmp1], ((-0, -2), (-1, -3)))
                proj_hlt = proj_hlt.fuse_legs(axes=(0, 1, (2, 3)))
                if tl is None:
                    proj_hlt = proj_hlt.remove_leg(axis=2)
                    proj_hlt = proj_hlt.add_leg(axis=2, leg=self[site].tl.get_legs(axes=0).conj())

                tmp0 = eye(self.config, legs=[l0.conj(), l0], isdiag=False)
                tmp1 = eye(self.config, legs=[l_b.conj(), l_b], isdiag=False)
                proj_hlb = ncon([tmp0, tmp1], ((-0, -2), (-1, -3)))
                proj_hlb = proj_hlb.fuse_legs(axes=(0, 1, (2, 3)))
                if bl is None:
                    proj_hlb = proj_hlb.remove_leg(axis=2)
                    proj_hlb = proj_hlb.add_leg(axis=2, leg=self[site].bl.get_legs(axes=1).conj())

                tmp = self[l].l @ proj_hlt
                tmp = tensordot(self.psi[l], tmp, axes=((0, 1), (2, 1)))
                tmp = tensordot(proj_hlb, tmp, axes=((0, 1), (2, 0)))
                env_tmp[site].l = tmp / tmp.norm(p='inf')

            #
            r = self.nn_site(site, d='r')
            if 'r' in self[site].fields() and r is not None:
                l0, _, l2 = self[r].r.get_legs()
                l_t, _, l_b, _ = self.psi[r].get_legs()

                tmp0 = eye(self.config, legs=[l2.conj(), l2], isdiag=False)
                tmp1 = eye(self.config, legs=[l_b.conj(), l_b], isdiag=False)
                proj_hrb = ncon([tmp0, tmp1], ((-0, -2), (-1, -3)))
                proj_hrb = proj_hrb.fuse_legs(axes=(0, 1, (2, 3)))
                if br is None:
                    proj_hrb = proj_hrb.remove_leg(axis=2)
                    proj_hrb = proj_hrb.add_leg(axis=2, leg=self[site].br.get_legs(axes=0).conj())

                tmp0 = eye(self.config, legs=[l0.conj(), l0], isdiag=False)
                tmp1 = eye(self.config, legs=[l_t.conj(), l_t], isdiag=False)
                proj_hrt = ncon([tmp0, tmp1], ((-0, -2), (-1, -3)))
                proj_hrt = proj_hrt.fuse_legs(axes=(0, 1, (2, 3)))
                if tr is None:
                    proj_hrt = proj_hrt.remove_leg(axis=2)
                    proj_hrt = proj_hrt.add_leg(axis=2, leg=self[site].tr.get_legs(axes=1).conj())

                tmp = self[r].r @ proj_hrb
                tmp = tensordot(self.psi[r], tmp, axes=((2, 3), (2, 1)))
                tmp = tensordot(proj_hrt, tmp, axes=((0, 1), (2, 0)))
                env_tmp[site].r = tmp / tmp.norm(p='inf')
            #
            t = self.nn_site(site, d='t')
            if 't' in self[site].fields() and t is not None:
                l0, _, l2 = self[t].t.get_legs()
                _, l_l, _, l_r = self.psi[t].get_legs()

                tmp0 = eye(self.config, legs=[l2.conj(), l2], isdiag=False)
                tmp1 = eye(self.config, legs=[l_r.conj(), l_r], isdiag=False)
                proj_vtr = ncon([tmp0, tmp1], ((-0, -2), (-1, -3)))
                proj_vtr = proj_vtr.fuse_legs(axes=(0, 1, (2, 3)))
                if tr is None:
                    proj_vtr = proj_vtr.remove_leg(axis=2)
                    proj_vtr = proj_vtr.add_leg(axis=2, leg=self[site].tr.get_legs(axes=0).conj())

                tmp0 = eye(self.config, legs=[l0.conj(), l0], isdiag=False)
                tmp1 = eye(self.config, legs=[l_l.conj(), l_l], isdiag=False)
                proj_vtl = ncon([tmp0, tmp1], ((-0, -2), (-1, -3)))
                proj_vtl = proj_vtl.fuse_legs(axes=(0, 1, (2, 3)))
                if tl is None:
                    proj_vtl = proj_vtl.remove_leg(axis=2)
                    proj_vtl = proj_vtl.add_leg(axis=2, leg=self[site].tl.get_legs(axes=1).conj())

                tmp = tensordot(proj_vtl, self[t].t, axes=(0, 0))
                tmp = tensordot(tmp, self.psi[t], axes=((2, 0), (0, 1)))
                tmp = tensordot(tmp, proj_vtr, axes=((1, 3), (0, 1)))
                env_tmp[site].t = tmp / tmp.norm(p='inf')

            #
            b = self.nn_site(site, d='b')
            if 'b' in self[site].fields() and b is not None:
                l0, _, l2 = self[b].b.get_legs()
                _, l_l, _, l_r = self.psi[b].get_legs()

                tmp0 = eye(self.config, legs=[l0.conj(), l0], isdiag=False)
                tmp1 = eye(self.config, legs=[l_r.conj(), l_r], isdiag=False)
                proj_vbr = ncon([tmp0, tmp1], ((-0, -2), (-1, -3)))
                proj_vbr = proj_vbr.fuse_legs(axes=(0, 1, (2, 3)))
                if br is None:
                    proj_vbr = proj_vbr.remove_leg(axis=2)
                    proj_vbr = proj_vbr.add_leg(axis=2, leg=self[site].br.get_legs(axes=1).conj())

                tmp0 = eye(self.config, legs=[l2.conj(), l2], isdiag=False)
                tmp1 = eye(self.config, legs=[l_l.conj(), l_l], isdiag=False)
                proj_vbl = ncon([tmp0, tmp1], ((-0, -2), (-1, -3)))
                proj_vbl = proj_vbl.fuse_legs(axes=(0, 1, (2, 3)))
                if bl is None:
                    proj_vbl = proj_vbl.remove_leg(axis=2)
                    proj_vbl = proj_vbl.add_leg(axis=2, leg=self[site].bl.get_legs(axes=0).conj())

                tmp = tensordot(proj_vbr, self[b].b, axes=(0, 0))
                tmp = tensordot(tmp, self.psi[b], axes=((2, 0), (2, 3)))
                tmp = tensordot(tmp, proj_vbl, axes=((1, 3), (0, 1)))
                env_tmp[site].b = tmp / tmp.norm(p='inf')
        #
        # modify existing environment in place
        update_storage_(self, env_tmp)

    def boundary_mps(self, n, dirn) -> mps.MpsMpoOBC:
        r""" Convert environmental tensors of Ctm to an MPS. """
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

    def calculate_corner_svd(env):
        """
        Return normalized SVD spectra, with largest singular value set to unity, of all corner tensors of environment.
        The corners are indexed by pair of Site and corner identifier.
        """
        _get_spec = lambda x: x.svd(compute_uv=False) if not (x is None) and not x.isdiag else x
        corner_sv = {}
        for site in env.sites():
            for dirn in env[site].fields(among=['tl', 'tr', 'bl', 'br']):
                corner_sv[site, dirn] = _get_spec(getattr(env[site], dirn))
        for k, v in corner_sv.items():
            if corner_sv[k] is not None:
                corner_sv[k] = v / v.norm(p='inf')
        return corner_sv

    def _partial_svd_predict_spec(self, leg0, leg1, sU):
        # TODO externalize defaults for extending number of singular values to solve for
        """
        Used in block-wise partial SVD solvers.

        Based on the projector spectra leg0, leg1, from (previous) projector pair,
        suggest number of singular value triples to solve for in each of the blocks.

        Parameters
        ----------
        leg0, leg1: yastn.Tensor
            Projector spectra for the previous projector pair.
        sU: int
            Signature of U in SVD decomposition. See :func:`proj_corners` and :func:`linalg.svd`.
        """
        # the projector spectra for projector pair are related by charge conjugation
        assert leg0 == leg1.conj(), f"Projector spectrum history mismatch between leg0={leg0} and leg1={leg1}"
        #
        l = leg0 if sU == leg0.s else leg1
        return {t: max(d + 10, int(d * 1.1)) for t, d in zip(l.t, l.D)}

    def update_(env, opts_svd, moves='hv', method='2x2 corner', **kwargs):
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

        moves: str
            Specify a sequence of moves forming a single sweep.
            Individual moves are 'l', 'r', 't', 'b', 'h', or 'v'.
            Horizontal 'h' and vertical 'v' moves have all sites updated simultaneously.
            Left 'l', right 'r', top 't', and bottom 'b' are executed causally,
            row after row or column after column.
            Argument specifies a sequence of individual moves, where sensible options are 'hv' and 'lrtb'.
            The default is 'hv'.

        method: str
            '2x2' or '1x2' in method. The default is '2x2 corner'.
            '2x2' uses the standard 2x2 enlarged corners forming 4x4 patch, allowing to enlarge EnvCTM bond dimension.
            '1x2' uses smaller 1x2 corners forming 2x4 patch. It is significantly faster, but is less stable and
            does not allow to grow EnvCTM bond dimension.

        checkpoint_move: bool
            Whether to use (reentrant) checkpointing for the move. The default is ``False``

        opts_si: dict | None
            Enable recycled subspace-iteration projectors with ``{'enabled': True}``.
            Supported options are ``oversampling`` (default 5), ``niter``
            (default 1), ``tol`` (default 1e-3), ``warmup`` (default 5 projector updates),
            ``correction_frequency`` (default 0, disabled), ``correct`` to
            force an immediate sector redistribution, ``refinement``
            (``'cwo'`` by default, or ``'asvr'``/``'rds'``), and
            ``recycle_grad`` (default False). ``'rds'`` distributes SI vectors
            proportionally to the charge-sector dimensions of the corners.

        Returns
        -------
        proj: Peps structure loaded with CTM projectors related to all lattice site.
        """
        if 'tol' not in opts_svd and 'tol_block' not in opts_svd:
            opts_svd['tol'] = 1e-14

        checkpoint_move = kwargs.get('checkpoint_move', False)
        for d in moves:
            if checkpoint_move:
                def f_update_core_(move_d, loc_im, *inputs_t):
                    loc_env = type(env).from_dict(combine_data_and_meta(inputs_t, loc_im))
                    loc_env._update_core_(move_d, opts_svd, method=method, **kwargs)
                    out_data, out_meta = split_data_and_meta(loc_env.to_dict(level=0))
                    return out_meta, *out_data

                if "torch" in env.config.backend.BACKEND_ID:
                    inputs_t, inputs_meta = split_data_and_meta(env.to_dict(level=0))

                    if checkpoint_move == 'reentrant':
                        use_reentrant = True
                    elif checkpoint_move == 'nonreentrant':
                        use_reentrant = False
                    checkpoint_F = env.config.backend.checkpoint
                    out_meta, *out_data = checkpoint_F(f_update_core_, d, inputs_meta, *inputs_t, \
                                      **{'use_reentrant': use_reentrant, 'debug': False})
                else:
                    raise RuntimeError(f"CTM update: checkpointing not supported for backend {env.config.BACKEND_ID}")

                # reconstruct env from output tensors
                env.update_from_dict_(combine_data_and_meta(out_data, out_meta))
            else:
                env._update_core_(d, opts_svd, method=method, **kwargs)
        return env

    def _update_core_(env, move: str, opts_svd: dict, method: str, **kwargs):
        r"""
        Core function updating CTM environment tensors pefrorming specified move.
        """
        assert move in ['h', 'v', 'l', 'r', 't', 'b'], "Invalid move"
        if (move in 'hv') or (len(env.sites()) < env.Nx * env.Ny):
            # For horizontal and vertical moves,
            # and unit cell with a nontrivial pattern like CheckerboardLattice or RectangularUnitcell,
            # all sites are updated simultaneously.
            shift_proj = None
            sitess = [env.sites()]
        elif move == 'l':  # Move done sequentially, column after column.
            shift_proj = 'l'
            sitess = [[Site(nx, ny) for nx in range(env.Nx)] for ny in range(env.Ny)]
        elif move == 'r':  # Move done sequentially, column after column.
            shift_proj = None
            sitess = [[Site(nx, ny) for nx in range(env.Nx)] for ny in range(env.Ny-1, -1, -1)]
        elif move == 't':  # Move done sequentially, row after row.
            shift_proj = 't'
            sitess = [[Site(nx, ny) for ny in range(env.Ny)] for nx in range(env.Nx)]
        elif move == 'b':  # Move done sequentially, row after row.
            shift_proj = None
            sitess = [[Site(nx, ny) for ny in range(env.Ny)] for nx in range(env.Nx-1, -1, -1)]

        for sites in sitess:
            sites_proj = [env.nn_site(site, shift_proj) for site in sites] if shift_proj else sites
            sites_proj = [site for site in sites_proj if site is not None]
            #
            # Projectors
            for site in sites_proj:
                env._update_projectors_(site, move, opts_svd, method, **kwargs)
            # fill (trivial) projectors on edges
            env._trivial_projectors_(move, sites_proj)
            #
            # Update move
            env_tmp = EnvCTM(env.psi, init=None)  # empty environments
            for site in sites:
                env_tmp._update_env_(site, env, move)
            update_storage_(env, env_tmp)

    def update_bond_(env, bond: tuple, opts_svd: dict | None = None, method: str = '2x2 corner', **kwargs):
        r"""
        Update EnvCTM tensors related to a specific nearest-neighbor bond.

        Intended primarily for FU evolution scheme -- assuming fixed sectorial bond dimensions.
        May require using a dictionary "D_block" specifying sectorial bond dimensions in
        opts_svd's passed to PEPS truncation and CTM.
        """
        if opts_svd is None:
            opts_svd = env.opts_svd

        dirn = env.nn_bond_dirn(*bond)
        s0, s1 = bond if dirn in ['lr', 'tb'] else bond[::-1]

        if dirn in 'lrl':
            env._update_env_(s0, env, move='r')
            env._update_env_(s1, env, move='l')
            env._update_projectors_(s0, 't', opts_svd, method, **kwargs)
            env._update_projectors_(env.nn_site(s0, d='t'), 'b', opts_svd, method, **kwargs)
        else:  # 'tbt'
            env._update_env_(s0, env, move='b')
            env._update_env_(s1, env, move='t')
            env._update_projectors_(s0, 'l', opts_svd, method, **kwargs)
            env._update_projectors_(env.nn_site(s0, d='l'), 'r', opts_svd, method, **kwargs)


    def _update_projectors_(env, site, move, opts_svd, method, **kwargs):
        r"""
        Calculate new projectors for CTM moves passing to specific method to create enlarged corners.
        """
        sites = [env.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))]
        # tl, tr, bl, br = sites
        if None in sites:
            return

        if '1x2' in method or '2x1' in method or method == '1site':
            return update_1x2_projectors_(env, *sites, move, opts_svd, **kwargs)
        elif '2x2' in method or method == '2site':
            return update_extended_2x2_projectors_(env, *sites, move, opts_svd, **kwargs)
        else:
            raise YastnError(f"CTM update {method=} not recognized. Should contain '1x2' or '2x2'")

    def _set_projector_pair_(env, anchor, pair, site0, name0, site1, name1,
                             r0, r1, opts_svd, **kwargs):
        """Update a projector pair and its recycled SI bases in place."""
        key = (anchor, pair)
        opts_si = kwargs.pop('opts_si', None)
        if opts_si is not None and opts_si.get('enabled', False):
            opts_si = dict(opts_si)
            age = env._si_age.get(key, 0)
            warmup = opts_si.get('warmup', 5)
            frequency = opts_si.get('correction_frequency', 0)
            opts_si['correct'] = (opts_si.get('correct', False)
                                  or age == warmup
                                  or (frequency > 0 and age > warmup
                                      and (age - warmup) % frequency == 0))
        p0, p1, X, Y = proj_corners(
            r0, r1, opts_svd=opts_svd, opts_si=opts_si,
            X=env.X.get(key), Y=env.Y.get(key), return_si_state=True,
            **kwargs)
        setattr(env.proj[site0], name0, p0)
        setattr(env.proj[site1], name1, p1)
        if X is not None:
            recycle_grad = opts_si.get('recycle_grad', False)
            env.X[key] = X if recycle_grad else X.detach()
            env.Y[key] = Y if recycle_grad else Y.detach()
            env._si_age[key] = env._si_age.get(key, 0) + 1

    def _trivial_projectors_(env, move, sites):
        r"""
        Adds trivial projectors if not present at the edges of the lattice with open boundary conditions.
        """
        if move == 'h':  move = 'lr'
        if move == 'v':  move = 'tb'
        for site in sites:
            for s0, s1, s2, s3, a0, a1, a2 in _for_trivial:
                if s2 in move and getattr(env.proj[site], s0) is None:
                    site_nn = env.nn_site(site, d=s1)
                    if site_nn is not None:
                        l0 = getattr(env[site], s2).get_legs(a0).conj()
                        l1 = env.psi[site].get_legs(a1).conj()
                        l2 = getattr(env[site_nn], s3).get_legs(a2).conj()
                        setattr(env.proj[site], s0, ones(env.config, legs=(l0, l1, l2)))

    def _update_env_(env_tmp, site, env, move: str):
        r"""
        Horizontal move of CTM step. Compute updated environment tensors given projectors for ``site``
        in left (``dir='l'``), right ``dir='r'``, or both directions (``dir='lr'``).
        Updated environment tensors are stored in ``env_tmp``.
        Vertical move of CTM step. Compute updated environment tensors given projectors for ``site``
        in top (``dir='t'``), bottom ``dir='b'``, or both directions (``dir='tb'``).
        Updated environment tensors are stored in ``env_tmp``.

        """
        psi = env.psi

        if move in 'lh':
            l = psi.nn_site(site, d='l')
            if l is not None:
                tmp = env[l].l @ env.proj[l].hlt
                tmp = tensordot(psi[l], tmp, axes=((0, 1), (2, 1)))
                tmp = tensordot(env.proj[l].hlb, tmp, axes=((0, 1), (2, 0)))
                env_tmp[site].l = tmp / tmp.norm(p='inf')

            tl = psi.nn_site(site, d='tl')
            if tl is not None:
                tmp = tensordot(env.proj[tl].hlb, env[l].tl @ env[l].t, axes=((0, 1), (0, 1)))
                env_tmp[site].tl = tmp / tmp.norm(p='inf')

            bl = psi.nn_site(site, d='bl')
            if bl is not None:
                tmp = tensordot(env[l].b, env[l].bl @ env.proj[bl].hlt, axes=((2, 1), (0, 1)))
                env_tmp[site].bl = tmp / tmp.norm(p='inf')

        if move in 'rh':
            r = psi.nn_site(site, d='r')
            if r is not None:
                tmp = env[r].r @ env.proj[r].hrb
                tmp = tensordot(psi[r], tmp, axes=((2, 3), (2, 1)))
                tmp = tensordot(env.proj[r].hrt, tmp, axes=((0, 1), (2, 0)))
                env_tmp[site].r = tmp / tmp.norm(p='inf')

            tr = psi.nn_site(site, d='tr')
            if tr is not None:
                tmp = tensordot(env[r].t, env[r].tr @ env.proj[tr].hrb, axes=((2, 1), (0, 1)))
                env_tmp[site].tr = tmp / tmp.norm(p='inf')

            br = psi.nn_site(site, d='br')
            if br is not None:
                tmp = tensordot(env.proj[br].hrt, env[r].br @ env[r].b, axes=((0, 1), (0, 1)))
                env_tmp[site].br = tmp / tmp.norm(p='inf')

        if move in 'tv':
            t = psi.nn_site(site, d='t')
            if t is not None:
                tmp = tensordot(env.proj[t].vtl, env[t].t, axes=(0, 0))
                tmp = tensordot(tmp, psi[t], axes=((2, 0), (0, 1)))
                tmp = tensordot(tmp, env.proj[t].vtr, axes=((1, 3), (0, 1)))
                env_tmp[site].t = tmp / tmp.norm(p='inf')

            tl = psi.nn_site(site, d='tl')
            if tl is not None:
                tmp = tensordot(env[t].l, env[t].tl @ env.proj[tl].vtr, axes=((2, 1), (0, 1)))
                env_tmp[site].tl = tmp / tmp.norm(p='inf')

            tr = psi.nn_site(site, d='tr')
            if tr is not None:
                tmp = tensordot(env.proj[tr].vtl, env[t].tr @ env[t].r, axes=((0, 1), (0, 1)))
                env_tmp[site].tr =  tmp / tmp.norm(p='inf')

        if move in 'bv':
            b = psi.nn_site(site, d='b')
            if b is not None:
                tmp = tensordot(env.proj[b].vbr, env[b].b, axes=(0, 0))
                tmp = tensordot(tmp, psi[b], axes=((2, 0), (2, 3)))
                tmp = tensordot(tmp, env.proj[b].vbl, axes=((1, 3), (0, 1)))
                env_tmp[site].b = tmp / tmp.norm(p='inf')

            bl = psi.nn_site(site, d='bl')
            if bl is not None:
                tmp = tensordot(env.proj[bl].vbr, env[b].bl @ env[b].l, axes=((0, 1), (0, 1)))
                env_tmp[site].bl = tmp / tmp.norm(p='inf')

            br = psi.nn_site(site, d='br')
            if br is not None:
                tmp = tensordot(env[b].r, env[b].br @ env.proj[br].vbl, axes=((2, 1), (0, 1)))
                env_tmp[site].br = tmp / tmp.norm(p='inf')

    def apply_patch(self):
        self.env.apply_patch()
        self.proj.apply_patch()

    def move_to_patch(self, sites):
        self.env.move_to_patch(sites)
        self.proj.move_to_patch(sites)

    def pre_truncation_(env, bond):
        pass
        #env.update_bond_(bond, opts_svd=env.opts_svd)

    def post_truncation_(env, bond, **kwargs):
        env.update_bond_(bond, **kwargs)

    def bond_metric(self, Q0, Q1, s0, s1, dirn) -> Tensor:
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
        if dirn in ("h", "lr"):
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
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))

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

    def iterate_(env, opts_svd=None, moves='hv', method='2x2 corner', max_sweeps=1, iterator=False, corner_tol=None, truncation_f: Callable = None, **kwargs):
        r"""
        Perform CTMRG updates :meth:`yastn.tn.fpeps.EnvCTM.update_` until convergence.
        Convergence can be measured based on singular values of CTM environment corner tensors.

        Outputs iterator if ``iterator`` is given, which allows
        inspecting ``env``, e.g., calculating expectation values,
        outside of ``ctmrg_`` function after every sweeps.

        Parameters
        ----------
        opts_svd: dict
            A dictionary of options to pass to SVD truncation algorithm.
            This sets EnvCTM bond dimension.

        moves: str
            Specify a sequence of moves forming a single sweep.
            Individual moves are 'l', 'r', 't', 'b', 'h', or 'v'.
            Horizontal 'h' and vertical 'v' moves have all sites updated simultaneously.
            Left 'l', right 'r', top 't', and bottom 'b' are executed causally,
            row after row or column after column.
            Argument specifies a sequence of individual moves, where sensible options are 'hv' and 'lrtb'.
            The default is 'hv'.

        method: str
            '2x2' or '1x2' contained in method. The default is '2x2'.

                * '2x2' uses the standard 2x2 enlarged corners (forming 4x4 patch), enabling enlargement of EnvCTM bond dimensions. When some PEPS bonds are rank-1, it recognizes it to use 3x2 corners to prevent artificial collapse of EnvCTM bond dimensions to 1, which is important for hexagonal lattice.
                * '1x2' uses smaller 1x2 corners (forming 2x4 patch). It is significantly faster, but is less stable and  does not allow for EnvCTM bond dimension growth.

        max_sweeps: int
            The maximal number of sweeps.

        iterator: bool
            If True, ``ctmrg_`` returns a generator that would yield output after every sweep.
            The default is False, in which case  ``ctmrg_`` sweeps are performed immediately.

        corner_tol: float
            Convergence tolerance for the change of singular values of all corners in a single update.
            The default is ``None``, in which case convergence is not checked and it is up to user to implement
            convergence check.

        truncation_f:
            Custom projector truncation function with signature ``truncation_f(S: Tensor)->Tensor``, consuming
            rank-1 tensor with singular values. If provided, truncation parameters passed to SVD decomposition
            are ignored.

        checkpoint_move: str | bool
            Whether to use checkpointing for the CTM updates. The default is ``False``.
            Otherwise, in case of PyTorch backend it can be set to 'reentrant' for reentrant checkpointing
            or 'nonreentrant' for non-reentrant checkpointing, see https://pytorch.org/docs/stable/checkpoint.html.

        use_qr: bool
            Whether to include intermediate QR decomposition while calculating projectors.
            The default is ``True``.

        Returns
        -------
        Generator if iterator is True.

        CTMRG_out(NamedTuple)
            NamedTuple including fields:

                * ``sweeps`` number of performed ctmrg updates.
                * ``max_dsv`` norm of singular values change in the worst corner in the last sweep.
                * ``max_D`` largest bond dimension of environment tensors virtual legs.
                * ``converged`` whether convergence based on ``corner_tol`` has been reached.
        """
        kwargs["iterator_step"] = kwargs.get("iterator_step", int(iterator))
        if ("checkpoint_move" in kwargs) and ("torch" in env.config.backend.BACKEND_ID):
            assert kwargs["checkpoint_move"] in ['reentrant', 'nonreentrant', False], f"Invalid choice for {kwargs['checkpoint_move']}"
        kwargs["truncation_f"] = truncation_f
        kwargs["iterator_step"] = kwargs.get("iterator_step", int(iterator))
        tmp = env._ctmrg_iterator_(opts_svd=opts_svd, moves=moves, method=method, max_sweeps=max_sweeps, corner_tol=corner_tol, **kwargs)
        return tmp if kwargs["iterator_step"] else next(tmp)

    ctmrg_ = iterate_   #  For backward compatibility, allow using EnvCtm.ctmrg_() instead of EnvCtm.iterate_().

    def _ctmrg_iterator_(env, opts_svd, moves, method, max_sweeps, corner_tol, **kwargs):
        """ Generator for ctmrg_. """
        iterator_step = kwargs.get("iterator_step", 0)
        max_dsv, converged, history = None, False, []
        for sweep in range(1, max_sweeps + 1):
            if env.profiling_mode in ["NVTX",]:
                env.config.backend.cuda.nvtx.range_push(f"update_")
                env.update_(opts_svd=opts_svd, moves=moves, method=method, **kwargs)
                env.config.backend.cuda.nvtx.range_pop()
            else:
                env.update_(opts_svd=opts_svd, moves=moves, method=method, **kwargs)

            # use default CTM convergence check
            if corner_tol is not None:
                converged, max_dsv, history = env.ctm_conv_corner_spec(history, corner_tol)
                logging.info(f'Sweep = {sweep:03d}; max_diff_corner_singular_values = {max_dsv:0.2e}')
                if converged:
                    break

            if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
                yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, max_D=env.max_D(), converged=converged)
        yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, max_D=env.max_D(), converged=converged)

    def ctm_conv_corner_spec(env: EnvCTM,
                             history: Sequence[dict[tuple[Site, str], Tensor]]=[],
                             corner_tol: None | float=1.0e-8) -> tuple[bool, float, Sequence[dict[tuple[Site, str], Tensor]]]:
        """
        Evaluate convergence of CTM by computing the difference of environment corner spectra between consecutive CTM steps.
        """
        corner_sv = env.calculate_corner_svd()
        max_dsv = max(spec_diff(history[-1][k], corner_sv[k]) for k in corner_sv) if history else float('Nan')
        corner_sv['max_dsv'] = max_dsv
        history.append(corner_sv)
        converged = (corner_tol is not None) and (max_dsv < corner_tol)
        return converged, max_dsv, history

    def is_consistent(env, verbosity = 2):
        out = {}
        env_legs = {}
        sites = set(s0 for s0, _ in env.bonds()) | set(s1 for _, s1 in env.bonds())
        for site in sites:
            env_legs[site, 'psi'] = env.psi[site].get_legs()
            for dirn in ['tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r']:
                ten = getattr(env[site], dirn)
                if ten is not None:
                    env_legs[site, dirn] = ten.get_legs()
            legs_consistent_(out, env_legs, (site, 'psi'), 0, (site, 't'), 1)
            legs_consistent_(out, env_legs, (site, 'psi'), 1, (site, 'l'), 1)
            legs_consistent_(out, env_legs, (site, 'psi'), 2, (site, 'b'), 1)
            legs_consistent_(out, env_legs, (site, 'psi'), 3, (site, 'r'), 1)
            legs_consistent_(out, env_legs, (site, 'tl'), 1, (site, 't'), 0)
            legs_consistent_(out, env_legs, (site, 't'), 2, (site, 'tr'), 0)
            legs_consistent_(out, env_legs, (site, 'tr'), 1, (site, 'r'), 0)
            legs_consistent_(out, env_legs, (site, 'r'), 2, (site, 'br'), 0)
            legs_consistent_(out, env_legs, (site, 'br'), 1, (site, 'b'), 0)
            legs_consistent_(out, env_legs, (site, 'b'), 2, (site, 'bl'), 0)
            legs_consistent_(out, env_legs, (site, 'bl'), 1, (site, 'l'), 0)
            legs_consistent_(out, env_legs, (site, 'l'), 2, (site, 'tl'), 0)

        for bond in env.bonds():
            dirn = env.nn_bond_dirn(*bond)
            s0, s1 = bond if dirn == 'lr' or 'tb' else bond[::-1]
            if 'l' in dirn:
                legs_consistent_(out, env_legs, (s0, 't'), 2, (s1, 't'), 0)
                legs_consistent_(out, env_legs, (s0, 'b'), 0, (s1, 'b'), 2)
                legs_consistent_(out, env_legs, (s0, 'tr'), 0, (s1, 'tl'), 1)
                legs_consistent_(out, env_legs, (s0, 'br'), 1, (s1, 'bl'), 0)
            if 't' in dirn:
                legs_consistent_(out, env_legs, (s0, 'l'), 0, (s1, 'l'), 2)
                legs_consistent_(out, env_legs, (s0, 'r'), 2, (s1, 'r'), 0)
                legs_consistent_(out, env_legs, (s0, 'bl'), 1, (s1, 'tl'), 0)
                legs_consistent_(out, env_legs, (s0, 'br'), 0, (s1, 'tr'), 1)

        not_consistent = [k for k, v in out.items() if not v]
        if verbosity > 0:
            if not_consistent:
                print("Unconsistent environment bonds: ")
                for x in not_consistent:
                    print(x)
        return len(not_consistent) == 0

    from ._env_ctm_measure import measure_1site, measure_nn, measure_2x2, measure_line, \
        measure_nsite, measure_2site, measure_nsite_exact, sample, transfer_matrix_spectrum


def legs_consistent_(out, env_legs, i0, l0, i1, l1):
    if i0 is None or i1 is None:
        return
    out[i0, l0, i1, l1] = env_legs[i0][l0].are_consistent(env_legs[i1][l1])


def spec_diff(x, y):
    if x is not None and y is not None:
        return (x - y).norm().item()
    elif x is None and y is None:
        return 0
    else:
        return float('Inf')


_for_trivial = (('hlt', 'r', 'l', 'tl', 2, 0, 0),
                ('hlb', 'r', 'l', 'bl', 0, 2, 1),
                ('hrt', 'l', 'r', 'tr', 0, 0, 1),
                ('hrb', 'l', 'r', 'br', 2, 2, 0),
                ('vtl', 'b', 't', 'tl', 0, 1, 1),
                ('vtr', 'b', 't', 'tr', 2, 3, 0),
                ('vbl', 't', 'b', 'bl', 2, 1, 0),
                ('vbr', 't', 'b', 'br', 0, 3, 1))


def update_extended_2x2_projectors_(env, tl, tr, bl, br, move, opts_svd, **kwargs):
    r"""
    Calculate new projectors for CTM moves from 4x4 extended corners
    which are enlarged to 5x4 if some virtual bond is one.
    Intended for a hexagonal lattice embedded on a square lattice.
    """
    psi = env.psi
    use_qr = kwargs.get("use_qr", True)
    kwargs["profiling_mode"]= env.profiling_mode
    psh = env.proj
    svd_predict_spec= lambda s0,p0,s1,p1,sign: opts_svd.get('D_block', float('inf')) \
        if psh is None or (getattr(psh[s0],p0) is None or getattr(psh[s1],p1) is None) else \
        env._partial_svd_predict_spec(getattr(psh[s0],p0).get_legs(-1), getattr(psh[s1],p1).get_legs(-1), sign)

    cor_tl = corner2x2('tl', env[tl].l, env[tl].tl, env[tl].t, psi[tl])
    cor_bl = corner2x2('bl', env[bl].b, env[bl].bl, env[bl].l, psi[bl])
    cor_tr = corner2x2('tr', env[tr].t, env[tr].tr, env[tr].r, psi[tr])
    cor_br = corner2x2('br', env[br].r, env[br].br, env[br].b, psi[br])

    if any(x in move for x in 'lrh'):
        cor_tt = cor_tl @ cor_tr  # b(left) b(right)
        cor_bb = cor_br @ cor_bl  # t(right) t(left)

    if any(x in move for x in 'rh'):
        sl = psi[tl].get_shape(axes=2)
        ltl = env.nn_site(tl, d='l')
        lbl = env.nn_site(bl, d='l')
        if sl == 1 and ltl and lbl:
            cor_ltl = env[ltl].l @ env[ltl].tl @ env[ltl].t
            cor_ltl = tensordot(cor_ltl, psi[ltl], axes=((2, 1), (0, 1)))
            cor_ltl = tensordot(cor_ltl, env[tl].t, axes=(1, 0))
            cor_ltl = tensordot(cor_ltl, psi[tl], axes=((3, 2), (0, 1)))
            cor_ltl = cor_ltl.fuse_legs(axes=((0, 1, 3), (2, 4)))

            cor_lbl = env[lbl].b @ env[lbl].bl @ env[lbl].l
            cor_lbl = tensordot(cor_lbl, psi[lbl], axes=((2, 1), (1, 2)))
            cor_lbl = env[bl].b @ cor_lbl
            cor_lbl = tensordot(cor_lbl, psi[bl], axes=((4, 1), (1, 2)))
            cor_lbl = cor_lbl.fuse_legs(axes=((0, 4), (1, 2, 3)))

            h1 = cor_ltl @ cor_tr  # b(left) b(right)
            h2 = cor_br @ cor_lbl  # t(right) t(left)
        else:
            h1,h2= cor_tt, cor_bb

        _, r_t = qr(h1, axes=(0, 1)) if use_qr else (None, h1)
        _, r_b = qr(h2, axes=(1, 0)) if use_qr else (None, h2.T)
        opts_svd["k_block"]= svd_predict_spec(tr, "hrb", br, "hrt", r_t.s[1])
        env._set_projector_pair_(tr, 'hrb', tr, 'hrb', br, 'hrt',
                                 r_t, r_b, opts_svd, **kwargs)

    if any(x in move for x in 'lh'):
        sr = psi[tr].get_shape(axes=2)
        rtr = env.nn_site(tr, d='r')
        rbr = env.nn_site(br, d='r')
        if sr == 1 and rtr and rbr:
            cor_rtr = env[rtr].t @ env[rtr].tr @ env[rtr].r
            cor_rtr = tensordot(cor_rtr, psi[rtr], axes=((1, 2), (0, 3)))
            cor_rtr = env[tr].t @ cor_rtr
            cor_rtr = tensordot(cor_rtr, psi[tr], axes=((1, 3), (0, 3)))
            cor_rtr = cor_rtr.fuse_legs(axes=((0, 3), (1, 2, 4)))

            cor_rbr = env[rbr].r @ env[rbr].br @ env[rbr].b
            cor_rbr = tensordot(cor_rbr, psi[rbr], axes=((2, 1), (2, 3)))
            cor_rbr = tensordot(cor_rbr, env[br].b, axes=(1, 0))
            cor_rbr = tensordot(cor_rbr, psi[br], axes=((3, 2), (2, 3)))
            cor_rbr = cor_rbr.fuse_legs(axes=((0, 1, 3), (2, 4)))

            h1 = cor_tl @ cor_rtr  # b(left) b(right)
            h2 = cor_rbr @ cor_bl  # t(right) t(left)
        else:
            h1,h2= cor_tt, cor_bb

        _, r_t = qr(h1, axes=(1, 0)) if use_qr else (None, h1.T)
        _, r_b = qr(h2, axes=(0, 1)) if use_qr else (None, h2)
        opts_svd["k_block"]= svd_predict_spec(tl, "hlb", bl, "hlt", r_t.s[1])
        env._set_projector_pair_(tl, 'hlb', tl, 'hlb', bl, 'hlt',
                                 r_t, r_b, opts_svd, **kwargs)

    if any(x in move for x in 'tbv'):
        cor_ll = cor_bl @ cor_tl  # l(bottom) l(top)
        cor_rr = cor_tr @ cor_br  # r(top) r(bottom)

    if any(x in move for x in 'tv'):
        sb = psi[bl].get_shape(axes=3)
        bbl = env.nn_site(bl, d='b')
        bbr = env.nn_site(br, d='b')
        if sb == 1 and bbl and bbr:
            cor_bbl = env[bbl].b @ env[bbl].bl @ env[bbl].l
            cor_bbl = tensordot(cor_bbl, psi[bbl], axes=((2, 1), (1, 2)))
            cor_bbl = tensordot(cor_bbl, env[bl].l, axes=(1, 0))
            cor_bbl = tensordot(cor_bbl, psi[bl], axes=((3, 1), (1, 2)))
            cor_bbl = cor_bbl.fuse_legs(axes=((0, 1, 4), (2, 3)))

            cor_bbr = env[bbr].r @ env[bbr].br @ env[bbr].b
            cor_bbr = tensordot(cor_bbr, psi[bbr], axes=((2, 1), (2, 3)))
            cor_bbr = env[br].r @ cor_bbr
            cor_bbr = tensordot(cor_bbr, psi[br], axes=((3, 1), (2, 3)))
            cor_bbr = cor_bbr.fuse_legs(axes=((0, 3), (1, 2, 4)))

            h1 = cor_bbl @ cor_tl  # l(bottom) l(top)
            h2 = cor_tr @ cor_bbr  # r(top) r(bottom)
        else:
            h1,h2= cor_ll, cor_rr

        _, r_l = qr(h1, axes=(0, 1)) if use_qr else (None, h1)
        _, r_r = qr(h2, axes=(1, 0)) if use_qr else (None, h2.T)
        opts_svd["k_block"]= svd_predict_spec(tl, "vtr", tr, "vtl", r_l.s[1])
        env._set_projector_pair_(tl, 'vtr', tl, 'vtr', tr, 'vtl',
                                 r_l, r_r, opts_svd, **kwargs)

    if any(x in move for x in 'bv'):
        st = psi[tl].get_shape(axes=3)
        ttl = env.nn_site(tl, d='t')
        ttr = env.nn_site(tr, d='t')
        if st == 1 and ttl and ttr:
            cor_ttl = env[ttl].l @ env[ttl].tl @ env[ttl].t
            cor_ttl = tensordot(cor_ttl, psi[ttl], axes=((2, 1), (0, 1)))
            cor_ttl = env[tl].l @ cor_ttl
            cor_ttl = tensordot(cor_ttl, psi[tl], axes=((3, 1), (0, 1)))
            cor_ttl = cor_ttl.fuse_legs(axes=((0, 3), (1, 2, 4)))

            cor_ttr = env[ttr].t @ env[ttr].tr @ env[ttr].r
            cor_ttr = tensordot(cor_ttr, psi[ttr], axes=((1, 2), (0, 3)))
            cor_ttr = tensordot(cor_ttr, env[tr].r, axes=(1, 0))
            cor_ttr = tensordot(cor_ttr, psi[tr], axes=((2, 3), (0, 3)))
            cor_ttr = cor_ttr.fuse_legs(axes=((0, 1, 3), (2, 4)))

            h1 = cor_bl @ cor_ttl  # l(bottom) l(top)
            h2 = cor_ttr @ cor_br  # r(top) r(bottom)
        else:
            h1,h2= cor_ll, cor_rr

        _, r_l = qr(h1, axes=(1, 0)) if use_qr else (None, h1.T)
        _, r_r = qr(h2, axes=(0, 1)) if use_qr else (None, h2)
        opts_svd["k_block"]= svd_predict_spec(bl, "vbr", br, "vbl", r_l.s[1])
        env._set_projector_pair_(bl, 'vbr', bl, 'vbr', br, 'vbl',
                                 r_l, r_r, opts_svd, **kwargs)


def update_1x2_projectors_(env, tl, tr, bl, br, move, opts_svd, **kwargs):
    r"""
    Calculate new projectors for CTM moves from 4x2 extended corners.
    """
    if move in 'lrh':
        cor_tl = (env[bl].tl @ env[bl].t).fuse_legs(axes=((0, 1), 2))
        cor_tr = (env[br].t @ env[br].tr).fuse_legs(axes=(0, (2, 1)))
        cor_br = (env[tr].br @ env[tr].b).fuse_legs(axes=((0, 1), 2))
        cor_bl = (env[tl].b @ env[tl].bl).fuse_legs(axes=(0, (2, 1)))
        r_tl, r_tr = regularize_1site_corners(cor_tl, cor_tr)
        r_br, r_bl = regularize_1site_corners(cor_br, cor_bl)

    if move in 'lh':
        env._set_projector_pair_(tr, 'hrb', tr, 'hrb', br, 'hrt',
                                 r_tr, r_br, opts_svd, **kwargs)

    if move in 'rh':
        env._set_projector_pair_(tl, 'hlb', tl, 'hlb', bl, 'hlt',
                                 r_tl, r_bl, opts_svd, **kwargs)

    if move in 'tbv':
        cor_bl = (env[br].bl @ env[br].l).fuse_legs(axes=((0, 1), 2))
        cor_tl = (env[tr].l @ env[tr].tl).fuse_legs(axes=(0, (2, 1)))
        cor_tr = (env[tl].tr @ env[tl].r).fuse_legs(axes=((0, 1), 2))
        cor_br = (env[bl].r @ env[bl].br).fuse_legs(axes=(0, (2, 1)))
        r_bl, r_tl = regularize_1site_corners(cor_bl, cor_tl)
        r_tr, r_br = regularize_1site_corners(cor_tr, cor_br)

    if move in 'tv':
        env._set_projector_pair_(tl, 'vtr', tl, 'vtr', tr, 'vtl',
                                 r_tl, r_tr, opts_svd, **kwargs)

    if move in 'bv':
        env._set_projector_pair_(bl, 'vbr', bl, 'vbr', br, 'vbl',
                                 r_bl, r_br, opts_svd, **kwargs)


def regularize_1site_corners(cor_0, cor_1):
    Q_0, R_0 = qr(cor_0, axes=(0, 1))
    Q_1, R_1 = qr(cor_1, axes=(1, 0))
    R01 = tensordot(R_0, R_1, axes=(1, 1))
    U_0, S, U_1 = R01.svd(axes=(0, 1), fix_signs=True)
    S = S.sqrt()
    r_0 = tensordot((U_0 @ S), Q_0, axes=(0, 1))
    r_1 = tensordot((S @ U_1), Q_1, axes=(1, 1))
    return r_0, r_1

def _si_rank(opts_svd, opts_si):
    """Total size of an SI basis, including oversampling."""
    oversampling = opts_si.get('oversampling', 5)
    D_total = opts_svd.get('D_total')
    if isinstance(D_total, int):
        return D_total + oversampling
    D_block = opts_svd.get('D_block')
    if isinstance(D_block, int):
        return D_block + oversampling
    raise YastnError("SI projectors require an integer D_total or D_block in opts_svd.")


def _distribute_si_rank(charges, rank):
    # basic distribution of rank over charge sectors, ignoring sector capacities
    """Distribute an SI rank evenly, filling remainders by charge order.
    Parameters
    ----------
    charges : list
        List of charges.
    rank : int
        Total number of columns to distribute over charge sectors.
    Returns
    -------
    dict
        Mapping ``charge -> amount``. Charges are always tuples, including ``()`` for tensors without symmetry and ``(q,)`` for U(1).
    """
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise YastnError("SI rank must be a positive integer.")

    full_charge_rank = rank // len(charges)
    rank_remainder = rank % len(charges)

    def charge_order(charge):
        # For U(1), this is 0, +1, -1, +2, -2, ... .  The tuple fallback
        # applies the same convention component by component.
        return tuple((abs(q), q < 0) for q in charge)

    charge_ordered = sorted(charges, key=charge_order)
    charges_map = {charge: full_charge_rank for charge in charge_ordered}
    for i in range(rank_remainder):
        charges_map[charge_ordered[i]] += 1

    return charges_map


def _distribute_si_rank_with_capacity(capacities, rank):
    r"""Distribute an SI rank without exceeding CTM-leg sector capacities.

    The returned mapping defines the auxiliary leg used by both SI bases, so
    ``X`` and ``Y`` always contain exactly the same charges and the same number
    of vectors in every charge sector. The allocation always has the requested
    oversampled rank; insufficient corner-leg capacity is an error.
    """
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise YastnError("SI rank must be a positive integer.")

    capacities = {charge: dimension
                  for charge, dimension in capacities.items()
                  if dimension > 0}
    if not capacities:
        raise YastnError("The CTM corner leg has no non-empty charge sectors.")

    total_capacity = sum(capacities.values())
    if rank > total_capacity:
        raise YastnError(
            f"Requested SI rank {rank} exceeds CTM corner-leg capacity "
            f"{total_capacity}; cannot construct an auxiliary leg of "
            f"dimension chi + oversampling.")

    def charge_order(charge):
        return tuple((abs(q), q < 0) for q in charge)

    ordered_charges = sorted(capacities, key=charge_order)
    allocation = {charge: 0 for charge in ordered_charges}
    remaining = rank

    # Round-robin allocation is even whenever capacities permit it. Once a
    # small sector is full, its remaining share is assigned to larger sectors.
    while remaining:
        for charge in ordered_charges:
            if allocation[charge] < capacities[charge]:
                allocation[charge] += 1
                remaining -= 1
                if remaining == 0:
                    break

    return {charge: dimension for charge, dimension in allocation.items()
            if dimension > 0}


def _validate_ctm_corner_pair(r0, r1):
    """Validate the two closures of a pair of CTM corner halves."""
    if not isinstance(r0, Tensor) or not isinstance(r1, Tensor):
        raise YastnError("CTM corner halves must be YASTN tensors.")
    if r0.ndim != 2 or r1.ndim != 2:
        raise YastnError("CTM corner halves must be rank-2 tensors.")
    if r0.config.sym.SYM_ID != r1.config.sym.SYM_ID:
        raise YastnError("CTM corner halves must use the same symmetry.")

    for axis in (0, 1):
        leg0 = r0.get_legs(axis)
        leg1 = r1.get_legs(axis)
        common_charges = leg0.tD.keys() & leg1.tD.keys()
        if any(leg0.tD[charge] != leg1.tD[charge]
               for charge in common_charges):
            raise YastnError(
                "CTM corner halves must have matching dimensions in every "
                "shared charge sector on both loop closures; "
                f"mismatch on axis {axis}.")


def _ctm_shared_sector_capacity(r0, r1):
    """Return capacities of sectors supported by both CTM corner halves."""
    capacity0 = r0.get_legs(0).tD
    capacity1 = r1.get_legs(0).tD
    return {charge: capacity0[charge]
            for charge in capacity0.keys() & capacity1.keys()}


def initialize_si_bases(r0, r1, rank, charges=None):
    r"""Initialize compatible column-isometric SI bases from Gaussian noise.

    The auxiliary rank is spread as uniformly as possible over charge sectors
    of the matching external CTM legs. A sector cannot be assigned more
    columns than that sector has rows.
    """
    _validate_ctm_corner_pair(r0, r1)

    x_input = r1.get_legs(0).conj() # right leg of r1
    y_input = r0.get_legs(0).conj() # left leg of r0
    sector_capacity = _ctm_shared_sector_capacity(r0, r1)

    if charges is None:
        charge_mapping = _distribute_si_rank_with_capacity(
            sector_capacity, rank)
    else:
        charge_mapping = dict(charges)
        unknown_charges = set(charge_mapping) - set(sector_capacity)
        if unknown_charges:
            raise YastnError(
                f"SI charge sectors are absent from the CTM corner leg: "
                f"{unknown_charges}.")
        if any(not isinstance(dimension, int) or isinstance(dimension, bool)
               or dimension <= 0
               for dimension in charge_mapping.values()):
            raise YastnError("SI charge-sector dimensions must be positive integers.")
        if not charge_mapping:
            raise YastnError("SI charge-sector mapping cannot be empty.")
        mapped_rank = sum(charge_mapping.values())
        if mapped_rank != rank:
            raise YastnError(
                f"Explicit SI charge-sector dimensions sum to {mapped_rank}, "
                f"but requested SI rank is {rank}.")
        for charge, dimension in charge_mapping.items():
            capacity = sector_capacity[charge]
            if dimension > capacity:
                raise YastnError(
                    f"SI dimension {dimension} exceeds capacity {capacity} "
                    f"in charge sector {charge}.")
    x_aux = Leg(
        r1.config,
        s=-x_input.s,
        t=tuple(charge_mapping.keys()),
        D=tuple(charge_mapping.values()),
    )

    X = rand(r1.config, legs=(x_input, x_aux), distribution='normal')
    Yh = rand(r0.config, legs=(y_input.conj(), x_aux),
              distribution='normal')

    X, _ = qr(X, axes=(0, 1), sQ=x_aux.s)
    Yh, _ = qr(Yh, axes=(0, 1), sQ=x_aux.s)
    return X, Yh.H

def si_bases_compatible(r0, r1, X, Y):
    """Whether recycled bases are compatible with the current corners."""
    if X is None or Y is None:
        return False

    def is_compatible_subspace(basis_leg, corner_leg):
        """A refined basis may intentionally contain only selected sectors."""
        return (basis_leg.s == corner_leg.s
                and all(charge in corner_leg.tD
                        and corner_leg.tD[charge] == dimension
                        for charge, dimension in basis_leg.tD.items()))

    try:
        return (
            is_compatible_subspace(
                X.get_legs(0), r1.get_legs(0).conj())
            and is_compatible_subspace(
                Y.get_legs(1), r0.get_legs(0).conj())
            and X.get_legs(1) == Y.get_legs(0).conj()
            and X.dtype == r1.dtype
            and Y.dtype == r0.dtype
            and X.device == r1.device
            and Y.device == r0.device
        )
    except (AttributeError, IndexError):
        return False

def si_subspace_error(Q, Q_old):
    r"""Mean squared sine of the principal angles between two SI bases.

    Both tensors are expected to be column-isometric.  The expression
    ``1 - ||Q_old.H @ Q||_F^2 / rank`` is invariant under rotations within
    either basis, unlike a direct tensor difference.
    """
    if Q_old is None or Q.get_legs() != Q_old.get_legs():
        return float('inf')

    rank = Q.get_shape(axes=1)
    overlap = Q_old.detach().H @ Q.detach()
    error = 1.0 - overlap.norm() ** 2 / rank
    # Roundoff can put the result just outside the mathematical interval.
    return max(0.0, min(1.0, error.item()))


def svd_charge_sector_dimensions(s):
    r"""Return the number of singular values in every charge sector.

    Parameters
    ----------
    s : Tensor
        Diagonal singular-value tensor returned by :meth:`Tensor.svd`.

    Returns
    -------
    dict
        Mapping ``charge -> amount``. Charges are always tuples, including
        ``()`` for tensors without symmetry and ``(q,)`` for U(1).
    """
    if not isinstance(s, Tensor) or not s.isdiag or s.ndim != 2:
        raise YastnError("Expected a diagonal rank-2 singular-value tensor.")
    return dict(s.get_legs(0).tD)


def svd_charge_sector_values(s):
    r"""Return singular values grouped by symmetry-charge sector.

    Parameters
    ----------
    s : Tensor
        Diagonal singular-value tensor returned by :meth:`Tensor.svd`.

    Returns
    -------
    dict
        Mapping ``charge -> list of singular values``. Charges are tuples,
        including ``()`` without symmetry and ``(q,)`` for U(1).
    """
    if not isinstance(s, Tensor) or not s.isdiag or s.ndim != 2:
        raise YastnError("Expected a diagonal rank-2 singular-value tensor.")

    return {
        charge: s[charge + charge].tolist()
        for charge in s.get_legs(0).t
    }


def _distribute_si_rank_proportionally(sector_weights, rank):
    """Distribute ``rank`` proportionally to nonnegative sector weights."""
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise YastnError("SI rank must be a positive integer.")
    if not sector_weights:
        raise YastnError("Cannot distribute SI rank without charge sectors.")
    if any(weight < 0 for weight in sector_weights.values()):
        raise YastnError("Charge-sector weights must be nonnegative.")

    total_weight = sum(sector_weights.values())
    if total_weight == 0:
        return _distribute_si_rank(sector_weights, rank)

    charge_mapping = {}
    fractional_numerators = {}
    for charge, weight in sector_weights.items():
        dimension, fractional_numerator = divmod(rank * weight,
                                                 total_weight)
        charge_mapping[charge] = dimension
        fractional_numerators[charge] = fractional_numerator

    remainder = rank - sum(charge_mapping.values())
    remainder_order = sorted(
        fractional_numerators,
        key=fractional_numerators.get,
        reverse=True)
    for charge in remainder_order[:remainder]:
        charge_mapping[charge] += 1

    return {charge: dimension for charge, dimension in charge_mapping.items()
            if dimension > 0}


def _si_refinement_asvr(r0, r1, X, Y, opts_svd, opts_si):
    """Return a stable SI charge mapping estimated from dominant spectra."""
    iterations = opts_si.get('asvr_iterations', 5)
    chip = _si_rank(opts_svd, opts_si)
    sector_capacity = _ctm_shared_sector_capacity(r0, r1)
    charge_mapping = dict(X.get_legs(1).tD)

    # A symmetry-preserving subspace iteration cannot generate a charge sector
    # absent from its input bases. Seed every sector shared by both corners so
    # that ASVR can compare their spectra before refining the allocation.
    missing_charges = set(sector_capacity) - set(charge_mapping)
    if missing_charges:
        exploratory_mapping = _distribute_si_rank_with_capacity(
            sector_capacity, chip)
        unexplored_charges = set(sector_capacity) - set(exploratory_mapping)
        if unexplored_charges:
            raise YastnError(
                "ASVR cannot probe every shared charge sector with SI rank "
                f"{chip}; increase D_total/D_block or oversampling. Missing "
                f"sectors: {unexplored_charges}.")
        charge_mapping = exploratory_mapping
        X, Y = initialize_si_bases(
            r0, r1, chip, charges=charge_mapping)

    for asvr_iteration in range(iterations):
        _, _, _, _, _, sall = si_projector_svd(
            r0, r1, X, Y, opts_svd, opts_si, return_spectrum=True)
        sector_values = svd_charge_sector_values(sall)
        # Keep values above the largest per-sector floor so dominant sectors
        # receive more columns in the next allocation.
        largest_smallest_sector_value = max(values[-1]
                                            for values in sector_values.values())
        sector_dominant_values = {
            charge: sum(value >= largest_smallest_sector_value for value in values)
            for charge, values in sector_values.items()
        }
        refined_mapping = _distribute_si_rank_proportionally(
            sector_dominant_values, chip)
        if refined_mapping == charge_mapping:
            break
        charge_mapping = refined_mapping
        if asvr_iteration + 1 < iterations:
            X, Y = initialize_si_bases(
                r0, r1, chip, charges=charge_mapping)
    return charge_mapping


def _si_refinement_rds(r0, r1, X, Y, opts_svd, opts_si):
    r"""Allocate SI rank from the relative sizes of CTM charge sectors.

    The auxiliary rank is distributed proportionally to the dimensions of the
    charge sectors shared by the external legs of ``r0`` and ``r1``. Integer
    dimensions are obtained by largest-remainder apportionment, and the result
    never exceeds a sector's capacity. ``X`` and ``Y`` are accepted to provide
    the same call signature as the other SI refinement methods.
    """
    sector_capacity = _ctm_shared_sector_capacity(r0, r1)
    rank = min(_si_rank(opts_svd, opts_si), sum(sector_capacity.values()))
    charge_mapping = _distribute_si_rank_proportionally(
        sector_capacity, rank)
    return charge_mapping


def _si_refinement_cwo(r0, r1, X, Y, opts_svd, opts_si):
    """Return an SI charge mapping estimated by per-sector oversampling."""
    chip = _si_rank(opts_svd, opts_si)
    oversampled_sector_values = {}
    for charge, capacity in _ctm_shared_sector_capacity(r0, r1).items():
        sector_rank = min(chip, capacity)
        X_charge, Y_charge = initialize_si_bases(
            r0, r1, sector_rank, charges={charge: sector_rank})
        _, _, _, _, _, sall = si_projector_svd(
            r0, r1, X_charge, Y_charge, opts_svd, opts_si,
            return_spectrum=True)
        oversampled_sector_values[charge] = (
            svd_charge_sector_values(sall).get(charge, []))

    top_values = sorted(
        ((value, charge) for charge, values in oversampled_sector_values.items()
         for value in values),
        key=lambda item: item[0], reverse=True)[:chip]
    charge_mapping = {}
    for _, charge in top_values:
        charge_mapping[charge] = charge_mapping.get(charge, 0) + 1
    if not charge_mapping:
        raise YastnError("CWO refinement found no singular values.")
    return charge_mapping


def si_refinement(r0, r1, X, Y, opts_svd, opts_si):
    r"""Refine and rebuild SI bases with the selected allocation strategy.

    This is the single dispatch point for SI charge-sector refinement. Each
    strategy returns a charge mapping; basis construction is centralized here
    so every method has the same public ``(X, Y)`` result.
    """
    _validate_ctm_corner_pair(r0, r1)
    refinement = opts_si.get('refinement', 'cwo')
    refinements = {
        'cwo': _si_refinement_cwo,
        'asvr': _si_refinement_asvr,
        'rds': _si_refinement_rds,
    }
    try:
        refine = refinements[refinement]
    except KeyError:
        raise YastnError(
            "Unknown SI refinement method "
            f"{refinement!r}; expected 'cwo', 'asvr', or 'rds'.") from None

    charge_mapping = refine(r0, r1, X, Y, opts_svd, opts_si)
    rank = sum(charge_mapping.values())
    return initialize_si_bases(
        r0, r1, rank, charges=charge_mapping)


def _apply_corner_product(r0, r1, X):
    r"""Apply A = tensordot(r0, r1, axes=(1, 1)) to X.

    r0 has indices (a, k), r1 has indices (b, k), and X has
    indices (b, p). The result has indices (a, p).
    """
    tmp = tensordot(r1, X, axes=(0, 0))       # (k, p)
    return tensordot(r0, tmp, axes=(1, 0))    # (a, p)


def _apply_corner_product_h(r0, r1, Z):
    r"""Apply A.H to Z without explicitly constructing A.

    Z has indices (a, p). The result has indices (b, p).
    """
    tmp = tensordot(r0.conj(), Z, axes=(0, 0))      # (k*, p)
    return tensordot(r1.conj(), tmp, axes=(1, 0))   # (b*, p)

def si_projector_svd(r0, r1, X, Y, opts_svd, opts_si,
                     return_spectrum=False):
    """Approximate the SVD of ``r0 @ r1.T`` using recycled subspaces."""
    _validate_ctm_corner_pair(r0, r1)
    niter = opts_si.get('niter', 5)
    tol = opts_si.get('tol', 1e-3)
    X_old, Yh_old = X, Y.H

    for _ in range(niter):
        AX = _apply_corner_product(r0, r1, X)
        X_next = _apply_corner_product_h(r0, r1, AX)
        X, _ = qr(X_next, axes=(0, 1), sQ=X.s[1])

        Yh = Y.H
        AHY = _apply_corner_product_h(r0, r1, Yh)
        Yh_next = _apply_corner_product(r0, r1, AHY)
        Yh, _ = qr(Yh_next, axes=(0, 1), sQ=Yh.s[1])

        error = max(si_subspace_error(X, X_old),
                    si_subspace_error(Yh, Yh_old))

        Y = Yh.H
        if error < tol:
            break
        X_old, Yh_old = X, Yh

    rho = Y @ _apply_corner_product(r0, r1, X)
    us, sall, vs = rho.svd(axes=(0, 1), sU=rho.s[1], fix_signs=True)

    X_new = X @ vs.H
    Y_new = us.H @ Y
    u = Y.H @ us
    v = vs @ X.H

    trunc_opts = {k: opts_svd[k] for k in ('tol', 'tol_block', 'D_block',
        'D_total', 'truncate_multiplets', 'mask_f') if k in opts_svd}
    mask = truncation_mask(sall, **trunc_opts)
    u, s, v = mask.apply_mask(u, sall, v, axes=(-1, 0, 0))
    result = (u, s, v, X_new, Y_new)
    return result + (sall,) if return_spectrum else result

def proj_corners(r0, r1, opts_svd, opts_si=None, X=None, Y=None,
                 return_si_state=False, **kwargs):
    r""" Projectors in between r0 @ r1.T corners. """
    # TODO: r1 matrix is defined as (right, left)
    _validate_ctm_corner_pair(r0, r1)
    opts_svd = dict(opts_svd)
    if 'truncation_f' in kwargs:
        opts_svd['mask_f'] = kwargs['truncation_f']
    opts_svd['fix_signs'] = opts_svd.get('fix_signs', True)
    verbosity = opts_svd.get('verbosity', 0)
    # only verbosity from opts_svd is to be passed down to svd_with_truncation
    kwargs.pop('verbosity', None)
    profiling_mode= kwargs.get('profiling_mode', None)

    si_enabled = opts_si is not None and opts_si.get('enabled', False)
    X_new = Y_new = None
    if si_enabled:
        # An eye-initialized CTM starts below its requested chi and grows over
        # the first updates.  During that growth the enlarged corners may not
        # yet accommodate chi + p rangefinder columns.  Use every currently
        # available shared direction; changed corner legs will invalidate and
        # enlarge the recycled bases on subsequent updates.
        rank = min(_si_rank(opts_svd, opts_si),
                   sum(_ctm_shared_sector_capacity(r0, r1).values()))
        recycled = si_bases_compatible(r0, r1, X, Y)
        if not recycled:
            X, Y = initialize_si_bases(r0, r1, rank)
        if opts_si.get('correct', False):
            X, Y = si_refinement(
                r0, r1, X, Y, opts_svd, opts_si)
        try:
            u, s, v, X_new, Y_new = si_projector_svd(
                r0, r1, X, Y, opts_svd, opts_si)
        except YastnError:
            # Aggregate charge dimensions can stay unchanged while an updated
            # CTM corner acquires incompatible dimensions inside a hard-fused
            # leg. In that case a recycled basis cannot be contracted, so
            # rebuild it for the current corner layout and retry once.
            if not recycled:
                raise
            X, Y = initialize_si_bases(r0, r1, rank)
            u, s, v, X_new, Y_new = si_projector_svd(
                r0, r1, X, Y, opts_svd, opts_si)
    elif profiling_mode in ["NVTX",]:
        rr = tensordot(r0, r1, axes=(1, 1))
        rr.config.backend.cuda.nvtx.range_push(f"svd_with_truncation")
        u, s, v = rr.svd_with_truncation(axes=(0, 1), sU=r0.s[1], **opts_svd, **kwargs)
        rr.config.backend.cuda.nvtx.range_pop()
    else:
        rr = tensordot(r0, r1, axes=(1, 1))
        u, s, v = rr.svd_with_truncation(axes=(0, 1), sU=r0.s[1], **opts_svd, **kwargs)

    if verbosity > 2:
        fname = sys._getframe().f_code.co_name
        logger.info(f"{fname} S {s.get_legs(0)}")

    cutoff = kwargs.get('cutoff', 0)
    rs = s.rsqrt(cutoff=cutoff)
    p0 = tensordot(r1, (rs @ v).conj(), axes=(0, 1)).unfuse_legs(axes=0)
    p1 = tensordot(r0, (u @ rs).conj(), axes=(0, 0)).unfuse_legs(axes=0)
    if return_si_state:
        return p0, p1, X_new, Y_new
    return p0, p1


def update_storage_(old, new):
    r"""
    Update projectors or environment tensor in ``old`` with the ones stored in ``new`` (ignoring unassigned projectors i.e. ``None``).

    Parameters
    ----------
    old: Peps | EnvCTM
        Has ``EnvCTM_projectors`` or ``EnvCTM_local`` assigned to each site
    new: Peps | EnvCTM
        Has ``EnvCTM_projectors`` or ``EnvCTM_local`` assigned to each site
    """
    for site in old.sites():
        for k, v in new[site].__dict__.items():
            if v is not None:
                setattr(old[site], k, v)


def _random_matrix_for_sector_test(sym, sectors, seed):
    """Create a random block-diagonal matrix for executable tests below."""
    from ....tensor import make_config

    config = make_config(backend='np', sym=sym)
    config.backend.random_seed(seed)
    matrix = Tensor(config=config, s=(1, -1))
    for charge, dimension in sectors:
        kwargs = {} if charge is None else {'ts': (charge, charge)}
        matrix.set_block(Ds=(dimension, dimension), val='rand', **kwargs)
    return matrix


def _charge_counts_for_sector_test(matrix):
    _, singular_values, _ = matrix.svd(
        axes=(0, 1), sU=matrix.s[1], fix_signs=True)
    return svd_charge_sector_dimensions(singular_values)


class TestSvdChargeSectorDimensions:
    """Executable tests for :func:`svd_charge_sector_dimensions`."""

    @staticmethod
    def plot_z2_singular_values(s_ref, s_si, D_total, plot_path):
        """Plot full-SVD and SI singular values for both Z2 sectors."""
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, spectrum, title in zip(
                axes, (s_ref, s_si), ('Full SVD', 'SI')):
            sector_values = {}
            for charge in (0, 1):
                block = (charge, charge)
                if block not in spectrum.get_blocks_charge():
                    continue
                values = np.asarray(spectrum[block]).reshape(-1)
                values = np.sort(values)[::-1]
                sector_values[charge] = values
                ax.semilogy(range(1, len(values) + 1), values,
                            marker='.', label=f'charge {charge}')
            all_values = np.concatenate(tuple(sector_values.values()))
            if 0 < D_total <= len(all_values):
                cutoff = np.sort(all_values)[::-1][D_total - 1]
                ax.axhline(cutoff, color='black', linestyle='--', linewidth=1.5,
                           label=f'D_total cutoff ({cutoff:.3g})')
            ax.set_title(title)
            ax.set_xlabel('index within charge sector')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()
        axes[0].set_ylabel('singular value')
        fig.suptitle(f'Z2 singular-value spectra (D_total={D_total})')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    def test_dense(self):
        rho = _random_matrix_for_sector_test('none', ((None, 3),), seed=0)
        counts = _charge_counts_for_sector_test(rho)
        assert counts == {(): 3}, counts

    def test_u1(self):
        rho = _random_matrix_for_sector_test(
            'U1', ((-1, 2), (0, 3), (2, 1)), seed=1)
        counts = _charge_counts_for_sector_test(rho)
        assert counts == {(-1,): 2, (0,): 3, (2,): 1}, counts

    def test_z2(self):
        rho = _random_matrix_for_sector_test('Z2', ((0, 2), (1, 3)), seed=2)
        counts = _charge_counts_for_sector_test(rho)
        assert counts == {(0,): 2, (1,): 3}, counts

    @staticmethod
    def z2_si_sector_distribution(r0_sector_dims, r1_sector_dims,
                                  x_sector_dims, y_sector_dims,
                                  D_total=12, scale=1,
                                  distribution='random', plot_path=None):
        """Return SI/reference sector distributions and their projector error.

        ``scale`` can be a single number applied to both Z2 sectors or a
        ``{charge: factor}`` mapping used to bias their singular spectra.
        ``distribution`` controls the spectrum within each sector and accepts
        ``'random'``, ``'flat'``, ``'linear'``, ``'exponential'``,
        ``'powerlaw'``, a callable ``f(dimension, charge)``, or a
        ``{charge: distribution}`` mapping.
        """
        import numpy as np

        if r0_sector_dims != r1_sector_dims:
            raise ValueError("r0 and r1 must have matching Z2 sector dimensions.")
        if x_sector_dims != y_sector_dims:
            raise ValueError("X and Y must have matching SI sector dimensions.")

        sectors = tuple(sorted(r1_sector_dims.items()))
        config = _random_matrix_for_sector_test('Z2', sectors, seed=3).config
        rng = np.random.default_rng(3)
        r1 = Tensor(config=config, s=(1, -1))
        r0 = Tensor(config=config, s=(1, -1))
        for charge, dimension in sorted(r0_sector_dims.items()):
            block = (charge, charge)
            r0.set_block(ts=block, Ds=(dimension, dimension),
                         val=np.eye(dimension))
            factor = scale.get(charge, 1) if isinstance(scale, dict) else scale
            sector_distribution = (distribution[charge]
                                   if isinstance(distribution, dict)
                                   else distribution)
            if callable(sector_distribution):
                singular_values = np.asarray(
                    sector_distribution(dimension, charge))
            elif sector_distribution == 'flat':
                singular_values = np.ones(dimension)
            elif sector_distribution == 'linear':
                singular_values = np.linspace(1, 1e-2, dimension)
            elif sector_distribution == 'exponential':
                singular_values = np.geomspace(1, 1e-8, dimension)
            elif sector_distribution == 'powerlaw':
                singular_values = 1 / np.arange(1, dimension + 1)
            elif sector_distribution == 'random':
                singular_values = np.sort(rng.random(dimension))[::-1]
            else:
                raise ValueError(
                    f"Unknown singular-value distribution for charge "
                    f"{charge}: {sector_distribution!r}.")
            if singular_values.shape != (dimension,):
                raise ValueError(
                    "A custom distribution must return one value per dimension.")

            q_left, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
            q_right, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
            matrix = q_left @ np.diag(factor * singular_values) @ q_right.T
            r1.set_block(ts=block, Ds=matrix.shape, val=matrix)

        biased_rho = r0 @ r1
        x_leg = Leg(config, s=-1, t=tuple(sorted(x_sector_dims)),
                    D=tuple(x_sector_dims[q] for q in sorted(x_sector_dims)))
        y_leg = Leg(config, s=-1, t=tuple(sorted(y_sector_dims)),
                    D=tuple(y_sector_dims[q] for q in sorted(y_sector_dims)))

        def random_isometry(outer_leg, si_leg):
            basis = rand(config, legs=(outer_leg, si_leg))
            return qr(basis, axes=(0, 1), sQ=si_leg.s)[0]

        X = random_isometry(biased_rho.get_legs(1).conj(), x_leg)
        Yh = random_isometry(biased_rho.get_legs(0), y_leg)

        opts_svd = {'D_total': D_total, 'D_block': float('inf'), 'tol': 0}
        opts_si = {'niter': 100, 'tol': 1e-12}
        u_si, s_si, v_si, _, _, s_si_all = si_projector_svd(
            r0, r1, X, Yh.H, opts_svd, opts_si, return_spectrum=True)
        _, s_ref_all, _ = biased_rho.svd(
            axes=(0, 1), sU=biased_rho.s[1], fix_signs=True)
        u_ref, s_ref, v_ref = biased_rho.svd_with_truncation(
            axes=(0, 1), sU=biased_rho.s[1], fix_signs=True, **opts_svd)

        si_counts = svd_charge_sector_dimensions(s_si)
        reference_counts = svd_charge_sector_dimensions(s_ref)

        left_error = (u_si @ u_si.H - u_ref @ u_ref.H).norm().item()
        right_error = (v_si.H @ v_si - v_ref.H @ v_ref).norm().item()
        error = max(left_error, right_error)

        if plot_path is not None:
            TestSvdChargeSectorDimensions.plot_z2_singular_values(
                s_ref_all, s_si_all, D_total, plot_path)

        return si_counts, reference_counts, error


    def test_rejects_nondiagonal(self):
        rho_u1 = _random_matrix_for_sector_test('U1', ((0, 2),), seed=4)
        try:
            svd_charge_sector_dimensions(rho_u1)
        except YastnError:
            pass
        else:
            raise AssertionError("A non-diagonal tensor should be rejected.")


def _test_svd_charge_sector_dimensions():
    """Run the executable tests for :func:`svd_charge_sector_dimensions`."""
    tests = TestSvdChargeSectorDimensions()
    # tests.test_dense()
    # tests.test_u1()
    # tests.test_z2()
    r0_sector_dims =r1_sector_dims = {0: 240, 1: 320}

    x_sector_dims = y_sector_dims ={0: 12, 1: 12}
    si_counts, reference_counts, error = tests.z2_si_sector_distribution(
        r0_sector_dims, r1_sector_dims, x_sector_dims, y_sector_dims,
        D_total=12,
        scale={0: 10, 1: 1},
        distribution={0: 'powerlaw', 1: 'exponential'},
        plot_path='z2_si_singular_values.png')
    print(f"si_counts={si_counts}, reference_counts={reference_counts}, error={error}")
    # tests.test_rejects_nondiagonal()
    print("svd_charge_sector_dimensions tests passed")


if __name__ == '__main__':
    _test_svd_charge_sector_dimensions()
    # method 1 exact 
    
