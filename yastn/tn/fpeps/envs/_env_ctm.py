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
from ....initialize import rand, ones, eye
from ....tensor import Tensor, YastnError, Leg, tensordot, qr, ncon
from ...._split_combine_dict import split_data_and_meta, combine_data_and_meta

logger = logging.getLogger(__name__)

class CTMRG_out(NamedTuple):
    sweeps: int = 0
    max_dsv: float = None
    converged: bool = False
    max_D: int = 1


class EnvCTM():
    def __init__(self, psi, init='rand', leg=None, ket=None):
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

        ket: Optional[yastn.tn.Peps]
            If provided, and ``psi`` has physical legs, forms a double-layer PEPS <psi | ket>.
        """
        self.geometry = psi.geometry
        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "f_ordered", "nn_bond_dirn"]:
            setattr(self, name, getattr(self.geometry, name))

        self.psi = Peps2Layers(bra=psi, ket=ket) if psi.has_physical() else psi
        self.env = Lattice(self.geometry, objects={site: EnvCTM_local() for site in self.sites()})
        self.proj = Lattice(self.geometry, objects={site: EnvCTM_projectors() for site in self.sites()})

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
        return env

    def shallow_copy(self) -> EnvCTM:
        cls = type(self)
        env = cls(self.psi, init=None)
        env.env = self.env.shallow_copy()
        env.proj = self.proj.shallow_copy()
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
        return env

    def detach_(self):
        r"""
        Detach all environment tensors from the computational graph.
        Data of environment tensors in detached environment is a `view` of the original data.
        """
        self.env.detach_()
        self.proj.detach_()

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
                'proj': self.proj.to_dict(level=level)}

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
            return env

    def update_from_dict_(self, d):
        psi = PEPS_CLASSES[d['psi']['type']].from_dict(d['psi'])
        tmp = type(self)(psi, init=None)
        self.psi = tmp.psi
        self.env = Lattice.from_dict(d['env'])
        self.proj = Lattice.from_dict(d['proj'])

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

    def _default_corner_signature(self):
        return (1, -1)

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

        if init == 'dl':
            self.reset_(init='eye')
            self.expand_outward_()
            for site in self.sites():
                for dirn in self[site].fields():
                    T = getattr(self[site], dirn)
                    T = T / T.norm(p=normalize) if isinstance(normalize, str) else normalize(T)
                    setattr(self[site], dirn, T)
            return

        cs = self._default_corner_signature()
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

    def _partial_svd_predict_spec(self,leg0,leg1,sU):
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
        l= leg0 if sU == leg0.s else leg1
        return { t: max(d+10,int(d*1.1)) for t,d in zip(l.t, l.D) }

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
                    return out_data, out_meta

                if "torch" in env.config.backend.BACKEND_ID:
                    inputs_t, inputs_meta = split_data_and_meta(env.to_dict(level=0))

                    if checkpoint_move == 'reentrant':
                        use_reentrant = True
                    elif checkpoint_move == 'nonreentrant':
                        use_reentrant = False
                    checkpoint_F = env.config.backend.checkpoint
                    out_data, out_meta = checkpoint_F(f_update_core_, d, inputs_meta, *inputs_t, \
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
        measure_nsite, measure_2site, sample


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
        env.proj[tr].hrb, env.proj[br].hrt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)

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
        env.proj[tl].hlb, env.proj[bl].hlt = proj_corners(r_t, r_b, opts_svd=opts_svd, **kwargs)

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
        env.proj[tl].vtr, env.proj[tr].vtl = proj_corners(r_l, r_r, opts_svd=opts_svd, **kwargs)

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
        env.proj[bl].vbr, env.proj[br].vbl = proj_corners(r_l, r_r, opts_svd=opts_svd, **kwargs)


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
        env.proj[tr].hrb, env.proj[br].hrt = proj_corners(r_tr, r_br, opts_svd=opts_svd, **kwargs)

    if move in 'rh':
        env.proj[tl].hlb, env.proj[bl].hlt = proj_corners(r_tl, r_bl, opts_svd=opts_svd, **kwargs)

    if move in 'tbv':
        cor_bl = (env[br].bl @ env[br].l).fuse_legs(axes=((0, 1), 2))
        cor_tl = (env[tr].l @ env[tr].tl).fuse_legs(axes=(0, (2, 1)))
        cor_tr = (env[tl].tr @ env[tl].r).fuse_legs(axes=((0, 1), 2))
        cor_br = (env[bl].r @ env[bl].br).fuse_legs(axes=(0, (2, 1)))
        r_bl, r_tl = regularize_1site_corners(cor_bl, cor_tl)
        r_tr, r_br = regularize_1site_corners(cor_tr, cor_br)

    if move in 'tv':
        env.proj[tl].vtr, env.proj[tr].vtl = proj_corners(r_tl, r_tr, opts_svd=opts_svd, **kwargs)

    if move in 'bv':
        env.proj[bl].vbr, env.proj[br].vbl = proj_corners(r_bl, r_br, opts_svd=opts_svd, **kwargs)


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

    opts_svd = dict(opts_svd)
    if 'truncation_f' in kwargs:
        opts_svd['mask_f'] = kwargs['truncation_f']
    opts_svd['fix_signs'] = opts_svd.get('fix_signs', True)
    verbosity = opts_svd.get('verbosity', 0)
    # only verbosity from opts_svd is to be passed down to svd_with_truncation
    kwargs.pop('verbosity', None)
    profiling_mode= kwargs.get('profiling_mode', None)

    if profiling_mode in ["NVTX",]:
        rr.config.backend.cuda.nvtx.range_push(f"svd_with_truncation")
        u, s, v = rr.svd_with_truncation(axes=(0, 1), sU=r0.s[1], **opts_svd, **kwargs)
        rr.config.backend.cuda.nvtx.range_pop()
    else:
        u, s, v = rr.svd_with_truncation(axes=(0, 1), sU=r0.s[1], **opts_svd, **kwargs)

    if verbosity > 2:
        fname = sys._getframe().f_code.co_name
        logger.info(f"{fname} S {s.get_legs(0)}")

    rs = s.rsqrt()
    p0 = tensordot(r1, (rs @ v).conj(), axes=(0, 1)).unfuse_legs(axes=0)
    p1 = tensordot(r0, (u @ rs).conj(), axes=(0, 0)).unfuse_legs(axes=0)
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
