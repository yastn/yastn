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
from math import isfinite
import os
import numpy as np

from ._env_contractions import *
from ._ntu_metric import NTUMetric
from .._evolution import BondMetric
from ....tensor import tensordot, YastnError, ncon
from ...._memory_override import memory_override
from ....initialize import eye
from ....tensor._auxiliary import get_blocks
from ....tensor._contractions import _apply_mask_axes
from ....tensor._merging import _mask_tensors_leg_intersection
from ....tensor._tests import _unpack_trans_test_axes_pair


# Target size of the selected boundary vector, not a bound on total memory.
# Closing each selection immediately also avoids the full append output.
_NNN_APPEND_CHUNK_BYTES = 16 << 30


def _nnn_vector_ranges(vector, axis, budget):
    """Partition reduced multiplicities by their actual input storage cost."""
    native = vector.trans[axis]
    blocks = get_blocks(vector.config.sym, vector.struct)
    itemsize = np.dtype(vector.yastn_dtype).itemsize
    costs = {}
    for charge, shape, (start, stop) in zip(blocks.t[:, native], blocks.D, blocks.slc):
        charge = tuple(charge)
        costs[charge] = costs.get(charge, 0) + int(stop - start) // int(shape[native]) * itemsize
    start, stop, used = 0, 0, 0
    leg = vector.get_legs(axis)
    for charge, dimension in zip(leg.t, leg.D):
        cost = costs.get(charge, 0)
        for _ in range(dimension):
            if stop > start and used + cost > budget:
                yield start, stop
                start, used = stop, 0
            stop += 1
            used += cost
    if stop > start:
        yield start, stop


def _append_vec_nnn(Ac, A, vector, closing, append, vertical):
    """Append and close one internal boundary selection at a time.

    The cut leg is summed over by the closing contraction. Partition its
    multiplicities in both operands, retaining complete irreps. This is an
    exact resolution of the identity; it does not partition an open QR leg
    or require a streamed representation of the final metric.
    """
    setting = os.environ.get('YASTN_NNN_APPEND_CHUNK_BYTES', str(_NNN_APPEND_CHUNK_BYTES))
    if memory_override():
        setting = str(1 << 62)
    try:
        budget = int(setting)
        if budget <= 0:
            raise ValueError
    except ValueError:
        raise YastnError(f'YASTN_NNN_APPEND_CHUNK_BYTES={setting!r}: '
                         'expected a positive integer number of bytes.') from None

    def close(v, c):
        if vertical:
            return tensordot(v, c, axes=((2, 3), (0, 1)))
        return tensordot(c, v, axes=((2, 1), (0, 1)))

    va, ca = (3, 0) if vertical else (0, 2)
    if (vector.size * np.dtype(vector.yastn_dtype).itemsize <= budget
            or vector.mfs[va] != (1,) or closing.mfs[ca] != (1,)):
        # Transfer the sole reference so append can release its input when
        # rebinding its working tensor, just as in the original call chain.
        pending = [vector]
        del vector
        return close(append(Ac, A, pending.pop()), closing)
    mask_needed, (nva, nca) = _unpack_trans_test_axes_pair(vector, closing, sgn=-1, axes=((va,), (ca,)))
    if mask_needed:
        mv, mc, hv, hc = _mask_tensors_leg_intersection(vector, closing, nva, nca)
        vector = _apply_mask_axes(vector, nva, mv)._replace(hfs=hv)
        closing = _apply_mask_axes(closing, nca, mc)._replace(hfs=hc)
    if vector.size == 0:
        return close(append(Ac, A, vector), closing)
    # This leg is never unfused again: only its total charge and multiplicity
    # enter append/close. Drop the same history on both ends before masking.
    vector = vector.drop_leg_history(va)
    closing = closing.drop_leg_history(ca)
    identity = eye(vector.config, legs=vector.get_legs(va), device=vector.device)
    result = None
    for start, stop in _nnn_vector_ranges(vector, va, budget):
        mask = identity * 0
        mask._data[start:stop] = 1
        selected, boundary = mask.apply_mask(vector, closing, axes=(va, ca))
        selected = append(Ac, A, selected)
        value = close(selected, boundary)
        result = value if result is None else result + value
        del selected, boundary, value, mask
    return result


class EnvNTU:
    def __init__(self, psi, which='NN', metric_chunk_size=None, opts_eat=None):
        r"""
        Supports a family of NTU environments for truncation of PEPS tensors during time evolution.
        The local exact contractions, and SVD-1 approximation used in some environments,
        preserve hermiticity and positivity of the metric tensor.
        Numerical noise breaking those properties can still appear and grow with the number of environment layers.

        Parameters
        ----------
        psi: yastn.tn.Peps
            Peps state being evolved.

        which: str
            Type of bond environment from 'NN', 'NN+', 'NN++', 'NNN', 'NNN+', 'NNN++',
            or 'LadderN' with N >= 1 layers on either side ('Ladder' means 'Ladder2').

        metric_chunk_size: int | None
            Stream the metric in blocks of this many multiplicities on the first
            open QR leg, keeping all bra/ket cross terms. SU(2) multiplets stay
            intact. None retains the full metric (default). With streaming, use
            fix_metric=None in evolution or truncation. Supports SVD and EAT
            initializations, and NN and multi-site/NNN gate paths.
            Large hard-fused NNN boundary vectors use internal chunks
            independently of this setting, including when it is None.
            Set ``YASTN_NNN_APPEND_CHUNK_BYTES`` to a positive integer number
            of bytes to change their input chunk target (default: 16 GiB).
            It is read on each append call and does not bound total memory.

        opts_eat: dict | None
            Streaming EAT power-iteration options: relative singular-pair
            residual ``tol`` (default 1e-10) and ``max_iter`` (default 100).
            These are independent of the subsequent ALS optimization options.
            Failure to converge raises YastnError without a full-metric fallback.
        """
        if metric_chunk_size is not None and (isinstance(metric_chunk_size, bool)
                or not isinstance(metric_chunk_size, int) or metric_chunk_size < 1):
            raise YastnError('metric_chunk_size must be a positive integer or None.')
        self.metric_chunk_size = metric_chunk_size
        self.opts_eat = {'tol': 1e-10, 'max_iter': 100}
        if opts_eat is not None:
            if not isinstance(opts_eat, dict) or set(opts_eat) - self.opts_eat.keys():
                raise YastnError('opts_eat accepts only tol and max_iter.')
            self.opts_eat.update(opts_eat)
        tol, max_iter = self.opts_eat['tol'], self.opts_eat['max_iter']
        if isinstance(tol, bool) or not isinstance(tol, (float, int)) or not isfinite(tol) or tol <= 0:
            raise YastnError('opts_eat["tol"] must be positive and finite.')
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise YastnError('opts_eat["max_iter"] must be a positive integer.')
        self.psi = psi
        self._set_which(which)

    def _get_which(self):
        return self._which

    def _set_which(self, which):
        if which not in ('NN', 'NN+', 'NN++', 'NNN', 'NNN+', 'NNN++') and "ladder" not in which.lower():
            raise YastnError(f"Type of EnvNTU bond_metric {which=} not recognized.")

        self._which = which

    which = property(fget=_get_which, fset=_set_which)

    def apply_patch(self):
        pass

    def move_to_patch(self, sites):
        pass

    def pre_truncation_(env, bond):
        pass

    def post_truncation_(env, bond, **kwargs):
        pass

    def bond_metric(self, Q0, Q1, s0, s1, dirn):
        r"""
        Calculate bond metric. The environment size is controlled by ``which``.

        Below, double lines indicate core bonds that are contracted exactly.
        Dashed lines indicate tree-like approximations, where border tensors are approximated using SVD-1,
        i.e., product boundary vectors from a rank-one SVD decomposition.
        Such approximation preserves positivity of the calculated metric.

        ::

            If which == 'NN':

                      (-1 +0)══(-1 +1)
                         ║        ║
               (+0 -1)═══Q0══   ══Q1═══(+0 +2)
                         ║        ║
                      (+1 +0)══(+1 +1)


            If which == 'NN+':

                            (-2 +0)┈┈(-2 +1)
                               ┊        ┊
                    (-1 -1)┄(-1 +0)══(-1 +1)┄(-1 +2)
                       ┊       ║        ║       ┆
            (+0 -2)═(+0 -1)════Q0══   ══Q1═══(+0 +2)═(+0 +3)
                       ┊       ║        ║       ┆
                    (+1 -1)┄(+1 +0)══(+1 +1)┄(+1 +2)
                               ┊        ┊
                            (+2 +0)┈┈(+2 +1)


            If which == 'NN++':

                                    (-3 +0)┈┈(-3 +1)
                                       ┊        ┊
                            (-2 -1)┈(-2 +0)┈┈(-2 +1)┈(-2 +2)
                               ┊       ┊        ┊       ┊
                    (-1 -2)┈(-1 -1)┈(-1 +0)══(-1 +1)┈(-1 +2)┈(-1 +3)
                       ┊       ┊       ║        ║       ┊       ┊
            (+0 -3)═(+0 -2)═(+0 -1)════Q0══   ══Q1═══(+0 +2)=(+0 +3)=(+0 +4)
                       ┊       ┊       ║        ║       ┊       ┊
                    (+1 -2)┈(+1 -1)┈(+1 +0)══(+1 +1)┈(+1 +2)┈(+1 +3)
                               ┊       ┊        ┊       ┊
                            (+2 -1)┈(+2 +0)┈┈(+2 +1)┈(+2 +2)
                                       ┊        ┊
                                    (+3 +0)┈┈(+3 +1)


            If which == 'NNN':

                (-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)
                   ║       ║        ║       ║
                (+0 -1)════Q0══   ══Q1═══(+0 +2)
                   ║       ║        ║       ║
                (+1 -1)=(+1 +0)══(+1 +1)=(+1 +2)


            If which == 'NNN+':

                    (-2 -1) (-2 +0)  (-2 +1) (-2 +2)
                       ║       ║        ║       ║
            (-1 -2)=(-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)=(-1 +3)
                       ║       ║        ║       ║
            (+0 -2)=(+0 -1)════Q0══   ══Q1═══(+0 +2)=(+0 +3)
                       ║       ║        ║       ║
            (+1 -2)=(+1 -1)=(+1 +0)══(+1 +1)=(+1 +2)=(+1 +3)
                       ║       ║        ║       ║
                    (+2 -1) (+2 +0)  (+2 +1) (+2 +2)


            If which == 'NNN++':

                    (-3 -2)┈(-3 -1)┈(-3 +0)┈┈(-3 +1)┈(-3 +2)┈(-3 +3)
                        ┊      ┊       ┊        ┊       ┊       ┊
            (-2 -3)┈(-2 -2)┈(-2 -1)┈(-2 +0)┈┈(-2 +1)┈(-2 +2)┈(-2 +3)┈(-2 +4)
               ┊        ┊      ┊       ┊        ┊       ┊       ┊       ┊
            (-1 -3)┈(-1 -2)┈(-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)┈(-1 +3)┈(-1 +4)
               ┊        ┊      ║       ║        ║       ║       ┊       ┊
            (+0 -3)=(+0 -2)=(+0 -1)════Q0══   ══Q1═══(+0 +2)=(+0 +3)=(+0 +4)
               ┊        ┊      ║       ║        ║       ║       ┊       ┊
            (+1 -3)┈(+1 -2)┈(+1 -1)=(+1 +0)══(+1 +1)=(+1 +2)┈(+1 +3)┈(+1 +4)
               ┊        ┊      ┊       ┊        ┊       ┊       ┊       ┊
            (+2 -3)┈(+2 -2)┈(+2 -1)┈(+2 +0)┈┈(+2 +1)┈(+2 +2)┈(+2 +3)┈(+2 +4)
                        ┊      ┊       ┊        ┊       ┊       ┊
                    (+3 -2)┈(+3 -1)┈(+3 +0)┈┈(+3 +1)┈(+3 +2)┈(+3 +3)

        """
        if self.metric_chunk_size is not None:
            return BondMetric(g=NTUMetric(self, Q0, Q1, s0, s1, dirn,
                                         self.metric_chunk_size))
        return self._bond_metric(Q0, Q1, s0, s1, dirn)

    def _bond_metric(self, Q0, Q1, s0, s1, dirn, Q0_ket=None, Q1_ket=None):
        """Build a metric block; *_ket supplies its conjugated (row) layer."""
        Q0_ket = Q0 if Q0_ket is None else Q0_ket
        Q1_ket = Q1 if Q1_ket is None else Q1_ket
        if self.which == 'NN':
            return self._g_NN(Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket)
        if self.which == 'NN+':
            return self._g_NNp(Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket)
        if self.which == 'NN++':
            return self._g_NNpp(Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket)
        if self.which == 'NNN':
            return self._g_NNN(Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket)
        if self.which == 'NNN+':
            return self._g_NNNp(Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket)
        if self.which == 'NNN++':
            return self._g_NNNpp(Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket)
        if 'ladder' in self.which.lower():
            return self._g_Ladder(Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket)
        raise YastnError(f" Type of EnvNTU which={self.which} not recognized.")


    def _g_NN(self, Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket):
        r"""
        Calculates metric tensor within "NTU-NN" approximation.

        For dirn == 'h':

                (-1 +0)══(-1 +1)
                   ║        ║
        (+0 -1)════Q0══   ══Q1═══(+0 +2)
                   ║        ║
                (+1 +0)══(+1 +1)

        For dirn == 'v':

                (-1 +0)
                   ║
        (+0 -1)═══0Q0═══(+0 +1)
           ║       ╳       ║
        (+1 -1)═══1Q1═══(+1 +1)
                   ║
                (+2 +0)
        """
        if dirn in ("h", "lr"):
            assert self.psi.nn_site(s0, (0, 1)) == s1
            m = {d: self.psi.nn_site(s0, d=d) for d in [(-1,0), (0,-1), (1,0), (1,1), (0,2), (-1,1)]}
            tensors_from_psi(m, self.psi)
            env_l = edge_l(Q0, hair_l(m[0, -1]), A_ket=Q0_ket)  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(Q1, hair_r(m[0,  2]), A_ket=Q1_ket)  # [tr tr'] [ll ll'] [br br']
            ctl = cor_tl(m[-1, 0])
            ctr = cor_tr(m[-1, 1])
            cbr = cor_br(m[ 1, 1])
            cbl = cor_bl(m[ 1, 0])
            g = tensordot((cbr @ cbl) @ env_l, (ctl @ ctr) @ env_r, axes=((0, 2), (2, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            m = {d: self.psi.nn_site(s0, d=d) for d in [(-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,1)]}
            tensors_from_psi(m, self.psi)
            env_t = edge_t(Q0, hair_t(m[-1, 0]), A_ket=Q0_ket)  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(Q1, hair_b(m[ 2, 0]), A_ket=Q1_ket)  # [rb rb'] [tt tt'] [lb lb']
            cbl = cor_bl(m[1,-1])
            ctl = cor_tl(m[0,-1])
            ctr = cor_tr(m[0, 1])
            cbr = cor_br(m[1, 1])
            g = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))


    def _g_NNp(self, Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket):
        r"""
        Calculates the metric tensor within "NTU-NN+" approximation.

        For dirn == 'h':

                        (-2 +0)┈┈(-2 +1)
                           ║        ║
                (-1 -1)┈(-1 +0)══(-1 +1)┈(-1 +2)
                   ┊       ║        ║       ┊
        (+0 -2)=(+0 -1)════Q0══   ══Q1═══(+0 +2)=(+0 +3)
                   ┊       ║        ║       ┊
                (+1 -1)┈(+1 +0)══(+1 +1)┈(+1 +2)
                           ║        ║
                        (+2 +0)┈┈(+2 +1)

        For dirn == 'v':

                        (-2 +0)
                           ║
                (-1 -1)┈(-1 +0)┈(-1 +1)
                    ┊      ║       ┊
        (+0 -2)=(+0 -1)═══0Q0═══(+0 +1)=(+0 +2)
           ┊       ║       ╳       ║       ┊
        (+1 -2)=(+1 -1)═══1Q1═══(+1 +1)=(+1 +2)
                    ┊      ║       ┊
                (+2 -1)┈(+2 +0)┈(+2 +1)
                           ║
                        (+3 +0)
        """
        if dirn in ("h", "lr"):
            assert self.psi.nn_site(s0, (0, 1)) == s1
            sts = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (1,2), (0,2), (-1,2),
                   (-1,1), (-1,0), (0,-2), (2,0), (2,1), (0,3), (-2,1), (-2,0)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)
            htl_t, htl_l = cut_into_hairs(cor_tl(m[-1,-1]))
            htr_r, htr_t = cut_into_hairs(cor_tr(m[-1, 2]))
            hbr_b, hbr_r = cut_into_hairs(cor_br(m[1, 2]))
            hbl_l, hbl_b = cut_into_hairs(cor_bl(m[1, -1]))

            env_l = edge_l(Q0, hair_l(m[0,-1], hl=hair_l(m[0,-2]), ht=htl_t, hb=hbl_b), A_ket=Q0_ket)  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(Q1, hair_r(m[0, 2], hr=hair_r(m[0, 3]), ht=htr_t, hb=hbr_b), A_ket=Q1_ket)  # [tr tr'] [ll ll'] [br br']

            hm1p0_t, hm1p1_t = cut_into_hairs(cor_tl(m[-2, 0]) @ cor_tr(m[-2, 1]))
            hp1p1_b, hp1p0_b = cut_into_hairs(cor_br(m[2, 1]) @ cor_bl(m[2, 0]))

            ctl = cor_tl(m[-1, 0], ht=hm1p0_t, hl=htl_l)
            ctr = cor_tr(m[-1, 1], ht=hm1p1_t, hr=htr_r)
            cbr = cor_br(m[ 1, 1], hb=hp1p1_b, hr=hbr_r)
            cbl = cor_bl(m[ 1, 0], hb=hp1p0_b, hl=hbl_l)
            g = tensordot((cbr @ cbl) @ env_l, (ctl @ ctr) @ env_r, axes=((0, 2), (2, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            sts = [(-1,-1), (0,-1), (1,-1), (2,-1), (2,0), (2,1), (1,1), (0,1),
                   (-1,1), (-1,0), (0,-2), (1,-2), (3,0), (1,2), (0,2), (-2,0)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            htl_t, htl_l = cut_into_hairs(cor_tl(m[-1,-1]))
            htr_r, htr_t = cut_into_hairs(cor_tr(m[-1, 1]))
            hbr_b, hbr_r = cut_into_hairs(cor_br(m[2, 1]))
            hbl_l, hbl_b = cut_into_hairs(cor_bl(m[2, -1]))

            env_t = edge_t(Q0, hair_t(m[-1, 0], ht=hair_t(m[-2, 0]), hl=htl_l, hr=htr_r), A_ket=Q0_ket)  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(Q1, hair_b(m[ 2, 0], hb=hair_b(m[ 3, 0]), hl=hbl_l, hr=hbr_r), A_ket=Q1_ket)  # [rb rb'] [tt tt'] [lb lb']

            hp1m1_l, hp0m1_l = cut_into_hairs(cor_bl(m[1, -2]) @ cor_tl(m[0, -2]))
            hp0p1_r, hp1p1_r = cut_into_hairs(cor_tr(m[0, 2]) @ cor_br(m[1, 2]))

            cbl = cor_bl(m[1,-1], hl=hp1m1_l, hb=hbl_b)
            ctl = cor_tl(m[0,-1], hl=hp0m1_l, ht=htl_t)
            ctr = cor_tr(m[0, 1], hr=hp0p1_r, ht=htr_t)
            cbr = cor_br(m[1, 1], hr=hp1p1_r, hb=hbr_b)
            g = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))


    def _g_NNpp(self, Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket):
        r"""
        Calculates the metric tensor within "NTU-NN++" approximation.

        For dirn == 'h':

                                (-3 +0)┈┈(-3 +1)
                                   ┊        ┊
                        (-2 -1)┈(-2 +0)┈┈(-2 +1)┈(-2 +2)
                           ┊       ┊        ┊       ┊
                (-1 -2)┈(-1 -1)┈(-1 +0)══(-1 +1)┈(-1 +2)┈(-1 +3)
                   ┊       ┊       ║        ║       ┊       ┊
        (+0 -3)=(+0 -2)=(+0 -1)════Q0══   ══Q1═══(+0 +2)=(+0 +3)=(+0 +4)
                   ┊       ┊       ║        ║       ┊       ┊
                (+1 -2)┈(+1 -1)┈(+1 +0)══(+1 +1)┈(+1 +2)┈(+1 +3)
                           ┊       ┊        ┊       ┊
                        (+2 -1)┈(+2 +0)┈┈(+2 +1)┈(+2 +2)
                                   ┊        ┊
                                (+3 +0)┈┈(+3 +1)

        For dirn == 'v':

                                (-3 +0)
                                   ║
                        (-2 -1)┈(-2 +0)┈(-2 +1)
                           ┊       ║       ┊
                (-1 -2)┈(-1 -1)┈(-1 +0)┈(-1 +1)┈(-1 +2)
                   ┊       ┊       ║       ┊        ┊
        (+0 -3)┈(+0 -2)┈(+0 -1)═══0Q0═══(+0 +1)┈(+0 +2)┈(+0 +3)
           ┊       ┊       ║       ╳       ║       ┊       ┊
        (+1 -3)┈(+1 -2)┈(+1 -1)═══1Q1═══(+1 +1)┈(+1 +2)┈(+1 +3)
                   ┊       ┊       ║       ┊       ┊
                (+2 -2)┈(+2 -1)┈(+2 +0)┈(+2 +1)┈(+2 +2)
                           ┊       ║       ┊
                        (+3 -1)┈(+3 +0)┈(+3 +1)
                                   ║
                                (+4 +0)
        """
        if dirn in ("h", "lr"):
            assert self.psi.nn_site(s0, (0, 1)) == s1
            sts =  [(-3,0), (-3,1), (-2,-1), (-2,0), (-2,1), (-2,2),
                    (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2), (-1,3),
                    (0,-3), (0,-2), (0,-1), (0,2), (0,3), (0,4),
                    (1,-2), (1,-1), (1,0), (1,1), (1,2), (1,3),
                    (2,-1), (2,0), (2,1), (2,2), (3,0), (3,1)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            hp0m2_t, hm1m1_l = cut_into_hairs(cor_tl(m[-1, -2]))
            hm1m1_t, hm2p0_l = cut_into_hairs(cor_tl(m[-2, -1]))
            hp0m1_t, hm1p0_l = cut_into_hairs(cor_tl(m[-1, -1], hl=hm1m1_l, ht=hm1m1_t))

            hm2p1_r, hm1p2_t = cut_into_hairs(cor_tr(m[-2, 2]))
            hm1p2_r, hp0p3_t = cut_into_hairs(cor_tr(m[-1, 3]))
            hm1p1_r, hp0p2_t = cut_into_hairs(cor_tr(m[-1, 2], ht=hm1p2_t, hr=hm1p2_r))

            hp0p3_b, hp1p2_r = cut_into_hairs(cor_br(m[1, 3]))
            hp1p2_b, hp2p1_r = cut_into_hairs(cor_br(m[2, 2]))
            hp0p2_b, hp1p1_r = cut_into_hairs(cor_br(m[1, 2], hr=hp1p2_r, hb=hp1p2_b))

            hp1m1_l, hp0m2_b = cut_into_hairs(cor_bl(m[1, -2]))
            hp2p0_l, hp1m1_b = cut_into_hairs(cor_bl(m[2, -1]))
            hp1p0_l, hp0m1_b = cut_into_hairs(cor_bl(m[1, -1], hb=hp1m1_b, hl=hp1m1_l))

            hp0m1_l = hair_l(m[0, -2], hl=hair_l(m[0, -3]), ht=hp0m2_t, hb=hp0m2_b)
            hp0m0_l = hair_l(m[0, -1], hl=hp0m1_l, ht=hp0m1_t, hb=hp0m1_b)
            env_l = edge_l(Q0, hp0m0_l, A_ket=Q0_ket)  # [bl bl'] [rr rr'] [tl tl']

            hp0p2_r = hair_r(m[0, 3], hr=hair_r(m[0, 4]), ht=hp0p3_t, hb=hp0p3_b)
            hp0p1_r = hair_r(m[0, 2], hr=hp0p2_r, ht=hp0p2_t, hb=hp0p2_b)
            env_r = edge_r(Q1, hp0p1_r, A_ket=Q1_ket)  # [tr tr'] [ll ll'] [br br']

            hm2p0_t, hm2p1_t = cut_into_hairs(cor_tl(m[-3, 0]) @ cor_tr(m[-3, 1]))
            hm1p0_t, hm1p1_t = cut_into_hairs(cor_tl(m[-2, 0], hl=hm2p0_l, ht=hm2p0_t) @ cor_tr(m[-2, 1], ht=hm2p1_t, hr=hm2p1_r))
            hp2p1_b, hp2p0_b = cut_into_hairs(cor_br(m[3, 1]) @ cor_bl(m[3, 0]))
            hp1p1_b, hp1p0_b = cut_into_hairs(cor_br(m[2, 1], hb=hp2p1_b, hr=hp2p1_r) @ cor_bl(m[2, 0], hl=hp2p0_l, hb=hp2p0_b))

            ctl = cor_tl(m[-1, 0], ht=hm1p0_t, hl=hm1p0_l)
            ctr = cor_tr(m[-1, 1], ht=hm1p1_t, hr=hm1p1_r)
            cbr = cor_br(m[1, 1], hb=hp1p1_b, hr=hp1p1_r)
            cbl = cor_bl(m[1, 0], hb=hp1p0_b, hl=hp1p0_l)

            g = tensordot((cbr @ cbl) @ env_l, (ctl @ ctr) @ env_r, axes=((0, 2), (2, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            sts = [(-3,-0), (-2,-1), (-2,0), (-2,1),
                   (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2),
                   (0,-3), (0,-2), (0,-1), (0,1), (0,2), (0,3),
                   (1,-3), (1,-2), (1,-1), (1,1), (1,2), (1,3),
                   (2,-2), (2,-1), (2,0), (2,1), (2,2),
                   (3,-1), (3,0), (3,1), (4,0)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            hp0m2_t, hm1m1_l = cut_into_hairs(cor_tl(m[-1, -2]))
            hm1m1_t, hm2p0_l = cut_into_hairs(cor_tl(m[-2, -1]))
            hp0m1_t, hm1p0_l = cut_into_hairs(cor_tl(m[-1, -1], hl=hm1m1_l, ht=hm1m1_t))

            hm1p1_r, hp0p2_t = cut_into_hairs(cor_tr(m[-1, 2]))
            hm2p0_r, hm1p1_t = cut_into_hairs(cor_tr(m[-2, 1]))
            hm1p0_r, hp0p1_t = cut_into_hairs(cor_tr(m[-1, 1], ht=hm1p1_t, hr=hm1p1_r))

            hp1p2_b, hp2p1_r = cut_into_hairs(cor_br(m[2, 2]))
            hp2p1_b, hp3p0_r = cut_into_hairs(cor_br(m[3, 1]))
            hp1p1_b, hp2p0_r = cut_into_hairs(cor_br(m[2, 1], hr=hp2p1_r, hb=hp2p1_b))

            hp2m1_l, hp1m2_b = cut_into_hairs(cor_bl(m[2, -2]))
            hp3p0_l, hp2m1_b = cut_into_hairs(cor_bl(m[3, -1]))
            hp2p0_l, hp1m1_b = cut_into_hairs(cor_bl(m[2, -1], hb=hp2m1_b, hl=hp2m1_l))

            hm1p0_t = hair_t(m[-2, 0], ht=hair_t(m[-3, 0]), hl=hm2p0_l, hr=hm2p0_r)
            hp0p0_t = hair_t(m[-1, 0], ht=hm1p0_t, hl=hm1p0_l, hr=hm1p0_r)
            env_t = edge_t(Q0, hp0p0_t, A_ket=Q0_ket)  # [lt lt'] [bb bb'] [rt rt']

            hp2p0_b = hair_b(m[3, 0], hb=hair_b(m[4, 0]), hl=hp3p0_l, hr=hp3p0_r)
            hp1p0_b = hair_b(m[2, 0], hb=hp2p0_b, hl=hp2p0_l, hr=hp2p0_r)
            env_b = edge_b(Q1, hp1p0_b, A_ket=Q1_ket)  # [rb rb'] [tt tt'] [lb lb']

            hp1m2_l, hp0m2_l = cut_into_hairs(cor_bl(m[1, -3]) @ cor_tl(m[0, -3]))
            hp1m1_l, hp0m1_l = cut_into_hairs(cor_bl(m[1, -2], hb=hp1m2_b, hl=hp1m2_l) @ cor_tl(m[0, -2], hl=hp0m2_l, ht=hp0m2_t))
            hp0p2_r, hp1p2_r = cut_into_hairs(cor_tr(m[0, 3]) @ cor_br(m[1, 3]))
            hp0p1_r, hp1p1_r = cut_into_hairs(cor_tr(m[0, 2], ht=hp0p2_t, hr=hp0p2_r) @ cor_br(m[1, 2], hb=hp1p2_b, hr=hp1p2_r))

            cbl = cor_bl(m[1,-1], hb=hp1m1_b, hl=hp1m1_l)
            ctl = cor_tl(m[0,-1], ht=hp0m1_t, hl=hp0m1_l)
            ctr = cor_tr(m[0, 1], ht=hp0p1_t, hr=hp0p1_r)
            cbr = cor_br(m[1, 1], hb=hp1p1_b, hr=hp1p1_r)

            g = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))


    def _g_NNN(self, Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket):
        r"""
        Calculates the metric tensor within "NTU-NNN" approximation.

        For dirn == 'h':

        (-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)
           ║       ║        ║       ║
        (+0 -1)════Q0══   ══Q1═══(+0 +2)
           ║       ║        ║       ║
        (+1 -1)=(+1 +0)══(+1 +1)=(+1 +2)

        For dirn == 'v':

        (-1 -1)=(-1 +0)=(-1 +1)
           ║       ║       ║
        (+0 -1)═══0Q0═══(+0 +1)
           ║       ╳       ║
        (+1 -1)═══0Q1═══(+1 +1)
           ║       ║       ║
        (+2 -1)=(+2 +0)=(+2 +1)
        """
        if dirn in ("h", "lr"):
            assert self.psi.nn_site(s0, (0, 1)) == s1
            sts = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (1,2), (0,2), (-1,2), (-1,1), (-1,0)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            ell = edge_l(m[0, -1])
            clt = cor_tl(m[-1, -1])
            elt = edge_t(m[-1, 0])
            elb = edge_b(m[1, 0])
            clb = cor_bl(m[1, -1])
            vecl = _append_vec_nnn(Q0_ket, Q0, ell @ (clt @ elt),
                                   elb @ clb, append_vec_tl, vertical=False)

            err = edge_r(m[0, 2])
            crb = cor_br(m[1, 2])
            erb = edge_b(m[1, 1])
            ert = edge_t(m[-1, 1])
            crt = cor_tr(m[-1, 2])
            vecr = _append_vec_nnn(Q1_ket, Q1, err @ (crb @ erb),
                                   ert @ crt, append_vec_br, vertical=False)
            g = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            sts = [(-1,-1), (0,-1), (1,-1), (2,-1), (2,0), (2,1), (1,1), (0,1), (-1,1), (-1,0)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            etl = edge_l(m[0, -1])
            ctl = cor_tl(m[-1, -1])
            ett = edge_t(m[-1, 0])
            ctr = cor_tr(m[-1, 1])
            etr = edge_r(m[0, 1])
            vect = _append_vec_nnn(Q0_ket, Q0, etl @ (ctl @ ett),
                                   ctr @ etr, append_vec_tl, vertical=True)

            ebr = edge_r(m[1, 1])
            cbr = cor_br(m[2, 1])
            ebb = edge_b(m[2, 0])
            cbl = cor_bl(m[2, -1])
            ebl = edge_l(m[1, -1])
            vecb = _append_vec_nnn(Q1_ket, Q1, ebr @ (cbr @ ebb),
                                   cbl @ ebl, append_vec_br, vertical=True)
            g = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))


    def _g_NNNp(self, Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket):
        r"""
        Calculates the metric tensor within "NTU-NNN+" approximation.

        For dirn == 'h':

                (-2 -1) (-2 +0)  (-2 +1) (-2 +2)
                   ║       ║        ║       ║
        (-1 -2)=(-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)═(-1 +3)
                   ║       ║        ║       ║
        (+0 -2)=(+0 -1)════Q0══   ══Q1═══(+0 +2)═(+0 +3)
                   ║       ║        ║       ║
        (+1 -2)=(+1 -1)=(+1 +0)══(+1 +1)═(+1 +2)═(+1 +3)
                   ║       ║        ║       ║
                (+2 -1) (+2 +0)  (+2 +1) (+2 +2)

        For dirn == 'v':

                (-2 -1) (-2 +0) (-2 +1)
                   ║       ║       ║
        (-1 -2)=(-1 -1)=(-1 +0)=(-1 +1)=(-1 +2)
                   ║       ║       ║
        (+0 -2)=(+0 -1)═══0Q0═══(+0 +1)=(+0 +2)
                   ║       ╳       ║
        (+1 -2)=(+1 -1)═══1Q1═══(+1 +1)=(+1 +2)
                   ║       ║       ║
        (+2 -2)=(+2 -1)=(+2 +0)=(+2 +1)=(+2 +2)
                   ║       ║       ║
                (+3 -1) (+3 +0) (+3 +1)
        """
        if dirn in ("h", "lr"):
            assert self.psi.nn_site(s0, (0, 1)) == s1
            sts = [(-2,-1), (-2,0), (-2,1), (-2,2),
                   (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2), (-1,3),
                   (0,-2), (0,-1), (0,2), (0,3),
                   (1,-2), (1,-1), (1,0), (1,1), (1,2), (1,3),
                   (2,-1), (2,0), (2,1), (2,2)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            ell = edge_l(m[0, -1], hl=hair_l(m[0, -2]))
            clt = cor_tl(m[-1, -1], hl=hair_l(m[-1, -2]), ht=hair_t(m[-2, -1]))
            elt = edge_t(m[-1, 0], ht=hair_t(m[-2, 0]))
            vecl = append_vec_tl(Q0_ket, Q0, ell @ (clt @ elt))
            elb = edge_b(m[1, 0], hb=hair_b(m[2, 0]))
            clb = cor_bl(m[1, -1], hb=hair_b(m[2, -1]), hl=hair_l(m[1, -2]))
            vecl = tensordot(elb @ clb, vecl, axes=((2, 1), (0, 1)))

            err = edge_r(m[0, 2], hr=hair_r(m[0, 3]))
            crb = cor_br(m[1, 2], hr=hair_r(m[1, 3]), hb=hair_b(m[2, 2]))
            erb = edge_b(m[1, 1], hb=hair_b(m[2, 1]))
            vecr = append_vec_br(Q1_ket, Q1, err @ (crb @ erb))
            ert = edge_t(m[-1, 1], ht=hair_t(m[-2, 1]))
            crt = cor_tr(m[-1, 2], ht=hair_t(m[-2, 2]), hr=hair_r(m[-1, 3]))
            vecr = tensordot(ert @ crt, vecr, axes=((2, 1), (0, 1)))

            g = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            sts =  [(-2,-1), (-2,0), (-2,1), (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2),
                    (0,-2), (0,-1), (0,1), (0,2), (1,-2), (1,-1), (1,1), (1,2),
                    (2,-2), (2,-1), (2,0), (2,1), (2,2), (3,-1), (3,0), (3,1)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            etl = edge_l(m[0, -1], hl=hair_l(m[0, -2]))
            ctl = cor_tl(m[-1, -1], hl=hair_l(m[-1, -2]), ht=hair_t(m[-2, -1]))
            ett = edge_t(m[-1, 0], ht=hair_t(m[-2, 0]))
            vect = append_vec_tl(Q0_ket, Q0, etl @ (ctl @ ett))
            ctr = cor_tr(m[-1, 1], ht=hair_t(m[-2, 1]), hr=hair_r(m[-1, 2]))
            etr = edge_r(m[0, 1], hr=hair_r(m[0, 2]))
            vect = tensordot(vect, ctr @ etr, axes=((2, 3), (0, 1)))

            ebr = edge_r(m[1, 1], hr=hair_r(m[1, 2]))
            cbr = cor_br(m[2, 1], hr=hair_r(m[2, 2]), hb=hair_b(m[3, 1]))
            cbb = edge_b(m[2, 0], hb=hair_b(m[3, 0]))
            vecb = append_vec_br(Q1_ket, Q1, ebr @ (cbr @ cbb))
            cbl = cor_bl(m[2, -1], hb=hair_b(m[3, -1]), hl=hair_l(m[2, -2]))
            ebl = edge_l(m[1, -1], hl=hair_l(m[1, -2]))
            vecb = tensordot(vecb, cbl @ ebl, axes=((2, 3), (0, 1)))

            g = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))


    def _g_NNNpp(self, Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket):
        r"""
        Calculates the metric tensor within using "NTU-NNN++" approximation.

        For dirn == 'h':

                (-3 -2)┈(-3 -1)┈(-3 +0)┈┈(-3 +1)┈(-3 +2)┈(-3 +3)
                   ┊       ┊       ┊        ┊       ┊       ┊
        (-2 -3)┈(-2 -2)┈(-2 -1)┈(-2 +0)┈┈(-2 +1)┈(-2 +2)┈(-2 +3)┈(-2 +4)
           ┊       ┊       ┊       ┊        ┊       ┊       ┊       ┊
        (-1 -3)┈(-1 -2)┈(-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)┈(-1 +3)┈(-1 +4)
           ┊       ┊       ║       ║        ║       ║       ┊       ┊
        (+0 -3)=(+0 -2)=(+0 -1)════Q0══   ══Q1═══(+0 +2)=(+0 +3)=(+0 +4)
           ┊       ┊       ║       ║        ║       ║       ┊       ┊
        (+1 -3)┈(+1 -2)┈(+1 -1)=(+1 +0)══(+1 +1)=(+1 +2)┈(+1 +3)┈(+1 +4)
           ┊       ┊       ┊       ┊        ┊       ┊       ┊       ┊
        (+2 -3)┈(+2 -2)┈(+2 -1)┈(+2 +0)┈┈(+2 +1)┈(+2 +2)┈(+2 +3)┈(+2 +4)
                   ┊       ┊       ┊        ┊       ┊       ┊
                (+3 -2)┈(+3 -1)┈(+3 +0)┈┈(+3 +1)┈(+3 +2)┈(+3 +3)

        For dirn == 'v':

                (-3 -2)┈(-3 -1)┈(-3 +0)┈(-3 +1)┈(-3 +2)
                   ┊       ┊       ║       ┊       ┊
        (-2 -3)┈(-2 -2)┈(-2 -1)┈(-2 +0)┈(-2 +1)┈(-2 +2)┈(-2 +3)
           ┊       ┊       ┊       ║       ┊       ┊       ┊
        (-1 -3)┈(-1 -2)┈(-1 -1)=(-1 +0)=(-1 +1)┈(-1 +2)┈(-1 +3)
           ┊       ┊       ║       ║       ║       ┊       ┊
        (+0 -3)┈(+0 -2)┈(+0 -1)═══0Q0═══(+0 +1)┈(+0 +2)┈(+0 +3)
           ┊       ┊       ║       ╳       ║       ┊       ┊
        (+1 -3)┈(+1 -2)┈(+1 -1)═══1Q1═══(+1 +1)┈(+1 +2)┈(+1 +3)
           ┊       ┊       ║       ║       ║       ┊       ┊
        (+2 -3)┈(+2 -2)┈(+2 -1)=(+2 +0)=(+2 +1)┈(+2 +2)┈(+2 +3)
           ┊       ┊       ┊       ║       ┊       ┊       ┊
        (+3 -3)┈(+3 -2)┈(+3 -1)┈(+3 +0)┈(+3 +1)┈(+3 +2)┈(+3 +3)
                   ┊       ┊       ║       ┊       ┊
                (+4 -2)┈(+4 -1)┈(+4 +0)┈(+4 +1)┈(+4 +2)
        """
        if dirn in ("h", "lr"):
            assert self.psi.nn_site(s0, (0, 1)) == s1
            sts = [(-3,-2), (-3,-1), (-3,0), (-3,1), (-3,2), (-3,3),
                   (-2,-3), (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2), (-2,3), (-2,4),
                   (-1,-3), (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2), (-1,3), (-1,4),
                   (0,-3), (0,-2), (0,-1), (0,2), (0,3), (0,4),
                   (1,-3), (1,-2), (1,-1), (1,0), (1,1), (1,2), (1,3), (1,4),
                   (2,-3), (2,-2), (2,-1), (2,0), (2,1), (2,2), (2,3), (2,4),
                   (3,-2), (3,-1), (3,0), (3,1), (3,2), (3,3)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            hm1m3_t, hm2m2_l = cut_into_hairs(cor_tl(m[-2, -3]))
            hm2m2_t, hm3m1_l = cut_into_hairs(cor_tl(m[-3, -2]))
            hm1m2_t, hm2m1_l = cut_into_hairs(cor_tl(m[-2, -2], hl=hm2m2_l, ht=hm2m2_t))
            hm2m1_t, hm3p0_l = cut_into_hairs(cor_tl(m[-3, -1], hl=hm3m1_l))
            hm1m1_t, hm2p0_l = cut_into_hairs(cor_tl(m[-2, -1], hl=hm2m1_l, ht=hm2m1_t))
            hp0m3_t, hm1m2_l = cut_into_hairs(cor_tl(m[-1, -3], ht=hm1m3_t))
            hp0m2_t, hm1m1_l = cut_into_hairs(cor_tl(m[-1, -2], hl=hm1m2_l, ht=hm1m2_t))

            hm3p2_r, hm2p3_t = cut_into_hairs(cor_tr(m[-3, 3]))
            hm2p3_r, hm1p4_t = cut_into_hairs(cor_tr(m[-2, 4]))
            hm2p2_r, hm1p3_t = cut_into_hairs(cor_tr(m[-2, 3], ht=hm2p3_t, hr=hm2p3_r))
            hm3p1_r, hm2p2_t = cut_into_hairs(cor_tr(m[-3, 2], hr=hm3p2_r))
            hm2p1_r, hm1p2_t = cut_into_hairs(cor_tr(m[-2, 2], ht=hm2p2_t, hr=hm2p2_r))
            hm1p3_r, hp0p4_t = cut_into_hairs(cor_tr(m[-1, 4], ht=hm1p4_t))
            hm1p2_r, hp0p3_t = cut_into_hairs(cor_tr(m[-1, 3], ht=hm1p3_t, hr=hm1p3_r))

            hp1p4_b, hp2p3_r = cut_into_hairs(cor_br(m[2, 4]))
            hp2p3_b, hp3p2_r = cut_into_hairs(cor_br(m[3, 3]))
            hp1p3_b, hp2p2_r = cut_into_hairs(cor_br(m[2, 3], hr=hp2p3_r, hb=hp2p3_b))
            hp0p4_b, hp1p3_r = cut_into_hairs(cor_br(m[1, 4], hb=hp1p4_b))
            hp0p3_b, hp1p2_r = cut_into_hairs(cor_br(m[1, 3], hr=hp1p3_r, hb=hp1p3_b))
            hp2p2_b, hp3p1_r = cut_into_hairs(cor_br(m[3, 2], hr=hp3p2_r))
            hp1p2_b, hp2p1_r = cut_into_hairs(cor_br(m[2, 2], hr=hp2p2_r, hb=hp2p2_b))

            hp2m2_l, hp1m3_b = cut_into_hairs(cor_bl(m[2, -3]))
            hp3m1_l, hp2m2_b = cut_into_hairs(cor_bl(m[3, -2]))
            hp2m1_l, hp1m2_b = cut_into_hairs(cor_bl(m[2, -2], hb=hp2m2_b, hl=hp2m2_l))
            hp1m2_l, hp0m3_b = cut_into_hairs(cor_bl(m[1, -3], hb=hp1m3_b))
            hp1m1_l, hp0m2_b = cut_into_hairs(cor_bl(m[1, -2], hb=hp1m2_b, hl=hp1m2_l))
            hp3p0_l, hp2m1_b = cut_into_hairs(cor_bl(m[3, -1], hl=hp3m1_l))
            hp2p0_l, hp1m1_b = cut_into_hairs(cor_bl(m[2, -1], hb=hp2m1_b, hl=hp2m1_l))

            hp0m2_l = hair_l(m[0, -3], ht=hp0m3_t, hb=hp0m3_b)
            hp0m1_l = hair_l(m[0, -2], ht=hp0m2_t, hb=hp0m2_b, hl=hp0m2_l)
            hp0p3_r = hair_r(m[0, 4], ht=hp0p4_t, hb=hp0p4_b)
            hp0p2_r = hair_r(m[0, 3], hr=hp0p3_r, ht=hp0p3_t, hb=hp0p3_b)

            hm2p0_t, hm2p1_t = cut_into_hairs(cor_tl(m[-3, 0], hl=hm3p0_l) @ cor_tr(m[-3, 1], hr=hm3p1_r))
            hm1p0_t, hm1p1_t = cut_into_hairs(cor_tl(m[-2, 0], hl=hm2p0_l, ht=hm2p0_t) @ cor_tr(m[-2, 1], ht=hm2p1_t, hr=hm2p1_r))
            hp2p1_b, hp2p0_b = cut_into_hairs(cor_br(m[3, 1], hr=hp3p1_r) @ cor_bl(m[3, 0], hl=hp3p0_l))
            hp1p1_b, hp1p0_b = cut_into_hairs(cor_br(m[2, 1], hb=hp2p1_b, hr=hp2p1_r) @ cor_bl(m[2, 0], hl=hp2p0_l, hb=hp2p0_b))

            ell = edge_l(m[0, -1], hl=hp0m1_l)
            clt = cor_tl(m[-1, -1], hl=hm1m1_l, ht=hm1m1_t)
            elt = edge_t(m[-1, 0], ht=hm1p0_t)
            vecl = append_vec_tl(Q0_ket, Q0, ell @ (clt @ elt))
            elb = edge_b(m[1, 0], hb=hp1p0_b)
            clb = cor_bl(m[1, -1], hb=hp1m1_b, hl=hp1m1_l)
            vecl = tensordot(elb @ clb, vecl, axes=((2, 1), (0, 1)))

            err = edge_r(m[0, 2], hr=hp0p2_r)
            crb = cor_br(m[1, 2], hr=hp1p2_r, hb=hp1p2_b)
            erb = edge_b(m[1, 1], hb=hp1p1_b)
            vecr = append_vec_br(Q1_ket, Q1, err @ (crb @ erb))
            ert = edge_t(m[-1, 1], ht=hm1p1_t)
            crt = cor_tr(m[-1, 2], ht=hm1p2_t, hr=hm1p2_r)
            vecr = tensordot(ert @ crt, vecr, axes=((2, 1), (0, 1)))

            g = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            sts = [(-3,-2), (-3,-1), (-3,0), (-3,1), (-3,2),
                   (-2,-3), (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2), (-2,3),
                   (-1,-3), (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2), (-1,3),
                   (0,-3), (0,-2), (0,-1), (0,1), (0,2), (0,3),
                   (1,-3), (1,-2), (1,-1), (1,1), (1,2), (1,3),
                   (2,-3), (2,-2), (2,-1), (2,0), (2,1), (2,2), (2,3),
                   (3,-3), (3,-2), (3,-1), (3,0), (3,1), (3,2), (3,3),
                   (4,-2), (4,-1), (4,0), (4,1), (4,2)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            hm1m3_t, hm2m2_l = cut_into_hairs(cor_tl(m[-2, -3]))
            hm2m2_t, hm3m1_l = cut_into_hairs(cor_tl(m[-3, -2]))
            hm1m2_t, hm2m1_l = cut_into_hairs(cor_tl(m[-2, -2], hl=hm2m2_l, ht=hm2m2_t))
            hm2m1_t, hm3p0_l = cut_into_hairs(cor_tl(m[-3, -1], hl=hm3m1_l))
            hm1m1_t, hm2p0_l = cut_into_hairs(cor_tl(m[-2, -1], hl=hm2m1_l, ht=hm2m1_t))
            hp0m3_t, hm1m2_l = cut_into_hairs(cor_tl(m[-1, -3], ht=hm1m3_t))
            hp0m2_t, hm1m1_l = cut_into_hairs(cor_tl(m[-1, -2], hl=hm1m2_l, ht=hm1m2_t))

            hm3p1_r, hm2p2_t = cut_into_hairs(cor_tr(m[-3, 2]))
            hm2p2_r, hm1p3_t = cut_into_hairs(cor_tr(m[-2, 3]))
            hm2p1_r, hm1p2_t = cut_into_hairs(cor_tr(m[-2, 2], ht=hm2p2_t, hr=hm2p2_r))
            hm3p0_r, hm2p1_t = cut_into_hairs(cor_tr(m[-3, 1], hr=hm3p1_r))
            hm2p0_r, hm1p1_t = cut_into_hairs(cor_tr(m[-2, 1], ht=hm2p1_t, hr=hm2p1_r))
            hm1p2_r, hp0p3_t = cut_into_hairs(cor_tr(m[-1, 3], ht=hm1p3_t))
            hm1p1_r, hp0p2_t = cut_into_hairs(cor_tr(m[-1, 2], ht=hm1p2_t, hr=hm1p2_r))

            hp2p3_b, hp3p2_r = cut_into_hairs(cor_br(m[3, 3]))
            hp3p2_b, hp4p1_r = cut_into_hairs(cor_br(m[4, 2]))
            hp2p2_b, hp3p1_r = cut_into_hairs(cor_br(m[3, 2], hr=hp3p2_r, hb=hp3p2_b))
            hp1p3_b, hp2p2_r = cut_into_hairs(cor_br(m[2, 3], hb=hp2p3_b))
            hp1p2_b, hp2p1_r = cut_into_hairs(cor_br(m[2, 2], hr=hp2p2_r, hb=hp2p2_b))
            hp3p1_b, hp4p0_r = cut_into_hairs(cor_br(m[4, 1], hr=hp4p1_r))
            hp2p1_b, hp3p0_r = cut_into_hairs(cor_br(m[3, 1], hr=hp3p1_r, hb=hp3p1_b))

            hp3m2_l, hp2m3_b = cut_into_hairs(cor_bl(m[3, -3]))
            hp4m1_l, hp3m2_b = cut_into_hairs(cor_bl(m[4, -2]))
            hp3m1_l, hp2m2_b = cut_into_hairs(cor_bl(m[3, -2], hb=hp3m2_b, hl=hp3m2_l))
            hp2m2_l, hp1m3_b = cut_into_hairs(cor_bl(m[2, -3], hb=hp2m3_b))
            hp2m1_l, hp1m2_b = cut_into_hairs(cor_bl(m[2, -2], hb=hp2m2_b, hl=hp2m2_l))
            hp4p0_l, hp3m1_b = cut_into_hairs(cor_bl(m[4, -1], hl=hp4m1_l))
            hp3p0_l, hp2m1_b = cut_into_hairs(cor_bl(m[3, -1], hb=hp3m1_b, hl=hp3m1_l))

            hm2p0_t = hair_t(m[-3, 0], hl=hm3p0_l, hr=hm3p0_r)
            hm1p0_t = hair_t(m[-2, 0], ht=hm2p0_t, hl=hm2p0_l, hr=hm2p0_r)
            hp3p0_b = hair_b(m[4, 0], hl=hp4p0_l, hr=hp4p0_r)
            hp2p0_b = hair_b(m[3, 0], hb=hp3p0_b, hl=hp3p0_l, hr=hp3p0_r)

            hp1m2_l, hp0m2_l = cut_into_hairs(cor_bl(m[1, -3], hb=hp1m3_b) @ cor_tl(m[0, -3], ht=hp0m3_t))
            hp1m1_l, hp0m1_l = cut_into_hairs(cor_bl(m[1, -2], hb=hp1m2_b, hl=hp1m2_l) @ cor_tl(m[0, -2], hl=hp0m2_l, ht=hp0m2_t))
            hp0p2_r, hp1p2_r = cut_into_hairs(cor_tr(m[0, 3], ht=hp0p3_t) @ cor_br(m[1, 3], hb=hp1p3_b))
            hp0p1_r, hp1p1_r = cut_into_hairs(cor_tr(m[0, 2], ht=hp0p2_t, hr=hp0p2_r) @ cor_br(m[1, 2], hb=hp1p2_b, hr=hp1p2_r))

            etl = edge_l(m[0, -1], hl=hp0m1_l)
            ctl = cor_tl(m[-1, -1], hl=hm1m1_l, ht=hm1m1_t)
            ett = edge_t(m[-1, 0], ht=hm1p0_t)
            vect = append_vec_tl(Q0_ket, Q0, etl @ (ctl @ ett))
            ctr = cor_tr(m[-1, 1], ht=hm1p1_t, hr=hm1p1_r)
            etr = edge_r(m[0, 1], hr=hp0p1_r)
            vect = tensordot(vect, ctr @ etr, axes=((2, 3), (0, 1)))

            ebr = edge_r(m[1, 1], hr=hp1p1_r)
            cbr = cor_br(m[2, 1], hr=hp2p1_r, hb=hp2p1_b)
            cbb = edge_b(m[2, 0], hb=hp2p0_b)
            vecb = append_vec_br(Q1_ket, Q1, ebr @ (cbr @ cbb))
            cbl = cor_bl(m[2, -1], hb=hp2m1_b, hl=hp2m1_l)
            ebl = edge_l(m[1, -1], hl=hp1m1_l)
            vecb = tensordot(vecb, cbl @ ebl, axes=((2, 3), (0, 1)))

            g = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))


    def _g_Ladder(self, Q0, Q1, s0, s1, dirn, Q0_ket, Q1_ket):
        r"""
        Contract an N-layer ladder around the updated bond.

        For dirn == 'h':

                (-2 +0)══(-2 +1)
                   ║        ║
                (-1 +0)══(-1 +1)
                   ║        ║
        (+0 -1)════Q0══   ══Q1═══(+0 +2)
                   ║        ║
                (+1 +0)══(+1 +1)
                   ║        ║
                (+2 +0)══(+2 +1)

        For dirn == 'v':

                          (-1 +0)
                             ║
         (+0 -2)══(+0 -1)═══0Q0═══(+0 +1)══(+0 +2)
            ║        ║       ╳       ║        ║
         (+1 -2)══(+1 -1)═══1Q1═══(+1 +1)══(+1 +2)
                             ║
                          (+2 +0)
        """
        digits = ''.join(c for c in self.which if c.isdigit())
        nn = int(digits) if digits else 2
        if nn < 1:
            raise YastnError('NTU Ladder requires at least one environment layer.')

        if dirn in ("h", "lr"):
            assert self.psi.nn_site(s0, (0, 1)) == s1
            sites = [(0, -1), (0, 2)]
            for ii in range(1, nn + 1):
                sites.extend([(-ii, 0), (-ii, 1), (ii, 0), (ii, 1)])

            m = {d: self.psi.nn_site(s0, d=d) for d in sites}
            tensors_from_psi(m, self.psi)

            env_l = edge_l(Q0, hair_l(m[0, -1]), A_ket=Q0_ket)  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(Q1, hair_r(m[0,  2]), A_ket=Q1_ket)  # [tr tr'] [ll ll'] [br br']

            ctl = cor_tl(m[-nn, 0])
            ctr = cor_tr(m[-nn, 1])
            cbr = cor_br(m[ nn, 1])
            cbl = cor_bl(m[ nn, 0])

            env_tl = {ii: edge_l(m[-ii, 0]) for ii in range(1, nn)}
            env_tr = {ii: edge_r(m[-ii, 1]) for ii in range(1, nn)}
            env_br = {ii: edge_r(m[ ii, 1]) for ii in range(1, nn)}
            env_bl = {ii: edge_l(m[ ii, 0]) for ii in range(1, nn)}

            et = ctl @ ctr
            eb = cbr @ cbl
            # Grow the boundary from the outermost corners toward the QR pair.
            for ii in range(nn - 1, 0, -1):
                et = ncon([et, env_tl[ii], env_tr[ii]], [[1, 3], [-0, 2, 1], [3, 2, -1]])
                eb = ncon([eb, env_br[ii], env_bl[ii]], [[1, 3], [-0, 2, 1], [3, 2, -1]])
            g = ncon([eb, env_l, et, env_r], [[3, 2], [2, -0, 1], [1, 4], [4, -1, 3]])

        else: # dirn == "v":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            sites = [(-1, 0), (2, 0)]
            for ii in range(1, nn + 1):
                sites.extend([(0, -ii), (1, -ii), (0, ii), (1, ii)])

            m = {d: self.psi.nn_site(s0, d=d) for d in sites}
            tensors_from_psi(m, self.psi)
            env_t = edge_t(Q0, hair_t(m[-1, 0]), A_ket=Q0_ket)  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(Q1, hair_b(m[ 2, 0]), A_ket=Q1_ket)  # [rb rb'] [tt tt'] [lb lb']

            env_tl = {ii: edge_t(m[0, -ii]) for ii in range(1, nn)}
            env_bl = {ii: edge_b(m[1, -ii]) for ii in range(1, nn)}
            env_tr = {ii: edge_t(m[0,  ii]) for ii in range(1, nn)}
            env_br = {ii: edge_b(m[1,  ii]) for ii in range(1, nn)}
            cbl = cor_bl(m[1, -nn])
            ctl = cor_tl(m[0, -nn])
            ctr = cor_tr(m[0,  nn])
            cbr = cor_br(m[1,  nn])

            el = cbl @ ctl
            er = ctr @ cbr
            for ii in range(nn - 1, 0, -1):
                el = ncon([el, env_bl[ii], env_tl[ii]], [[1, 3], [-0, 2, 1], [3, 2, -1]])
                er = ncon([er, env_tr[ii], env_br[ii]], [[1, 3], [-0, 2, 1], [3, 2, -1]])
            g = ncon([el, env_t, er, env_b], [[4, 1], [1, -0, 2], [2, 3], [3, -1, 4]])

        # ncon can leave the two open double-layer legs in a pending transpose.
        # Materialize their order before exposing the non-Abelian fusion trees.
        g = g.consume_transpose()
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))
