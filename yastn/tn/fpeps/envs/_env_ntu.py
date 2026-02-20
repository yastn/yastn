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
from ._env_contractions import *
from .._evolution import BondMetric
from ....tensor import tensordot, YastnError, ncon


class EnvNTU:
    def __init__(self, psi, which='NN'):
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
            Type of bond environment from 'NN', 'NN+', 'NN++', 'NNN', 'NNN+', 'NNN++'
        """
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
        if self.which == 'NN':
            return self._g_NN(Q0, Q1, s0, s1, dirn)
        if self.which == 'NN+':
            return self._g_NNp(Q0, Q1, s0, s1, dirn)
        if self.which == 'NN++':
            return self._g_NNpp(Q0, Q1, s0, s1, dirn)
        if self.which == 'NNN':
            return self._g_NNN(Q0, Q1, s0, s1, dirn)
        if self.which == 'NNN+':
            return self._g_NNNp(Q0, Q1, s0, s1, dirn)
        if self.which == 'NNN++':
            return self._g_NNNpp(Q0, Q1, s0, s1, dirn)
        if 'ladder' in self.which.lower():
            return self._g_Ladder(Q0, Q1, s0, s1, dirn)
        raise YastnError(f" Type of EnvNTU which={self.which} not recognized.")


    def _g_NN(self, Q0, Q1, s0, s1, dirn):
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
            env_l = edge_l(Q0, hair_l(m[0, -1]))  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(Q1, hair_r(m[0,  2]))  # [tr tr'] [ll ll'] [br br']
            ctl = cor_tl(m[-1, 0])
            ctr = cor_tr(m[-1, 1])
            cbr = cor_br(m[ 1, 1])
            cbl = cor_bl(m[ 1, 0])
            g = tensordot((cbr @ cbl) @ env_l, (ctl @ ctr) @ env_r, axes=((0, 2), (2, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            m = {d: self.psi.nn_site(s0, d=d) for d in [(-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,1)]}
            tensors_from_psi(m, self.psi)
            env_t = edge_t(Q0, hair_t(m[-1, 0]))  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(Q1, hair_b(m[ 2, 0]))  # [rb rb'] [tt tt'] [lb lb']
            cbl = cor_bl(m[1,-1])
            ctl = cor_tl(m[0,-1])
            ctr = cor_tr(m[0, 1])
            cbr = cor_br(m[1, 1])
            g = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))


    def _g_NNp(self, Q0, Q1, s0, s1, dirn):
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

            env_l = edge_l(Q0, hair_l(m[0,-1], hl=hair_l(m[0,-2]), ht=htl_t, hb=hbl_b))  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(Q1, hair_r(m[0, 2], hr=hair_r(m[0, 3]), ht=htr_t, hb=hbr_b))  # [tr tr'] [ll ll'] [br br']

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

            env_t = edge_t(Q0, hair_t(m[-1, 0], ht=hair_t(m[-2, 0]), hl=htl_l, hr=htr_r))  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(Q1, hair_b(m[ 2, 0], hb=hair_b(m[ 3, 0]), hl=hbl_l, hr=hbr_r))  # [rb rb'] [tt tt'] [lb lb']

            hp1m1_l, hp0m1_l = cut_into_hairs(cor_bl(m[1, -2]) @ cor_tl(m[0, -2]))
            hp0p1_r, hp1p1_r = cut_into_hairs(cor_tr(m[0, 2]) @ cor_br(m[1, 2]))

            cbl = cor_bl(m[1,-1], hl=hp1m1_l, hb=hbl_b)
            ctl = cor_tl(m[0,-1], hl=hp0m1_l, ht=htl_t)
            ctr = cor_tr(m[0, 1], hr=hp0p1_r, ht=htr_t)
            cbr = cor_br(m[1, 1], hr=hp1p1_r, hb=hbr_b)
            g = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))


    def _g_NNpp(self, Q0, Q1, s0, s1, dirn):
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
            env_l = edge_l(Q0, hp0m0_l)  # [bl bl'] [rr rr'] [tl tl']

            hp0p2_r = hair_r(m[0, 3], hr=hair_r(m[0, 4]), ht=hp0p3_t, hb=hp0p3_b)
            hp0p1_r = hair_r(m[0, 2], hr=hp0p2_r, ht=hp0p2_t, hb=hp0p2_b)
            env_r = edge_r(Q1, hp0p1_r)  # [tr tr'] [ll ll'] [br br']

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
            env_t = edge_t(Q0, hp0p0_t)  # [lt lt'] [bb bb'] [rt rt']

            hp2p0_b = hair_b(m[3, 0], hb=hair_b(m[4, 0]), hl=hp3p0_l, hr=hp3p0_r)
            hp1p0_b = hair_b(m[2, 0], hb=hp2p0_b, hl=hp2p0_l, hr=hp2p0_r)
            env_b = edge_b(Q1, hp1p0_b)  # [rb rb'] [tt tt'] [lb lb']

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


    def _g_NNN(self, Q0, Q1, s0, s1, dirn):
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
            vecl = append_vec_tl(Q0, Q0, ell @ (clt @ elt))
            elb = edge_b(m[1, 0])
            clb = cor_bl(m[1, -1])
            vecl = tensordot(elb @ clb, vecl, axes=((2, 1), (0, 1)))

            err = edge_r(m[0, 2])
            crb = cor_br(m[1, 2])
            erb = edge_b(m[1, 1])
            vecr = append_vec_br(Q1, Q1, err @ (crb @ erb))
            ert = edge_t(m[-1, 1])
            crt = cor_tr(m[-1, 2])
            vecr = tensordot(ert @ crt, vecr, axes=((2, 1), (0, 1)))
            g = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            sts = [(-1,-1), (0,-1), (1,-1), (2,-1), (2,0), (2,1), (1,1), (0,1), (-1,1), (-1,0)]
            m = {d: self.psi.nn_site(s0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            etl = edge_l(m[0, -1])
            ctl = cor_tl(m[-1, -1])
            ett = edge_t(m[-1, 0])
            vect = append_vec_tl(Q0, Q0, etl @ (ctl @ ett))
            ctr = cor_tr(m[-1, 1])
            etr = edge_r(m[0, 1])
            vect = tensordot(vect, ctr @ etr, axes=((2, 3), (0, 1)))

            ebr = edge_r(m[1, 1])
            cbr = cor_br(m[2, 1])
            ebb = edge_b(m[2, 0])
            vecb = append_vec_br(Q1, Q1, ebr @ (cbr @ ebb))
            cbl = cor_bl(m[2, -1])
            ebl = edge_l(m[1, -1])
            vecb = tensordot(vecb, cbl @ ebl, axes=((2, 3), (0, 1)))
            g = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))


    def _g_NNNp(self, Q0, Q1, s0, s1, dirn):
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
            vecl = append_vec_tl(Q0, Q0, ell @ (clt @ elt))
            elb = edge_b(m[1, 0], hb=hair_b(m[2, 0]))
            clb = cor_bl(m[1, -1], hb=hair_b(m[2, -1]), hl=hair_l(m[1, -2]))
            vecl = tensordot(elb @ clb, vecl, axes=((2, 1), (0, 1)))

            err = edge_r(m[0, 2], hr=hair_r(m[0, 3]))
            crb = cor_br(m[1, 2], hr=hair_r(m[1, 3]), hb=hair_b(m[2, 2]))
            erb = edge_b(m[1, 1], hb=hair_b(m[2, 1]))
            vecr = append_vec_br(Q1, Q1, err @ (crb @ erb))
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
            vect = append_vec_tl(Q0, Q0, etl @ (ctl @ ett))
            ctr = cor_tr(m[-1, 1], ht=hair_t(m[-2, 1]), hr=hair_r(m[-1, 2]))
            etr = edge_r(m[0, 1], hr=hair_r(m[0, 2]))
            vect = tensordot(vect, ctr @ etr, axes=((2, 3), (0, 1)))

            ebr = edge_r(m[1, 1], hr=hair_r(m[1, 2]))
            cbr = cor_br(m[2, 1], hr=hair_r(m[2, 2]), hb=hair_b(m[3, 1]))
            cbb = edge_b(m[2, 0], hb=hair_b(m[3, 0]))
            vecb = append_vec_br(Q1, Q1, ebr @ (cbr @ cbb))
            cbl = cor_bl(m[2, -1], hb=hair_b(m[3, -1]), hl=hair_l(m[2, -2]))
            ebl = edge_l(m[1, -1], hl=hair_l(m[1, -2]))
            vecb = tensordot(vecb, cbl @ ebl, axes=((2, 3), (0, 1)))

            g = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))


    def _g_NNNpp(self, Q0, Q1, s0, s1, dirn):
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
            vecl = append_vec_tl(Q0, Q0, ell @ (clt @ elt))
            elb = edge_b(m[1, 0], hb=hp1p0_b)
            clb = cor_bl(m[1, -1], hb=hp1m1_b, hl=hp1m1_l)
            vecl = tensordot(elb @ clb, vecl, axes=((2, 1), (0, 1)))

            err = edge_r(m[0, 2], hr=hp0p2_r)
            crb = cor_br(m[1, 2], hr=hp1p2_r, hb=hp1p2_b)
            erb = edge_b(m[1, 1], hb=hp1p1_b)
            vecr = append_vec_br(Q1, Q1, err @ (crb @ erb))
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
            vect = append_vec_tl(Q0, Q0, etl @ (ctl @ ett))
            ctr = cor_tr(m[-1, 1], ht=hm1p1_t, hr=hm1p1_r)
            etr = edge_r(m[0, 1], hr=hp0p1_r)
            vect = tensordot(vect, ctr @ etr, axes=((2, 3), (0, 1)))

            ebr = edge_r(m[1, 1], hr=hp1p1_r)
            cbr = cor_br(m[2, 1], hr=hp2p1_r, hb=hp2p1_b)
            cbb = edge_b(m[2, 0], hb=hp2p0_b)
            vecb = append_vec_br(Q1, Q1, ebr @ (cbr @ cbb))
            cbl = cor_bl(m[2, -1], hb=hp2m1_b, hl=hp2m1_l)
            ebl = edge_l(m[1, -1], hl=hp1m1_l)
            vecb = tensordot(vecb, cbl @ ebl, axes=((2, 3), (0, 1)))

            g = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))


    def _g_Ladder(self, Q0, Q1, s0, s1, dirn):
        r"""
        Calculates metric tensor within "NTU-NN" approximation.

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

        if dirn in ("h", "lr"):
            assert self.psi.nn_site(s0, (0, 1)) == s1
            sites = [(0, -1), (0, 2)]
            for ii in range(1, nn + 1):
                sites.extend([(-ii, 0), (-ii, 1), (ii, 0), (ii, 1)])

            m = {d: self.psi.nn_site(s0, d=d) for d in sites}
            tensors_from_psi(m, self.psi)

            env_l = edge_l(Q0, hair_l(m[0, -1]))  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(Q1, hair_r(m[0,  2]))  # [tr tr'] [ll ll'] [br br']

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
            for ii in range(1, nn):
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
            env_t = edge_t(Q0, hair_t(m[-1, 0]))  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(Q1, hair_b(m[ 2, 0]))  # [rb rb'] [tt tt'] [lb lb']

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
            for ii in range(1, nn):
                el = ncon([el, env_bl[ii], env_tl[ii]], [[1, 3], [-0, 2, 1], [3, 2, -1]])
                er = ncon([er, env_tr[ii], env_br[ii]], [[1, 3], [-0, 2, 1], [3, 2, -1]])
            g = ncon([el, env_t, er, env_b], [[4, 1], [1, -0, 2], [2, 3], [3, -1, 4]])

        return BondMetric(g=g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2))))
