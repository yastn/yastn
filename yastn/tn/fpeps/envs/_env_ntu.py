from .... import tensordot, YastnError
from ._env_auxlliary import *


class EnvNTU:
    def __init__(self, psi, which='NN'):
        """
        NTU environments for truncation of PEPS tensors during time evolution.

        Parameters
        ----------
        psi: yastn.tn.Peps
            Evolved state

        which: str
            Type of environment from 'NN', 'NN+', 'NN++', 'NNN', 'NNN+', 'NNN++'
        """

        if which not in ('NN', 'NN+', 'NN++', 'NNN', 'NNN+', 'NNN++'):  # 'SU+'
            raise YastnError(f" Type of EnvNTU {which} not recognized.")
        self.psi = psi
        self.which = which
        self._dict_gs = {'NN': self._g_NN,
                         'NN+': self._g_NNp,
                         'NN++': self._g_NNpp,
                         'NNN': self._g_NNN,
                         'NNN+': self._g_NNNp,
                         'NNN++': self._g_NNNpp
                        }


    def bond_metric(self, bd, QA, QB):
        """
        Calculates bond metric.

        Double lines indicate a core tensors that are contracted exactly.
        Single lines are tree-like approximation of border tensors resulting from their rank-one SVD decomposition.

        If which == 'NN':
                 (-1, +0)==(-1, +1)
                    ||        ||
        (+0, -1) == QA++    ++QB == (+0, +2)
                    ||        ||
                 (+1, +0)==(+1, +1)

        If which == 'NN+':
                          (-2, +0)  (-2, +1)
                             ||        ||
                 (-1, -1)-(-1, +0)==(-1, +1)-(-1, +2)
                     :       ||        ||       :
        (+0, -2)=(+0, -1) == QA++    ++QB == (+0, +2)=(+0, +3)
                     :       ||        ||       :
                 (+1, -1)-(+1, +0)==(+1, +1)-(+1, +2)
                             ||        ||
                          (+2, +0)  (+2, +1)

        If which == 'NN++':
                                   (-3, +0)--(-3, +1)
                                      :         :
                          (-2, -1)-(-2, +0)--(-2, +1)-(-2, +2)
                              :       :         :        :
                 (-1, -2)-(-1, -1)-(-1, +0)==(-1, +1)-(-1, +2)-(-1, +3)
                     :        :       ||        ||       :        :
        (+0, -3)=(+0, -2)=(+0, -1) == QA++    ++QB == (+0, +2)=(+0, +3)=(+0, +4)
                     :        :       ||        ||       :        :
                 (+1, -2)-(+1, -1)-(+1, +0)==(+1, +1)-(+1, +2)-(+1, +3)
                              :       :         :        :
                          (+2, -1)-(+2, +0)--(+2, +1)-(+2, +2)
                                      :         :
                                   (+3, +0)--(+3, +1)

        If which == 'NNN':
        (-1, -1)=(-1, +0)==(-1, +1)=(-1, +2)
           ||       ||        ||       ||
        (+0, -1) == GA++    ++GB == (+0, +2)
           ||       ||        ||       ||
        (+1, -1)=(+1, +0)==(+1, +1)=(+1, +2)

        If which == 'NNN+':
                 (-2, -1) (-2, +0)  (-2, +1) (-2, +2)
                    ||       ||        ||       ||
        (-1, -2)=(-1, -1)=(-1, +0)==(-1, +1)=(-1, +2)=(-1, +3)
                    ||       ||        ||       ||
        (+0, -2)=(+0, -1) == GA++    ++GB == (+0, +2)=(+0, +3)
                    ||       ||        ||       ||
        (+1, -2)=(+1, -1)=(+1, +0)==(+1, +1)=(+1, +2)=(+1, +3)
                    ||       ||        ||       ||
                 (+2, -1) (+2, +0)  (+2, +1) (+2, +2)

        If which == 'NNN++':
                 (-3, -2)-(-3, -1)-(-3, +0)--(-3, +1)-(-3, +2)-(-3, +3)
                    :        :        :         :        :        :
        (-2, -3)-(-2, -2)-(-2, -1)-(-2, +0)--(-2, +1)-(-2, +2)-(-2, +3)-(-2, +4)
           :        :        :        :         :        :        :        :
        (-1, -3)-(-1, -2)-(-1, -1)=(-1, +0)==(-1, +1)=(-1, +2)-(-1, +3)-(-1, +4)
           :        :        ||       ||        ||       ||       :        :
        (+0, -3)=(+0, -2)=(+0, -1) == GA++    ++GB == (+0, +2)=(+0, +3)=(+0, +4)
           :        :        ||       ||        ||       ||       :        :
        (+1, -3)-(+1, -2)-(+1, -1)=(+1, +0)==(+1, +1)=(+1, +2)-(+1, +3)-(+1, +4)
           :        :        :        :         :        :        :        :
        (+2, -3)-(+2, -2)-(+2, -1)-(+2, +0)--(+2, +1)-(+2, +2)-(+2, +3)-(+2, +4)
                    :        :        :         :        :        :
                 (+3, -2)-(+3, -1)-(+3, +0)--(+3, +1)-(+3, +2)-(+3, +3)

        """
        return self._dict_gs[self.which](bd, QA, QB)


    def _g_NN(self, bd, QA, QB):
        """
        Calculates metric tensor within "NTU-NN" approximation.

        For bd.dirn == 'h':

                 (-1, +0)==(-1, +1)
                    ||        ||
        (+0, -1) == QA++    ++QB == (+0, +2)
                    ||        ||
                 (+1, +0)==(+1, +1)

        For bd.dirn == 'v':

                 (-1, +0)
                    ||
        (+0, -1) == QA == (+0, +1)
           ||       ++       ||
        (+1, -1) == QB == (+1, +1)
                    ||
                 (+2, +0)
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1,0), (0,-1), (1,0), (1,1), (0,2), (-1,1)]}
            tensors_from_psi(m, self.psi)
            env_l = edge_l(QA, hair_l(m[0, -1]))  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(QB, hair_r(m[0,  2]))  # [tr tr'] [ll ll'] [br br']
            ctl = cor_tl(m[-1, 0])
            ctr = cor_tr(m[-1, 1])
            cbr = cor_br(m[ 1, 1])
            cbl = cor_bl(m[ 1, 0])
            g = tensordot((cbr @ cbl) @ env_l, (ctl @ ctr) @ env_r, axes=((0, 2), (2, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,1)]}
            tensors_from_psi(m, self.psi)
            env_t = edge_t(QA, hair_t(m[-1, 0]))  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(QB, hair_b(m[ 2, 0]))  # [rb rb'] [tt tt'] [lb lb']
            cbl = cor_bl(m[1,-1])
            ctl = cor_tl(m[0,-1])
            ctr = cor_tr(m[0, 1])
            cbr = cor_br(m[1, 1])
            g = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))


    def _g_NNp(self, bd, QA, QB):
        """
        Calculates the metric tensor within "NTU-NN+" approximation.

        For bd.dirn == 'h':

                          (-2, +0)  (-2, +1)
                             ||        ||
                 (-1, -1)-(-1, +0)==(-1, +1)-(-1, +2)
                     :       ||        ||       :
        (+0, -2)=(+0, -1) == QA++    ++QB == (+0, +2)=(+0, +3)
                     :       ||        ||       :
                 (+1, -1)-(+1, +0)==(+1, +1)-(+1, +2)
                             ||        ||
                          (+2, +0)  (+2, +1)

        For bd.dirn == 'v':

                          (-2, +0)
                             ||
                 (-1, -1)-(-1, +0)-(-1, +1)
                     :       ||       :
        (+0, -2)=(+0, -1) == GA == (+0, +1)=(+0, +2)
                    ||       ++       ||
        (+1, -2)=(+1, -1) == GB == (+1, +1)=(+1, +2)
                     :       ||       :
                 (+2, -1)-(+2, +0)-(+2, +1)
                             ||
                          (+3, +0)
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            sts = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (1,2), (0,2), (-1,2),
                   (-1,1), (-1,0), (0,-2), (2,0), (2,1), (0,3), (-2,1), (-2,0)]
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)
            htl_t, htl_l = cut_into_hairs(cor_tl(m[-1,-1]))
            htr_r, htr_t = cut_into_hairs(cor_tr(m[-1, 2]))
            hbr_b, hbr_r = cut_into_hairs(cor_br(m[1, 2]))
            hbl_l, hbl_b = cut_into_hairs(cor_bl(m[1, -1]))

            env_l = edge_l(QA, hair_l(m[0,-1], hl=hair_l(m[0,-2]), ht=htl_t, hb=hbl_b))  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(QB, hair_r(m[0, 2], hr=hair_r(m[0, 3]), ht=htr_t, hb=hbr_b))  # [tr tr'] [ll ll'] [br br']

            ctl = cor_tl(m[-1, 0], ht=hair_t(m[-2, 0]), hl=htl_l)
            ctr = cor_tr(m[-1, 1], ht=hair_t(m[-2, 1]), hr=htr_r)
            cbr = cor_br(m[ 1, 1], hb=hair_b(m[ 2, 1]), hr=hbr_r)
            cbl = cor_bl(m[ 1, 0], hb=hair_b(m[ 2, 0]), hl=hbl_l)
            g = tensordot((cbr @ cbl) @ env_l, (ctl @ ctr) @ env_r, axes=((0, 2), (2, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            sts = [(-1,-1), (0,-1), (1,-1), (2,-1), (2,0), (2,1), (1,1), (0,1),
                   (-1,1), (-1,0), (0,-2), (1,-2), (3,0), (1,2), (0,2), (-2,0)]
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            htl_t, htl_l = cut_into_hairs(cor_tl(m[-1,-1]))
            htr_r, htr_t = cut_into_hairs(cor_tr(m[-1, 1]))
            hbr_b, hbr_r = cut_into_hairs(cor_br(m[2, 1]))
            hbl_l, hbl_b = cut_into_hairs(cor_bl(m[2, -1]))

            env_t = edge_t(QA, hair_t(m[-1, 0], ht=hair_t(m[-2, 0]), hl=htl_l, hr=htr_r))  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(QB, hair_b(m[ 2, 0], hb=hair_b(m[ 3, 0]), hl=hbl_l, hr=hbr_r))  # [rb rb'] [tt tt'] [lb lb']
            cbl = cor_bl(m[1,-1], hl=hair_l(m[1,-2]), hb=hbl_b)
            ctl = cor_tl(m[0,-1], hl=hair_l(m[0,-2]), ht=htl_t)
            ctr = cor_tr(m[0, 1], hr=hair_r(m[0, 2]), ht=htr_t)
            cbr = cor_br(m[1, 1], hr=hair_r(m[1, 2]), hb=hbr_b)
            g = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))


    def _g_NNpp(self, bd, QA, QB):
        """
        Calculates the metric tensor within "NTU-NN++" approximation.

        For bd.dirn == 'h':

                                   (-3, +0)--(-3, +1)
                                      :         :
                          (-2, -1)-(-2, +0)--(-2, +1)-(-2, +2)
                              :       :         :        :
                 (-1, -2)-(-1, -1)-(-1, +0)==(-1, +1)-(-1, +2)-(-1, +3)
                     :        :       ||        ||       :        :
        (+0, -3)=(+0, -2)=(+0, -1) == QA++    ++QB == (+0, +2)=(+0, +3)=(+0, +4)
                     :        :       ||        ||       :        :
                 (+1, -2)-(+1, -1)-(+1, +0)==(+1, +1)-(+1, +2)-(+1, +3)
                              :       :         :        :
                          (+2, -1)-(+2, +0)--(+2, +1)-(+2, +2)
                                      :         :
                                   (+3, +0)--(+3, +1)

        For bd.dirn == 'v':

                                   (-3, +0)
                                      ||
                          (-2, -1)-(-2, +0)-(-2, +1)
                             :        ||       :
                 (-1, -2)-(-1, -1)-(-1, +0)-(-1, +1)-(-1, +2)
                    :        :        ||       :        :
        (+0, -3)-(+0, -2)-(+0, -1) == GA == (+0, +1)-(+0, +2)-(+0, +3)
           :        :        ||       ++       ||       :        :
        (+1, -3)-(+1, -2)-(+1, -1) == GB == (+1, +1)-(+1, +2)-(+1, +3)
                    :        :        ||       :        :
                 (+2, -2)-(+2, -1)-(+2, +0)-(+2, +1)-(+2, +2)
                             :        ||       :
                          (+3, -1)-(+3, +0)-(+3, +1)
                                      ||
                                   (+4, +0)
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            sts =  [(-3,0), (-3,1), (-2,-1), (-2,0), (-2,1), (-2,2),
                    (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2), (-1,3),
                    (0,-3), (0,-2), (0,-1), (0,2), (0,3), (0,4),
                    (1,-2), (1,-1), (1,0), (1,1), (1,2), (1,3),
                    (2,-1), (2,0), (2,1), (2,2), (3,0), (3,1)]
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in sts}
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

            hp0m1_l = hair_l(m[0, -2], hl=hair_l(m[0, -2]), ht=hp0m2_t, hb=hp0m2_b)
            hp0m0_l = hair_l(m[0, -1], hl=hp0m1_l, ht=hp0m1_t, hb=hp0m1_b)
            env_l = edge_l(QA, hp0m0_l)  # [bl bl'] [rr rr'] [tl tl']

            hp0p2_r = hair_r(m[0, 3], hr=hair_r(m[0, 4]), ht=hp0p3_t, hb=hp0p3_b)
            hp0p1_r = hair_r(m[0, 2], hr=hp0p2_r, ht=hp0p2_t, hb=hp0p2_b)
            env_r = edge_r(QB, hp0p1_r)  # [tr tr'] [ll ll'] [br br']

            hm2p0_t, hm2p1_t = cut_into_hairs(cor_tl(m[-3, 0]) @ cor_tr(m[-3, 1]))
            cm2p0 = cor_tl(m[-2, 0], hl=hm2p0_l, ht=hm2p0_t)
            cm2p1 = cor_tr(m[-2, 1], ht=hm2p1_t, hr=hm2p1_r)
            hm1p0_t, hm1p1_t = cut_into_hairs(cm2p0 @ cm2p1)
            ctl = cor_tl(m[-1, 0], ht=hm1p0_t, hl=hm1p0_l)
            ctr = cor_tr(m[-1, 1], ht=hm1p1_t, hr=hm1p1_r)

            hp2p1_b, hp2p0_b = cut_into_hairs(cor_br(m[3, 1]) @ cor_bl(m[3, 0]))
            cp2p1 = cor_br(m[2, 1], hb=hp2p1_b, hr=hp2p1_r)
            cp2p0 = cor_bl(m[2, 0], hl=hp2p0_l, hb=hp2p0_b)
            hp1p1_b, hp1p0_b = cut_into_hairs(cp2p1 @ cp2p0)
            cbr = cor_br(m[1, 1], hb=hp1p1_b, hr=hp1p1_r)
            cbl = cor_bl(m[1, 0], hb=hp1p0_b, hl=hp1p0_l)

            g = tensordot((cbr @ cbl) @ env_l, (ctl @ ctr) @ env_r, axes=((0, 2), (2, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            sts = [(-3,-0), (-2,-1), (-2,0), (-2,1),
                   (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2),
                   (0,-3), (0,-2), (0,-1), (0,1), (0,2), (0,3),
                   (1,-3), (1,-2), (1,-1), (1,1), (1,2), (1,3),
                   (2,-2), (2,-1), (2,0), (2,1), (2,2),
                   (3,-1), (3,0), (3,1), (4,0)]
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in sts}
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
            env_t = edge_t(QA, hp0p0_t)  # [lt lt'] [bb bb'] [rt rt']

            hp2p0_b = hair_b(m[3, 0], hb=hair_b(m[4, 0]), hl=hp3p0_l, hr=hp3p0_r)
            hp1p0_b = hair_b(m[2, 0], hb=hp2p0_b, hl=hp2p0_l, hr=hp2p0_r)
            env_b = edge_b(QB, hp1p0_b)  # [rb rb'] [tt tt'] [lb lb']

            hp1m2_l, hp0m2_l = cut_into_hairs(cor_bl(m[1, -3]) @ cor_tl(m[0, -3]))
            cp1m2 = cor_bl(m[1, -2], hb=hp1m2_b, hl=hp1m2_l)
            cp0m2 = cor_tl(m[0, -2], hl=hp0m2_l, ht=hp0m2_t)
            hp1m1_l, hp0m1_l = cut_into_hairs(cp1m2 @ cp0m2)
            cbl = cor_bl(m[1,-1], hb=hp1m1_b, hl=hp1m1_l)
            ctl = cor_tl(m[0,-1], ht=hp0m1_t, hl=hp0m1_l)

            hp0p2_r, hp1p2_r  = cut_into_hairs(cor_tr(m[0, 3]) @ cor_br(m[1, 3]))
            cp0p2 = cor_tr(m[0, 2], ht=hp0p2_t, hr=hp0p2_r)
            cp1p2 = cor_br(m[1, 2], hb=hp1p2_b, hr=hp1p2_r)
            hp0p1_r, hp1p1_r  = cut_into_hairs(cp0p2 @ cp1p2)
            ctr = cor_tr(m[0, 1], ht=hp0p1_t, hr=hp0p1_r)
            cbr = cor_br(m[1, 1], hb=hp1p1_b, hr=hp1p1_r)

            g = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))


    def _g_NNN(self, bd, QA, QB):
        """
        Calculates the metric tensor within "NTU-NNN" approximation.

        For bd.dirn == 'h':

        (-1, -1)=(-1, +0)==(-1, +1)=(-1, +2)
           ||       ||        ||       ||
        (+0, -1) == GA-+    +-GB == (+0, +2)
           ||       ||        ||       ||
        (+1, -1)=(+1, +0)==(+1, +1)=(+1, +2)

        For bd.dirn == 'v':

        (-1, -1)=(-1, +0)=(-1, +1)
           ||       ||       ||
        (+0, -1) == GA == (+0, +1)
           ||       ++       ||
        (+1, -1) == GB == (+1, +1)
           ||       ||       ||
        (+2, -1)=(+2, +0)=(+2, +1)
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            sts = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (1,2), (0,2), (-1,2), (-1,1), (-1,0)]
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            ell = edge_l(m[0, -1])
            clt = cor_tl(m[-1, -1])
            elt = edge_t(m[-1, 0])
            vecl = append_vec_tl(QA, QA, ell @ (clt @ elt))
            elb = edge_b(m[1, 0])
            clb = cor_bl(m[1, -1])
            vecl = tensordot(elb @ clb, vecl, axes=((2, 1), (0, 1)))

            err = edge_r(m[0, 2])
            crb = cor_br(m[1, 2])
            erb = edge_b(m[1, 1])
            vecr = append_vec_br(QB, QB, err @ (crb @ erb))
            ert = edge_t(m[-1, 1])
            crt = cor_tr(m[-1, 2])
            vecr = tensordot(ert @ crt, vecr, axes=((2, 1), (0, 1)))
            g = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            sts = [(-1,-1), (0,-1), (1,-1), (2,-1), (2,0), (2,1), (1,1), (0,1), (-1,1), (-1,0)]
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            etl = edge_l(m[0, -1])
            ctl = cor_tl(m[-1, -1])
            ett = edge_t(m[-1, 0])
            vect = append_vec_tl(QA, QA, etl @ (ctl @ ett))
            ctr = cor_tr(m[-1, 1])
            etr = edge_r(m[0, 1])
            vect = tensordot(vect, ctr @ etr, axes=((2, 3), (0, 1)))

            ebr = edge_r(m[1, 1])
            cbr = cor_br(m[2, 1])
            ebb = edge_b(m[2, 0])
            vecb = append_vec_br(QB, QB, ebr @ (cbr @ ebb))
            cbl = cor_bl(m[2, -1])
            ebl = edge_l(m[1, -1])
            vecb = tensordot(vecb, cbl @ ebl, axes=((2, 3), (0, 1)))
            g = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))


    def _g_NNNp(self, bd, QA, QB):
        """
        Calculates the metric tensor within "NTU-NNN+" approximation.

        For bd.dirn == 'h':

                 (-2, -1) (-2, +0)  (-2, +1) (-2, +2)
                    ||       ||        ||       ||
        (-1, -2)=(-1, -1)=(-1, +0)==(-1, +1)=(-1, +2)=(-1, +3)
                    ||       ||        ||       ||
        (+0, -2)=(+0, -1) == GA+      +GB == (+0, +2)=(+0, +3)
                    ||       ||        ||       ||
        (+1, -2)=(+1, -1)=(+1, +0)==(+1, +1)=(+1, +2)=(+1, +3)
                    ||       ||        ||       ||
                 (+2, -1) (+2, +0)  (+2, +1) (+2, +2)

        For bd.dirn == 'v':

                 (-2, -1) (-2, +0) (-2, +1)
                    ||       ||       ||
        (-1, -2)=(-1, -1)=(-1, +0)=(-1, +1)=(-1, +2)
                    ||       ||       ||
        (+0, -2)=(+0, -1) == GA == (+0, +1)=(+0, +2)
                    ||       ++       ||
        (+1, -2)=(+1, -1) == GB == (+1, +1)=(+1, +2)
                    ||       ||       ||
        (+2, -2)=(+2, -1)=(+2, +0)=(+2, +1)=(+2, +2)
                    ||       ||       ||
                 (+3, -1) (+3, +0) (+3, +1)
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            sts = [(-2,-1), (-2,0), (-2,1), (-2,2),
                   (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2), (-1,3),
                   (0,-2), (0,-1), (0,2), (0,3),
                   (1,-2), (1,-1), (1,0), (1,1), (1,2), (1,3),
                   (2,-1), (2,0), (2,1), (2,2)]
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            ell = edge_l(m[0, -1], hl=hair_l(m[0, -2]))
            clt = cor_tl(m[-1, -1], hl=hair_l(m[-1, -2]), ht=hair_t(m[-2, -1]))
            elt = edge_t(m[-1, 0], ht=hair_t(m[-2, 0]))
            vecl = append_vec_tl(QA, QA, ell @ (clt @ elt))
            elb = edge_b(m[1, 0], hb=hair_b(m[2, 0]))
            clb = cor_bl(m[1, -1], hb=hair_b(m[2, -1]), hl=hair_l(m[1, -2]))
            vecl = tensordot(elb @ clb, vecl, axes=((2, 1), (0, 1)))

            err = edge_r(m[0, 2], hr=hair_r(m[0, 3]))
            crb = cor_br(m[1, 2], hr=hair_r(m[1, 3]), hb=hair_b(m[2, 2]))
            erb = edge_b(m[1, 1], hb=hair_b(m[2, 1]))
            vecr = append_vec_br(QB, QB, err @ (crb @ erb))
            ert = edge_t(m[-1, 1], ht=hair_t(m[-2, 1]))
            crt = cor_tr(m[-1, 2], ht=hair_t(m[-2, 2]), hr=hair_r(m[-1, 3]))
            vecr = tensordot(ert @ crt, vecr, axes=((2, 1), (0, 1)))

            g = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            sts =  [(-2,-1), (-2,0), (-2,1), (-1,-2), (-1,1), (-1,0), (-1,1), (-1,2),
                    (0,-2), (0,-1), (0,1), (0,2), (1,-2), (1,-1), (1,1), (1,2),
                    (2,-2), (2,-1), (2,0), (2,1), (2,2), (-3,-1), (-3,0), (-3,1)]
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            etl = edge_l(m[0, -1], hl=hair_l(m[0, -2]))
            ctl = cor_tl(m[-1, -1], hl=hair_l(m[-1, -2]), ht=hair_t(m[-2, -1]))
            ett = edge_t(m[-1, 0], ht=hair_t(m[-2, 0]))
            vect = append_vec_tl(QA, QA, etl @ (ctl @ ett))
            ctr = cor_tr(m[-1, 1], ht=hair_t(m[-2, 1]), hr=hair_r(m[-1, 2]))
            etr = edge_r(m[0, 1], hr=hair_r(m[0, 2]))
            vect = tensordot(vect, ctr @ etr, axes=((2, 3), (0, 1)))

            ebr = edge_r(m[1, 1], hr=hair_r(m[1, 2]))
            cbr = cor_br(m[2, 1], hr=hair_r(m[2, 2]), hb=hair_b(m[3, 1]))
            cbb = edge_b(m[2, 0], hb=hair_b(m[3, 0]))
            vecb = append_vec_br(QB, QB, ebr @ (cbr @ cbb))
            cbl = cor_bl(m[2, -1], hb=hair_b(m[3, -1]), hl=hair_l(m[2, -2]))
            ebl = edge_l(m[1, -1], hl=hair_l(m[1, -2]))
            vecb = tensordot(vecb, cbl @ ebl, axes=((2, 3), (0, 1)))

            g = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))


    def _g_NNNpp(self, bd, QA, QB):
        """
        Calculates the metric tensor within using "NTU-NNN++" approximation.

        For bd.dirn == 'h':

                 (-3, -2)-(-3, -1)-(-3, +0)--(-3, +1)-(-3, +2)-(-3, +3)
                    :        :        :         :        :        :
        (-2, -3)-(-2, -2)-(-2, -1)-(-2, +0)--(-2, +1)-(-2, +2)-(-2, +3)-(-2, +4)
           :        :        :        :         :        :        :        :
        (-1, -3)-(-1, -2)-(-1, -1)=(-1, +0)==(-1, +1)=(-1, +2)-(-1, +3)-(-1, +4)
           :        :        ||       ||        ||       ||       :        :
        (+0, -3)=(+0, -2)=(+0, -1) == GA++    ++GB == (+0, +2)=(+0, +3)=(+0, +4)
           :        :        ||       ||        ||       ||       :        :
        (+1, -3)-(+1, -2)-(+1, -1)=(+1, +0)==(+1, +1)=(+1, +2)-(+1, +3)-(+1, +4)
           :        :        :        :         :        :        :        :
        (+2, -3)-(+2, -2)-(+2, -1)-(+2, +0)--(+2, +1)-(+2, +2)-(+2, +3)-(+2, +4)
                    :        :        :         :        :        :
                 (+3, -2)-(+3, -1)-(+3, +0)--(+3, +1)-(+3, +2)-(+3, +3)

        For bd.dirn == 'v':

                 (-3, -2)-(-3, -1)-(-3, +0)-(-3, +1)-(-3, +2)
                    :        :        ||       :        :
        (-2, -3)-(-2, -2)-(-2, -1)-(-2, +0)-(-2, +1)-(-2, +2)-(-2, +3)
           :        :        :        ||       :        :        :
        (-1, -3)-(-1, -2)-(-1, -1)=(-1, +0)=(-1, +1)-(-1, +2)-(-1, +3)
           :        :        ||       ||       ||       :        :
        (+0, -3)-(+0, -2)-(+0, -1) == GA == (+0, +1)-(+0, +2)-(+0, +3)
           :        :        ||       ++       ||       :        :
        (+1, -3)-(+1, -2)-(+1, -1) == GB == (+1, +1)-(+1, +2)-(+1, +3)
           :        :        ||       ||       ||       :        :
        (+2, -3)-(+2, -2)-(+2, -1)=(+2, +0)=(+2, +1)-(+2, +2)-(+2, +3)
           :        :        :        ||       :        :        :
        (+3, -3)-(+3, -2)-(+3, -1)-(+3, +0)-(+3, +1)-(+3, +2)-(+3, +3)
                    :        :        ||       :        :
                 (+4, -2)-(+4, -1)-(+4, +0)-(+4, +1)-(+4, +2)
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            sts = [(-3,-2), (-3,-1), (-3,0), (-3,1), (-3,2), (-3,3),
                   (-2,-3), (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2), (-2,3), (-2,4),
                   (-1,-3), (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2), (-1,3), (-1,4),
                   (0,-3), (0,-2), (0,-1), (0,2), (0,3), (0,4),
                   (1,-3), (1,-2), (1,-1), (1,0), (1,1), (1,2), (1,3), (1,4),
                   (2,-3), (2,-2), (2,-1), (2,0), (2,1), (2,2), (2,3), (2,4),
                   (3,-2), (3,-1), (3,0), (3,1), (3,2), (3,3)]
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            ell = edge_l(m[0, -1], hl=hair_l(m[0, -2], hl=hair_l(m[0, -3])))
            cclt = hair_t(m[-2, -2], ht=hair_t(m[-3, -2]), hl=hair_l(m[-2, -3]))
            clt = cor_tl(m[-1, -1], hl=hair_l(m[-1, -2], hl=hair_l(m[-1, -3]), ht=cclt), ht=hair_t(m[-2, -1], ht=hair_t(m[-3, -1])))
            elt = edge_t(m[-1, 0], ht=hair_t(m[-2, 0], ht=hair_t(m[-3, 0])))
            vecl = append_vec_tl(QA, QA, ell @ (clt @ elt))
            elb = edge_b(m[1, 0], hb=hair_b(m[2, 0], hb=hair_b(m[3, 0])))
            cclb = hair_b(m[2, -2], hb=hair_b(m[3, -2]), hl=hair_l(m[2, -3]))
            clb = cor_bl(m[1, -1], hb=hair_b(m[2, -1], hb=hair_b(m[3, -1])), hl=hair_l(m[1, -2], hl=hair_l(m[1, -3]), hb=cclb))
            vecl = tensordot(elb @ clb, vecl, axes=((2, 1), (0, 1)))

            err = edge_r(m[0, 2], hr=hair_r(m[0, 3], hr=hair_r(m[0, 4])))
            ccrb = hair_b(m[2, 3], hb=hair_b(m[3, 3]), hr=hair_r(m[2, 4]))
            crb = cor_br(m[1, 2], hr=hair_r(m[1, 3], hr=hair_r(m[1, 4]), hb=ccrb), hb=hair_b(m[2, 2], hb=hair_b(m[3, 2])))
            erb = edge_b(m[1, 1], hb=hair_b(m[2, 1], hb=hair_b(m[3, 1])))
            vecr = append_vec_br(QB, QB, err @ (crb @ erb))
            ert = edge_t(m[-1, 1], ht=hair_t(m[-2, 1], ht=hair_t(m[-3, 1])))
            ccrt = hair_t(m[-2, 3], hr=hair_r(m[-2, 4]), ht=hair_t(m[-3, 3]))
            crt = cor_tr(m[-1, 2], ht=hair_t(m[-2, 2], ht=hair_t(m[-3, 2])), hr=hair_r(m[-1, 3], hr=hair_r(m[-1, 4]), ht=ccrt))
            vecr = tensordot(ert @ crt, vecr, axes=((2, 1), (0, 1)))

            g = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            sts = [(-3,-2), (-3,-1), (-3,0), (-3,1), (-3,2),
                   (-2,-3), (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2), (-2,3),
                   (-1,-3), (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2), (-1,3),
                   (0,-3), (0,-2), (0,-1), (0,1), (0,2), (0,3),
                   (1,-3), (1,-2), (1,-1), (1,1), (1,2), (1,3),
                   (2,-3), (2,-2), (2,-1), (2,0), (2,1), (2,2), (2,3),
                   (3,-3), (3,-2), (3,-1), (3,0), (3,1), (3,2), (3,3),
                   (4,-2), (4,-1), (4,0), (4,1), (4,2)]
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)

            etl = edge_l(m[0, -1], hl=hair_l(m[0, -2], hl=hair_l(m[0, -3])))
            cctl = hair_t(m[-2, -2], hl=hair_l(m[-2, -3]), ht=hair_t(m[-3, -2]))
            ctl = cor_tl(m[-1, -1], hl=hair_l(m[-1, -2], hl=hair_l(m[-1, -3]), ht=cctl), ht=hair_t(m[-2, -1], ht=hair_t(m[-3, -1])))
            ett = edge_t(m[-1, 0], ht=hair_t(m[-2, 0], ht=hair_t(m[-3, 0])))
            vect = append_vec_tl(QA, QA, etl @ (ctl @ ett))
            cctr = hair_t(m[-2, 2], hr=hair_r(m[-2, 3]), ht=hair_t(m[-3, 2]))
            ctr = cor_tr(m[-1, 1], ht=hair_t(m[-2, 1], ht=hair_t(m[-3, 1])), hr=hair_r(m[-1, 2], hr=hair_r(m[-1, 3]), ht=cctr))
            etr = edge_r(m[0, 1], hr=hair_r(m[0, 2], hr=hair_r(m[0, 3])))
            vect = tensordot(vect, ctr @ etr, axes=((2, 3), (0, 1)))

            ebr = edge_r(m[1, 1], hr=hair_r(m[1, 2], hr=hair_r(m[1, 3])))
            ccbr = hair_b(m[3, 2], hr=hair_r(m[3, 3]), hb=hair_b(m[4, 2]))
            cbr = cor_br(m[2, 1], hr=hair_r(m[2, 2], hr=hair_r(m[2, 3]), hb=ccbr), hb=hair_b(m[3, 1], hb=hair_b(m[4, 1])))
            cbb = edge_b(m[2, 0], hb=hair_b(m[3, 0], hb=hair_b(m[4, 0])))
            vecb = append_vec_br(QB, QB, ebr @ (cbr @ cbb))
            ccbl = hair_b(m[3, -2], hl=hair_l(m[3, -3]), hb=hair_b(m[4, -2]))
            cbl = cor_bl(m[2, -1], hb=hair_b(m[3, -1], hb=hair_b(m[4, -1])), hl=hair_l(m[2, -2], hl=hair_l(m[2, -3]), hb=ccbl))
            ebl = edge_l(m[1, -1], hl=hair_l(m[1, -2], hl=hair_l(m[1, -3])))
            vecb = tensordot(vecb, cbl @ ebl, axes=((2, 3), (0, 1)))

            g = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))
