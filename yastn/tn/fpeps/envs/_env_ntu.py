from .... import tensordot, YastnError
from ._env_auxlliary import *


class EnvNTU:
    def __init__(self, psi, which='NN'):
        if which not in ('NN', 'NN+', 'NN++', 'NNN', 'NNN+', 'NNN++'):  # 'SU+'
            raise YastnError(f" Type of EnvNTU {which} not recognized.")
        self.psi = psi
        self.which = which
        self._dict_gs = {#'SU': self._g_p,
                         'NN': self._g_NN,
                         'NN+': self._g_NNp,
                         'NN++': self._g_NNpp,
                         'NNN': self._g_NNN,
                         'NNN+': self._g_NNNp,
                         'NNN++': self._g_NNNpp
                        }


    def bond_metric(self, bd, QA, QB):
        """ Calculates bond metric. """
        return self._dict_gs[self.which](bd, QA, QB)

    # def _g_p(self, bd, QA, QB):
    #     """
    #     Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
    #     """
    #     if bd.dirn == "h":
    #         assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
    #         #          (-1, 0)  (-1, 1)
    #         #             ||       ||
    #         #   (0, -1) = GA +   + GB = (0, +2)
    #         #             ||       ||
    #         #          (+1, 0)  (+1, 1)
    #         m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1, 0), (0, -1), (1, 0), (1, 1), (0, 2), (-1, 1)]}
    #         tensors_from_psi(m, self.psi)
    #         env_l = hair_l(QA, hl=hair_l(m[0, -1]), ht=hair_t(m[-1, 0]), hb=hair_b(m[1, 0]))
    #         env_r = hair_r(QB, hr=hair_r(m[0,  2]), ht=hair_t(m[-1, 1]), hb=hair_b(m[1, 1]))
    #         g = tensordot(env_l, env_r, axes=((), ()))  # [rr' rr] [ll' ll]
    #     else: # dirn == "v":
    #         assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
    #         #          (-1, 0)
    #         #             ||
    #         #   (0, -1) = GA = (0, 1)
    #         #             ++
    #         #   (1, -1) = GB = (1, 1)
    #         #             ||
    #         #          (+2, 0)
    #         m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1, 0), (0, -1), (1, -1), (2, 0), (1, 1), (0, 1)]}
    #         tensors_from_psi(m, self.psi)
    #         env_t = hair_t(QA, ht=hair_t(m[-1, 0]), hl=hair_l(m[0, -1]), hr=hair_r(m[0, 1]))
    #         env_b = hair_b(QB, hb=hair_b(m[ 2, 0]), hl=hair_l(m[1, -1]), hr=hair_r(m[1, 1]))
    #         g = tensordot(env_t, env_b, axes=((), ()))  # [bb' bb] [tt' tt]
    #     return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((0, 2), (1, 3)))


    def _g_NN(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network within "NTU-NN" approximation.

        For bd.dirn == 'h':

                 (-1, +0)==(-1, +1)
                    ||        ||
        (+0, -1) == QA+      +QB == (+0, +2)
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
        Calculates the metric tensor g for the given PEPS tensor network within "NTU-NN+" approximation.

        For bd.dirn == 'h':

                          (-2, +0)  (-2, +1)
                             ||        ||
                 (-1, -1)-(-1, +0)==(-1, +1)-(-1, +2)
                     :       ||        ||       :
        (+0, -2)=(+0, -1) == QA+      +QB == (+0, +2)=(+0, +3)
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
        Calculates the metric tensor g for the given PEPS tensor network "NTU-NN++" approximation.

        For bd.dirn == 'h':

                                   (-2, +0)--(-2, +1)
                                      ||        ||
                          (-2, -1)-(-2, +0)--(-2, +1)-(-2, +1)
                             :        ||        ||       :
                 (-1, -2)-(-1, -1)-(-1, +0)==(-1, +1)-(-1, +2)-(-1, +3)
                    :        :        ||        ||       :        :
        (+0, -2)=(+0, -2)=(+0, -1) == QA+      +QB == (+0, +2)=(+0, +3)=(+0, +4)
                    :        :        ||        ||       :        :
                 (+1, -2)-(+1, -1)-(+1, +0)==(+1, +1)-(+1, +2)-(+1, +3)
                             :        ||        ||       :
                          (+2, -1)-(+2, +0)--(+2, +1)-(+2, +2)
                                      ||        ||
                                   (+2, +0)--(+2, +1)

        For bd.dirn == 'v':

                                   (-3, +0)
                                      ||
                          (-2, -1)-(-2, +0)-(-2, +1)
                             :        ||       :
                 (-1, -2)-(-1, -1)-(-1, +0)-(-1, +1)-(-1, +2)
                    :        :        ||       :        :
        (+0, -3)=(+0, -2)=(+0, -1) == GA == (+0, +1)=(+0, +2)=(+0, +3)
           :        :        ||       ++       ||       :        :
        (+1, -3)=(+1, -2)=(+1, -1) == GB == (+1, +1)=(+1, +2)=(+1, +3)
                    :        :        ||       :        :
                 (+2, -2)-(+2, -1)-(+2, +0)-(+2, +1)-(+2, +2)
                             :        ||       :
                          (+3, -1)-(+3, +0)-(+3, +1)
                                      ||
                                   (+4, +0)
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            sts =  [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (1,2), (0,2), (-1,2), (-1,1), (-1,0), (-1,-2), (0,-2),
                    (1,-2), (2,-1), (2,0), (2,1), (2,2), (1,3), (0,3), (-1,3), (-2,2), (-2,1), (-2,0), (-2,-1)]
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)
            env_l = edge_l(QA, hair_l(m[0,-1], hl=hair_l(m[0,-2])))  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(QB, hair_r(m[0, 2], hr=hair_r(m[0, 3])))  # [tr tr'] [ll ll'] [br br']
            ctr = cor_tr(m[-1, 1], ht=hair_t(m[-2, 1]), hr=hair_r(m[-1, 2], hr=hair_r(m[-1, 3]), ht=hair_t(m[-2, 2])))
            ctl = cor_tl(m[-1, 0], ht=hair_t(m[-2, 0]), hl=hair_l(m[-1,-1], hl=hair_l(m[-1,-2]), ht=hair_t(m[-2,-1])))
            cbr = cor_br(m[ 1, 1], hb=hair_b(m[ 2, 1]), hr=hair_r(m[ 1, 2], hr=hair_r(m[ 1, 3]), hb=hair_b(m[ 2, 2])))
            cbl = cor_bl(m[ 1, 0], hb=hair_b(m[ 2, 0]), hl=hair_l(m[ 1,-1], hl=hair_l(m[ 1,-2]), hb=hair_b(m[ 2,-1])))
            g = tensordot((cbr @ cbl) @ env_l, (ctl @ ctr) @ env_r, axes=((0, 2), (2, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            sts = [(-1,-1), (0,-1), (1,-1), (2,-1), (2,0), (2,1), (1,1), (0,1), (-1,1), (-1,0), (-1,-2), (0,-2),
                    (1,-2), (2,-2), (3,-1), (3,0), (3,1), (2,2), (1,2), (0,2), (-1,2), (-2,1), (-2,0), (-2,-1)]
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in sts}
            tensors_from_psi(m, self.psi)
            env_t = edge_t(QA, hair_t(m[-1, 0], ht=hair_t(m[-2, 0])))  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(QB, hair_b(m[ 2, 0], hb=hair_b(m[ 3, 0])))  # [rb rb'] [tt tt'] [lb lb']
            cbl = cor_bl(m[1,-1], hl=hair_l(m[1,-2]), hb=hair_b(m[2, -1], hb=hair_b(m[3, -1]), hl=hair_l(m[2, -2])))
            ctl = cor_tl(m[0,-1], hl=hair_l(m[0,-2]), ht=hair_t(m[-1,-1], ht=hair_t(m[-2,-1]), hl=hair_l(m[-1,-2])))
            ctr = cor_tr(m[0, 1], hr=hair_r(m[0, 2]), ht=hair_t(m[-1, 1], ht=hair_t(m[-2, 1]), hr=hair_r(m[-1, 2])))
            cbr = cor_br(m[1, 1], hr=hair_r(m[1, 2]), hb=hair_b(m[2,  1], hb=hair_b(m[3,  1]), hr=hair_r(m[2,  2])))
            g = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))


    def _g_NNN(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network "NTU-NNN" approximation.

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
        Calculates the metric tensor g for the given PEPS tensor network "NTU-NNN+" approximation.

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
            sts = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (1,2), (0,2), (-1,2), (-1,1), (-1,0), (-1,-2), (0,-2),
                   (1,-2), (2,-1), (2,0), (2,1), (2,2), (1,3), (0,3), (-1,3), (-2,2), (-2,1), (-2,0), (-2,-1)]
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
            sts =  [(-1,-1), (0,-1), (1,-1), (2,-1), (2,0), (2,1), (1,1), (0,1), (-1,1), (-1,0), (-1,-2), (0,-2),
                    (1,-2), (2,-2), (3,-1), (3,0), (3,1), (2,2), (1,2), (0,2), (-1,2), (-2,1), (-2,0), (-2,-1)]
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
        Calculates the metric tensor g for the given PEPS tensor network using "NTU-NNN++" approximation.

        For bd.dirn == 'h':

                 (-3, -2) (-3, -1) (-3, +0)  (-3, +1) (-3, +2) (-3, +3)
                             ||       ||        ||       ||
        (-2, -3) (-2, -2) (-2, -1) (-2, +0)  (-2, +1) (-2, +2) (-2, +3) (-2, +4)
                             ||       ||        ||       ||
        (-1, -3)=(-1, -2)=(-1, -1)=(-1, +0)==(-1, +1)=(-1, +2)=(-1, +3)=(-1, +4)
                             ||       ||        ||       ||
        (+0, -3)=(+0, -2)=(+0, -1) == GA+      +GB == (+0, +2)=(+0, +3)=(+0, +4)
                             ||       ||        ||       ||
        (+1, -3)=(+1, -2)=(+1, -1)=(+1, +0)==(+1, +1)=(+1, +2)=(+1, +3)=(+1, +4)
                             ||       ||        ||       ||
        (+2, -3) (+2, -2) (+2, -1) (+2, +0)  (+2, +1) (+2, +2) (+2, +3) (+2, +4)
                             ||       ||        ||       ||
                 (+3, -2) (+3, -1) (+3, +0)  (+3, +1) (+3, +2) (+3, +3)

        For bd.dirn == 'v':

                 (-3, -2) (-3, -1) (-3, +0) (-3, +1) (-3, +2)

        (-2, -3) (-2, -2) (-2, -1) (-2, +0) (-2, +1) (-2, +2) (-2, +3)

        (-1, -3) (-1, -2) (-1, -1)=(-1, +0)=(-1, +1) (-1, +2) (-1, +3)
                             ||       ||       ||
        (+0, -3) (+0, -2) (+0, -1) == GA == (+0, +1) (+0, +2) (+0, +3)
                             ||       ++       ||
        (+1, -3) (+1, -2) (+1, -1) == GB == (+1, +1) (+1, +2) (+1, +3)
                             ||       ||       ||
        (+2, -3) (+2, -2) (+2, -1)=(+2, +0)=(+2, +1) (+2, +2) (+2, +3)

        (+3, -3) (+3, -2) (+3, -1) (+3, +0) (+3, +1) (+3, +2) (+3, +3)

                 (+4, -2) (+4, -1) (+4, +0) (+4, +1) (+4, +2)
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            sts = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (1,2), (0,2), (-1,2), (-1,1), (-1,0), (-1,-2), (0,-2),
                   (1,-2), (2,-1), (2,0), (2,1), (2,2), (1,3), (0,3), (-1,3), (-2,2), (-2,1), (-2,0), (-2,-1), (-1,-3),
                   (0,-3), (1,-3), (3,-1), (3,0), (3,1), (3,2), (1,4), (0,4), (-1,4), (-3,2), (-3,1), (-3,0), (-3,-1),
                   (-2,-3), (-2,-2), (-3,-2), (-2,3), (-3,3), (-2,4), (2,-3), (2,-2), (3,-2), (2,3), (2,4), (3,3)]
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
            sts = [(-1,-1), (0,-1), (1,-1), (2,-1), (2,0), (2,1), (1,1), (0,1), (-1,1), (-1,0), (-1,-2), (0,-2),
                   (1,-2), (2,-2), (3,-1), (3,0), (3,1), (2,2), (1,2), (0,2), (-1,2), (-2,1), (-2,0), (-2,-1), (-1,-3),
                   (0,-3), (1,-3), (2,-3), (4,-1), (4,0), (4,1), (2,3), (1,3), (0,3), (-1,3), (-3,1), (-3,0), (-3,-1),
                   (-2,-2), (-2,-3), (-3,-2), (-2,2), (-2,3), (-3,2), (3,-2), (3,-3), (4,-2), (3,2), (3,3), (4,2)]
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
