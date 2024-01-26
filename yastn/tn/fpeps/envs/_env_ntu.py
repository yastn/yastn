from .... import tensordot, YastnError, ones, Leg
from ._env_auxlliary import *

class EnvNTU:
    def __init__(self, psi, which='NN'):
        if which not in ('h', 'NN', 'NNh', 'NNhh', 'NNhh', 'NNN', 'NNNh'):
            raise YastnError(f" Type of EnvNTU {which} not recognized.")
        self.psi = psi
        self.which = which

    def bond_metric(self, bd, QA, QB):
        """ Calculates bond metric. """
        if self.which == 'NN':
            return self._g_NN(bd, QA, QB)
        if self.which == 'NNh':
            return self._g_NNh(bd, QA, QB)
        if self.which == 'NNhh':
            return self._g_NNhh(bd, QA, QB)

        if self.which == 'NNN':
            return self._g_NNN(bd, QA, QB)
        if self.which == 'NNNh':
            return self._g_NNNh(bd, QA, QB)
        if self.which == 'h':
            return self._g_h(bd, QA, QB)

    def _g_h(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            #          (-1, 0)  (-1, 1)
            #             ||       ||
            #   (0, -1) = GA +   + GB = (0, +2)
            #             ||       ||
            #          (+1, 0)  (+1, 1)
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1, 0), (0, -1), (1, 0), (1, 1), (0, 2), (-1, 1)]}
            tensors_from_psi(m, self.psi)
            env_l = hair_l(QA, hl=hair_l(m[0, -1]), ht=hair_t(m[-1, 0]), hb=hair_b(m[1, 0]))
            env_r = hair_r(QB, hr=hair_r(m[0,  2]), ht=hair_t(m[-1, 1]), hb=hair_b(m[1, 1]))
            G = tensordot(env_l, env_r, axes=((), ()))  # [rr' rr] [ll' ll]
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            #          (-1, 0)
            #             ||
            #   (0, -1) = GA = (0, 1)
            #             ++
            #   (1, -1) = GB = (1, 1)
            #             ||
            #          (+2, 0)
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1, 0), (0, -1), (1, -1), (2, 0), (1, 1), (0, 1)]}
            tensors_from_psi(m, self.psi)
            env_t = hair_t(QA, ht=hair_t(m[-1, 0]), hl=hair_l(m[0, -1]), hr=hair_r(m[0, 1]))
            env_b = hair_b(QB, hb=hair_b(m[ 2, 0]), hl=hair_l(m[1, -1]), hr=hair_r(m[1, 1]))
            G = tensordot(env_t, env_b, axes=((), ()))  # [bb' bb] [tt' tt]
        return G.unfuse_legs(axes=(0, 1)).transpose(axes=(1, 0, 3, 2))


    def _g_NN(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            #          (-1, 0)==(-1, 1)
            #             ||       ||
            #   (0, -1) = GA +   + GB = (0, +2)
            #             ||       ||
            #          (+1, 0)==(+1, 1)
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1, 0), (0, -1), (1, 0), (1, 1), (0, 2), (-1, 1)]}
            tensors_from_psi(m, self.psi)
            env_l = edge_l(QA, hair_l(m[0, -1]))  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(QB, hair_r(m[0,  2]))  # [tr tr'] [ll ll'] [br br']
            ctl = cor_tl(m[-1, 0])
            ctr = cor_tr(m[-1, 1])
            cbr = cor_br(m[ 1, 1])
            cbl = cor_bl(m[ 1, 0])
            G = tensordot((cbr @ cbl) @ env_l, (ctl @ ctr) @ env_r, axes=((0, 2), (2, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            #         (-1, 0)
            #            ||
            #   (0,-1) = GA = (0, 1)
            #     ||     ++     ||
            #   (1,-1) = GB = (1, 1)
            #            ||
            #         (+2, 0)
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1, 0), (0, -1), (1, -1), (2, 0), (1, 1), (0, 1)]}
            tensors_from_psi(m, self.psi)
            env_t = edge_t(QA, hair_t(m[-1, 0]))  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(QB, hair_b(m[ 2, 0]))  # [rb rb'] [tt tt'] [lb lb']
            cbl = cor_bl(m[1,-1])
            ctl = cor_tl(m[0,-1])
            ctr = cor_tr(m[0, 1])
            cbr = cor_br(m[1, 1])
            G = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return G.unfuse_legs(axes=(0, 1))


    def _g_NNh(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            #                 (-2,0)  (-2,1)
            #        (-1,-1)==(-1,0)==(-1,1)==(-1,2)
            #                   ||      ||
            # (0, -2)(0, -1) == GA  ++  GB == (0, 2)(0, 3)
            #                   ||      ||
            #        (1, -1)==(1, 0)==(1, 1)==(1, 2)
            #                 (2, 0)  (2, 1)
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1,-1), (0,-1), (1,-1), (1, 0), (1, 1), (1, 2), (0, 2), (-1, 2), (-1, 1), (-1, 0),
                                                              (0, -2), (2, 0), (2, 1), (0, 3), (-2, 1), (-2, 0)]}
            tensors_from_psi(m, self.psi)
            env_l = edge_l(QA, hair_l(m[0,-1], hl=hair_l(m[0,-2])))  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(QB, hair_r(m[0, 2], hr=hair_r(m[0, 3])))  # [tr tr'] [ll ll'] [br br']
            ctl = cor_tl(m[-1, 0], ht=hair_t(m[-2, 0]), hl=hair_l(m[-1,-1]))
            ctr = cor_tr(m[-1, 1], ht=hair_t(m[-2, 1]), hr=hair_r(m[-1, 2]))
            cbr = cor_br(m[ 1, 1], hb=hair_b(m[ 2, 1]), hr=hair_r(m[ 1, 2]))
            cbl = cor_bl(m[ 1, 0], hb=hair_b(m[ 2, 0]), hl=hair_l(m[ 1,-1]))
            G = tensordot((cbr @ cbl) @ env_l, (ctl @ ctr) @ env_r, axes=((0, 2), (2, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            #                 (-2, 0)
            #        (-1,-1)  (-1, 0)  (-1,1)
            #           ||       ||      ||
            # (0, -2)(0, -1)===  GA ===(0, 1)(0, 2)
            #           ||       ++      ||
            # (1, -2)(1, -1)===  GB ===(1, 1)(1, 2)
            #           ||       ||      ||
            #        (2, -1)  (2,  0)  (2, 1)
            #                 (3,  0)
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1,-1), (0,-1), (1,-1), (2,-1), (2, 0), (2, 1), (1, 1), (0, 1), (-1, 1), (-1, 0),
                                                              (0, -2), (1,-2), (3, 0), (1, 2), (0, 2), (-2, 0)]}
            tensors_from_psi(m, self.psi)
            env_t = edge_t(QA, hair_t(m[-1, 0], ht=hair_t(m[-2, 0])))  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(QB, hair_b(m[ 2, 0], hb=hair_b(m[ 3, 0])))  # [rb rb'] [tt tt'] [lb lb']
            cbl = cor_bl(m[1,-1], hl=hair_l(m[1,-2]), hb=hair_b(m[2, -1]))
            ctl = cor_tl(m[0,-1], hl=hair_l(m[0,-2]), ht=hair_t(m[-1,-1]))
            ctr = cor_tr(m[0, 1], hr=hair_r(m[0, 2]), ht=hair_t(m[-1, 1]))
            cbr = cor_br(m[1, 1], hr=hair_r(m[1, 2]), hb=hair_b(m[2,  1]))
            G = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return G.unfuse_legs(axes=(0, 1))

    def _g_NNhh(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            #        (-2,-1)  (-2,0)  (-2,1)  (-2,2)
            # (-1,-2)(-1,-1)==(-1,0)==(-1,1)==(-1,2)(-1,3)
            #                   ||      ||
            # (0, -2)(0, -1) == GA  ++  GB == (0, 2)(0, 3)
            #                   ||      ||
            # (1, -2)(1, -1)==(1, 0)==(1, 1)==(1, 2)(1, 3)
            #        (2, -1)  (2, 0)  (2, 1)  (2, 2)
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1,-1), (0,-1), (1,-1), (1, 0), (1, 1), (1, 2), (0, 2), (-1, 2), (-1, 1), (-1, 0),
                                                              (-1,-2), (0,-2), (1,-2), (2,-1), (2, 0), (2, 1), (2, 2), ( 1, 3), ( 0, 3), (-1, 3), (-2, 2), (-2, 1), (-2, 0), (-2, -1)]}
            tensors_from_psi(m, self.psi)
            env_l = edge_l(QA, hair_l(m[0,-1], hl=hair_l(m[0,-2])))  # [bl bl'] [rr rr'] [tl tl']
            env_r = edge_r(QB, hair_r(m[0, 2], hr=hair_r(m[0, 3])))  # [tr tr'] [ll ll'] [br br']
            ctr = cor_tr(m[-1, 1], ht=hair_t(m[-2, 1]), hr=hair_r(m[-1, 2], hr=hair_r(m[-1, 3]), ht=hair_t(m[-2, 2])))
            ctl = cor_tl(m[-1, 0], ht=hair_t(m[-2, 0]), hl=hair_l(m[-1,-1], hl=hair_l(m[-1,-2]), ht=hair_t(m[-2,-1])))
            cbr = cor_br(m[ 1, 1], hb=hair_b(m[ 2, 1]), hr=hair_r(m[ 1, 2], hr=hair_r(m[ 1, 3]), hb=hair_b(m[ 2, 2])))
            cbl = cor_bl(m[ 1, 0], hb=hair_b(m[ 2, 0]), hl=hair_l(m[ 1,-1], hl=hair_l(m[ 1,-2]), hb=hair_b(m[ 2,-1])))
            G = tensordot((cbr @ cbl) @ env_l, (ctl @ ctr) @ env_r, axes=((0, 2), (2, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            #        (-2,-1)  (-2, 0)  (-2,1)
            # (-1,-2)(-1,-1)  (-1, 0)  (-1,1)(-1,2)
            #           ||       ||      ||
            # (0, -2)(0, -1)===  GA ===(0, 1)(0, 2)
            #           ||       ++      ||
            # (1, -2)(1, -1)===  GB ===(1, 1)(1, 2)
            #           ||       ||      ||
            # (2, -2)(2, -1)  (2,  0)  (2, 1)(2, 2)
            #        (3, -1)  (3,  0)  (3, 1)
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1,-1), (0,-1), (1,-1), (2,-1), (2, 0), (2, 1), (1, 1), (0, 1), (-1, 1), (-1, 0),
                                                              (-1,-2), (0,-2), (1,-2), (2,-2), (3,-1), (3, 0), (3, 1), (2, 2), (1, 2), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-2,-1)]}
            tensors_from_psi(m, self.psi)
            env_t = edge_t(QA, hair_t(m[-1, 0], ht=hair_t(m[-2, 0])))  # [lt lt'] [bb bb'] [rt rt']
            env_b = edge_b(QB, hair_b(m[ 2, 0], hb=hair_b(m[ 3, 0])))  # [rb rb'] [tt tt'] [lb lb']
            cbl = cor_bl(m[1,-1], hl=hair_l(m[1,-2]), hb=hair_b(m[2, -1], hb=hair_b(m[3, -1]), hl=hair_l(m[2, -2])))
            ctl = cor_tl(m[0,-1], hl=hair_l(m[0,-2]), ht=hair_t(m[-1,-1], ht=hair_t(m[-2,-1]), hl=hair_l(m[-1,-2])))
            ctr = cor_tr(m[0, 1], hr=hair_r(m[0, 2]), ht=hair_t(m[-1, 1], ht=hair_t(m[-2, 1]), hr=hair_r(m[-1, 2])))
            cbr = cor_br(m[1, 1], hr=hair_r(m[1, 2]), hb=hair_b(m[2,  1], hb=hair_b(m[3,  1]), hr=hair_r(m[2,  2])))
            G = tensordot((cbl @ ctl) @ env_t, (ctr @ cbr) @ env_b, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return G.unfuse_legs(axes=(0, 1))


    def _g_NNN(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            #   (-1,-1)==(-1,0)==(-1,1)==(-1,2)
            #      ||      ||      ||      ||
            #   (0, -1) == GA  ++  GB == (0, 2)
            #      ||      ||      ||      ||
            #   (1, -1)==(1, 0)==(1, 1)==(1, 2)
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1, -1), (0, -1), (1, -1), (1,  0), (1,  1),  (1,  2), (0,  2), (-1, 2), (-1, 1), (-1, 0)]}
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
            G = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            #   (-1,-1)==(-1, 0)==(-1,1)
            #      ||       ||      ||
            #   (0, -1)===  GA ===(0, 1)
            #      ||       ++      ||
            #   (1, -1)===  GB ===(1, 1)
            #      ||       ||      ||
            #   (2, -1)==(2,  0)==(2, 1)
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1, -1), (0, -1), (1, -1), (2, -1), (2, 0), (2, 1), (1, 1), (0, 1), (-1, 1), (-1, 0)]}
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
            G = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return G.unfuse_legs(axes=(0, 1))


    def _g_NNNh(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            #        (-2,-1)  (-2,0)  (-2,1)  (-2,2)
            # (-1,-2)(-1,-1)==(-1,0)==(-1,1)==(-1,2)(-1,3)
            #           ||      ||      ||      ||
            # (0, -2)(0, -1) == GA  ++  GB == (0, 2)(0, 3)
            #           ||      ||      ||      ||
            # (1, -2)(1, -1)==(1, 0)==(1, 1)==(1, 2)(1, 3)
            #        (2, -1)  (2, 0)  (2, 1)  (2, 2)
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1,-1), (0,-1), (1,-1), (1, 0), (1, 1), (1, 2), (0, 2), (-1, 2), (-1, 1), (-1, 0),
                                                              (-1,-2), (0,-2), (1,-2), (2,-1), (2, 0), (2, 1), (2, 2), ( 1, 3), ( 0, 3), (-1, 3), (-2, 2), (-2, 1), (-2, 0), (-2, -1)]}
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

            G = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(bd.site0, (1, 0)) == bd.site1
            #        (-2,-1)  (-2, 0)  (-2,1)
            # (-1,-2)(-1,-1)==(-1, 0)==(-1,1)(-1,2)
            #           ||       ||      ||
            # (0, -2)(0, -1)===  GA ===(0, 1)(0, 2)
            #           ||       ++      ||
            # (1, -2)(1, -1)===  GB ===(1, 1)(1, 2)
            #           ||       ||      ||
            # (2, -2)(2, -1)==(2,  0)==(2, 1)(2, 2)
            #        (3, -1)  (3,  0)  (3, 1)
            m = {d: self.psi.nn_site(bd.site0, d=d) for d in [(-1,-1), (0,-1), (1,-1), (2,-1), (2, 0), (2, 1), (1, 1), (0, 1), (-1, 1), (-1, 0),
                                                              (-1,-2), (0,-2), (1,-2), (2,-2), (3,-1), (3, 0), (3, 1), (2, 2), (1, 2), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-2,-1)]}
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

            G = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']
        return G.unfuse_legs(axes=(0, 1))


def tensors_from_psi(m, psi):
    if any(v is None for v in m.values()):
        cfg = psi[(0, 0)].config
        triv = ones(cfg, legs=[Leg(cfg, t=((0,) * cfg.sym.NSYM,), D=(1,))])
        for s in (-1, 1, 1, -1):
            triv = triv.add_leg(axis=0, s=s)
        triv = triv.fuse_legs(axes=((0, 1), (2, 3), 4))
    for k, v in m.items():
        m[k] = triv if v is None else psi[v]
