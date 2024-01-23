from .... import tensordot, YastnError, ones, Leg
from ._env_auxlliary import *
from .._doublePepsTensor import DoublePepsTensor

class EnvNTU:
    def __init__(self, psi, which='NN'):
        if which not in ('p', 'NN', 'NNN', 'NNN+'):
            raise YastnError(f" Type of EnvNTU {which} not recognized.")
        self.psi = psi
        self.which = which

    def bond_metric(self, bd, QA, QB):
        """ Calculates bond metric. """
        if self.which == 'NN':
            return self._g_NN(bd, QA, QB)
        if self.which == 'NNN':
            return self._g_NNN(bd, QA, QB)
        if self.which == 'NNNp':
            return self._g_NNNp(bd, QA, QB)
        if self.which == 'p':
            return self._g_p(bd, QA, QB)

    def _g_p(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        psi = self.psi
        dirn = bd.dirn

        if dirn == "h":
            m = {'tl': psi.nn_site(bd.site0, d='t'),
                 'l' : psi.nn_site(bd.site0, d='l'),
                 'bl': psi.nn_site(bd.site0, d='b'),
                 'tr': psi.nn_site(bd.site1, d='t'),
                 'r' : psi.nn_site(bd.site1, d='r'),
                 'br': psi.nn_site(bd.site1, d='b')}
            tensors_from_psi(m, psi)

            env_l = edge_l(QA, leaf_l(m['l']))  # [t t'] [b b'] [rr rr']
            env_r = edge_r(QB, leaf_r(m['r']))  # [ll ll'] [t t'] [b b']
            env_t = cor_tl(m['tl']) @ cor_tr(m['tr'])  # [tl tl'] [tr tr']
            env_b = cor_br(m['br']) @ cor_bl(m['bl'])  # [br br'] [bl bl']
            env_l = env_b @ env_l
            env_r = env_t @ env_r
            G = tensordot(env_l, env_r, axes=((0, 2), (2, 0)))  # [ll ll'] [rr rr']
        else: # dirn == "v":
            m = {'tl': psi.nn_site(bd.site0, d='l'),
                 't' : psi.nn_site(bd.site0, d='t'),
                 'tr': psi.nn_site(bd.site0, d='r'),
                 'bl': psi.nn_site(bd.site1, d='l'),
                 'b' : psi.nn_site(bd.site1, d='b'),
                 'br': psi.nn_site(bd.site1, d='r')}
            tensors_from_psi(m, psi)

            env_t = edge_t(QA, leaf_t(m['t']))  # [l l'] [r r'] [bb bb']
            env_b = edge_b(QB, leaf_b(m['b']))  # [tt tt'] [l l'] [r r']
            env_l = cor_bl(m['bl']) @ cor_tl(m['tl'])  # [bl bl'] [tl tl']
            env_r = cor_tr(m['tr']) @ cor_br(m['br'])  # [tr tr'] [br br']
            env_t = env_l @ env_t
            env_b = env_r @ env_b
            G = tensordot(env_t, env_b, axes=((0, 2), (2, 0)))
        return G.unfuse_legs(axes=(0, 1))


    def _g_NN(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        psi = self.psi
        dirn = bd.dirn

        if dirn == "h":
            m = {'tl': psi.nn_site(bd.site0, d='t'),
                 'l' : psi.nn_site(bd.site0, d='l'),
                 'bl': psi.nn_site(bd.site0, d='b'),
                 'tr': psi.nn_site(bd.site1, d='t'),
                 'r' : psi.nn_site(bd.site1, d='r'),
                 'br': psi.nn_site(bd.site1, d='b')}
            tensors_from_psi(m, psi)

            env_l = edge_l(QA, leaf_l(m['l']))  # [t t'] [b b'] [rr rr']
            env_r = edge_r(QB, leaf_r(m['r']))  # [ll ll'] [t t'] [b b']
            env_t = cor_tl(m['tl']) @ cor_tr(m['tr'])  # [tl tl'] [tr tr']
            env_b = cor_br(m['br']) @ cor_bl(m['bl'])  # [br br'] [bl bl']
            env_l = env_b @ env_l
            env_r = env_t @ env_r
            G = tensordot(env_l, env_r, axes=((0, 2), (2, 0)))  # [ll ll'] [rr rr']
        else: # dirn == "v":
            m = {'tl': psi.nn_site(bd.site0, d='l'),
                 't' : psi.nn_site(bd.site0, d='t'),
                 'tr': psi.nn_site(bd.site0, d='r'),
                 'bl': psi.nn_site(bd.site1, d='l'),
                 'b' : psi.nn_site(bd.site1, d='b'),
                 'br': psi.nn_site(bd.site1, d='r')}
            tensors_from_psi(m, psi)

            env_t = edge_t(QA, leaf_t(m['t']))  # [l l'] [r r'] [bb bb']
            env_b = edge_b(QB, leaf_b(m['b']))  # [tt tt'] [l l'] [r r']
            env_l = cor_bl(m['bl']) @ cor_tl(m['tl'])  # [bl bl'] [tl tl']
            env_r = cor_tr(m['tr']) @ cor_br(m['br'])  # [tr tr'] [br br']
            env_t = env_l @ env_t
            env_b = env_r @ env_b
            G = tensordot(env_t, env_b, axes=((0, 2), (2, 0)))
        return G.unfuse_legs(axes=(0, 1))

    def _g_NNN(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        psi = self.psi
        dirn = bd.dirn

        if dirn == "h":
            m = {'lt' : psi.nn_site(bd.site0, d='t'),
                 'ltl': psi.nn_site(bd.site0, d='tl'),
                 'll' : psi.nn_site(bd.site0, d='l'),
                 'lbl': psi.nn_site(bd.site0, d='bl'),
                 'lb' : psi.nn_site(bd.site0, d='b'),
                 'rt' : psi.nn_site(bd.site1, d='t'),
                 'rtr': psi.nn_site(bd.site1, d='tr'),
                 'rr' : psi.nn_site(bd.site1, d='r'),
                 'rbr': psi.nn_site(bd.site1, d='br'),
                 'rb' : psi.nn_site(bd.site1, d='b')}
            tensors_from_psi(m, psi)

            lt  = edge_t(m['lt'])
            ltl = cor_tl(m['ltl'])
            ll  = edge_l(m['ll'])
            lbl = cor_bl(m['lbl'])
            lb  = edge_b(m['lb'])
            rt  = edge_t(m['rt'])
            rtr = cor_tr(m['rtr'])
            rr  = edge_r(m['rr'])
            rbr = cor_br(m['rbr'])
            rb  = edge_b(m['rb'])

            vecl = (lbl @ ll) @ (ltl @ lt)
            vecl = append_vec_tl(QA, QA, vecl)
            vecl = tensordot(lb, vecl, axes=((2, 1), (0, 1)))

            vecr = (rtr @ rr) @ (rbr @ rb)
            vecr = append_vec_br(QB, QB, vecr)
            vecr = tensordot(rt, vecr, axes=((2, 1), (0, 1)))

            G = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [ll ll'] [rr rr']
        else: # dirn == "v":
            m = {'tl' : psi.nn_site(bd.site0, d='l'),
                 'ttl': psi.nn_site(bd.site0, d='tl'),
                 'tt' : psi.nn_site(bd.site0, d='t'),
                 'ttr': psi.nn_site(bd.site0, d='tr'),
                 'tr' : psi.nn_site(bd.site0, d='r'),
                 'bl' : psi.nn_site(bd.site1, d='l'),
                 'bbl': psi.nn_site(bd.site1, d='bl'),
                 'bb' : psi.nn_site(bd.site1, d='b'),
                 'bbr': psi.nn_site(bd.site1, d='br'),
                 'br' : psi.nn_site(bd.site1, d='r')}
            tensors_from_psi(m, psi)

            tl  = edge_l(m['tl'])
            ttl = cor_tl(m['ttl'])
            tt  = edge_t(m['tt'])
            ttr = cor_tr(m['ttr'])
            tr  = edge_r(m['tr'])
            bl  = edge_l(m['bl'])
            bbl = cor_bl(m['bbl'])
            bb  = edge_b(m['bb'])
            bbr = cor_br(m['bbr'])
            br  = edge_r(m['br'])

            vect = (tl @ ttl) @ (tt @ ttr)
            vect = append_vec_tl(QA, QA, vect)
            vect = tensordot(vect, tr, axes=((2, 3), (0, 1)))

            vecb = (br @ bbr) @ (bb @ bbl)
            vecb = append_vec_br(QB, QB, vecb)
            vecb = tensordot(vecb, bl, axes=((2, 3), (0, 1)))

            G = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [tt tt'] [bb bb']
        return G.unfuse_legs(axes=(0, 1))


    def _g_NNNp(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        psi = self.psi
        dirn = bd.dirn

        if dirn == "h":
            m = {'lt' : psi.nn_site(bd.site0, d='t'),
                 'ltl': psi.nn_site(bd.site0, d='tl'),
                 'll' : psi.nn_site(bd.site0, d='l'),
                 'lbl': psi.nn_site(bd.site0, d='bl'),
                 'lb' : psi.nn_site(bd.site0, d='b'),
                 'rt' : psi.nn_site(bd.site1, d='t'),
                 'rtr': psi.nn_site(bd.site1, d='tr'),
                 'rr' : psi.nn_site(bd.site1, d='r'),
                 'rbr': psi.nn_site(bd.site1, d='br'),
                 'rb' : psi.nn_site(bd.site1, d='b')}
            m['lt_t'] = psi.nn_site(m['lt'], d='t')
            m['ltl_t'] = psi.nn_site(m['ltl'], d='t')
            m['ltl_l'] = psi.nn_site(m['ltl'], d='l')
            m['ll_l'] = psi.nn_site(m['ll'], d='l')
            m['lbl_l'] = psi.nn_site(m['lbl'], d='l')
            m['lbl_b'] = psi.nn_site(m['lbl'], d='b')
            m['lb_b'] = psi.nn_site(m['lb'], d='b')
            m['rt_t'] = psi.nn_site(m['rt'], d='t')
            m['rtr_t'] = psi.nn_site(m['rtr'], d='t')
            m['rtr_r'] = psi.nn_site(m['rtr'], d='r')
            m['rr_r'] = psi.nn_site(m['rr'], d='r')
            m['rbr_r'] = psi.nn_site(m['rbr'], d='r')
            m['rbr_b'] = psi.nn_site(m['rbr'], d='b')
            m['rb_b'] = psi.nn_site(m['rb'], d='b')

            tensors_from_psi(m, psi)

            lt  = edge_t(m['lt'], lft=leaf_t(m['lt_t']))
            ltl = cor_tl(m['ltl'], leafs=(leaf_t(m['ltl_t']), leaf_l(m['ltl_l'])))
            ll  = edge_l(m['ll'], lfl=leaf_l(m['ll_l']))
            lbl = cor_bl(m['lbl'], leafs=(leaf_b(m['lbl_b']), leaf_l(m['lbl_l'])))
            lb  = edge_b(m['lb'], lfb=leaf_b(m['lb_b']))
            rt  = edge_t(m['rt'], lft=leaf_t(m['rt_t']))
            rtr = cor_tr(m['rtr'], leafs=(leaf_t(m['rtr_t']), leaf_r(m['rtr_r'])))
            rr  = edge_r(m['rr'], lfr=leaf_r(m['rr_r']))
            rbr = cor_br(m['rbr'], leafs=(leaf_b(m['rbr_b']), leaf_r(m['rbr_r'])))
            rb  = edge_b(m['rb'], lfb=leaf_b(m['rb_b']))

            vecl = (lbl @ ll) @ (ltl @ lt)
            vecl = append_vec_tl(QA, QA, vecl)
            vecl = tensordot(lb, vecl, axes=((2, 1), (0, 1)))

            vecr = (rtr @ rr) @ (rbr @ rb)
            vecr = append_vec_br(QB, QB, vecr)
            vecr = tensordot(rt, vecr, axes=((2, 1), (0, 1)))

            G = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [ll ll'] [rr rr']
        else: # dirn == "v":
            m = {'tl' : psi.nn_site(bd.site0, d='l'),
                 'ttl': psi.nn_site(bd.site0, d='tl'),
                 'tt' : psi.nn_site(bd.site0, d='t'),
                 'ttr': psi.nn_site(bd.site0, d='tr'),
                 'tr' : psi.nn_site(bd.site0, d='r'),
                 'bl' : psi.nn_site(bd.site1, d='l'),
                 'bbl': psi.nn_site(bd.site1, d='bl'),
                 'bb' : psi.nn_site(bd.site1, d='b'),
                 'bbr': psi.nn_site(bd.site1, d='br'),
                 'br' : psi.nn_site(bd.site1, d='r')}
            m['tl_l'] = psi.nn_site(m['tl'], d='l')
            m['ttl_l'] = psi.nn_site(m['ttl'], d='l')
            m['ttl_t'] = psi.nn_site(m['ttl'], d='t')
            m['tt_t'] = psi.nn_site(m['tt'], d='t')
            m['ttr_t'] = psi.nn_site(m['ttr'], d='t')
            m['ttr_r'] = psi.nn_site(m['ttr'], d='r')
            m['tr_r'] = psi.nn_site(m['tr'], d='r')
            m['bl_l'] = psi.nn_site(m['bl'], d='l')
            m['bbl_l'] = psi.nn_site(m['bbl'], d='l')
            m['bbl_b'] = psi.nn_site(m['bbl'], d='b')
            m['bb_b'] = psi.nn_site(m['bb'], d='b')
            m['bbr_b'] = psi.nn_site(m['bbr'], d='b')
            m['bbr_r'] = psi.nn_site(m['bbr'], d='r')
            m['br_r'] = psi.nn_site(m['br'], d='r')

            tensors_from_psi(m, psi)

            tl  = edge_l(m['tl'], lfl=leaf_l(m['tl_l']))
            ttl = cor_tl(m['ttl'], leafs=(leaf_t(m['ttl_t']), leaf_l(m['ttl_l'])))
            tt  = edge_t(m['tt'], lft=leaf_t(m['tt_t']))
            ttr = cor_tr(m['ttr'], leafs=(leaf_t(m['ttr_t']), leaf_r(m['ttr_r'])))
            tr  = edge_r(m['tr'], lfr=leaf_r(m['tr_r']))
            bl  = edge_l(m['bl'], lfl=leaf_l(m['bl_l']))
            bbl = cor_bl(m['bbl'], leafs=(leaf_b(m['bbl_b']), leaf_l(m['bbl_l'])))
            bb  = edge_b(m['bb'], lfb=leaf_b(m['bb_b']))
            bbr = cor_br(m['bbr'], leafs=(leaf_b(m['bbr_b']), leaf_r(m['bbr_r'])))
            br  = edge_r(m['br'], lfr=leaf_r(m['br_r']))

            vect = (tl @ ttl) @ (tt @ ttr)
            vect = append_vec_tl(QA, QA, vect)
            vect = tensordot(vect, tr, axes=((2, 3), (0, 1)))

            vecb = (br @ bbr) @ (bb @ bbl)
            vecb = append_vec_br(QB, QB, vecb)
            vecb = tensordot(vecb, bl, axes=((2, 3), (0, 1)))

            G = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [tt tt'] [bb bb']
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
