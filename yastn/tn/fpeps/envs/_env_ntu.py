from .... import tensordot, YastnError, ones, Leg
from ._env_auxlliary import *
from .._doublePepsTensor import DoublePepsTensor

class EnvNTU:
    def __init__(self, psi, which='NN'):
        if which not in ('NN', 'NNN'):
            raise YastnError(f" Type of EnvNTU {which} not recognized.")
        self.psi = psi
        self.which = which

    def bond_metric(self, bd, QA, QB):
        """ Calculates bond metric. """
        if self.which == 'NN':
            return self._g_NN(bd, QA, QB)
        if self.which == 'NNN':
            return self._g_NNN(bd, QA, QB)

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

            env_l = con_Al(QA, con_l(m['l']))  # [t t'] [b b'] [rr rr']
            env_r = con_Ar(QB, con_r(m['r']))  # [ll ll'] [t t'] [b b']
            env_t = con_tl(m['tl']) @ con_tr(m['tr'])  # [tl tl'] [tr tr']
            env_b = con_br(m['br']) @ con_bl(m['bl'])  # [br br'] [bl bl']
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

            env_t = con_At(QA, con_t(m['t']))  # [l l'] [r r'] [bb bb']
            env_b = con_Ab(QB, con_b(m['b']))  # [tt tt'] [l l'] [r r']
            env_l = con_bl(m['bl']) @ con_tl(m['tl'])  # [bl bl'] [tl tl']
            env_r = con_tr(m['tr']) @ con_br(m['br'])  # [tr tr'] [br br']
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

            lt  = con_At(m['lt'])
            ltl = con_tl(m['ltl'])
            ll  = con_Al(m['ll'])
            lbl = con_bl(m['lbl'])
            lb  = con_Ab(m['lb'])
            rt  = con_At(m['rt'])
            rtr = con_tr(m['rtr'])
            rr  = con_Ar(m['rr'])
            rbr = con_br(m['rbr'])
            rb  = con_Ab(m['rb'])

            vecl = (lbl @ ll) @ (ltl @ lt)
            uQA = QA.unfuse_legs(axes=(0, 1))
            dQA = DoublePepsTensor(uQA, uQA.swap_gate(axes=(0, 1, 2, 3)))
            vecl = dQA.append_a_tl(vecl)
            vecl = tensordot(lb, vecl, axes=((2, 1), (0, 1)))

            vecr = (rtr @ rr) @ (rbr @ rb)
            uQB = QB.unfuse_legs(axes=(0, 1))
            dQB = DoublePepsTensor(uQB, uQB.swap_gate(axes=(0, 1, 2, 3)))
            vecr = dQB.append_a_br(vecr)
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

            tl  = con_Al(m['tl'])
            ttl = con_tl(m['ttl'])
            tt  = con_At(m['tt'])
            ttr = con_tr(m['ttr'])
            tr  = con_Ar(m['tr'])
            bl  = con_Al(m['bl'])
            bbl = con_bl(m['bbl'])
            bb  = con_Ab(m['bb'])
            bbr = con_br(m['bbr'])
            br  = con_Ar(m['br'])

            vect = (tl @ ttl) @ (tt @ ttr)
            uQA = QA.unfuse_legs(axes=(0, 1))
            dQA = DoublePepsTensor(uQA, uQA.swap_gate(axes=(0, 1, 2, 3)))
            vect = dQA.append_a_tl(vect)
            vect = tensordot(vect, tr, axes=((2, 3), (0, 1)))

            vecb = (br @ bbr) @ (bb @ bbl)
            uQB = QB.unfuse_legs(axes=(0, 1))
            dQB = DoublePepsTensor(uQB, uQB.swap_gate(axes=(0, 1, 2, 3)))
            vecb = dQB.append_a_br(vecb)
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
