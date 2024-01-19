from .... import ncon, YastnError, leg_undo_product, eye, ones, Leg
from ._env_auxlliary import *

class EnvNTU:
    def __init__(self, psi, which='NN'):
        if which not in ('NN',):
            raise YastnError(f" Type of EnvNTU {which} not recognized.")
        self.psi = psi
        self.which = which

    def bond_metric(self, bd, QA, QB):
        """ Calculates bond metric. """
        if self.which == 'NN':
            return self._g_NN(bd, QA, QB)


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

            env_l = con_Q_l(QA, con_l(m['l']))  # [t t'] [b b'] [rr rr']
            env_r = con_Q_r(QB, con_r(m['r']))  # [ll ll'] [t t'] [b b']
            env_t = con_tl(m['tl']) @ con_tr(m['tr'])  # [tl tl'] [tr tr']
            env_b = con_br(m['br']) @ con_bl(m['bl'])  # [br br'] [bl bl']
            G = ncon((env_t, env_l, env_r, env_b), ((1, 4), (1, 3, -1), (-2, 4, 2), (2, 3)))  # [ll ll'] [rr rr']
        else: # dirn == "v":
            m = {'tl': psi.nn_site(bd.site0, d='l'),
                 't' : psi.nn_site(bd.site0, d='t'),
                 'tr': psi.nn_site(bd.site0, d='r'),
                 'bl': psi.nn_site(bd.site1, d='l'),
                 'b' : psi.nn_site(bd.site1, d='b'),
                 'br': psi.nn_site(bd.site1, d='r')}
            tensors_from_psi(m, psi)

            env_t = con_Q_t(QA, con_t(m['t']))  # [l l'] [r r'] [bb bb']
            env_b = con_Q_b(QB, con_b(m['b']))  # [tt tt'] [l l'] [r r']
            env_l = con_bl(m['bl']) @ con_tl(m['tl'])  # [bl bl'] [tl tl']
            env_r = con_tr(m['tr']) @ con_br(m['br'])  # [tr tr'] [br br']
            G = ncon((env_l, env_t, env_b, env_r), ((4, 1), (1, 3, -1), (-2, 4, 2), (3, 2)))  # [tt tt'] [bb bb']

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
