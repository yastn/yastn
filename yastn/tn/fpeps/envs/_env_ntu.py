import yastn
from yastn import ncon
from ._env_auxlliary import con_tl, con_tr, con_br, con_bl, con_l, con_r, con_b, con_t, con_Q_l, con_Q_r, con_Q_t, con_Q_b

class EnvNTU:
    def __init__(self, psi, which='NN'):
        self.psi = psi
        self.which = which

    def bond_metric(self, bd, QA, QB):
        """ Calculates bond metric. """
        if self.which == 'NN':
            return self._g_NN(bd, QA, QB)
        else:
            pass

    def _g_NN(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        env = self.ntu_tensors(bd)
        dirn = bd.dirn

        G = {}
        for k, v in env.items():
            if v is None:
                leg = self.psi[(0, 0)].get_legs(axes=-1)
                leg, _ = yastn.leg_undo_product(leg)  # last leg of A should be fused
                fid = yastn.eye(config=self.psi[(0, 0)].config, legs=[leg, leg.conj()]).diag()
                G[k] = trivial_tensor(fid)
            else:
                G[k] = self.psi[v]

        if dirn == "h":
            m_tl, m_l, m_bl = con_tl(G['tl']), con_l(G['l']), con_bl(G['bl'])
            m_tr, m_r, m_br = con_tr(G['tr']), con_r(G['r']), con_br(G['br'])
            env_l = con_Q_l(QA, m_l)  # [t t'] [b b'] [rr rr']
            env_r = con_Q_r(QB, m_r)  # [ll ll'] [t t'] [b b']
            env_t = m_tl @ m_tr  # [tl tl'] [tr tr']
            env_b = m_br @ m_bl  # [br br'] [bl bl']
            G = ncon((env_t, env_l, env_r, env_b), ((1, 4), (1, 3, -1), (-2, 4, 2), (2, 3)))  # [ll ll'] [rr rr']
        elif dirn == "v":
            m_tl, m_t, m_tr = con_tl(G['tl']), con_t(G['t']), con_tr(G['tr'])
            m_bl, m_b, m_br = con_bl(G['bl']), con_b(G['b']), con_br(G['br'])
            env_t = con_Q_t(QA, m_t)  # [l l'] [r r'] [bb bb']
            env_b = con_Q_b(QB, m_b)  # [tt tt'] [l l'] [r r']
            env_l = m_bl @ m_tl  # [bl bl'] [tl tl']
            env_r = m_tr @ m_br  # [tr tr'] [br br']
            G = ncon((env_l, env_t, env_b, env_r), ((4, 1), (1, 3, -1), (-2, 4, 2), (3, 2)))  # [tt tt'] [bb bb']

        return G.unfuse_legs(axes=(0, 1))

    def ntu_tensors(self, bds):
        """
        Returns a dictionary containing the neighboring sites of the bond `bds`.
        """
        neighbors = {}
        site1, site_2 = bds.site0, bds.site1
        psi = self.psi
        if bds.dirn == 'h':
            neighbors['tl'], neighbors['l'], neighbors['bl'] = psi.nn_site(site1, d='t'), psi.nn_site(site1, d='l'), psi.nn_site(site1, d='b')
            neighbors['tr'], neighbors['r'], neighbors['br'] = psi.nn_site(site_2, d='t'), psi.nn_site(site_2, d='r'), psi.nn_site(site_2, d='b')
        elif bds.dirn == 'v':
            neighbors['tl'], neighbors['t'], neighbors['tr'] = psi.nn_site(site1, d='l'), psi.nn_site(site1, d='t'), psi.nn_site(site1, d='r')
            neighbors['bl'], neighbors['b'], neighbors['br'] = psi.nn_site(site_2, d='l'), psi.nn_site(site_2, d='b'), psi.nn_site(site_2, d='r')
        return neighbors


def trivial_tensor(fid):
    """ fid is identity operator in local space with desired symmetry """
    A = fid.fuse_legs(axes=[(0, 1)])
    for s in (-1, 1, 1, -1):
        A = A.add_leg(axis=0, s=s)
    return A.fuse_legs(axes=((0, 1), (2, 3), 4))
