import yastn
from yastn.tn.fpeps.gates._gates import trivial_tensor
from yastn import tensordot, swap_gate, fuse_legs, ncon
from ._env_auxlliary import con_tl, con_tr, con_br, con_bl, con_l, con_r, con_b, con_t, con_Q_l, con_Q_r, con_Q_t, con_Q_b

class EnvCluster:
    def __init__(self, psi, depth=1):
        self.psi = psi
        self.depth = depth
        self.G = None  # Environmental tensors
        # env class should not have data container; create everything in place

    def bond_metric(self, bd, QA, QB, dirn):
        """ Calculates bond metric. """
        if self.depth == 0:
            pass
        if self.depth == 1:
            return self.env_NTU(bd, QA, QB, dirn)

    def env_NTU(self, bd, QA, QB, dirn):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        env = self.ntu_tensors(bd)
        G = {}
        for ms in env.keys():
            if env[ms] is None:
                leg = self.psi[(0, 0)].get_legs(axes=-1)
                leg, _ = yastn.leg_undo_product(leg)  # last leg of A should be fused
                fid = yastn.eye(config=self.psi[(0, 0)].config, legs=[leg, leg.conj()]).diag()
                G[ms] = trivial_tensor(fid)
            else:
                G[ms] = self.psi[env[ms]]

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
        if self.psi.lattice == 'checkerboard':
            if bds.dirn == 'h':
                neighbors['tl'], neighbors['l'], neighbors['bl'] = site_2, site_2, site_2
                neighbors['tr'], neighbors['r'], neighbors['br'] = site1, site1, site1
            elif bds.dirn == 'v':
                neighbors['tl'], neighbors['t'], neighbors['tr'] = site_2, site_2, site_2
                neighbors['bl'], neighbors['b'], neighbors['br'] = site1, site1, site1
        else:
            if bds.dirn == 'h':
                neighbors['tl'], neighbors['l'], neighbors['bl'] = self.nn_site(site1, d='t'), self.nn_site(site1, d='l'), self.nn_site(site1, d='b')
                neighbors['tr'], neighbors['r'], neighbors['br'] = self.nn_site(site_2, d='t'), self.nn_site(site_2, d='r'), self.nn_site(site_2, d='b')
            elif bds.dirn == 'v':
                neighbors['tl'], neighbors['t'], neighbors['tr'] = self.nn_site(site1, d='l'), self.nn_site(site1, d='t'), self.nn_site(site1, d='r')
                neighbors['bl'], neighbors['b'], neighbors['br'] = self.nn_site(site_2, d='l'), self.nn_site(site_2, d='b'), self.nn_site(site_2, d='r')

        return neighbors

