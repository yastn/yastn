from .... import tensordot, YastnError, ones, Leg
from ._env_auxlliary import *

class EnvApprox:
    def __init__(self, psi, which='65', D=4):
        if which not in ('65', '65h'):
            raise YastnError(f" Type of EnvApprox {which} not recognized.")
        self.psi = psi
        self.which = which
        self.D = D
        self.data = {}

    def bond_metric(self, bd, QA, QB):
        """ Calculates bond metric. """
        if self.which == '65':
            return self._g_NN(bd, QA, QB)
        if self.which == '65h':
            return self._g_NNh(bd, QA, QB)



    def _g_6x5(self, bd, QA, QB):
        """
        Calculates the metric tensor g for the given PEPS tensor network using the NTU algorithm.
        """
        if bd.dirn == "h":
            assert self.psi.nn_site(bd.site0, (0, 1)) == bd.site1
            # (-2,-2)==(-2,-1)==(-2,0)==(-2,1)==(-2,2)==(-2,3)
            #    ||      ||       ||      ||      ||      ||
            # (-1,-2)==(-1,-1)==(-1,0)==(-1,1)==(-1,2)==(-1,3)
            #    ||      ||       ||      ||      ||      ||
            # (0, -2)==(0, -1)=== GA  ++  GB ===(0, 2)==(0, 3)
            #    ||      ||       ||      ||      ||      ||
            # (1, -2)==(1, -1)==(1, 0)==(1, 1)==(1, 2)==(1, 3)
            #    ||      ||       ||      ||      ||      ||
            # (2, -2)==(2, -1)==(2, 0)==(2, 1)==(2, 2)==(2, 3)

            m = {(nx, ny): self.psi.nn_site(bd.site0, d=(nx, ny)) for nx in range(-2, 3) for ny in range(-2, 4)}
            [m.pop(k) for k in [(0, 0), (0, 1)]]
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
            #   (-2,-2)==(-2,-1)==(-2,0)==(-2,1)==(-2,2)
            #      ||       ||      ||      ||      ||
            #   (-1,-2)==(-1,-1)==(-1,0)==(-1,1)==(-1,2)
            #      ||       ||      ||      ||      ||
            #   (0, -2)==(0, -1)=== GA ===(0, 1)==(0, 2)
            #      ||       ||      ++      ||      ||
            #   (1, -2)==(1, -1)=== GB ===(1, 1)==(1, 2)
            #      ||       ||      ||      ||      ||
            #   (2, -2)==(2, -1)==(2, 0)==(2, 1)==(2, 2)
            #      ||       ||      ||      ||      ||
            #   (3, -2)==(3, -1)==(3, 0)==(3, 1)==(3, 2)

            m = {(nx, ny): self.psi.nn_site(bd.site0, d=(nx, ny)) for nx in range(-2, 4) for ny in range(-2, 3)}
            [m.pop(k) for k in [(0, 0), (1, 0)]]
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


def tensors_from_psi(m, psi):
    if any(v is None for v in m.values()):
        cfg = psi[(0, 0)].config
        triv = ones(cfg, legs=[Leg(cfg, t=((0,) * cfg.sym.NSYM,), D=(1,))])
        for s in (-1, 1, 1, -1):
            triv = triv.add_leg(axis=0, s=s)
        triv = triv.fuse_legs(axes=((0, 1), (2, 3), 4))
    for k, v in m.items():
        m[k] = triv if v is None else psi[v]
