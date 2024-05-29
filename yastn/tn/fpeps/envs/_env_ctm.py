from dataclasses import dataclass
from .... import rand, ones, eye, YastnError, Leg, tensordot, qr, truncation_mask_multiplets, truncation_mask
from ... import mps
from .._peps import Peps, Peps2Layers
from .._gates_auxiliary import apply_gate_onsite, gate_product_operator, gate_fix_order
from .._geometry import Bond
from ._env_auxlliary import *


@dataclass()
class EnvCTM_local():
    r""" Dataclass for CTM environment tensors associated with Peps lattice site. """
    tl = None # top-left
    tr = None # top-right
    bl = None # bottom-left
    br = None # bottom-right
    t = None  # top
    b = None  # bottom
    l = None  # left
    r = None  # right


@dataclass()
class EnvCTM_projectors():
    r""" Dataclass for CTM projectors associated with Peps lattice site. """
    hlt : any = None  # horizontal left top
    hlb : any = None  # horizontal left bottom
    hrt : any = None  # horizontal right top
    hrb : any = None  # horizontal right bottom
    vtl : any = None  # vertical top left
    vtr : any = None  # vertical top right
    vbl : any = None  # vertical bottom left
    vbr : any = None  # vertical bottom right


class EnvCTM(Peps):
    def __init__(self, psi, init='rand', leg=None):
        r"""
        Environment used in Corner Transfer Matrix Renormalization algorithm.

        Parameters
        ----------
        psi: yastn.tn.Peps
            Peps lattice to be contracted using CTM.
            If psi has physical legs, 2-layers peps with no physical legs is formed.
        """
        super().__init__(psi.geometry)
        self.psi = Peps2Layers(psi) if psi.has_physical() else psi
        if init not in (None, 'rand', 'eye'):
            raise YastnError(f"EnvCTM {init=} not recognized. Should be 'rand', 'eye', None.")
        for site in self.sites():
            self[site] = EnvCTM_local()
        if init is not None:
            self.reset_(init=init, leg=leg)

    def copy(self):
        env = EnvCTM(self.psi, init=None)
        for site in env.sites():
            for dirn in ['tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r']:
                setattr(env[site], dirn, getattr(self[site], dirn).copy())
        return env

    def reset_(self, init='rand', leg=None):
        r""" Initialize random CTMRG environments of peps tensors A. """
        config = self.psi.config
        leg0 = Leg(config, s=1, t=((0,) * config.sym.NSYM,), D=(1,))

        if init == 'nn':
            pass
        else:
            if leg is None:
                leg = leg0

            for site in self.sites():
                legs = self.psi[site].get_legs()

                for dirn in ('tl', 'tr', 'bl', 'br'):
                    if self.nn_site(site, d=dirn) is None or init == 'eye':
                        setattr(self[site], dirn, eye(config, legs=[leg0, leg0.conj()], isdiag=False))
                    else:
                        setattr(self[site], dirn, rand(config, legs=[leg, leg.conj()]))

                for ind, dirn in enumerate('tlbr'):
                    if self.nn_site(site, d=dirn) is None or init == 'eye':
                        tmp1 = identity_boundary(config, legs[ind].conj())
                        tmp0 = eye(config, legs=[leg0, leg0.conj()], isdiag=False)
                        tmp = tensordot(tmp0, tmp1, axes=((), ())).transpose(axes=(0, 2, 1))
                        setattr(self[site], dirn, tmp)
                    else:
                        setattr(self[site], dirn, rand(config, legs=[leg, legs[ind].conj(), leg.conj()]))

    def boundary_mps(self, n, dirn):
        r""" Convert environmental tensors of Ctm to an MPS """
        if dirn == 'b':
            H = mps.Mps(N=self.Ny)
            for ny in range(self.Ny):
                H.A[ny] = self[n, ny].b.transpose(axes=(2, 1, 0))
            H = H.conj()
        elif dirn == 'r':
            H = mps.Mps(N=self.Nx)
            for nx in range(self.Nx):
                H.A[nx] = self[nx, n].r
        elif dirn == 't':
            H = mps.Mps(N=self.Ny)
            for ny in range(self.Ny):
                H.A[ny] = self[n, ny].t
        elif dirn == 'l':
            H = mps.Mps(N=self.Nx)
            for nx in range(self.Nx):
                H.A[nx] = self[nx, n].l.transpose(axes=(2, 1, 0))
            H = H.conj()
        return H

    def measure_1site(self, op, site=None) -> dict:
        r"""
        Calculate local expectation values within CTM environment.

        Return a number if site is provided.
        If None, returns a dictionary {site: value} for all unique lattice sites.

        Parameters
        ----------
        env: class CtmEnv
            class containing ctm environment tensors along with lattice structure data
        op: single site operator
        """
        if site is None:
            return {site: self.measure_1site(op, site) for site in self.sites()}

        lenv = self[site]
        ten = self.psi[site]
        vect = (lenv.l @ lenv.tl) @ (lenv.t @ lenv.tr)
        vecb = (lenv.r @ lenv.br) @ (lenv.b @ lenv.bl)

        tmp = ten._attach_01(vect)
        val_no = tensordot(vecb, tmp, axes=((0, 1, 2, 3), (2, 3, 1, 0))).to_number()

        ten.top = apply_gate_onsite(ten.top, op)
        tmp = ten._attach_01(vect)
        val_op = tensordot(vecb, tmp, axes=((0, 1, 2, 3), (2, 3, 1, 0))).to_number()

        return val_op / val_no

    def measure_nn(self, O0, O1, bond=None):
        r"""
        Calculate nearest-neighbor expectation values within CTM environment.

        Return a number if the nearest-neighbor bond is provided.
        If None, returns a dictionary {bond: value} for all unique lattice bonds.

        Parameters
        ----------
        O0, O1: yastn.Tensor
            Calculate <O0_s0 O1_s1>.
            O1 is applied first, which might matter for fermionic operators.
        bond: yastn.tn.fpeps.Bond | tuple[tuple[int, int], tuple[int, int]]
            Bond of the form (s0, s1). Sites s0 and s1 should be nearest-neighbors on the lattice.
        """

        if bond is None:
             return {bond: self.measure_nn(O0, O1, bond) for bond in self.bonds()}

        bond = Bond(*bond)
        dirn, l_ordered = self.nn_bond_type(bond)
        f_ordered = self.f_ordered(bond)
        s0, s1 = bond if l_ordered else bond[::-1]
        env0, env1 = self[s0], self[s1]
        ten0, ten1 = self.psi[s0], self.psi[s1]

        if O0.ndim == 2 and O1.ndim == 2:
            G0, G1 = gate_product_operator(O0, O1, l_ordered, f_ordered)
        elif O0.ndim == 3 and O1.ndim == 3:
            G0, G1 = gate_fix_order(O0, O1, l_ordered, f_ordered)
        else:
            raise YastnError("Both operators O0 and O1 should have the same ndim==2, or ndim=3.")

        if dirn == 'h':
            vecl = (env0.bl @ env0.l) @ (env0.tl @ env0.t)
            vecr = (env1.tr @ env1.r) @ (env1.br @ env1.b)

            tmp0 = ten0._attach_01(vecl)
            tmp0 = tensordot(env0.b, tmp0, axes=((2, 1), (0, 1)))
            tmp1 = ten1._attach_23(vecr)
            tmp1 = tensordot(env1.t, tmp1, axes=((2, 1), (0, 1)))
            val_no = tensordot(tmp0, tmp1, axes=((0, 1, 2), (1, 0, 2))).to_number()

            ten0.top = apply_gate_onsite(ten0.top, G0, dirn='l')
            ten1.top = apply_gate_onsite(ten1.top, G1, dirn='r')

            tmp0 = ten0._attach_01(vecl)
            tmp0 = tensordot(env0.b, tmp0, axes=((2, 1), (0, 1)))
            tmp1 = ten1._attach_23(vecr)
            tmp1 = tensordot(env1.t, tmp1, axes=((2, 1), (0, 1)))
            val_op = tensordot(tmp0, tmp1, axes=((0, 1, 2), (1, 0, 2))).to_number()
        else:  # dirn == 'v':
            vect = (env0.l @ env0.tl) @ (env0.t @ env0.tr)
            vecb = (env1.r @ env1.br) @ (env1.b @ env1.bl)

            tmp0 = ten0._attach_01(vect)
            tmp0 = tensordot(tmp0, env0.r, axes=((2, 3), (0, 1)))
            tmp1 = ten1._attach_23(vecb)
            tmp1 = tensordot(tmp1, env1.l, axes=((2, 3), (0, 1)))
            val_no = tensordot(tmp0, tmp1, axes=((0, 1, 2), (2, 1, 0))).to_number()

            ten0.top = apply_gate_onsite(ten0.top, G0, dirn='t')
            ten1.top = apply_gate_onsite(ten1.top, G1, dirn='b')

            tmp0 = ten0._attach_01(vect)
            tmp0 = tensordot(tmp0, env0.r, axes=((2, 3), (0, 1)))
            tmp1 = ten1._attach_23(vecb)
            tmp1 = tensordot(tmp1, env1.l, axes=((2, 3), (0, 1)))
            val_op = tensordot(tmp0, tmp1, axes=((0, 1, 2), (2, 1, 0))).to_number()

        return val_op / val_no

    def update_(env, opts_svd, method='2site'):
        r"""
        Perform one step of CTMRG update. Environment tensors are updated in place.

        The function performs a CTMRG update for a square lattice using the corner transfer matrix
        renormalization group (CTMRG) algorithm. The update is performed in two steps: a horizontal move
        and a vertical move. The projectors for each move are calculated first, and then the tensors in
        the CTM environment are updated using the projectors. The boundary conditions of the lattice
        determine whether trivial projectors are needed for the move.

        Parameters
        ----------
        opts_svd: dict
            A dictionary of options to pass to the SVD algorithm.
        method: str
            '2site' or '1site'. Default is '2site'
            '2site' uses the standard 4x4 enlarged corners, allowing to enlarge EnvCTM bond dimension.
            '1site' uses smaller 4x2 corners. It is faster, but does not allow to grow EnvCTM bond dimension.

        Returns
        -------
        proj: Peps structure loaded with CTM projectors related to all lattice site.
        """
        if all(s not in opts_svd for s in ('tol', 'tol_block')):
            opts_svd['tol'] = 1e-14
        if method not in ('1site', '2site'):
            raise YastnError(f"CTM update {method=} not recognized. Should be '1site' or '2site'")
        update_proj_ = update_2site_projectors_ if method == '2site' else update_1site_projectors_
        #
        # Empty structure for projectors
        proj = Peps(env.geometry)
        for site in proj.sites():
            proj[site] = EnvCTM_projectors()
        #
        # horizontal projectors
        for site in env.sites():
            update_proj_(proj, site, 'lr', env, opts_svd)
        trivial_projectors_(proj, 'lr', env)  # fill None's
        #
        # horizontal move
        env_tmp = EnvCTM(env.psi, init=None)  # empty environments
        for site in env.sites():
            update_env_horizontal_(env_tmp, site, env, proj)
        update_old_env_(env, env_tmp)
        #
        # vertical projectors
        for site in env.sites():
            update_proj_(proj, site, 'tb', env, opts_svd)
        trivial_projectors_(proj, 'tb', env)
        #
        # vertical move
        env_tmp = EnvCTM(env.psi, init=None)
        for site in env.sites():
            update_env_vertical_(env_tmp, site, env, proj)
        update_old_env_(env, env_tmp)
        #
        return proj

    def bond_metric(self, Q0, Q1, s0, s1, dirn):
        r"""
        Calculates Full-Update metric tensor.

        For dirn == 'h':

        tl == tt  ==  tt == tr
        |     |        |     |
        ll == GA-+  +-GB == rr
        |     |        |     |
        bl == bb  ==  bb == br

        For dirn == 'v':

        tl == tt == tr
        |     |      |
        ll == GA == rr
        |     ++     |
        ll == GB == rr
        |     |      |
        bl == bb == br
        """

        env0, env1 = self[s0], self[s1]
        if dirn == "h":
            assert self.psi.nn_site(s0, (0, 1)) == s1
            vecl = append_vec_tl(Q0, Q0, env0.l @ (env0.tl @ env0.t))
            vecl = tensordot(env0.b @ env0.bl, vecl, axes=((2, 1), (0, 1)))
            vecr = append_vec_br(Q1, Q1, env1.r  @ (env1.br @ env1.b))
            vecr = tensordot(env1.t @ env1.tr, vecr, axes=((2, 1), (0, 1)))
            g = tensordot(vecl, vecr, axes=((0, 1), (1, 0)))  # [rr rr'] [ll ll']
        else: # dirn == "v":
            assert self.psi.nn_site(s0, (1, 0)) == s1
            vect = append_vec_tl(Q0, Q0, env0.l @ (env0.tl @ env0.t))
            vect = tensordot(vect, env0.tr @ env0.r, axes=((2, 3), (0, 1)))
            vecb = append_vec_br(Q1, Q1, env1.r @ (env1.br @ env1.b))
            vecb = tensordot(vecb, env1.bl @ env1.l, axes=((2, 3), (0, 1)))
            g = tensordot(vect, vecb, axes=((0, 2), (2, 0)))  # [bb bb'] [tt tt']

        g = g / g.trace(axes=(0, 1)).to_number()
        return g.unfuse_legs(axes=(0, 1)).fuse_legs(axes=((1, 3), (0, 2)))

    def save_to_dict(self):
        """
        Serialize EnvCTM into a dictionary.
        """
        psi = self.psi
        if isinstance(psi, Peps2Layers):
            psi = psi.ket

        d = {'class': 'EnvCTM',
             'psi': psi.save_to_dict(),
             'data': {}}
        for site in self.sites():
            d_local = {dirn: getattr(self[site], dirn).save_to_dict()
                       for dirn in ['tl', 'tr', 'bl', 'br', 't', 'l', 'b', 'r']}
            d['data'][site] = d_local
        return d

    def check_corner_bond_dimension(env, disp=False):

        dict_bond_dimension = {}
        dict_symmetric_sector = {}
        for site in env.sites():
            if disp:
                print(site)
            corners = [env[site].tl, env[site].bl, env[site].br, env[site].tr]
            corners_id = ["tl", "bl", "br", "tr"]
            for ii in range(4):
                dict_symmetric_sector[site, corners_id[ii]] = []
                dict_bond_dimension[site, corners_id[ii]] = []
                if disp:
                    print(corners_id[ii])
                for leg in range (0, 2):
                    temp_t = []
                    temp_D = []
                    for it in range(len(corners[ii].get_legs()[leg].t)):
                        temp_t.append(corners[ii].get_legs()[leg].t[it])
                        temp_D.append(corners[ii].get_legs()[leg].D[it])
                    if disp:
                        print(temp_t)
                        print(temp_D)
                    dict_symmetric_sector[site, corners_id[ii]].append(temp_t)
                    dict_bond_dimension[site, corners_id[ii]].append(temp_D)
        return [dict_bond_dimension, dict_symmetric_sector]




def update_2site_projectors_(proj, site, dirn, env, opts_svd):
    r"""
    Calculate new projectors for CTM moves from 4x4 extended corners.
    """
    psi = env.psi
    sites = [psi.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))]
    if None in sites:
        return

    tl, tr, bl, br = sites

    cor_tl = psi[tl]._attach_01(env[tl].l @ env[tl].tl @ env[tl].t)
    cor_tl = cor_tl.fuse_legs(axes=((0, 1), (2, 3)))
    cor_bl = psi[bl]._attach_12(env[bl].b @ env[bl].bl @ env[bl].l)
    cor_bl = cor_bl.fuse_legs(axes=((0, 1), (2, 3)))
    cor_tr = psi[tr]._attach_30(env[tr].t @ env[tr].tr @ env[tr].r)
    cor_tr = cor_tr.fuse_legs(axes=((0, 1), (2, 3)))
    cor_br = psi[br]._attach_23(env[br].r @ env[br].br @ env[br].b)
    cor_br = cor_br.fuse_legs(axes=((0, 1), (2, 3)))

    if ('l' in dirn) or ('r' in dirn):
        cor_tt = cor_tl @ cor_tr
        cor_bb = cor_br @ cor_bl

    if 'r' in dirn:
        _, r_t = qr(cor_tt, axes=(0, 1))
        _, r_b = qr(cor_bb, axes=(1, 0))
        proj[tr].hrb, proj[br].hrt = proj_corners(r_t, r_b, opts_svd=opts_svd)

    if 'l' in dirn:
        _, r_t = qr(cor_tt, axes=(1, 0))
        _, r_b = qr(cor_bb, axes=(0, 1))
        proj[tl].hlb, proj[bl].hlt = proj_corners(r_t, r_b, opts_svd=opts_svd)

    if ('t' in dirn) or ('b' in dirn):
        cor_ll = cor_bl @ cor_tl
        cor_rr = cor_tr @ cor_br

    if 't' in dirn:
        _, r_l = qr(cor_ll, axes=(0, 1))
        _, r_r = qr(cor_rr, axes=(1, 0))
        proj[tl].vtr, proj[tr].vtl = proj_corners(r_l, r_r, opts_svd=opts_svd)

    if 'b' in dirn:
        _, r_l = qr(cor_ll, axes=(1, 0))
        _, r_r = qr(cor_rr, axes=(0, 1))
        proj[bl].vbr, proj[br].vbl = proj_corners(r_l, r_r, opts_svd=opts_svd)


def update_1site_projectors_(proj, site, dirn, env, opts_svd):
    r"""
    Calculate new projectors for CTM moves from 4x2 extended corners.
    """
    psi = env.psi
    sites = [psi.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))]
    if None in sites:
        return

    tl, tr, bl, br = sites

    if ('l' in dirn) or ('r' in dirn):
        cor_tl = (env[bl].tl @ env[bl].t).fuse_legs(axes=((0, 1), 2))
        cor_tr = (env[br].t @ env[br].tr).fuse_legs(axes=(0, (2, 1)))
        cor_br = (env[tr].br @ env[tr].b).fuse_legs(axes=((0, 1), 2))
        cor_bl = (env[tl].b @ env[tl].bl).fuse_legs(axes=(0, (2, 1)))
        r_tl, r_tr = regularize_1site_corners(cor_tl, cor_tr)
        r_br, r_bl = regularize_1site_corners(cor_br, cor_bl)

    if 'r' in dirn:
        proj[tr].hrb, proj[br].hrt = proj_corners(r_tr, r_br, opts_svd=opts_svd)

    if 'l' in dirn:
        proj[tl].hlb, proj[bl].hlt = proj_corners(r_tl, r_bl, opts_svd=opts_svd)

    if ('t' in dirn) or ('b' in dirn):
        cor_bl = (env[br].bl @ env[br].l).fuse_legs(axes=((0, 1), 2))
        cor_tl = (env[tr].l @ env[tr].tl).fuse_legs(axes=(0, (2, 1)))
        cor_tr = (env[tl].tr @ env[tl].r).fuse_legs(axes=((0, 1), 2))
        cor_br = (env[bl].r @ env[bl].br).fuse_legs(axes=(0, (2, 1)))
        r_bl, r_tl = regularize_1site_corners(cor_bl, cor_tl)
        r_tr, r_br = regularize_1site_corners(cor_tr, cor_br)

    if 't' in dirn:
        proj[tl].vtr, proj[tr].vtl = proj_corners(r_tl, r_tr, opts_svd=opts_svd)

    if 'b' in dirn:
        proj[bl].vbr, proj[br].vbl = proj_corners(r_bl, r_br, opts_svd=opts_svd)


def regularize_1site_corners(cor_0, cor_1):
    Q_0, R_0 = qr(cor_0, axes=(0, 1))
    Q_1, R_1 = qr(cor_1, axes=(1, 0))
    R01 = tensordot(R_0, R_1, axes=(1, 1))
    U_0, S, U_1 = R01.svd(axes=(0, 1), fix_signs=True)
    S = S.sqrt()
    r_0 = tensordot((U_0 @ S), Q_0, axes=(0, 1))
    r_1 = tensordot((S @ U_1), Q_1, axes=(1, 1))
    return r_0, r_1


def proj_corners(r0, r1, fix_signs, opts_svd):
    r""" Projectors in between r0 @ r1.T corners. """
    rr = tensordot(r0, r1, axes=(1, 1))
    u, s, v = rr.svd(axes=(0, 1), sU=r0.s[1], fix_signs=fix_signs)

    Smask = truncation_mask(s, **opts_svd)

    print("-----------------")
    if sum(Smask[(1,1,0), (1,1,0)]) != sum(Smask[(-1,-1,0), (-1,-1,0)]):
        for t in [(1,1,0), (-1,-1,0)]:
            try:
                st = s[t, t]
                print("t, s = ", t, st.tolist())
                st = Smask[t, t]
                print("t, s = ", t, st.tolist())
            except YastnError:
                pass

    u, s, v = Smask.apply_mask(u, s, v, axes=(-1, 0, 0))

    # u, s, v = rr.svd_with_truncation(axes=(0, 1), sU=r0.s[1], fix_signs=fix_signs, **opts_svd)
    rs = s.rsqrt()
    p0 = tensordot(r1, (rs @ v).conj(), axes=(0, 1)).unfuse_legs(axes=0)
    p1 = tensordot(r0, (u @ rs).conj(), axes=(0, 0)).unfuse_legs(axes=0)
    return p0, p1


_trivial = (('hlt', 'r', 'l', 'tl', 2, 0, 0),
            ('hlb', 'r', 'l', 'bl', 0, 2, 1),
            ('hrt', 'l', 'r', 'tr', 0, 0, 1),
            ('hrb', 'l', 'r', 'br', 2, 2, 0),
            ('vtl', 'b', 't', 'tl', 0, 1, 1),
            ('vtr', 'b', 't', 'tr', 2, 3, 0),
            ('vbl', 't', 'b', 'bl', 2, 1, 0),
            ('vbr', 't', 'b', 'br', 0, 3, 1))


def trivial_projectors_(proj, dirn, env):
    r"""
    Adds trivial projectors if not present at the edges of the lattice with open boundary conditions.
    """
    config = env.psi.config
    for site in env.sites():
        for s0, s1, s2, s3, a0, a1, a2 in _trivial:
            if s2 in dirn and getattr(proj[site], s0) is None:
                site_nn = env.nn_site(site, d=s1)
                if site_nn is not None:
                    l0 = getattr(env[site], s2).get_legs(a0).conj()
                    l1 = env.psi[site].get_legs(a1).conj()
                    l2 = getattr(env[site_nn], s3).get_legs(a2).conj()
                    setattr(proj[site], s0, ones(config, legs=(l0, l1, l2)))


def update_env_horizontal_(env_tmp, site, env, proj):
    r"""
    Horizontal move of CTM step. Calculate environment tensors for given site.
    """
    psi = env.psi

    l = psi.nn_site(site, d='l')
    if l is not None:
        tmp = env[l].l @ proj[l].hlt
        tmp = psi[l]._attach_01(tmp)
        tmp = tensordot(proj[l].hlb, tmp, axes=((0, 1), (0, 1))).transpose(axes=(0, 2, 1))
        env_tmp[site].l = tmp / tmp.norm(p='inf')

    r = psi.nn_site(site, d='r')
    if r is not None:
        tmp = env[r].r @ proj[r].hrb
        tmp = psi[r]._attach_23(tmp)
        tmp = tensordot(proj[r].hrt, tmp, axes=((0, 1), (0, 1))).transpose(axes=(0, 2, 1))
        env_tmp[site].r = tmp / tmp.norm(p='inf')

    tl = psi.nn_site(site, d='tl')
    if tl is not None:
        tmp = tensordot(proj[tl].hlb, env[l].tl @ env[l].t, axes=((0, 1), (0, 1)))
        env_tmp[site].tl = tmp / tmp.norm(p='inf')

    tr = psi.nn_site(site, d='tr')
    if tr is not None:
        tmp = tensordot(env[r].t, env[r].tr @ proj[tr].hrb, axes=((2, 1), (0, 1)))
        env_tmp[site].tr = tmp / tmp.norm(p='inf')

    bl = psi.nn_site(site, d='bl')
    if bl is not None:
        tmp = tensordot(env[l].b, env[l].bl @ proj[bl].hlt, axes=((2, 1), (0, 1)))
        env_tmp[site].bl = tmp / tmp.norm(p='inf')

    br = psi.nn_site(site, d='br')
    if br is not None:
        tmp = tensordot(proj[br].hrt, env[r].br @ env[r].b, axes=((0, 1), (0, 1)))
        env_tmp[site].br = tmp / tmp.norm(p='inf')


def update_env_vertical_(env_tmp, site, env, proj):
    r"""
    Vertical move of CTM step. Calculate environment tensors for given site.
    """
    psi = env.psi

    t = psi.nn_site(site, d='t')
    if t is not None:
        tmp = proj[t].vtl.transpose(axes=(2, 1, 0)) @ env[t].t
        tmp = psi[t]._attach_01(tmp)
        tmp = tensordot(tmp, proj[t].vtr, axes=((2, 3), (0, 1)))
        env_tmp[site].t = tmp / tmp.norm(p='inf')

    b = psi.nn_site(site, d='b')
    if b is not None:
        tmp = proj[b].vbr.transpose(axes=(2, 1, 0)) @ env[b].b
        tmp = psi[b]._attach_23(tmp)
        tmp = tensordot(tmp, proj[b].vbl, axes=((2, 3), (0, 1)))
        env_tmp[site].b = tmp / tmp.norm(p='inf')

    tl = psi.nn_site(site, d='tl')
    if tl is not None:
        tmp = tensordot(env[t].l, env[t].tl @ proj[tl].vtr, axes=((2, 1), (0, 1)))
        env_tmp[site].tl = tmp / tmp.norm(p='inf')

    tr = psi.nn_site(site, d='tr')
    if tr is not None:
        tmp = tensordot(proj[tr].vtl, env[t].tr @ env[t].r, axes=((0, 1), (0, 1)))
        env_tmp[site].tr =  tmp / tmp.norm(p='inf')

    bl = psi.nn_site(site, d='bl')
    if bl is not None:
        tmp = tensordot(proj[bl].vbr, env[b].bl @ env[b].l, axes=((0, 1), (0, 1)))
        env_tmp[site].bl = tmp / tmp.norm(p='inf')

    br = psi.nn_site(site, d='br')
    if br is not None:
        tmp = tensordot(env[b].r, env[b].br @ proj[br].vbl, axes=((2, 1), (0, 1)))
        env_tmp[site].br = tmp / tmp.norm(p='inf')


def update_old_env_(env, env_tmp):
    r"""
    Update tensors in env with the ones from env_tmp that are not None.
    """
    for site in env.sites():
        for k, v in env_tmp[site].__dict__.items():
            if v is not None:
                setattr(env[site], k, v)
