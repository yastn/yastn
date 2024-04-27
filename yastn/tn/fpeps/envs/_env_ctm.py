from typing import NamedTuple
from dataclasses import dataclass
from .... import rand, ones, YastnError, Leg, tensordot, ncon, qr
from ... import mps
from .._peps import Peps, Peps2Layers
from .._gates_auxiliary import apply_gate_onsite, gate_product_operator, gate_fix_order
from .._geometry import Bond
from ._env_auxlliary import append_vec_tl, append_vec_br


class ctm_window(NamedTuple):
    """ elements of a 2x2 window for the CTM algorithm. """
    nw : tuple  # north-west
    ne : tuple  # north-east
    sw : tuple  # south-west
    se : tuple  # south-east


class EnvCTM(Peps):
    r""" Geometric information about the lattice provided to ctm tensors """
    def __init__(self, psi):
        super().__init__(psi.geometry)
        self.psi = Peps2Layers(psi) if psi.has_physical() else psi

        windows = []
        for site in self.sites():
            win = [self.nn_site(site, d=d) for d in ('r', 'b', 'br')]
            if None not in win:
                windows.append(ctm_window(site, *win))
        self._windows = tuple(windows)
        self.init_()

    def copy(self):
        env = EnvCTM(self.psi)
        env._data = {k : v.copy() for k, v in self._data.items()}
        return env

    def tensors_CtmEnv(self):
        return self._windows

    def boundary_mps(self, n, dirn):
        """ Convert environmental tensors of Ctm to an MPS """
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

    def init_(self, t0=None, D0=1, type='rand'):
        """ Initialize random CTMRG environments of peps tensors A. """
        if type not in ('rand', 'ones'):
            raise YastnError(f"type = {type} not recognized in EnvCTM init. Should be in ('rand', 'ones')")

        config = self.psi.config
        tensor_init = rand if type == 'rand' else ones

        if t0 is None:
            t0 = (0,) * config.sym.NSYM
        leg0 = Leg(config, s=1, t=[t0], D=[D0])

        for site in self.sites():
            legsA = self.psi[site].get_legs()

            self[site] = Local_CTMEnv()
            self[site].tl = tensor_init(config, legs=[leg0, leg0.conj()])
            self[site].tr = tensor_init(config, legs=[leg0, leg0.conj()])
            self[site].bl = tensor_init(config, legs=[leg0, leg0.conj()])
            self[site].br = tensor_init(config, legs=[leg0, leg0.conj()])
            self[site].t = tensor_init(config, legs=[leg0, legsA[0].conj(), leg0.conj()])
            self[site].b = tensor_init(config, legs=[leg0, legsA[2].conj(), leg0.conj()])
            self[site].l = tensor_init(config, legs=[leg0, legsA[1].conj(), leg0.conj()])
            self[site].r = tensor_init(config, legs=[leg0, legsA[3].conj(), leg0.conj()])

    def measure_1site(self, op, site=None):
        r"""
        dictionary containing site coordinates as keys and their corresponding expectation values

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


    def update_(env, opts_svd=None, fix_signs=None):
        r"""
        Generator for ctmrg().
        Perform one step of CTMRG update for a mxn lattice.

        Parameters
        ----------
            env : class CTMEnv
                The current CTM environment tensor.
            AAb : CtmBond
                The CtmBond tensor for the lattice.
            fix_signs : bool
                Whether to fix the signs of the environment tensors.
            opts_svd : dict, optional
                A dictionary of options to pass to the SVD algorithm, by default None.

        Returns
        -------
            envn_ver : class CTMEnv
                The updated CTM environment tensor
            proj     :  dictionary of CTM projectors.


        The function performs a CTMRG update for a square lattice using the corner transfer matrix
        renormalization group (CTMRG) algorithm. The update is performed in two steps: a horizontal move
        and a vertical move. The projectors for each move are calculated first, and then the tensors in
        the CTM environment are updated using the projectors. The boundary conditions of the lattice
        determine whether trivial projectors are needed for the move. If the boundary conditions are
        'infinite', no trivial projectors are needed; if they are 'obc', four trivial projectors
        are needed for each move. The signs of the environment tensors can also be fixed during the
        update if `fix_signs` is set to True. The latter is important when we set the criteria for
        stopping the algorithm based on singular values of the corners.
        """

        psi = env.psi
        Nx, Ny = psi.Nx, psi.Ny

        proj = Peps(psi.geometry)
        for site in proj.sites():
            proj[site] = Local_ProjectorEnv()


        for ms in env.tensors_CtmEnv():
            proj = proj_horizontal(env[ms.nw], env[ms.ne], env[ms.sw], env[ms.se], proj, ms, psi[ms.nw], psi[ms.ne], psi[ms.sw], psi[ms.se], fix_signs, opts_svd)

        if env.boundary == 'obc':
            # we need trivial projectors on the boundary for horizontal move for finite lattices
            for ms in range(psi.Ny - 1):
                proj[0, ms].hlt = trivial_projector(env[0,ms].l, psi[0,ms], env[0,ms+1].tl, dirn='hlt')
                proj[Nx-1,ms].hlb = trivial_projector(env[Nx-1,ms].l, psi[Nx-1,ms], env[Nx-1,ms+1].bl, dirn='hlb')
                proj[0, ms+1].hrt = trivial_projector(env[0,ms+1].r, psi[0,ms+1], env[0,ms].tr, dirn='hrt')
                proj[Nx-1, ms+1].hrb = trivial_projector(env[Nx-1,ms+1].r, psi[Nx-1,ms+1], env[Nx-1,ms].br, dirn='hrb')

        # print('######## Horizontal Move ###########')

        env0 = env.copy()
        for ms in psi.sites():
            # print('move ctm horizontal', ms)
            move_horizontal_(env, env0, psi, proj, ms)

        # print('######## Calculating projectors for vertical move ###########')
        for ms in env.tensors_CtmEnv():   # vertical absorption and renormalization
            # print('projector calculation ctm cluster vertical', ms)
            proj = proj_vertical(env[ms.nw], env[ms.sw], env[ms.ne], env[ms.se], proj, ms, psi[ms.nw], psi[ms.sw], psi[ms.ne], psi[ms.se], fix_signs, opts_svd)

        if psi.boundary == 'obc':
            # we need trivial projectors on the boundary for vertical move for finite lattices
            for ms in range(Nx-1):
                proj[ms,0].vtl = trivial_projector(env[ms,0].t, psi[ms,0], env[ms+1,0].tl, dirn='vtl')
                proj[ms+1,0].vbl = trivial_projector(env[ms+1,0].b, psi[ms+1,0], env[ms,0].bl, dirn='vbl')
                proj[ms,Ny-1].vtr = trivial_projector(env[ms, Ny-1].t, psi[ms,Ny-1], env[ms+1,Ny-1].tr, dirn='vtr')
                proj[ms+1,Ny-1].vbr = trivial_projector(env[ms+1,Ny-1].b, psi[ms+1,Ny-1], env[ms,Ny-1].br, dirn='vbr')

        # print('######### Vertical Move ###########')

        env0 = env.copy()
        for ms in psi.sites():   # vertical absorption and renormalization
            move_vertical_(env, env0, psi, proj, ms)


    def bond_metric(self, Q0, Q1, s0, s1, dirn):
        """
        Calculates Full-update metric tensor.

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


@dataclass()
class Local_CTMEnv():
    """ data class for CTM environment tensors associated with each peps tensor """

    tl : any = None # top-left
    tr : any = None # top-right
    bl : any = None # bottom-left
    br : any = None # bottom-right
    t : any = None  # top
    b : any = None  # bottom
    l : any = None  # left
    r : any = None  # right

    def copy(self):
        return Local_CTMEnv(tl=self.tl, tr=self.tr, bl=self.bl, br=self.br, t=self.t, b=self.b, l=self.l, r=self.r)


@dataclass()
class Local_ProjectorEnv():
    """ data class for projectors labelled by a single lattice site calculated during ctm renormalization step """

    hlt : any = None  # horizontal left top
    hlb : any = None  # horizontal left bottom
    hrt : any = None  # horizontal right top
    hrb : any = None  # horizontal right bottom
    vtl : any = None  # vertical top left
    vtr : any = None  # vertical top right
    vbl : any = None  # vertical bottom left
    vbr : any = None  # vertical bottom right

    def copy(self):
        return Local_ProjectorEnv(hlt=self.hlt, hlb=self.hlb, hrt=self.hrt, hrb=self.hrb, vtl=self.vtl, vtr=self.vtr, vbl=self.vbl, vbr=self.vbr)




def proj_horizontal(env_nw, env_ne, env_sw, env_se, out, ms, AAb_nw, AAb_ne, AAb_sw, AAb_se, fix_signs, opts_svd=None):

    """
    Calculates horizontal environment projectors and stores the resulting tensors in the output dictionary.

    Parameters
    ----------
    env_nw: environment at north west of 2x2 ctm window
    env_sw: environment at south west of 2x2 ctm window
    env_ne: environment at north east of 2x2 ctm window
    env_se: environment at south east of 2x2 ctm window
    out: dictionary that stores the horizontal projectors
    ms: current 2x2 ctm window
    AAb_nw: double PEPS tensor at north west of 2x2 ctm window
    AAb_sw: double PEPS tensor at south west of 2x2 ctm window
    AAb_ne: double PEPS tensor at north east of 2x2 ctm window
    AAb_se: double PEPS tensor at south east of 2x2 ctm window
    fix_signs: whether to fix the signs of the projectors or not
    opts_svd: options for the SVD (singular value decomposition)

    Returns
    ----------
    out: dictionary that stores the vertical projectors

    """
    # ms in the CTM unit cell
    cortlm = fcor_tl(env_nw.l, env_nw.tl, env_nw.t, AAb_nw) # contracted matrix at top-left constructing a projector with cut in the middle
    cortrm = fcor_tr(env_ne.t, env_ne.tr, env_ne.r, AAb_ne) # contracted matrix at top-right constructing a projector with cut in the middle
    corttm = tensordot(cortrm, cortlm, axes=((0, 1), (2, 3))) # top half for constructing middle projector
    del cortlm
    del cortrm

    corblm = fcor_bl(env_sw.b, env_sw.bl, env_sw.l, AAb_sw) # contracted matrix at bottom-left constructing a projector with cut in the middle
    corbrm = fcor_br(env_se.r, env_se.br, env_se.b, AAb_se) # contracted matrix at bottom-right constructing a projector with cut in the middle
    corbbm = tensordot(corbrm, corblm, axes=((2, 3), (0, 1)))
    del corblm
    del corbrm

    _, rt = qr(corttm, axes=((0, 1), (2, 3)))
    _, rb = qr(corbbm, axes=((0, 1), (2, 3)))

    out[ms.nw].hlb, out[ms.sw].hlt = proj_Cor(rt, rb, fix_signs, opts_svd=opts_svd)   # projector left-middle

    corttm = corttm.transpose(axes=(2, 3, 0, 1))
    corbbm = corbbm.transpose(axes=(2, 3, 0, 1))

    _, rt = qr(corttm, axes=((0, 1), (2, 3)))
    _, rb = qr(corbbm, axes=((0, 1), (2, 3)))

    out[ms.ne].hrb, out[ms.se].hrt = proj_Cor(rt, rb, fix_signs, opts_svd=opts_svd)   # projector right-middle
    return out


def proj_vertical(env_nw, env_sw, env_ne, env_se, out, ms, AAb_nw, AAb_sw, AAb_ne, AAb_se, fix_signs, opts_svd=None):

    r"""
    Calculates vertical environment projectors and stores the resulting tensors in the output dictionary.

    Parameters
    ----------
    env_nw: environment at north west of 2x2 ctm window
    env_sw: environment at south west of 2x2 ctm window
    env_ne: environment at north east of 2x2 ctm window
    env_se: environment at south east of 2x2 ctm window
    out: dictionary that stores the vertical projectors
    ms: current 2x2 ctm window
    AAb_nw: double PEPS tensor at north west of 2x2 ctm window
    AAb_sw: double PEPS tensor at south west of 2x2 ctm window
    AAb_ne: double PEPS tensor at north east of 2x2 ctm window
    AAb_se: double PEPS tensor at south east of 2x2 ctm window
    fix_signs: whether to fix the signs of the projectors or not
    opts_svd: options for the SVD (singular value decomposition)

    Returns
    ----------
    out: dictionary that stores the vertical projectors

    """
    cortlm = fcor_tl(env_nw.l, env_nw.tl, env_nw.t, AAb_nw) # contracted matrix at top-left constructing a projector with a cut in the middle
    corblm = fcor_bl(env_sw.b, env_sw.bl, env_sw.l, AAb_sw) # contracted matrix at bottom-left constructing a projector with a cut in the middle

    corvvm = tensordot(corblm, cortlm, axes=((2, 3), (0, 1))) # left half for constructing middle projector
    del corblm
    del cortlm

    cortrm = fcor_tr(env_ne.t, env_ne.tr, env_ne.r, AAb_ne) # contracted matrix at top-right constructing a projector with cut in the middle
    corbrm = fcor_br(env_se.r, env_se.br, env_se.b, AAb_se) # contracted matrix at bottom-right constructing a projector with cut in the middle

    corkkr = tensordot(corbrm, cortrm, axes=((0, 1), (2, 3))) # right half for constructing middle projector

    _, rl = qr(corvvm, axes=((0, 1), (2, 3)))
    _, rr = qr(corkkr, axes=((0, 1), (2, 3)))

    out[ms.nw].vtr, out[ms.ne].vtl = proj_Cor(rl, rr, fix_signs, opts_svd=opts_svd) # projector top-middle

    corvvm = corvvm.transpose(axes=(2, 3, 0, 1))
    corkkr = corkkr.transpose(axes=(2, 3, 0, 1))

    _, rl = qr(corvvm, axes=((0, 1), (2, 3)))
    _, rr = qr(corkkr, axes=((0, 1), (2, 3)))

    out[ms.sw].vbr, out[ms.se].vbl = proj_Cor(rl, rr, fix_signs, opts_svd=opts_svd) # projector bottom-middle
    return out


def trivial_projector(a, b, c, dirn):
    """ projectors which fix the bond dimension of the environment CTM tensors
     corresponding to boundary PEPS tensors to 1 """

    if dirn == 'hlt':
        la, lb, lc = a.get_legs(axes=2), b.get_legs(axes=0), c.get_legs(axes=0)
    elif dirn == 'hlb':
        la, lb, lc = a.get_legs(axes=0), b.get_legs(axes=2), c.get_legs(axes=1)
    elif dirn == 'hrt':
        la, lb, lc = a.get_legs(axes=0), b.get_legs(axes=0), c.get_legs(axes=1)
    elif dirn == 'hrb':
        la, lb, lc = a.get_legs(axes=2), b.get_legs(axes=2), c.get_legs(axes=0)
    elif dirn == 'vtl':
        la, lb, lc = a.get_legs(axes=0), b.get_legs(axes=1), c.get_legs(axes=1)
    elif dirn == 'vbl':
        la, lb, lc = a.get_legs(axes=2), b.get_legs(axes=1), c.get_legs(axes=0)
    elif dirn == 'vtr':
        la, lb, lc = a.get_legs(axes=2), b.get_legs(axes=3), c.get_legs(axes=0)
    elif dirn == 'vbr':
        la, lb, lc = a.get_legs(axes=0), b.get_legs(axes=3), c.get_legs(axes=1)

    return ones(b.config, legs=[la.conj(), lb.conj(), lc.conj()])


def move_horizontal_(envn, env, AAb, proj, ms):

    r"""
    Horizontal move of CTM step calculating environment tensors corresponding
     to a particular site on a lattice.

    Parameters
    ----------
        envn : class CtmEnv
            Class containing data for the new environment tensors renormalized by the horizontal move at each site + lattice info.
        env : class CtmEnv
            Class containing data for the input environment tensors at each site + lattice info.
        AAb : Contains top and bottom Peps tensors
        proj : Class Lattice with info on lattice
            Projectors to be applied for renormalization.
        ms : site whose environment tensors are to be renormalized

    Returns
    -------
        yastn.fpeps.CtmEnv
        Contains data of updated environment tensors along with those not updated
    """
    _, y = ms
    left = AAb.nn_site(ms, d='l')
    right = AAb.nn_site(ms, d='r')

    if AAb.boundary == 'obc' and y == 0:
        l_abv = None
        l_bel = None
    else:
        l_abv = AAb.nn_site(left, d='t')
        l_bel = AAb.nn_site(left, d='b')

    if AAb.boundary == 'obc' and y == (env.Ny-1):
        r_abv = None
        r_bel = None
    else:
        r_abv = AAb.nn_site(right, d='t')
        r_bel = AAb.nn_site(right, d='b')

    if l_abv is not None:
        envn[ms].tl = ncon((env[left].tl, env[left].t, proj[l_abv].hlb),
                                   ([2, 3], [3, 1, -1], [2, 1, -0]))

    if r_abv is not None:
        envn[ms].tr = ncon((env[right].tr, env[right].t, proj[r_abv].hrb),
                               ([3, 2], [-0, 1, 3], [2, 1, -1]))
    if l_bel is not None:
        envn[ms].bl = ncon((env[left].bl, env[left].b, proj[l_bel].hlt),
                               ([3, 2], [-0, 1, 3], [2, 1, -1]))
    if r_bel is not None:
        envn[ms].br = ncon((proj[r_bel].hrt, env[right].br, env[right].b),
                               ([2, 1, -0], [2, 3], [3, 1, -1]))

    if not(left is None):
        tt_l = tensordot(env[left].l, proj[left].hlt, axes=(2, 0))
        tt_l = AAb[left]._attach_01(tt_l)
        envn[ms].l = ncon((proj[left].hlb, tt_l), ([2, 1, -0], [2, 1, -2, -1]))

    if not(right is None):
        tt_r = tensordot(env[right].r, proj[right].hrb, axes=(2,0))
        tt_r = AAb[right]._attach_23(tt_r)
        envn[ms].r = ncon((tt_r, proj[right].hrt), ([1, 2, -3, -2], [1, 2, -1]))

    envn[ms].tl = envn[ms].tl/ envn[ms].tl.norm(p='inf')
    envn[ms].l = envn[ms].l/ envn[ms].l.norm(p='inf')
    envn[ms].tr = envn[ms].tr/ envn[ms].tr.norm(p='inf')
    envn[ms].bl = envn[ms].bl/ envn[ms].bl.norm(p='inf')
    envn[ms].r = envn[ms].r/ envn[ms].r.norm(p='inf')
    envn[ms].br = envn[ms].br/ envn[ms].br.norm(p='inf')


def move_vertical_(envn, env, AAb, proj, ms):

    r"""
    Vertical move of CTM step calculating environment tensors corresponding
     to a particular site on a lattice.

    Parameters
    ----------
        envn : class CtmEnv
            Class containing data for the new environment tensors renormalized by the vertical move at each site + lattice info.
        env : class CtmEnv
            Class containing data for the input environment tensors at each site + lattice info.
        AAb : Contains top and bottom Peps tensors
        proj : Class Lattice with info on lattice
              Projectors to be applied for renormalization.
        ms : site whose environment tensors are to be renormalized

    Returns
    -------
        yastn.fpeps.CtmEnv
        Contains data of updated environment tensors along with those not updated
    """
    x, _ = ms
    top = AAb.nn_site(ms, d='t')
    bottom = AAb.nn_site(ms, d='b')

    if AAb.boundary == 'obc' and x == 0:
        t_left = None
        t_right = None
    else:
        t_left = AAb.nn_site(top, d='l')
        t_right =  AAb.nn_site(top, d='r')
    if AAb.boundary == 'obc' and x == (env.Nx-1):
        b_left = None
        b_right = None
    else:
        b_left = AAb.nn_site(bottom, d='l')
        b_right = AAb.nn_site(bottom, d='r')


    if t_left is not None:
        envn[ms].tl = ncon((env[top].tl, env[top].l, proj[t_left].vtr),
                               ([3, 2], [-0, 1, 3], [2, 1, -1]))
    if t_right is not None:
        envn[ms].tr = ncon((proj[t_right].vtl, env[top].tr, env[top].r),
                               ([2, 1, -0], [2, 3], [3, 1, -1]))
    if b_left is not None:
        envn[ms].bl = ncon((env[bottom].bl, env[bottom].l, proj[b_left].vbr),
                               ([1, 3], [3, 2, -1], [1, 2, -0]))
    if b_right is not None:
        envn[ms].br = ncon((proj[b_right].vbl, env[bottom].br, env[bottom].r),
                               ([2, 1, -1], [3, 2], [-0, 1, 3]))
    if not(top is None):
        ll_t = ncon((proj[top].vtl, env[top].t), ([1, -1, -0], [1, -2, -3]))
        ll_t = AAb[top]._attach_01(ll_t)
        envn[ms].t = ncon((ll_t, proj[top].vtr), ([-0, -1, 2, 1], [2, 1, -2]))
    if not(bottom is None):
        ll_b = ncon((proj[bottom].vbr, env[bottom].b), ([1, -2, -1], [1, -3, -4]))
        ll_b = AAb[bottom]._attach_23(ll_b)
        envn[ms].b = ncon((ll_b, proj[bottom].vbl), ([-0, -1, 1, 2], [1, 2, -3]))

    envn[ms].tl = envn[ms].tl/ envn[ms].tl.norm(p='inf')
    envn[ms].t = envn[ms].t/ envn[ms].t.norm(p='inf')
    envn[ms].tr = envn[ms].tr/ envn[ms].tr.norm(p='inf')
    envn[ms].bl = envn[ms].bl/ envn[ms].bl.norm(p='inf')
    envn[ms].b = envn[ms].b/ envn[ms].b.norm(p='inf')
    envn[ms].br = envn[ms].br/ envn[ms].br.norm(p='inf')


def fcor_bl(env_b, env_bl, env_l, AAb):
    """ Creates extended bottom left corner. Order of indices see _attach_12. """
    corbln = tensordot(env_b, env_bl, axes=(2, 0))
    corbln = tensordot(corbln, env_l, axes=(2, 0))
    return AAb._attach_12(corbln)

def fcor_tl(env_l, env_tl, env_t, AAb):
    """ Creates extended top left corner. """
    cortln = tensordot(env_l, env_tl, axes=(2, 0))
    cortln = tensordot(cortln, env_t, axes=(2, 0))
    return AAb._attach_01(cortln)

def fcor_tr(env_t, env_tr, env_r, AAb):
    """ Creates extended top right corner. """
    cortrn = tensordot(env_t, env_tr, axes=(2, 0))
    cortrn = tensordot(cortrn, env_r, axes=(2, 0))
    return AAb._attach_30(cortrn)

def fcor_br(env_r, env_br, env_b, AAb):
    """ Creates extended bottom right corner. """
    corbrn = tensordot(env_r, env_br, axes=(2, 0))
    corbrn = tensordot(corbrn, env_b, axes=(2, 0))
    return AAb._attach_23(corbrn)


def proj_Cor(rt, rb, fix_signs, opts_svd):
    """
    tt upper half of the CTM 4 extended corners diagram indices from right (e,t,b) to left (e,t,b)
    tb lower half of the CTM 4 extended corners diagram indices from right (e,t,b) to left (e,t,b)
    """

    rr = tensordot(rt, rb, axes=((1, 2), (1, 2)))
    u, s, v = rr.svd_with_truncation(axes=(0, 1), sU =rt.get_signature()[1], fix_signs=fix_signs, **opts_svd)
    s = s.rsqrt()
    pt = s.broadcast(tensordot(rb, v, axes=(0, 1), conj=(0, 1)), axes=2)
    pb = s.broadcast(tensordot(rt, u, axes=(0, 0), conj=(0, 1)), axes=2)
    return pt, pb
