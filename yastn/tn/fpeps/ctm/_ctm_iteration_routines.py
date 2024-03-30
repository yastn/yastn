"""
Routines for CTMRG of layers PEPS on an arbitrary mxn square lattice.

PEPS tensors A[0] and A[1] corresponds to "black" and "white" sites of the checkerboard.
PEPs tensors habe 6 legs: [top, left, bottom, right, ancilla, system] with signature s=(-1, 1, 1, -1, -1, 1).
The routines include swap_gates to incorporate fermionic anticommutation relations into the lattice.
"""

import yastn
from yastn import tensordot, ncon, svd_with_truncation, qr, vdot, initialize
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps._doublePepsTensor import DoublePepsTensor
from ._ctm_env import Local_ProjectorEnv
import numpy as np

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
    u, s, v = svd_with_truncation(rr, axes=(0, 1), sU =rt.get_signature()[1], fix_signs=fix_signs, **opts_svd)
    s = s.rsqrt()
    pt = s.broadcast(tensordot(rb, v, axes=(0, 1), conj=(0, 1)), axes=2)
    pb = s.broadcast(tensordot(rt, u, axes=(0, 0), conj=(0, 1)), axes=2)
    return pt, pb


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

    if opts_svd is None:
        opts_svd = {'tol':1e-10, 'D_total': round(5*np.max(AAb_nw.A.get_shape()))}   # D_total if not given can be a multiple of the total bond dimension

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
    del corttm
    del corbbm

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

    if opts_svd is None:
        opts_svd = {'tol':1e-10, 'D_total': round(5*np.max(AAb_nw.A.get_shape()))}   # D_total if not given can be a multiple of the total bond dimension

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

    del corvvm
    del corkkr

    return out


def move_horizontal(envn, env, AAb, proj, ms):

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

    (_,y) = ms
    left = AAb.nn_site(ms, d='l')
    right = AAb.nn_site(ms, d='r')

    if AAb.boundary == 'obc' and y == 0:
        l_abv = None
        l_bel = None
    else:
        l_abv = AAb.nn_site(left, d='t')
        l_bel =  AAb.nn_site(left, d='b')

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

    return envn


def move_vertical(envn, env, AAb, proj, ms):

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


    (x,_) = ms
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
    return envn



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

    tmp = initialize.ones(b.A.config, legs=[la.conj(), lb.conj(), lc.conj()])

    return tmp

def CTM_it(env, AAb, fix_signs, opts_svd=None):
    r"""Perform one step of CTMRG update for a mxn lattice.

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

    proj = fpeps.Peps(AAb) # ctm projectors defined as an instance of Lattice class
    for ms in proj.sites():
        proj[ms] = Local_ProjectorEnv()

    # print('######## Calculating projectors for horizontal move ###########')
    envn_hor = env.copy()

    for ms in AAb.tensors_CtmEnv():   #ctm_wndows(trajectory='h', index=y): # horizontal absorption and renormalization
        # print('projector calculation ctm cluster horizontal', ms)
        proj = proj_horizontal(env[ms.nw], env[ms.ne], env[ms.sw], env[ms.se], proj, ms, AAb[ms.nw], AAb[ms.ne], AAb[ms.sw], AAb[ms.se], fix_signs, opts_svd)

    if AAb.boundary == 'obc':
        # we need trivial projectors on the boundary for horizontal move for finite lattices
        for ms in range(AAb.Ny-1):
            proj[0,ms].hlt = trivial_projector(env[0,ms].l, AAb[0,ms], env[0,ms+1].tl, dirn='hlt')
            proj[AAb.Nx-1,ms].hlb = trivial_projector(env[AAb.Nx-1,ms].l, AAb[AAb.Nx-1,ms], env[AAb.Nx-1,ms+1].bl, dirn='hlb')
            proj[0, ms+1].hrt = trivial_projector(env[0,ms+1].r, AAb[0,ms+1], env[0,ms].tr, dirn='hrt')
            proj[AAb.Nx-1, ms+1].hrb = trivial_projector(env[AAb.Nx-1,ms+1].r, AAb[AAb.Nx-1,ms+1], env[AAb.Nx-1,ms].br, dirn='hrb')

    # print('######## Horizontal Move ###########')

    for ms in AAb.sites():
        # print('move ctm horizontal', ms)
        envn_hor = move_horizontal(envn_hor, env, AAb, proj, ms)
    envn_ver = envn_hor.copy()

    # print('######## Calculating projectors for vertical move ###########')
    for ms in AAb.tensors_CtmEnv():   # vertical absorption and renormalization
        # print('projector calculation ctm cluster vertical', ms)
        proj = proj_vertical(envn_hor[ms.nw], envn_hor[ms.sw], envn_hor[ms.ne], envn_hor[ms.se], proj, ms, AAb[ms.nw], AAb[ms.sw], AAb[ms.ne], AAb[ms.se], fix_signs, opts_svd)

    if AAb.boundary == 'obc':
        # we need trivial projectors on the boundary for vertical move for finite lattices
        for ms in range(AAb.Nx-1):
            proj[ms,0].vtl = trivial_projector(envn_hor[ms,0].t, AAb[ms,0], envn_hor[ms+1,0].tl, dirn='vtl')
            proj[ms+1,0].vbl = trivial_projector(envn_hor[ms+1,0].b, AAb[ms+1,0], envn_hor[ms,0].bl, dirn='vbl')
            proj[ms,AAb.Ny-1].vtr = trivial_projector(envn_hor[ms, AAb.Ny-1].t, AAb[ms,AAb.Ny-1], envn_hor[ms+1,AAb.Ny-1].tr, dirn='vtr')
            proj[ms+1,AAb.Ny-1].vbr = trivial_projector(envn_hor[ms+1,AAb.Ny-1].b, AAb[ms+1,AAb.Ny-1], envn_hor[ms,AAb.Ny-1].br, dirn='vbr')

    # print('######### Vertical Move ###########')

    for ms in AAb.sites():   # vertical absorption and renormalization
        # print('move ctm vertical', ms)
        envn_ver = move_vertical(envn_ver, envn_hor, AAb, proj, ms)
    return envn_ver, proj


def fPEPS_l(A, op):
    """
    attaches operator to the tensor located at the left (left according to
    chosen fermionic order) while calulating expectation values of non-local
    fermionic operators in vertical direction
    """
    Aop = tensordot(A, op, axes=(4, 1)) # t l b r [s a] c
    Aop = Aop.swap_gate(axes=(2, 5))
    Aop = Aop.fuse_legs(axes=(0, 1, 2, (3, 5), 4)) # t l b [r c] [s a]
    return Aop


def fPEPS_r(A, op):
    """
    attaches operator to the tensor located at the right (right according to
    chosen fermionic order) while calulating expectation values of non-local
    fermionic operators in vertical direction
    """
    Aop = tensordot(A, op, axes=(4, 1))
    Aop = Aop.fuse_legs(axes=(0, (1, 5), 2, 3, 4)) # t [l c] b r [s a]
    return Aop


def fPEPS_t(A, op):
    """
    attaches operator to the tensor located at the top (left according to
    chosen fermionic order) while calulating expectation values of non-local
    fermionic operators in vertical direction
    """
    Aop = tensordot(A, op, axes=(4, 1))
    Aop = Aop.fuse_legs(axes=(0, 1, (2, 5), 3, 4)) # t l [b c] r [s a]
    return Aop


def fPEPS_b(A, op):
    """
    attaches operator to the tensor located at the bottom (right according to
    chosen fermionic order) while calulating expectation values of non-local
    fermionic operators in vertical direction
    """
    Aop = tensordot(A, op, axes=(4, 1))
    Aop = Aop.swap_gate(axes=(1, 5))
    Aop = Aop.fuse_legs(axes=((0, 5), 1, 2, 3, 4)) # [t c] l b r [s a]
    return Aop

def fPEPS_op1s(A, op):
    """
    attaches operator to the tensor while calulating expectation
    values of local operators, no need to fuse auxiliary legs
    """
    Aop = tensordot(A, op, axes=(4, 1)) # t l b r [s a]
    return Aop

def fPEPS_fuse_physical(A):
    return {ind: x.fuse_legs(axes=(0, 1, 2, 3, (4, 5))) for ind, x in A.items()}

def fPEPS_unfuse_physical(A):
    return {ind: x.unfuse_legs(axes=4) for ind, x in A.items()}

def fuse_ancilla_wos(op, fid):
    """ kron and fusion of local operator with identity for ancilla --- without string """
    op = ncon((op, fid), ((-0, -2), (-1, -3)))
    return op.fuse_legs(axes=((0, 1), (2, 3)))

def fuse_ancilla_ws(op, fid, dirn):
    """ kron and fusion of nn operator with identity for ancilla --- with string """
    if dirn == 'l':
        op= op.add_leg(s=1).swap_gate(axes=(0, 2))
        op = ncon((op, fid), ((-0, -2, -4), (-1, -3)))
        op = op.swap_gate(axes=(3, 4)) # swap of connecting axis with ancilla is always in GA gate
        op = op.fuse_legs(axes=((0, 1), (2, 3), 4))
    elif dirn == 'r':
        op = op.add_leg(s=-1)
        op = ncon((op, fid), ((-0, -2, -4), (-1, -3)))
        op = op.fuse_legs(axes=((0, 1), (2, 3), 4))
    return op

def fPEPS_2layers(A, B=None, op=None, dir=None):
    """
    Prepare top and bottom peps tensors for CTM procedures.
    Applies operators on top if provided, with dir = 'l', 'r', 't', 'b', '1s'
    If dir = '1s', no auxiliary indices are introduced as the operator is local.
    Here spin and ancilla legs of tensors are fused
    """
    leg = A.get_legs(axes=-1)

    if not leg.is_fused():  # when there is no ancilla on A, only the physical index is present
        A = A.add_leg(s=-1)
        A = A.fuse_legs(axes=(0, 1, 2, 3, (4, 5)))  # new ancilla on outgoing leg
        leg = A.get_legs(axes=-1)

    _, leg = yastn.leg_undo_product(leg) # last leg of A should be fused
    fid = yastn.eye(config=A.config, legs=[leg, leg.conj()]).diag()

    if op is not None:
        if dir == 't':
            op_aux = fuse_ancilla_ws(op,fid,dirn='l')
            Ao = fPEPS_t(A,op_aux)
        elif dir == 'b':
            op_aux = fuse_ancilla_ws(op,fid,dirn='r')
            Ao = fPEPS_b(A,op_aux)
        elif dir == 'l':
            op_aux = fuse_ancilla_ws(op,fid,dirn='l')
            Ao = fPEPS_l(A,op_aux)
        elif dir == 'r':
            op_aux = fuse_ancilla_ws(op,fid,dirn='r')
            Ao = fPEPS_r(A,op_aux)
        elif dir == '1s':
            op = fuse_ancilla_wos(op,fid)
            Ao = fPEPS_op1s(A,op)
        else:
            raise RuntimeError("dir should be equal to 'l', 'r', 't', 'b' or '1s'")
    else:
        if A.ndim == 5:
            Ao = A # t l b r [s a]
        elif A.ndim == 6:
            Ao = A # t l b r str [s a]
            Ao = Ao.unfuse_legs(axes=5) # t l b r str s a
            Ao = Ao.fuse_legs(axes=(0, 1, 2, 3, 5, (6, 4))).fuse_legs(axes=(0, 1, 2, 3, (4, 5))) # t l b r [s [a str]]

    if B is None:
        if A.ndim == 5:
            B = A # t l b r [s a]
        if A.ndim == 6:
            B_int = A # t l b r str [s a]
            B_int = B_int.unfuse_legs(axes=5) # t l b r str s a
            B = B_int.fuse_legs(axes=(0, 1, 2, 3, 5, (6, 4))).fuse_legs(axes=(0, 1, 2, 3, (4, 5))) # t l b r [s [a str]]
    elif B is not None:
        B = B
    AAb = DoublePepsTensor(top=Ao, btm=B)
    return AAb


def check_consistency_tensors(A):
    # to check if the A tensors have the appropriate configuration of legs i.e. t l b r [s a]

    Ab = fpeps.Peps(A)
    if A[0, 0].ndim == 6:
        for ms in Ab.sites():
            Ab[ms] = A[ms].fuse_legs(axes=(0, 1, 2, 3, (4, 5)))
    elif A[0, 0].ndim == 3:
        for ms in Ab.sites():
            Ab[ms] = A[ms].unfuse_legs(axes=(0, 1)) # system and ancila are fused by default
    else:
        for ms in Ab.sites():
            Ab[ms] = A[ms]   # system and ancila are fused by default
    return Ab

