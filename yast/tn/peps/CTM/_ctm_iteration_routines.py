"""
Routines for CTMRG of layers PEPS on an arbitrary mxn rectangular lattice.

PEPS tensors A[0] and A[1] corresponds to "black" and "white" sites of the checkerboard.
PEPs tensors habe 6 legs: [top, left, bottom, right, ancilla, system] with signature s=(-1, 1, 1, -1, -1, 1).
The routines include swap_gates to incorporate fermionic anticommutation relations into the lattice.
"""

import yast
from yast import tensordot, ncon, svd_with_truncation, qr, vdot, initialize
import yast.tn.peps as peps
from yast.tn.peps._doublePepsTensor import DoublePepsTensor
from ._ctm_env import Proj, Local_Projector_Env
import multiprocess as mp
import time
import numpy as np


def append_a_bl(tt, AAb):
    """
    tensor indices have counterclockwise order.
    A indices counterclockwise order starting from top, then spin and ancilla indices.
    tt indices counterclockwise order (e1,[t1,b1],e2,[t2,b2]).
    """
    if isinstance(AAb, DoublePepsTensor):
        return AAb.append_a_bl(tt)
    return append_a_bl2(tt, AAb)


def append_a_bl2(tt, AAb):
    # tt: e3 (2t 2b) (1t 1b) e0
    # AAb['bl']: ((2t 2b) (1t 1b)) ((0t 0b) (3t 3b))
    tt = tt.fuse_legs(axes=((0, 3), (1, 2)))  # (e3 e0) ((2t 2b) (1t 1b))
    tt = tensordot(tt, AAb['bl'], axes=(1, 0))
    tt = tt.unfuse_legs(axes=(0, 1))  # e3 e0 (0t 0b) (3t 3b)
    tt = tt.transpose(axes=(0, 3, 1, 2)) # e3 (3t 3b) e0 (0t 0b)
    return tt

def append_a_tr(tt, AAb):
    """ ind like in append_a_bl """
    if isinstance(AAb, DoublePepsTensor):
        return AAb.append_a_tr(tt)
    return append_a_tr2(tt, AAb)


def append_a_tr2(tt, AAb):
    # tt:  e1 (0t 0b) (3t 3b) e2
    # AAb['bl']: ((2t 2b) (1t 1b)) ((0t 0b) (3t 3b))
    tt = tt.fuse_legs(axes=((1, 2), (0, 3)))  # ((0t 0b) (3t 3b)) (e1 e2)
    tt = tensordot(AAb['bl'], tt, axes=(1, 0))  # ((2t 2b) (1t 1b)) (e1 e2)
    tt = tt.unfuse_legs(axes=(0, 1))  # (2t 2b) (1t 1b) e1 e2
    tt = tt.transpose(axes=(2, 1, 3, 0))  # e1 (1t 1b) e2 (2t 2b)
    return tt

def append_a_tl(tt, AAb):
    """ ind like in append_a_bl """
    if isinstance(AAb, DoublePepsTensor):
        return AAb.append_a_tl(tt)
    return append_a_tl2(tt, AAb)


def append_a_tl2(tt, AAb):
    # tt: e2 (1t 1b) (0t 0b) e3
    # AAb['tl']:  ((1t 1b) (0t 0b)) ((3t 3b) (2t 2b))
    tt = tt.fuse_legs(axes=((0, 3), (1, 2)))  # (e2 e3) ((1t 1b) (0t 0b))
    tt = tensordot(tt, AAb['tl'], axes=(1, 0))  
    tt = tt.unfuse_legs(axes=(0, 1))  # e2 e3 (3t 3b) (2t 2b)
    tt = tt.transpose(axes=(0, 3, 1, 2))  # (e2 (2t 2b)) (e3 (3t 3b))
    return tt


def append_a_br(tt, AAb):
    """ ind like in append_a_bl """
    if isinstance(AAb, DoublePepsTensor):
        return AAb.append_a_br(tt)
    return append_a_br2(tt, AAb)

def append_a_br2(tt, AAb):
    # tt: e0 (3t 3b) (2t 2b) e1
    # AAb['tl']:  ((1t 1b) (0t 0b)) ((3t 3b) (2t 2b))
    tt = tt.fuse_legs(axes=((1, 2), (0, 3)))  # ((3t 3b) (2t 2b)) (e0 e1)
    tt = tensordot(AAb['tl'], tt, axes=(1, 0))  # ((1t 1b) (0t 0b)) (e0 e1)
    tt = tt.unfuse_legs(axes=(0, 1))  # (1t 1b) (0t 0b) e0 e1
    tt = tt.transpose(axes=(2, 1, 3, 0))  # e0 (0t 0b) e1 (1t 1b)
    return tt

def fcor_bl(env_b, env_bl, env_l, AAb):
    """ Creates extended bottom left corner. Order of indices see append_a_bl. """
    corbln = tensordot(env_b, env_bl, axes=(2, 0))
    corbln = tensordot(corbln, env_l, axes=(2, 0))
    return append_a_bl(corbln, AAb)

def fcor_tl(env_l, env_tl, env_t, AAb):
    """ Creates extended top left corner. """
    cortln = tensordot(env_l, env_tl, axes=(2, 0))
    cortln = tensordot(cortln, env_t, axes=(2, 0))
    return append_a_tl(cortln, AAb)

def fcor_tr(env_t, env_tr, env_r, AAb):
    """ Creates extended top right corner. """
    cortrn = tensordot(env_t, env_tr, axes=(2, 0))
    cortrn = tensordot(cortrn, env_r, axes=(2, 0))
    return append_a_tr(cortrn, AAb)

def fcor_br(env_r, env_br, env_b, AAb):
    """ Creates extended bottom right corner. """
    corbrn = tensordot(env_r, env_br, axes=(2, 0))
    corbrn = tensordot(corbrn, env_b, axes=(2, 0))
    return append_a_br(corbrn, AAb)


def apply_TMO_left(vecl, env, indten, indop, AAb):
    """ apply TM (bottom-AAB-top) to left boundary vector"""
    new_vecl = tensordot(vecl, env[indten].t, axes=(2, 0))
    new_vecl = append_a_tl(new_vecl, AAb[indop])
    new_vecl = ncon((new_vecl, env[indten].b), ([1, 2, -3, -2], [-1, 2, 1]))
    new_vecl = new_vecl.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if new_vecl.ndim == 5:
        new_vecl = new_vecl.swap_gate(axes=((0, 3), 2))
        new_vecl = new_vecl.fuse_legs(axes=((0, 2), (1, 3), 4))
    elif new_vecl.ndim == 4:
        new_vecl =  new_vecl.fuse_legs(axes=(0, (1, 2), 3))
    return new_vecl


def apply_TMO_right(vecr, env, indten, indop, AAb):
    """ apply TM (top-AAB-bottom) to right boundary vector"""
    new_vecr = tensordot(vecr, env[indten].b, axes=(2, 0))
    new_vecr = append_a_br(new_vecr, AAb[indop])
    new_vecr = ncon((new_vecr, env[indten].t), ([1, 2, -3, -2], [-1, 2, 1]))
    new_vecr = new_vecr.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if new_vecr.ndim == 5:
        new_vecr = new_vecr.swap_gate(axes=((3, 4), 2))
        new_vecr =  new_vecr.fuse_legs(axes=(0, (1, 3), (4, 2)))
    elif new_vecr.ndim == 4:
        new_vecr =  new_vecr.fuse_legs(axes=(0, (1, 2), 3))
    return new_vecr


def apply_TM_left(vecl, env, indten, AAb):
    """ apply TM (bottom-AAB-top) to left boundary vector"""
    new_vecl = tensordot(vecl, env[indten].t, axes=(2, 0))
    new_vecl = append_a_tl(new_vecl, AAb)
    new_vecl = new_vecl.unfuse_legs(axes=0)
    if new_vecl.ndim == 5:
        new_vecl = ncon((new_vecl, env[indten].b), ([1, -4, 2, -3, -2], [-1, 2, 1]))
        new_vecl = new_vecl.fuse_legs(axes=((0, 3), 1, 2))
    elif new_vecl.ndim == 4:
        new_vecl = ncon((new_vecl, env[indten].b), ([1, 2, -3, -2], [-1, 2, 1]))
    return new_vecl


def apply_TMO_top(vect, env, indten, indop, AAb):
    """ apply TMO (top-AAB-bottom) to top boundary vector"""
    new_vect = tensordot(env[indten].l, vect, axes=(2, 0))
    new_vect = append_a_tl(new_vect, AAb[indop])
    new_vect = ncon((new_vect, env[indten].r), ([-1, -2, 2, 1], [2, 1, -3]))
    new_vect = new_vect.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if new_vect.ndim == 5:
        new_vect = new_vect.swap_gate(axes=((1, 4), 2))
        new_vect =  new_vect.fuse_legs(axes=(0, (1, 3), (4, 2)))
    elif new_vect.ndim == 4:
        new_vect =  new_vect.fuse_legs(axes=(0, (1, 2), 3))
    return new_vect


def apply_TMO_bottom(vecb, env, indten, indop, AAb):
    """ apply TMO (bottom-AAB-top) to bottom boundary vector"""
    new_vecb = tensordot(env[indten].r, vecb, axes=(2, 0))
    new_vecb = append_a_br(new_vecb, AAb[indop])
    new_vecb = ncon((new_vecb, env['strl', indten]), ([-1, -2, 1, 2], [1, 2, -3]))
    new_vecb = new_vecb.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if new_vecb.ndim == 5:
        new_vecb = new_vecb.swap_gate(axes=((0, 1), 2))
        new_vecb =  new_vecb.fuse_legs(axes=((0, 2), (1, 3), 4)) 
    elif new_vecb.ndim == 4:
        new_vecb =  new_vecb.fuse_legs(axes=(0, (1, 2), 3))
    return new_vecb


def apply_TM_top(vect, env, indten, AAb):
    """ apply TM (left-AAB-right)   to top boundary vector"""
    new_vect = tensordot(env[indten].l, vect, axes=(2, 0))
    new_vect = append_a_tl(new_vect, AAb)
    new_vect = new_vect.unfuse_legs(axes=2)
    if new_vect.ndim == 5:
        new_vect = ncon((new_vect, env[indten].r), ([-1, -2, 2, -4, 1], [2, 1, -3]))
        new_vect = new_vect.fuse_legs(axes=(0, 1, (2, 3)))
    elif new_vect.ndim == 4:
        new_vect = ncon((new_vect, env[indten].r), ([-1, -2, 2, 1], [2, 1, -3]))
    return new_vect



def EV2sNN_ver(env, indten_t, indten_b, indop_t, indop_b, AAb):
    """
    Calculates 2-site verical NN expectation value.
    indt is the top PEPS index. indb is the bottom PEPS index.
    """
    cortln = fcor_tl(env, indten_t, indop_t, AAb)
    corttn = ncon((cortln, env[indten_t].tr, env[indten_t].r), ((-0, -1, 2, 3), (2, 1), (1, 3, -2)))
    del cortln
    corbrn = fcor_br(env, indten_b, indop_b, AAb)
    corbbn = ncon((corbrn, env[indten_b].l, env[indten_b].bl), ((-2, -1, 2, 3), (1, 3, -0), (2, 1)))
    del corbrn
    return vdot(corttn, corbbn, conj=(0, 0))


def EV1s(env, indop, AAb):
    """
    Calculates 1-site expectation value.
    indt is the top PEPS index. indb is the bottom PEPS index.
    """
    indten = 0 if indop == 'l' else 1
    cortln = fcor_tl(env, indten, indop, AAb)
    top_part = ncon((cortln, env[indten].tr, env[indten].r), ((-0, -1, 2, 3), (2, 1), (1, 3, -2)))
    bot_part = ncon((env[indten].br, env[indten].b, env[indten].bl), ((-2, 1), (1, -1, 2), (2, -0)))
    return vdot(top_part, bot_part, conj=(0, 0))


def proj_Cor(rt, rb, fix_signs, opts_svd):
    """
    tt upper half of the CTM 4 extended corners diagram indices from right (e,t,b) to left (e,t,b)
    tb lower half of the CTM 4 extended corners diagram indices from right (e,t,b) to left (e,t,b)
    """

    rr = tensordot(rt, rb, axes=((1, 2), (1, 2)))
    u, s, v = svd_with_truncation(rr, axes=(0, 1), sU =rt.get_signature()[1], fix_signs=fix_signs, **opts_svd)
    s = s.rsqrt()
    pt = s.broadcast(tensordot(rb, v, axes=(0, 1), conj=(0, 1)), axis=2)
    pb = s.broadcast(tensordot(rt, u, axes=(0, 0), conj=(0, 1)), axis=2)
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
        Class containing data for the new environment tensors 
        renormalized by the horizontal move at each site + lattice info.
    env : class CtmEnv
        Class containing data for the input environment tensors at each site + lattice info.
    AAb : Contains top and bottom Peps tensors
    proj : Class Proj with info on lattice
        Projectors to be applied for renormalization.
    ms : site whose environment tensors are to be renormalized

    Returns
    -------
    envn :  class CtmEnv
        Contains data of updated environment tensors along with those not updated
    """

    (_,y) = ms
    left = AAb.nn_site(ms, d='l')
    right = AAb.nn_site(ms, d='r')
    
    if AAb.boundary == 'finite' and y == 0:
        l_abv = None
        l_bel = None
    else:
        l_abv = AAb.nn_site(left, d='t')
        l_bel =  AAb.nn_site(left, d='b')

    if AAb.boundary == 'finite' and y == (env.Ny-1):
        r_abv = None
        r_bel = None
    else:
        r_abv = AAb.nn_site(right, d='t')
        r_bel = AAb.nn_site(right, d='b')

    if AAb.lattice == 'checkerboard':
        if ms == (0,0):
            left =  right = (0,1)
            l_abv = r_abv = r_bel = l_bel = (0,0)
        elif ms == (0,1):
            left =  right = (0,0)
            l_abv = r_abv = r_bel = l_bel = (0,1)

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
        tt_l = append_a_tl(tt_l, AAb[left])
        envn[ms].l = ncon((proj[left].hlb, tt_l), ([2, 1, -0], [2, 1, -2, -1]))

    if not(right is None):
        tt_r = tensordot(env[right].r, proj[right].hrb, axes=(2,0))
        tt_r = append_a_br(tt_r, AAb[right])
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
        Class containing data for the new environment tensors 
        renormalized by the vertical move at each site + lattice info.
    env : class CtmEnv
        Class containing data for the input environment tensors at each site + lattice info.
    AAb : Contains top and bottom Peps tensors
    proj : Class Proj with info on lattice
        Projectors to be applied for renormalization.
    ms : site whose environment tensors are to be renormalized

    Returns
    -------
    envn :  class CtmEnv
        Contains data of updated environment tensors along with those not updated
    """


    (x,_) = ms
    top = AAb.nn_site(ms, d='t')
    bottom = AAb.nn_site(ms, d='b')

    if AAb.boundary == 'finite' and x == 0:
        t_left = None
        t_right = None
    else:
        t_left = AAb.nn_site(top, d='l')
        t_right =  AAb.nn_site(top, d='r')
    if AAb.boundary == 'finite' and x == (env.Nx-1):
        b_left = None
        b_right = None
    else:
        b_left = AAb.nn_site(bottom, d='l')
        b_right = AAb.nn_site(bottom, d='r')

    if AAb.lattice == 'checkerboard':
        if ms == (0,0):
            top =  bottom = (0,1)
            t_left = b_left = b_right = t_right = (0,0)
        elif ms == (0,1):
            top =  bottom = (0,0)
            t_left = b_left = b_right = t_right = (0,1)


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
        ll_t = append_a_tl(ll_t, AAb[top])
        envn[ms].t = ncon((ll_t, proj[top].vtr), ([-0, -1, 2, 1], [2, 1, -2]))
    if not(bottom is None):
        ll_b = ncon((proj[bottom].vbr, env[bottom].b), ([1, -2, -1], [1, -3, -4]))
        ll_b = append_a_br(ll_b, AAb[bottom])
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
        la, lb, lc = a.get_legs(axis=2), b.get_legs(axis=0), c.get_legs(axis=0)
    elif dirn == 'hlb':
        la, lb, lc = a.get_legs(axis=0), b.get_legs(axis=2), c.get_legs(axis=1)
    elif dirn == 'hrt':
        la, lb, lc = a.get_legs(axis=0), b.get_legs(axis=0), c.get_legs(axis=1)
    elif dirn == 'hrb':
        la, lb, lc = a.get_legs(axis=2), b.get_legs(axis=2), c.get_legs(axis=0)
    elif dirn == 'vtl':
        la, lb, lc = a.get_legs(axis=0), b.get_legs(axis=1), c.get_legs(axis=1)
    elif dirn == 'vbl':
        la, lb, lc = a.get_legs(axis=2), b.get_legs(axis=1), c.get_legs(axis=0)
    elif dirn == 'vtr':
        la, lb, lc = a.get_legs(axis=2), b.get_legs(axis=3), c.get_legs(axis=0)
    elif dirn == 'vbr':
        la, lb, lc = a.get_legs(axis=0), b.get_legs(axis=3), c.get_legs(axis=1)

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

    Procedure
    ---------
    The function performs a CTMRG update for a rectangular lattice using the corner transfer matrix
    renormalization group (CTMRG) algorithm. The update is performed in two steps: a horizontal move
    and a vertical move. The projectors for each move are calculated first, and then the tensors in
    the CTM environment are updated using the projectors. The boundary conditions of the lattice
    determine whether trivial projectors are needed for the move. If the boundary conditions are
    'infinite', no trivial projectors are needed; if they are 'finite', four trivial projectors
    are needed for each move. The signs of the environment tensors can also be fixed during the
    update if `fix_signs` is set to True. The latter is important when we set the criteria for 
    stopping the algorithm based on singular values of the corners.

    """

    proj = Proj(lattice=AAb.lattice, dims=AAb.dims, boundary=AAb.boundary) 
    for ms in proj.sites():
        proj[ms] = Local_Projector_Env()

    print('######## Calculating projectors for horizontal move ###########')
    envn_hor = env.copy()

    for ms in AAb.tensors_CtmEnv():   #ctm_wndows(trajectory='h', index=y): # horizontal absorption and renormalization
        print('projector calculation ctm cluster horizontal', ms)
        proj = proj_horizontal(env[ms.nw], env[ms.ne], env[ms.sw], env[ms.se], proj, ms, AAb[ms.nw], AAb[ms.ne], AAb[ms.sw], AAb[ms.se], fix_signs, opts_svd)

    if AAb.boundary == 'finite': 
        # we need trivial projectors on the boundary for horizontal move for finite lattices
        for ms in range(AAb.Ny-1):
            proj[0,ms].hlt = trivial_projector(env[0,ms].l, AAb[0,ms], env[0,ms+1].tl, dirn='hlt')
            proj[AAb.Nx-1,ms].hlb = trivial_projector(env[AAb.Nx-1,ms].l, AAb[AAb.Nx-1,ms], env[AAb.Nx-1,ms+1].bl, dirn='hlb')
            proj[0, ms+1].hrt = trivial_projector(env[0,ms+1].r, AAb[0,ms+1], env[0,ms].tr, dirn='hrt')
            proj[AAb.Nx-1, ms+1].hrb = trivial_projector(env[AAb.Nx-1,ms+1].r, AAb[AAb.Nx-1,ms+1], env[AAb.Nx-1,ms].br, dirn='hrb')
    
    print('######## Horizontal Move ###########')
        

    if AAb.lattice == 'checkerboard':
        envn_hor = move_horizontal(envn_hor, env, AAb, proj, (0,0))
        envn_hor = move_horizontal(envn_hor, env, AAb, proj, (0,1))
    else:
        for ms in AAb.sites():
            print('move ctm horizontal', ms)
            envn_hor = move_horizontal(envn_hor, env, AAb, proj, ms)

    envn_ver = envn_hor.copy()

    print('######## Calculating projectors for vertical move ###########')    
    for ms in AAb.tensors_CtmEnv():   # vertical absorption and renormalization
        print('projector calculation ctm cluster vertical', ms)
        proj = proj_vertical(envn_hor[ms.nw], envn_hor[ms.sw], envn_hor[ms.ne], envn_hor[ms.se], proj, ms, AAb[ms.nw], AAb[ms.sw], AAb[ms.ne], AAb[ms.se], fix_signs, opts_svd)

    if AAb.boundary == 'finite': 
        # we need trivial projectors on the boundary for vertical move for finite lattices
        for ms in range(AAb.Nx-1):
            proj[ms,0].vtl = trivial_projector(envn_hor[ms,0].t, AAb[ms,0], envn_hor[ms+1,0].tl, dirn='vtl')
            proj[ms+1,0].vbl = trivial_projector(envn_hor[ms+1,0].b, AAb[ms+1,0], envn_hor[ms,0].bl, dirn='vbl')
            proj[ms,AAb.Ny-1].vtr = trivial_projector(envn_hor[ms, AAb.Ny-1].t, AAb[ms,AAb.Ny-1], envn_hor[ms+1,AAb.Ny-1].tr, dirn='vtr')
            proj[ms+1,AAb.Ny-1].vbr = trivial_projector(envn_hor[ms+1,AAb.Ny-1].b, AAb[ms+1,AAb.Ny-1], envn_hor[ms,AAb.Ny-1].br, dirn='vbr')

    print('######### Vertical Move ###########')

    if AAb.lattice == 'checkerboard':
        envn_ver = move_vertical(envn_ver, envn_hor, AAb, proj, (0,0))
        envn_ver = move_vertical(envn_ver, envn_hor, AAb, proj, (0,1))
    else:
        for ms in AAb.sites():   # vertical absorption and renormalization
            print('move ctm vertical', ms)
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
    leg = A.get_legs(axis=-1)
    _, leg = yast.leg_undo_product(leg) # last leg of A should be fused
    fid = yast.eye(config=A.config, legs=[leg, leg.conj()]).diag()

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
            B = A.swap_gate(axes=(0, 1, 2, 3)) # t l b r [s a]
        if A.ndim == 6:
            B_int = A.swap_gate(axes=(0, 1, 2, 3)) # t l b r str [s a]
            B_int = B_int.unfuse_legs(axes=5) # t l b r str s a
            B = B_int.fuse_legs(axes=(0, 1, 2, 3, 5, (6, 4))).fuse_legs(axes=(0, 1, 2, 3, (4, 5))) # t l b r [s [a str]]
    elif B is not None:
        B = B.swap_gate(axes=(0, 1, 2, 3))
    AAb = DoublePepsTensor(top=Ao, btm=B)
    return AAb


def fPEPS_fuse_layers(AAb, EVonly=False):

    r"""
    Fuse two layers of PEPS tensors along their vertical and horizontal bonds to obtain a new, thicker
    layer.

    Parameters
    ----------
        AAb (dict): A dictionary of PEPS tensors for the two adjacent layers, with keys 'A', 'Ab', 'B', 'Bb', etc.
        EVonly (bool): If True, only compute and return the top-left corner of the fused layer, containing the
            eigenvectors. If False (default), also compute and return the bottom-left corner, containing the
            singular values.

    Returns
    -------
        dict: A dictionary containing the fused PEPS tensor for the new, thicker layer, with keys 'tl' (top-left
            corner) and 'bl' (bottom-left corner, only if EVonly is False).
    """

    for st in AAb.values():
        fA = st.top.fuse_legs(axes=((0, 1), (2, 3), 4))  # (0t 1t) (2t 3t) 4t
       # st.pop('A')
        fAb = st.btm.fuse_legs(axes=((0, 1), (2, 3), 4))  # (0b 1b) (2b 3b) 4b
       # st.pop('Ab')
        tt = tensordot(fA, fAb, axes=(2, 2), conj=(0, 1))  # (0t 1t) (2t 3t) (0b 1b) (2b 3b)
        tt = tt.fuse_legs(axes=(0, 2, (1, 3)))  # (0t 1t) (0b 1b) ((2t 3t) (2b 3b))
        tt = tt.unfuse_legs(axes=(0, 1))  # 0t 1t 0b 1b ((2t 3t) (2b 3b))
        tt = tt.swap_gate(axes=(1, 2))  # 0t 1t 0b 1b ((2t 3t) (2b 3b))
        tt = tt.fuse_legs(axes=((0, 2), (1, 3), 4))  # (0t 0b) (1t 1b) ((2t 3t) (2b 3b))
        tt = tt.fuse_legs(axes=((1, 0), 2))  # ((1t 1b) (0t 0b)) ((2t 3t) (2b 3b))
        tt = tt.unfuse_legs(axes=1)  # ((1t 1b) (0t 0b)) (2t 3t) (2b 3b)
        tt = tt.unfuse_legs(axes=(1, 2))  # ((1t 1b) (0t 0b)) 2t 3t 2b 3b
        tt = tt.swap_gate(axes=(1, 4))  # ((1t 1b) (0t 0b)) 2t 3t 2b 3b
        tt = tt.fuse_legs(axes=(0, (1, 3), (2, 4)))  # ((1t 1b) (0t 0b)) (2t 2b) (3t 3b)
        st['tl'] = tt.fuse_legs(axes=(0, (2, 1))) # ((1t 1b) (0t 0b)) ((3t 3b) (2t 2b))
        if not EVonly:
            tt = tt.unfuse_legs(axes=0)  # (1t 1b) (0t 0b) (2t 2b) (3t 3b)
            st['bl'] = tt.fuse_legs(axes=((2, 0), (1, 3)))  # ((2t 2b) (1t 1b)) ((0t 0b) (3t 3b))


def check_consistency_tensors(A):
    # to check if the A tensors have the appropriate configuration of legs i.e. t l b r [s a]

    Ab = peps.Peps(A.lattice, A.dims, A.boundary)
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
        