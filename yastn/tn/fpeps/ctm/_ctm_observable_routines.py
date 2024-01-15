# this routine just calculates nearest neighbor correlators
import numpy as np
from yastn import tensordot, ncon
from ._ctm_iteration_routines import append_a_tl, append_a_br, append_a_tr, append_a_bl, fPEPS_2layers

def ret_AAbs(A, bds, op, orient):
    """ preparing the nearest neighbor tensor before contraction by attaching them with operators"""
    if orient == 'h':
        AAb = {'l': fPEPS_2layers(A[bds.site0], op=op['l'], dir='l'), 'r': fPEPS_2layers(A[bds.site1], op=op['r'], dir='r')}
    elif orient == 'v':
        AAb = {'l': fPEPS_2layers(A[bds.site0], op=op['l'], dir='t'), 'r': fPEPS_2layers(A[bds.site1], op=op['r'], dir='b')}
    elif orient == '1s':
        AAb = {'l': fPEPS_2layers(A[bds.site0], op=op['l'], dir='1s'), 'r': fPEPS_2layers(A[bds.site1], op=op['r'], dir='1s')}
    return AAb

def apply_TM_left(vecl, env, site, AAb):
    """ apply TM (bottom-AAB-top) to left boundary vector"""
    new_vecl = tensordot(vecl, env[site].t, axes=(2, 0))
    new_vecl = append_a_tl(new_vecl, AAb)
    new_vecl = new_vecl.unfuse_legs(axes=0)
    if new_vecl.ndim == 5:
        new_vecl = ncon((new_vecl, env[site].b), ([1, -4, 2, -3, -2], [-1, 2, 1]))
        new_vecl = new_vecl.fuse_legs(axes=((0, 3), 1, 2))
    elif new_vecl.ndim == 4:
        new_vecl = ncon((new_vecl, env[site].b), ([1, 2, -3, -2], [-1, 2, 1]))
    return new_vecl

def apply_TM_top(vect, env, site, AAb):
    """ apply TM (left-AAB-right)   to top boundary vector"""
    new_vect = tensordot(env[site].l, vect, axes=(2, 0))
    new_vect = append_a_tl(new_vect, AAb)
    new_vect = new_vect.unfuse_legs(axes=2)

    if new_vect.ndim == 5:
        new_vect = ncon((new_vect, env[site].r), ([-1, -2, 2, -4, 1], [2, 1, -3]))
        new_vect =  new_vect.fuse_legs(axes=(0, 1, (2, 3)))
    elif new_vect.ndim == 4:
        new_vect = ncon((new_vect, env[site].r), ([-1, -2, 2, 1], [2, 1, -3]))
    return new_vect

def apply_TMO_left(vecl, env, site, AAb):
    """ apply TM (bottom-AAB-top) to left boundary vector"""
    new_vecl = tensordot(vecl, env[site].t, axes=(2, 0))
    new_vecl = append_a_tl(new_vecl, AAb)
    new_vecl = ncon((new_vecl, env[site].b), ([1, 2, -3, -2], [-1, 2, 1]))
    new_vecl = new_vecl.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if new_vecl.ndim == 5:
        new_vecl = new_vecl.swap_gate(axes=((0, 3), 2))
        new_vecl = new_vecl.fuse_legs(axes=((0, 2), (1, 3), 4))
    elif new_vecl.ndim == 4:
        new_vecl =  new_vecl.fuse_legs(axes=(0, (1, 2), 3))
    return new_vecl

def apply_TMO_right(vecr, env, site, AAb):
    """ apply TM (top-AAB-bottom) to right boundary vector"""
    new_vecr = tensordot(vecr, env[site].b, axes=(2, 0))
    new_vecr = append_a_br(new_vecr, AAb)
    new_vecr = ncon((new_vecr, env[site].t), ([1, 2, -3, -2], [-1, 2, 1]))
    new_vecr = new_vecr.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if new_vecr.ndim == 5:
        new_vecr = new_vecr.swap_gate(axes=((3, 4), 2))
        new_vecr =  new_vecr.fuse_legs(axes=(0, (1, 3), (4, 2)))
    elif new_vecr.ndim == 4:
        new_vecr =  new_vecr.fuse_legs(axes=(0, (1, 2), 3))
    return new_vecr

def apply_TMO_top(vect, env, site, AAb):
    """ apply TMO (top-AAB-bottom) to top boundary vector"""
    new_vect = tensordot(env[site].l, vect, axes=(2, 0))
    new_vect = append_a_tl(new_vect, AAb)
    new_vect = ncon((new_vect, env[site].r), ([-1, -2, 2, 1], [2, 1, -3]))
    new_vect = new_vect.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if new_vect.ndim == 5:
        new_vect = new_vect.swap_gate(axes=((1, 4), 2))
        new_vect =  new_vect.fuse_legs(axes=(0, (1, 3), (4, 2)))
    elif new_vect.ndim == 4:
        new_vect =  new_vect.fuse_legs(axes=(0, (1, 2), 3))
    return new_vect

def apply_TMO_bottom(vecb, env, site, AAb):
    """ apply TMO (bottom-AAB-top) to bottom boundary vector"""
    new_vecb = tensordot(env[site].r, vecb, axes=(2, 0))
    new_vecb = append_a_br(new_vecb, AAb)
    new_vecb = ncon((new_vecb, env[site].l), ([-1, -2, 1, 2], [1, 2, -3]))
    new_vecb = new_vecb.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if new_vecb.ndim == 5:
        new_vecb = new_vecb.swap_gate(axes=((0, 1), 2))
        new_vecb =  new_vecb.fuse_legs(axes=((0, 2), (1, 3), 4))
    elif new_vecb.ndim == 4:
        new_vecb =  new_vecb.fuse_legs(axes=(0, (1, 2), 3))
    return new_vecb


def left_right_op_vectors(env, site0, site1, AAbl, AAbr):
    """ form the left and right part in the process of evaluating a horizontal correlator before contracting them"""

    vecl = tensordot(env[site0].l, env[site0].tl, axes=(2, 0))
    vecl = tensordot(env[site0].bl, vecl, axes=(1, 0))

    vecr = tensordot(env[site1].tr, env[site1].r, axes=(1, 0))
    vecr = tensordot(vecr, env[site1].br, axes=(2, 0))

    new_vecl = apply_TMO_left(vecl, env, site0, AAbl)
    new_vecr = apply_TMO_right(vecr, env, site1, AAbr)

    return new_vecl, new_vecr


def top_bottom_op_vectors(env, site0, site1, AAbt, AAbb):
    """ form the top and bottom part in the process of evaluating a vertical correlator before contracting them"""

    vect = tensordot(env[site0].tl, env[site0].t, axes=(1, 0))
    vect = tensordot(vect, env[site0].tr, axes=(2, 0))

    vecb = tensordot(env[site1].b, env[site1].bl, axes=(2, 0))
    vecb = tensordot(env[site1].br, vecb, axes=(1, 0))

    new_vect = apply_TMO_top(vect, env, site0, AAbt)
    new_vecb = apply_TMO_bottom(vecb, env, site1, AAbb)

    return new_vect, new_vecb




def array_EV2pt(peps, env, site0, site1, op=None):

    r"""
    Calculate two-point axial correlators between two sites site0 and site1.

    Parameters
    ----------
        peps: Class Peps.
        env: Class CtmEnv contaning data for CTM environment tensors .
        site0: The coordinate of the first site as a tuple of two integers.
        site1: The coordinate of the second site as a tuple of two integers.
        op: An optional dictionary specifying the operators to be applied to the sites.

    Raises
    ------
        ValueError: If site0 and site1 are the same, or if they are not aligned either horizontally or vertically.
    """

    # check if horizontal or vertical

    dx, dy = site1[0] - site0[0], site1[1] - site0[1]
    if dx == dy == 0:
        raise ValueError("Both sites are the same.")
    elif dx != 0 and dy != 0:
        raise ValueError("Both sites should be aligned vertically or horizontally.")

    orient = 'horizontal' if dx == 0 else 'vertical'
    num_correlators = abs(dy) if orient == 'horizontal' else abs(dx)
    nxt_site = peps.nn_site(site0, d='r' if orient == 'horizontal' else 'b')

    if orient == 'horizontal':
        AAbl = fPEPS_2layers(peps[site0])
        AAbr = fPEPS_2layers(peps[nxt_site])
    elif orient == 'vertical':
        AAbt = fPEPS_2layers(peps[site0])
        AAbb = fPEPS_2layers(peps[nxt_site])

    if op is not None:
        if orient == 'horizontal':
            AAbl = fPEPS_2layers(peps[site0], op=op['l'], dir='l')
            AAbr = fPEPS_2layers(peps[nxt_site], op=op['r'], dir='r')
        elif orient == 'vertical':
            AAbt = fPEPS_2layers(peps[site0], op=op['l'], dir='t')
            AAbb = fPEPS_2layers(peps[nxt_site], op=op['r'], dir='b')

    array_vals = np.zeros((num_correlators), dtype=np.complex128)
    if orient == 'horizontal':
        left_bound_vec, right_bound_vec = left_right_op_vectors(env, site0, nxt_site, AAbl, AAbr)
    elif orient == 'vertical':
        left_bound_vec, right_bound_vec = top_bottom_op_vectors(env, site0, nxt_site, AAbt, AAbb)

    array_vals[0] = con_bi(left_bound_vec, right_bound_vec)
    vecl = left_bound_vec

    for num in range(2, num_correlators+1):

        AAb_nxt = fPEPS_2layers(peps[nxt_site])
        new_vecl = apply_TM_left(vecl, env, nxt_site, AAb_nxt) if orient == 'horizontal' else apply_TM_top(vecl, env, nxt_site, AAb_nxt)
        nxt_site = peps.nn_site(nxt_site, d='r' if orient == 'horizontal' else 'b')

        if op is not None:
            if orient == 'horizontal':
                AAbr = fPEPS_2layers(peps[nxt_site], op=op['r'], dir='r')
            elif orient == 'vertical':
                AAbb = fPEPS_2layers(peps[nxt_site], op=op['r'], dir='b')
        else:
            if orient == 'horizontal':
                AAbr = fPEPS_2layers(peps[nxt_site])
            elif orient == 'vertical':
                AAbb = fPEPS_2layers(peps[nxt_site])

        if orient == 'horizontal':
            vecr = tensordot(env[nxt_site].tr, env[nxt_site].r, axes=(1, 0))
            vecr = tensordot(vecr, env[nxt_site].br, axes=(2, 0))
            right_bound_vec = apply_TMO_right(vecr, env, nxt_site, AAbr)
        elif orient == 'vertical':
            vecb = tensordot(env[nxt_site].b, env[nxt_site].bl, axes=(2, 0))
            vecb = tensordot(env[nxt_site].br, vecb, axes=(1, 0))
            right_bound_vec = apply_TMO_bottom(vecb, env, nxt_site, AAbb)

        array_vals[num-1] = con_bi(new_vecl, right_bound_vec)
        vecl = new_vecl

    return array_vals

def con_bi(new_vecl, new_vecr):
    return tensordot(new_vecl, new_vecr, axes=((0, 1, 2), (2, 1, 0))).to_number()

def hor_extension(env, bd, AAbo, AAb):
    """ merge the left and right vecs + TM """
    site0, site1 = bd.site0, bd.site1

    left_bound_vec, right_bound_vec = left_right_op_vectors(env, site0, site1, AAbo)
    hor = con_bi(left_bound_vec, right_bound_vec)
    left_bound_vec_norm, right_bound_vec_norm = left_right_op_vectors(env, site0, site1, AAb)
    hor_norm = con_bi(left_bound_vec_norm, right_bound_vec_norm)

    return (hor/hor_norm)

def ver_extension(env, bd, AAbo, AAb):
    """ merge the left and right vecs + TM """
    site0, site1 = bd.site0, bd.site1
    top_bound_vec, bottom_bound_vec = top_bottom_op_vectors(env, site0, site1, AAbo)
    ver = con_bi(top_bound_vec, bottom_bound_vec)
    top_bound_vec_norm, bottom_bound_vec_norm = top_bottom_op_vectors(env, site0, site1, AAb)
    ver_norm = con_bi(top_bound_vec_norm, bottom_bound_vec_norm)

    return (ver/ver_norm)

def make_ext_corner_tl(cortl, str_t, str_l, AAb):
    """ Returns the newly created extended top-left corner tensor """
    vec_cor_tl = str_l @ cortl @ str_t
    new_vec_cor_tl = append_a_tl(vec_cor_tl, AAb).fuse_legs(axes=(0, 1, 3, 2))
    nftl = new_vec_cor_tl.unfuse_legs(axes=2).unfuse_legs(axes=2)
    if nftl.ndim == 6:
        new_vec_cor_tl = nftl.swap_gate(axes=(4, 3))
        new_vec_cor_tl = new_vec_cor_tl.fuse_legs(axes=(0, 1, (2, 4), 5, 3))
    return new_vec_cor_tl

def make_ext_corner_tr(cortr, str_t, str_r, AAb):
    """ Returns the newly created extended top-right corner tensor """
    vec_cor_tr = str_t @ cortr @ str_r
    new_vec_cor_tr = append_a_tr(vec_cor_tr, AAb).fuse_legs(axes=(0, 1, 3, 2))
    nftr = new_vec_cor_tr.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if nftr.ndim == 6:
        new_vec_cor_tr = nftr.swap_gate(axes=(2, 3))
        new_vec_cor_tr = new_vec_cor_tr.fuse_legs(axes=(0, (1, 3), 4, 5, 2))
    return new_vec_cor_tr

def make_ext_corner_bl(corbl, str_b, str_l, AAb):
    """ Returns the newly created extended bottom-left corner tensor """
    vec_cor_bl = str_b @ corbl @ str_l
    new_vec_cor_bl = append_a_bl(vec_cor_bl, AAb).fuse_legs(axes=(0, 1, 3, 2))
    nfbl = new_vec_cor_bl.unfuse_legs(axes=1).unfuse_legs(axes=1)
    if nfbl.ndim == 6:
        new_vec_cor_bl = nfbl.swap_gate(axes=(2, 1))
        new_vec_cor_bl = new_vec_cor_bl.fuse_legs(axes=(0, (1, 3), 4, 5, 2))
    return new_vec_cor_bl

def make_ext_corner_br(corbr, str_b, str_r, AAb):
    """ Returns the newly created extended bottom-right corner tensor """
    vec_cor_br = str_r @ corbr @ str_b
    new_vec_cor_br = append_a_br(vec_cor_br, AAb).fuse_legs(axes=(0, 1, 3, 2))
    nfbr = new_vec_cor_br.unfuse_legs(axes=2).unfuse_legs(axes=2)
    if nfbr.ndim == 6:
        new_vec_cor_br = nfbr.swap_gate(axes=(2, 3))
        new_vec_cor_br = new_vec_cor_br.fuse_legs(axes=(0, 1, (2, 4), 5, 3))
    return new_vec_cor_br

def array2ptdiag(peps, env, AAb_top, AAb_bottom, site0, site1, flag=None):
    r""" Returns diagonal correlators

    Note: 1. site0 has to be to at left and site1 at right according to the defined fermionic order.
          2. Flag should be set to 'y' when observables are included.
    """

    x0, y0 = site0
    x1, y1 = site1

    if x0<x1:
        ptl,ptr, pbr, pbl = site0, peps.nn_site(site0, d='r'), site1, peps.nn_site(site1, d='l')
    elif x1<x0:
        ptr, ptl, pbl, pbr = site1, peps.nn_site(site1, d='l'), site0, peps.nn_site(site0, d='r')

    vec_tl = make_ext_corner_tl(env[ptl].tl, env[ptl].t, env[ptl].l, AAb_top['l'])
    vec_tr = make_ext_corner_tr(env[ptr].tr, env[ptr].t, env[ptr].r, AAb_top['r'])
    vec_bl = make_ext_corner_bl(env[pbl].bl, env[pbl].b, env[pbl].l, AAb_bottom['l'])
    vec_br = make_ext_corner_br(env[pbr].br, env[pbr].b, env[pbr].r, AAb_bottom['r'])

    if flag == 'y':
        if x0<x1:
            corr_l = ncon((vec_tl, vec_bl), ([2, 1, -3, -4, -5], [-1, -2, 1, 2]))
            corr_r = ncon((vec_br, vec_tr), ([2, 1, -2, -1, -5], [-4, -3, 1, 2]))
            corr = tensordot(corr_l, corr_r, axes=((0, 1, 2, 3, 4), (0, 1, 2, 3, 4))).to_number()
        elif x1<x0:
            corr_l = ncon((vec_tl, vec_bl), ([2, 1, -3, -4], [-1, -2, 1, 2, -5]))
            corr_r = ncon((vec_br, vec_tr), ([2, 1, -2, -1], [-4, -3, 1, 2, -5]))
            corr = tensordot(corr_l, corr_r, axes=((0, 1, 2, 3, 4), (0, 1, 2, 3, 4))).to_number()
    elif flag is None:
        corr_l = ncon((vec_tl, vec_bl), ([2, 1, -3, -4], [-1, -2, 1, 2]))
        corr_r = ncon((vec_br, vec_tr), ([2, 1, -2, -1], [-4, -3, 1, 2]))
        corr = tensordot(corr_l, corr_r, axes=((0, 1, 2, 3), (0, 1, 2, 3))).to_number()

    return corr
