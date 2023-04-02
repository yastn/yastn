# this routine just calculates nearest neighbor correlators

from yastn import tensordot, svd_with_truncation, rand, ncon
from ._ctm_iteration_routines import append_a_tl, append_a_br, append_a_tr, append_a_bl, fPEPS_2layers

def ret_AAbs(A, bds, op, orient):
    """ preparing the nearest neighbor tensor before contraction by attaching them with operators"""
    if orient == 'h':
        AAb = {'l': fPEPS_2layers(A[bds.site_0], op=op['l'], dir='l'), 'r': fPEPS_2layers(A[bds.site_1], op=op['r'], dir='r')}
    elif orient == 'v':
        AAb = {'l': fPEPS_2layers(A[bds.site_0], op=op['l'], dir='t'), 'r': fPEPS_2layers(A[bds.site_1], op=op['r'], dir='b')}
    elif orient == '1s':
        AAb = {'l': fPEPS_2layers(A[bds.site_0], op=op['l'], dir='1s'), 'r': fPEPS_2layers(A[bds.site_1], op=op['r'], dir='1s')}
    return AAb


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


def left_right_op_vectors(env, site_0, site_1, AAb):
    """ form the left and right part in the process of evaluating a horizontal correlator before contracting them"""

    vecl = tensordot(env[site_0].l, env[site_0].tl, axes=(2, 0))
    vecl = tensordot(env[site_0].bl, vecl, axes=(1, 0))

    vecr = tensordot(env[site_1].tr, env[site_1].r, axes=(1, 0)) 
    vecr = tensordot(vecr, env[site_1].br, axes=(2, 0))

    new_vecl = apply_TMO_left(vecl, env, site_0, AAb['l'])
    new_vecr = apply_TMO_right(vecr, env, site_1, AAb['r'])

    return new_vecl, new_vecr


def top_bottom_op_vectors(env, site_0, site_1, AAb):
    """ form the top and bottom part in the process of evaluating a vertical correlator before contracting them"""

    vect = tensordot(env[site_0].tl, env[site_0].t, axes=(1, 0))
    vect = tensordot(vect, env[site_0].tr, axes=(2, 0))

    vecb = tensordot(env[site_1].b, env[site_1].bl, axes=(2, 0))
    vecb = tensordot(env[site_1].br, vecb, axes=(1, 0))

    new_vect = apply_TMO_top(vect, env, site_0, AAb['l'])
    new_vecb = apply_TMO_bottom(vecb, env, site_1, AAb['r'])

    return new_vect, new_vecb

def con_bi(new_vecl, new_vecr):
    return tensordot(new_vecl, new_vecr, axes=((0, 1, 2), (2, 1, 0))).to_number()

def hor_extension(env, bd, AAbo, AAb):
    """ merge the left and right vecs + TM """
    site_0, site_1 = bd.site_0, bd.site_1

    left_bound_vec, right_bound_vec = left_right_op_vectors(env, site_0, site_1, AAbo) 
    hor = con_bi(left_bound_vec, right_bound_vec)
    left_bound_vec_norm, right_bound_vec_norm = left_right_op_vectors(env, site_0, site_1, AAb) 
    hor_norm = con_bi(left_bound_vec_norm, right_bound_vec_norm)

    return (hor/hor_norm)

def ver_extension(env, bd, AAbo, AAb):
    """ merge the left and right vecs + TM """
    site_0, site_1 = bd.site_0, bd.site_1
    top_bound_vec, bottom_bound_vec = top_bottom_op_vectors(env, site_0, site_1, AAbo)
    ver = con_bi(top_bound_vec, bottom_bound_vec) 
    top_bound_vec_norm, bottom_bound_vec_norm = top_bottom_op_vectors(env, site_0, site_1, AAb)
    ver_norm = con_bi(top_bound_vec_norm, bottom_bound_vec_norm) 

    return (ver/ver_norm)

