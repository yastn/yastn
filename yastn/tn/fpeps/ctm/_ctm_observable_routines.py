# this routine just calculates nearest neighbor correlators
from yastn import tensordot, ncon
from ._ctm_iteration_routines import fPEPS_2layers



def apply_TMO_left(vecl, env, site, AAb):
    """ apply TM (bottom-AAB-top) to left boundary vector"""
    new_vecl = tensordot(vecl, env[site].t, axes=(2, 0))
    new_vecl = AAb.append_a_tl(new_vecl)
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
    new_vecr = AAb.append_a_br(new_vecr)
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
    new_vect = AAb.append_a_tl(new_vect)
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
    new_vecb = AAb.append_a_br(new_vecb)
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

def EV2pt(peps, env, site0, site1, op=None):

    r"""
    Calculate correlators between two nearest neighbor sites.

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

    dx, dy = site1.nx - site0.nx, site1.ny - site0.ny
    if dx == dy == 0:
        raise ValueError("Both sites are the same.")
    elif dx > 1 or dy > 1:
        raise ValueError("Sites should be nearest neighbor.")

    orient = 'horizontal' if dx == 0 else 'vertical'

    if orient == 'horizontal':
        AAbl = fPEPS_2layers(peps[site0])
        AAbr = fPEPS_2layers(peps[site1])
    elif orient == 'vertical':
        AAbt = fPEPS_2layers(peps[site0])
        AAbb = fPEPS_2layers(peps[site1])

    if op is not None:
        if orient == 'horizontal':
            AAbl = fPEPS_2layers(peps[site0], op=op['l'], dir='l')
            AAbr = fPEPS_2layers(peps[site1], op=op['r'], dir='r')
        elif orient == 'vertical':
            AAbt = fPEPS_2layers(peps[site0], op=op['l'], dir='t')
            AAbb = fPEPS_2layers(peps[site1], op=op['r'], dir='b')

    if orient == 'horizontal':
        left_bound_vec, right_bound_vec = left_right_op_vectors(env, site0, site1, AAbl, AAbr)
    elif orient == 'vertical':
        left_bound_vec, right_bound_vec = top_bottom_op_vectors(env, site0, site1, AAbt, AAbb)

    result = tensordot(left_bound_vec, right_bound_vec, axes=((0, 1, 2), (2, 1, 0))).to_number()


    return result

