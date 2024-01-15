from ._ctm_iteration_routines import check_consistency_tensors
from ._ctm_iteration_routines import fPEPS_2layers
from ._ctm_observable_routines import apply_TMO_left, con_bi, array_EV2pt, array2ptdiag
import yastn

def nn_exp_dict(peps, env, op):

    r"""
    Returns two dictionaries 'obs_hor' and 'obs_ver' that store horizontal and vertical
    nearest-neighbor expectation values, respectively. Each such dictionary has as its keys
    labels of the input two-site operators which further stores a dictionary of NN sites as
    keys and the corresponding obervables as values.

    Parameters
    ----------
    peps : class Lattice
        class containing peps data along with the lattice structure data
    env: class CtmEnv
        class containing ctm environment tensors along with lattice structure data
    op: dictionary of two-site observables

    """

    peps = check_consistency_tensors(peps)
    obs_hor = {}
    obs_ver = {}

    for ms in op.keys():
        obs_hor[ms] = {}
        obs_ver[ms] = {}

        opt = op.get(ms)

        for bds_h in peps.nn_bonds(dirn='h'):  # correlators on all horizontal bonds
            val_hor = EV2ptcorr(peps, env, opt, bds_h.site0, bds_h.site1)
            obs_hor[ms][(bds_h.site0, bds_h.site1)] = val_hor[0]

        for bds_v in peps.nn_bonds(dirn='v'):  # correlators on all vertical bonds
            val_ver = EV2ptcorr(peps, env, opt, bds_v.site0, bds_v.site1)
            obs_ver[ms][(bds_v.site0, bds_v.site1)] = val_ver[0]

    return obs_hor, obs_ver

def one_site_dict(peps, env, op):
    r"""
    dictionary containing site coordinates as keys and their corresponding expectation values

    Parameters
    ----------
    peps : class Lattice
        class containing peps data along with the lattice structure data
    env: class CtmEnv
        class containing ctm environment tensors along with lattice structure data
    op: single site operator

    """

    site_exp_dict = {}  # Dictionary to store site-wise expectation values
    peps = check_consistency_tensors(peps)

    if peps.lattice == 'checkerboard':
        lists = [(0,0), (0,1)]
    else:
        lists = peps.sites()

    for ms in lists:
        Am = peps[ms]
        val_op = measure_one_site_spin(Am, ms, env, op=op)
        val_norm = measure_one_site_spin(Am, ms, env, op=None)
        site_exp_value = val_op / val_norm  # expectation value of particular target site
        # Store the site and its expectation value in the dictionary
        site_exp_dict[ms] = site_exp_value

    return site_exp_dict

def measure_one_site_spin(A, ms, env, op=None):
    r"""
    Returns the overlap of bra and ket on a single site.

    Parameters
    ----------
    A : single peps tensor at site ms
    ms : site where we want to measure some observable
    env: class CtmEnv
        class containing ctm environmental tensors along with lattice structure data
    op: single site operator
    """

    if op is not None:
        AAb = fPEPS_2layers(A, op=op, dir='1s')
    elif op is None:
        AAb = fPEPS_2layers(A)
    vecl = yastn.tensordot(env[ms].l, env[ms].tl, axes=(2, 0))
    vecl = yastn.tensordot(env[ms].bl, vecl, axes=(1, 0))
    new_vecl = apply_TMO_left(vecl, env, ms, AAb)
    vecr = yastn.tensordot(env[ms].tr, env[ms].r, axes=(1, 0))
    vecr = yastn.tensordot(vecr, env[ms].br, axes=(2, 0))
    hor = con_bi(new_vecl, vecr)
    return hor


def EV2ptcorr(peps, env, op, site0, site1):

    r"""
    Returns two-point correlators given any two sites.

    Parameters
    ----------
    peps : class Lattice
    env: class CtmEnv
    op: observable whose two-point correlators need to be calculated
    site0, site1: sites where the two-point correlator is to be calculated

    ### note for now it has only arbitary axial correlator and nearest diagonal coorelator
    ### to be done arbitrary coorelator
    """

    x0, y0 = site0
    x1, y1 = site1

    if (x0-x1) == 0 or (y0-y1) == 0:   # check if axial
        exp_corr = EV2ptcorr_axial(peps, env, op, site0, site1)
    elif abs(x0-x1) == 1 and abs(y0-y1) == 1:   # check if diagonal
        exp_corr = EV2ptcorr_diagonal(peps, env, op, site0, site1)

    return exp_corr


def EV2ptcorr_axial(peps, env, op, site0, site1):

    r"""
    Returns two-point correlators along axial direction (horizontal or vertical) given any two sites and observables
    to be evaluated on those sites. Directed from EV2ptcorr when sites lie in the axial (horizontal or vertical) direction
    """

    peps = check_consistency_tensors(peps) # to check if A has the desired fused form of legs i.e. t l b r [s a]
    norm_array = array_EV2pt(peps, env, site0, site1)
    op_array = array_EV2pt(peps, env, site0, site1, op)
    array_corr = op_array / norm_array

    return array_corr


def EV2ptcorr_diagonal(peps, env, ops, site0, site1):
    r"""
    Returns two-point correlators along diagonal direction given any two sites and observables
    to be evaluated on those sites. Directed from EV2ptcorr when sites lie diagonally.

    Note: site0 has to be to at left and site1 at right according to the defined fermionic order
    """

    peps = check_consistency_tensors(peps) # to check if A has the desired fused form of legs i.e. t l b r [s a]
    x0, y0 = site0
    x1, y1 = site1

    if x0<x1:
        ptl,ptr, pbr, pbl = site0, peps.nn_site(site0, d='r'), site1, peps.nn_site(site1, d='l')
    elif x1<x0:
        ptr, ptl, pbl, pbr = site1, peps.nn_site(site1, d='l'), site0, peps.nn_site(site0, d='r')

    AAb_top = {'l': fPEPS_2layers(peps[ptl]), 'r': fPEPS_2layers(peps[ptr])}     # top layer of double peps tensors without operator
    AAb_bottom = {'l': fPEPS_2layers(peps[pbl]), 'r': fPEPS_2layers(peps[pbr])}  # bottom layer of double peps tensors without operator
    normd = array2ptdiag(peps, env, AAb_top, AAb_bottom, site0, site1)

    # print('#####################')
    # print('site left', site0)
    # print('site right', site1)
    # print('norm ',normd)

    if x0<x1:
        AAbop_top = {'l': fPEPS_2layers(peps[ptl], op=ops['l'], dir='l'), 'r': fPEPS_2layers(peps[ptr])} # top layer of double peps tensors with operator
        AAbop_bottom = {'l': fPEPS_2layers(peps[pbl]), 'r': fPEPS_2layers(peps[pbr], op=ops['r'], dir='r')} # bottom layer of double peps tensors with operator
    elif x1<x0:
        AAbop_top = {'l': fPEPS_2layers(peps[ptl]), 'r': fPEPS_2layers(peps[ptr], op=ops['r'], dir='r')} # top layer of dounle peps tensors with operator
        AAbop_bottom = {'l': fPEPS_2layers(peps[pbl], op=ops['l'], dir='l'), 'r': fPEPS_2layers(peps[pbr])} # bottom layer of double peps tensors with operator

    expdg = array2ptdiag(peps, env, AAbop_top, AAbop_bottom, site0, site1, flag='y')
    exp_diag = expdg/normd

    print('normalized expectation value of diagonal correlator', exp_diag)
    print('#####################')

    return exp_diag