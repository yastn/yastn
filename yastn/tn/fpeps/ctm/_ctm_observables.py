from ._ctm_iteration_routines import check_consistency_tensors
from ._ctm_iteration_routines import fPEPS_2layers
from ._ctm_observable_routines import apply_TMO_left, con_bi, array_EV2pt, array2ptdiag
import yastn
import numpy as np

def nn_avg(peps, env, op):

    r"""
    Calculate two-site nearest-neighbor calculation of observables with CTM environments.

    Parameters
    ----------
    peps : class Peps
           class containing peps data along with the lattice structure data

    env: class CtmEnv
        class containing ctm environmental tensors along with lattice structure data

    op: dict
        dictionary containing observables placed on NN sites

    Returns
    -------
    obs_hor: dict
        dictionary containing name of the horizontal observables as keys and their averaged values over all horizontal bonds.
    obs_ver: dict
        dictionary containing name of the vertical observables as keys and their averaged values over all vertical bonds.
    """

    peps = check_consistency_tensors(peps)
    obs_hor_mean = {}
    obs_ver_mean = {}
    obs_hor_sum = {}
    obs_ver_sum = {}
    for ms in op.keys():
        res_hor = []
        res_ver = []
        opt = op.get(ms)
        for bds_h in peps.bonds(dirn='h'):  # correlators on all horizontal bonds
            val_hor = EV2ptcorr(peps, env, opt, bds_h.site_0, bds_h.site_1)
            res_hor.append(val_hor[0])
        for bds_v in peps.bonds(dirn='v'): # correlators on all vertical bonds
            val_ver = EV2ptcorr(peps, env, opt, bds_v.site_0, bds_v.site_1)
            res_ver.append(val_ver[0])
        
        dic_hor_mean = {ms: np.mean(res_hor)}
        dic_ver_mean = {ms: np.mean(res_ver)}
        dic_hor_sum = {ms: np.sum(res_hor)}
        dic_ver_sum = {ms: np.sum(res_ver)}
        obs_hor_mean.update(dic_hor_mean)
        obs_ver_mean.update(dic_ver_mean)
        obs_hor_sum.update(dic_hor_sum)
        obs_ver_sum.update(dic_ver_sum)

    return obs_hor_mean, obs_ver_mean, obs_hor_sum, obs_ver_sum



def nn_bond(peps, env, op, bd):

    r"""
    Returns expecation value for specified nearest-neighbor bond.

    Parameters
    ----------
    peps : class Peps
           class containing peps data along with the lattice structure data

    env  : class CtmEnv
         Class containing ctm environmental tensors along with lattice structure data

    op   : dict
         Contains NN pair of obsevables with dictionary key 'l' corresponding to
         the observable on the left and key 'r' corresponding to
         the observable on the right

    bd: NamedTuple
        contains info about NN sites where the oexpectation value is to be calculated
 
    """

    peps = check_consistency_tensors(peps) 
    val = EV2ptcorr(peps, env, op, bd.site_0, bd.site_1)
    return val

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


def one_site_avg(peps, env, op):
    r"""
    Measures the expectation value of single site operators all over the lattice.

    Parameters
    ----------
    peps : class Peps
        class containing peps data along with the lattice structure data

    env: class CtmEnv
        class containing ctm environment tensors along with lattice structure data

    op: single site operator

    Returns
    -------
    mean_one_site: expectation value of one site observables averaged over all the lattice sites
    
    mat: expectation value of all the sites in a 2D table form
    """

    mat = np.zeros((peps.Nx, peps.Ny))  # output returns the expectation value on every site

    peps = check_consistency_tensors(peps)
    s = 0

    if peps.lattice == 'checkerboard':
        lists = [(0,0), (0,1)]
    else:
        lists = peps.sites()
    
    one_site_exp = np.zeros(len(lists))

    for ms in lists:
        Am = peps[ms]
        val_op = measure_one_site_spin(Am, ms, env, op=op)
        val_norm = measure_one_site_spin(Am, ms, env, op=None)
        one_site_exp[s] = val_op/val_norm   # expectation value of particular target site
        mat[ms[0], ms[1]] = one_site_exp[s]
        s = s+1
    mean_one_site = np.mean(one_site_exp)

    return mean_one_site, mat


def EV2ptcorr(peps, env, op, site0, site1):

    r"""
    Returns two-point correlators given any two sites.

    Parameters
    ----------
    peps : class Peps
        class containing peps data along with the lattice structure data

    env: class CtmEnv
        class containing ctm environment tensors along with lattice structure data
    
    op: observable whose two-point correlators need to be calculated

    site0, site1: sites where the two-point correlator is to be calculated
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


def EV2ptcorr_diagonal(peps, env, op, site0, site1):
    
    r"""
    Returns two-point correlators along diagonal direction given any two sites and observables
    to be evaluated on those sites. Directed from EV2ptcorr when sites lie diagonally.
    """

    peps = check_consistency_tensors(peps) # to check if A has the desired fused form of legs i.e. t l b r [s a]
    op_array = diagonal(peps, env, op, site0, site1)   

    return op_array


def diagonal(peps, env, ops, site0, site1):
    
    # site0 has to be to at left and site1 at right according to the defined fermionic order
    # decide if site0 is at top or bottom
    x0, y0 = site0
    x1, y1 = site1

    if x0<x1:
        ptl,ptr, pbr, pbl = site0, peps.nn_site(site0, d='r'), site1, peps.nn_site(site1, d='l')
    elif x1<x0:
        ptr, ptl, pbl, pbr = site1, peps.nn_site(site1, d='l'), site0, peps.nn_site(site0, d='r') 

    AAb_top = {'l': fPEPS_2layers(peps[ptl]), 'r': fPEPS_2layers(peps[ptr])}     # top layer of double peps tensors without operator
    AAb_bottom = {'l': fPEPS_2layers(peps[pbl]), 'r': fPEPS_2layers(peps[pbr])}  # bottom layer of double peps tensors without operator
    normd = array2ptdiag(peps, env, AAb_top, AAb_bottom, site0, site1)

    print('#####################')
    print('site left', site0)
    print('site right', site1)
    print('norm ',normd)
        
    if x0<x1:  
        AAbop_top = {'l': fPEPS_2layers(peps[ptl], op=ops['l'], dir='l'), 'r': fPEPS_2layers(peps[ptr])} # top layer of double peps tensors with operator
        AAbop_bottom = {'l': fPEPS_2layers(peps[pbl]), 'r': fPEPS_2layers(peps[pbr], op=ops['r'], dir='r')} # bottom layer of double peps tensors with operator
    elif x1<x0:
        AAbop_top = {'l': fPEPS_2layers(peps[ptl]), 'r': fPEPS_2layers(peps[ptr], op=ops['r'], dir='r')} # top layer of dounle peps tensors with operator
        AAbop_bottom = {'l': fPEPS_2layers(peps[pbl], op=ops['l'], dir='l'), 'r': fPEPS_2layers(peps[pbr])} # bottom layer of double peps tensors with operator

    expdg = array2ptdiag(peps, env, AAbop_top, AAbop_bottom, site0, site1, flag_str='ws') 
    exp_diag = expdg/normd 
        
    
    print('expectation value of diagonal correlator', exp_diag)
    print('#####################')

    return exp_diag