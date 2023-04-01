from ._ctm_iteration_routines import check_consistency_tensors
from ._ctm_iteration_routines import fPEPS_2layers
from ._ctm_observable_routines import ret_AAbs, hor_extension, ver_extension, apply_TMO_left, con_bi
import yast
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
    obs_hor = {}
    obs_ver = {}
    for ms in op.keys():
        res_hor = []
        res_ver = []
        opt = op.get(ms)
        
        for bds_h in peps.bonds(dirn='h'):  # correlators on all horizontal bonds
            AAb = {'l': fPEPS_2layers(peps[bds_h.site_0]), 'r': fPEPS_2layers(peps[bds_h.site_1])}
            AAbo = ret_AAbs(peps, bds_h, opt, orient='h')
            val_hor = hor_extension(env, bds_h, AAbo, AAb)
            res_hor.append(val_hor)
        for bds_v in peps.bonds(dirn='v'): # correlators on all vertical bonds
            AAb = {'l': fPEPS_2layers(peps[bds_v.site_0]), 'r': fPEPS_2layers(peps[bds_v.site_1])}
            AAbo = ret_AAbs(peps, bds_v, opt, orient='v')
            val_ver = ver_extension(env, bds_v, AAbo, AAb)
            res_ver.append(val_ver)
        
        dic_hor = {ms: np.mean(res_hor)}
        dic_ver = {ms: np.mean(res_ver)}
        obs_hor.update(dic_hor)
        obs_ver.update(dic_ver)

    return obs_hor, obs_ver


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

    AAbo = ret_AAbs(peps, bd, op, orient=bd.dirn)
    AAb = {'l': fPEPS_2layers(peps[bd.site_0]), 'r': fPEPS_2layers(peps[bd.site_1])}
    val = hor_extension(env, bd, AAbo, AAb) if bd.dirn == 'h' else ver_extension(env, bd, AAbo, AAb)
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
    vecl = yast.tensordot(env[ms].l, env[ms].tl, axes=(2, 0))
    vecl = yast.tensordot(env[ms].bl, vecl, axes=(1, 0))
    new_vecl = apply_TMO_left(vecl, env, ms, AAb)
    vecr = yast.tensordot(env[ms].tr, env[ms].r, axes=(1, 0)) 
    vecr = yast.tensordot(vecr, env[ms].br, axes=(2, 0))
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

    target_site = (round((peps.Nx-1)*0.5), round((peps.Ny-1)*0.5))
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
       
