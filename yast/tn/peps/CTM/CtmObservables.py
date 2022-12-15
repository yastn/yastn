from .CtmIterationRoutines import check_consistency_tensors
from .CtmIterationRoutines import fPEPS_2layers, fPEPS_only_spin
from .CtmObservableRoutines import ret_AAbs, hor_extension, ver_extension, apply_TMO_left, con_bi
import yast
import numpy as np

def nn_avg(Gamma, net, env, op):
    # main function which calculates the nearest neighbor correlators on all bonds and averages them
    Gamma = check_consistency_tensors(Gamma, net)
    obs_hor = {}
    obs_ver = {}
    for ms in op.keys():
        res_hor = []
        res_ver = []
        opt = op.get(ms)
        for bds_h in net.bonds(dirn='h'):  # correlators on all horizontal bonds
            AAbo = ret_AAbs(Gamma, bds_h, opt, orient='h')
            AAb = {'l': fPEPS_2layers(Gamma._data[bds_h.site_0]), 'r': fPEPS_2layers(Gamma._data[bds_h.site_1])}
            val_hor = hor_extension(env, bds_h, AAbo, AAb)
            res_hor.append(val_hor)
        for bds_v in net.bonds(dirn='v'): # correlators on all vertical bonds
            AAbo = ret_AAbs(Gamma, bds_v, opt, orient='v')
            AAb = {'l': fPEPS_2layers(Gamma._data[bds_v.site_0]), 'r': fPEPS_2layers(Gamma._data[bds_v.site_1])}
            val_ver = ver_extension(env, bds_v, AAbo, AAb)
            res_ver.append(val_ver)
        
        dic_hor = {ms: np.mean(res_hor)}
        dic_ver = {ms: np.mean(res_ver)}
        obs_hor.update(dic_hor)
        obs_ver.update(dic_ver)

    return obs_hor, obs_ver


def nn_bond(Gamma, net, env, op, bd):
    # calculation of correlator on a desired bond in the lattice
    # useful when you don't need the average value

    Gamma = check_consistency_tensors(Gamma, net)
    AAbo = ret_AAbs(Gamma, bd, op, orient=bd.dirn)
    AAb = {'l': fPEPS_2layers(Gamma._data[bd.site_0]), 'r': fPEPS_2layers(Gamma._data[bd.site_1])}
    val = hor_extension(env, bd, AAbo, AAb) if bd.dirn == 'h' else ver_extension(env, bd, AAbo, AAb)
    return val

def measure_one_site_spin(A, ms, env, op=None):

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

def measure_one_site_only_spin(A, ms, env, op=None):

    if op is not None:
        AAb = fPEPS_only_spin(A, op=op)
    elif op is None:
        AAb = fPEPS_only_spin(A)
    vecl = yast.tensordot(env[ms].l, env[ms].tl, axes=(2, 0))
    vecl = yast.tensordot(env[ms].bl, vecl, axes=(1, 0))
    new_vecl = apply_TMO_left(vecl, env, ms, AAb)
    vecr = yast.tensordot(env[ms].tr, env[ms].r, axes=(1, 0)) 
    vecr = yast.tensordot(vecr, env[ms].br, axes=(2, 0))
    hor = con_bi(new_vecl, vecr)
    return hor

def one_site_avg(Gamma, net, env, op):

    Gamma = check_consistency_tensors(Gamma, net)
    one_site_exp = np.zeros((net.Nx*net.Ny))
    s = 0
    for ms in net.sites():
        print('site: ', ms)
        val_op = measure_one_site_spin(Gamma[ms], ms, env, op=op)
        val_norm = measure_one_site_spin(Gamma[ms], ms, env, op=None)
        one_site_exp[s] = val_op/val_norm
        print(one_site_exp[s])
        s = s+1
    
    return np.mean(one_site_exp)