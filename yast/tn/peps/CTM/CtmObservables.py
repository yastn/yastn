from .CtmIterationRoutines import check_consistency_tensors
from .CtmIterationRoutines import fPEPS_2layers
from .CtmObservableRoutines import ret_AAbs, hor_extension, ver_extension
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
            AAb = {'l': fPEPS_2layers(Gamma[bds_h.site_0]), 'r': fPEPS_2layers(Gamma[bds_h.site_1])}
            val_hor = hor_extension(env, bds_h, AAbo, AAb)
            res_hor.append(val_hor)

        for bds_v in net.bonds(dirn='v'): # correlators on all vertical bonds
            AAbo = ret_AAbs(Gamma, bds_v, opt, orient='v')
            AAb = {'l': fPEPS_2layers(Gamma[bds_v.site_0]), 'r': fPEPS_2layers(Gamma[bds_v.site_1])}
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
    AAb = {'l': fPEPS_2layers(Gamma[bd.site_0]), 'r': fPEPS_2layers(Gamma[bd.site_1])}
    val = hor_extension(env, bd, AAbo, AAb) if bd.dirn == 'h' else ver_extension(env, bd, AAbo, AAb)
    return val