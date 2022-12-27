from .CtmIterationRoutines import check_consistency_tensors
from .CtmIterationRoutines import fPEPS_2layers
from .CtmObservableRoutines import ret_AAbs, hor_extension, ver_extension, apply_TMO_left, con_bi, diagonal_correlation
import yast
import numpy as np

def nn_avg(Gamma, env, op):
    # main function which calculates the nearest neighbor correlators on all bonds and averages them
    Gamma = check_consistency_tensors(Gamma)
    obs_hor = {}
    obs_ver = {}
    for ms in op.keys():
        res_hor = []
        res_ver = []
        opt = op.get(ms)
        for bds_h in Gamma.bonds(dirn='h'):  # correlators on all horizontal bonds
            AAbo = ret_AAbs(Gamma, bds_h, opt, orient='h')
            AAb = {'l': fPEPS_2layers(Gamma[bds_h.site_0]), 'r': fPEPS_2layers(Gamma[bds_h.site_1])}
            val_hor = hor_extension(env, bds_h, AAbo, AAb)
            res_hor.append(val_hor)
        for bds_v in Gamma.bonds(dirn='v'): # correlators on all vertical bonds
            AAbo = ret_AAbs(Gamma, bds_v, opt, orient='v')
            AAb = {'l': fPEPS_2layers(Gamma[bds_v.site_0]), 'r': fPEPS_2layers(Gamma[bds_v.site_1])}
            val_ver = ver_extension(env, bds_v, AAbo, AAb)
            res_ver.append(val_ver)
        
        dic_hor = {ms: np.mean(res_hor)}
        dic_ver = {ms: np.mean(res_ver)}
        obs_hor.update(dic_hor)
        obs_ver.update(dic_ver)

    return obs_hor, obs_ver


def nn_bond(Gamma, env, op, bd):
    # calculation of correlator on a desired bond in the lattice
    # useful when you don't need the average value

    Gamma = check_consistency_tensors(Gamma)
    AAbo = ret_AAbs(Gamma, bd, op, orient=bd.dirn)
    AAb = {'l': fPEPS_2layers(Gamma[bd.site_0]), 'r': fPEPS_2layers(Gamma[bd.site_1])}
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


def one_site_avg(Gamma, env, op, flag=None):

    mat = np.zeros((Gamma.Nx, Gamma.Ny))

    target_site = (round((Gamma.Nx-1)*0.5), round((Gamma.Ny-1)*0.5))
    Gamma = check_consistency_tensors(Gamma)
    one_site_exp = np.zeros((Gamma.Nx*Gamma.Ny))
    s = 0
    for ms in Gamma.sites():
        print('site: ', ms)
        Am = Gamma[ms]

        if flag == 'hole':
            if ms == target_site:
                Am = Am.unfuse_legs(axes=4).unfuse_legs(axes=5).fuse_legs(axes=(0, 1, 2, 3, 6, (4, 5))) # t l b r str [s a]
                val_op = measure_one_site_spin(Am, ms, env, op=op)
                val_norm = measure_one_site_spin(Am, ms, env, op=None)
                one_site_exp[s] = val_op/val_norm
                print(one_site_exp[s])
                cs_val = one_site_exp[s]
                mat[ms[0],ms[1]] = one_site_exp[s]

                s = s+1
            else:
                val_op = measure_one_site_spin(Am, ms, env, op=op)
                val_norm = measure_one_site_spin(Am, ms, env, op=None)
                one_site_exp[s] = val_op/val_norm
                print(one_site_exp[s])
                mat[ms[0],ms[1]] = one_site_exp[s]
                s = s+1
            return np.mean(one_site_exp), cs_val, mat

        else:
        
            val_op = measure_one_site_spin(Am, ms, env, op=op)
            val_norm = measure_one_site_spin(Am, ms, env, op=None)
            one_site_exp[s] = val_op/val_norm
            print(one_site_exp[s])
            mat[ms[0], ms[1]] = one_site_exp[s]
            if ms == target_site:
                cs_val = one_site_exp[s]
            s = s+1

    return np.mean(one_site_exp), cs_val, mat
       

def EVcorr_diagonal(env, ops, Gamma, x_l, y_l, x_r, y_r):

    Gamma = check_consistency_tensors(Gamma) # to check if A has the desired fused form of legs i.e. t l b r [s a]

    if x_l > x_r:
        orient = 'ne'
    elif x_l < x_r:
        orient = 'se'

    EV_diag = {}

    if orient == 'ne':

        AAb_top = {'l': fPEPS_2layers(Gamma[(x_r, y_r-1)]), 'r': fPEPS_2layers(Gamma[(x_r, y_r)])}     # top layer of A tensors without operator
        AAb_bottom = {'l': fPEPS_2layers(Gamma[(x_l, y_l)]), 'r': fPEPS_2layers(Gamma[(x_l, y_l+1)])}  # bottom layer of A tensors without operator
        norm = diagonal_correlation(env, x_l, y_l, x_r, y_r, AAb_top, AAb_bottom, AAb_top, AAb_bottom, orient='ne') 
        print(norm)

        for k, op in ops.items():
            print(k)
            AAbop_top = {'l': fPEPS_2layers(Gamma[(x_r, y_r-1)], op=op['l'], dir='l'), 'r': fPEPS_2layers(Gamma[(x_r, y_r)], op=op['r'], dir='r')} # top layer of A tensors with operator
            AAbop_bottom = {'l': fPEPS_2layers(Gamma[(x_l, y_l)], op=op['l'], dir='l'), 'r': fPEPS_2layers(Gamma[(x_l, y_l+1)], op=op['r'], dir='r')} # bottom layer of A tensors with operator
            val = diagonal_correlation(env, x_l, y_l, x_r, y_r, AAb_top, AAb_bottom, AAbop_top, AAbop_bottom, orient='ne') # diagonal correlation with
            corr = val/norm
            EV_diag[k] = corr
    
    elif orient == 'se':
 
        AAb_top = {'l': fPEPS_2layers(Gamma[(x_l, y_l)]), 'r': fPEPS_2layers(Gamma[(x_l, y_l+1)])}     # top layer of A tensors without operator
        AAb_bottom = {'l': fPEPS_2layers(Gamma[(x_r, y_r-1)]), 'r': fPEPS_2layers(Gamma[(x_r, y_r)])}  # bottom layer of A tensors without operator
        norm = diagonal_correlation(env, x_l, y_l, x_r, y_r, AAb_top, AAb_bottom, AAb_top, AAb_bottom, orient='se') 
        print(norm)

        for k, op in ops.items():
            print(k)
            AAbop_top = {'l': fPEPS_2layers(Gamma[(x_l, y_l)], op=op['l'], dir='l'), 'r': fPEPS_2layers(Gamma[(x_l, y_l+1)], op=op['r'], dir='r')} # top layer of A tensors with operator
            AAbop_bottom = {'l': fPEPS_2layers(Gamma[(x_r, y_r-1)], op=op['l'], dir='l'), 'r': fPEPS_2layers(Gamma[(x_r, y_r)], op=op['r'], dir='r')} # bottom layer of A tensors with operator
            val = diagonal_correlation(env, x_l, y_l, x_r, y_r, AAb_top, AAb_bottom, AAbop_top, AAbop_bottom, orient='se') # diagonal correlation with
            corr = val/norm
            EV_diag[k] = corr

        return EV_diag