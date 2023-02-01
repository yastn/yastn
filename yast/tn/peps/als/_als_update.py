""" Function performing NTU update on all four unique bonds corresponding to a two site unit cell. """
from ._routines import single_bond_local_update, ntu_machine
from typing import NamedTuple

class Gate(NamedTuple):
    """ site_0 should be before site_1 in the fermionic order. """
    A : tuple = None
    B : tuple = None

def _als_update(gamma, Gates, Ds, step, truncation_mode, env_type):
    
    infos = []
   # local gate
    if Gates['loc'] is not None:
        gamma = single_bond_local_update(gamma, Gates['loc'])
   
    for iter in GB_list(gamma, Gates['nn']):
        bd = iter['bond'] 
        gamma, info = ntu_machine(gamma, bd, Gate(iter['gateA'], iter['gateB']), Ds, truncation_mode, step, env_type)
        # show_leg_structure(gamma)
        infos.append(info)

    # local gate
    if Gates['loc'] is not None:
        gamma = single_bond_local_update(gamma, Gates['loc'])

    if step=='svd-update':
        return gamma, info 
    else: 
        info['ntu_error'] = [record['ntu_error'] for record in infos]
        info['optimal_cutoff'] = [record['optimal_cutoff'] for record in infos]
        info['svd_error'] = [record['svd_error'] for record in infos]
        return gamma, info


def GB_list(gamma, nn_gates):
    # len(nn_gates) indicates the physical degrees of freedom; option to add more
    list_tuple = gamma.bonds(dirn='h') + gamma.bonds(dirn='v') + gamma.bonds(dirn='v', reverse=True) + gamma.bonds(dirn='h', reverse=True)

    coll = []
    if len(nn_gates) == 2:
        for m in list_tuple:
            coll.append({'gateA':nn_gates['GA'], 'gateB':nn_gates['GB'], 'bond':m})
    elif len(nn_gates) == 4:
        for m in list_tuple:
            coll.append({'gateA':nn_gates['GA_up'], 'gateB':nn_gates['GB_up'], 'bond':m})
            coll.append({'gateA':nn_gates['GA_dn'], 'gateB':nn_gates['GB_dn'], 'bond':m})

    return coll


def show_leg_structure(gamma):
   for ms in gamma.sites():
        xs = gamma[ms].unfuse_legs((0, 1))
        print("site ", str(ms), xs.get_shape()) 
  
