""" Function performing NTU update on all four unique bonds corresponding to a two site unit cell. """
from .NtuEssentials import single_bond_local_update, ntu_machine
from typing import NamedTuple

class Gate(NamedTuple):
    """ site_0 should be before site_1 in the fermionic order. """
    A : tuple = None
    B : tuple = None

def ntu_update(Gamma, Gates, Ds, step, truncation_mode, fix_bd, flag=None):
    
    infos = []
   # local gate
    if Gates['loc'] is not None:
        Gamma = single_bond_local_update(Gamma, Gates['loc'], flag)
   
    for iter in GB_list(Gamma, Gates['nn']):
        bd = iter['bond'] 
        Gamma, info = ntu_machine(Gamma, bd, Gate(iter['gateA'], iter['gateB']), Ds, truncation_mode, step, fix_bd, flag)
        # show_leg_structure(Gamma)
        infos.append(info)

    # local gate
    if Gates['loc'] is not None:
        Gamma = single_bond_local_update(Gamma, Gates['loc'], flag)

    if step=='svd-update':
        return Gamma, info 
    else: 
        info['ntu_error'] = [record['ntu_error'] for record in infos]
        info['optimal_cutoff'] = [record['optimal_cutoff'] for record in infos]
        info['svd_error'] = [record['svd_error'] for record in infos]
        return Gamma, info


def GB_list(Gamma, nn_gates):
    # len(nn_gates) indicates the physical degrees of freedom; option to add more
    list_tuple = Gamma.bonds(dirn='h') + Gamma.bonds(dirn='v') + Gamma.bonds(dirn='v', reverse=True) + Gamma.bonds(dirn='h', reverse=True)

    coll = []
    if len(nn_gates) == 2:
        for m in list_tuple:
            coll.append({'gateA':nn_gates['GA'], 'gateB':nn_gates['GB'], 'bond':m})
    elif len(nn_gates) == 4:
        for m in list_tuple:
            coll.append({'gateA':nn_gates['GA_up'], 'gateB':nn_gates['GB_up'], 'bond':m})
            coll.append({'gateA':nn_gates['GA_dn'], 'gateB':nn_gates['GB_dn'], 'bond':m})

    return coll


def show_leg_structure(Gamma):
   for ms in Gamma.sites():
        xs = Gamma[ms].unfuse_legs((0, 1))
        print("site ", str(ms), xs.get_shape()) 
  
