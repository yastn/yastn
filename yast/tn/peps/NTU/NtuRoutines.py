""" Function performing NTU update on all four unique bonds corresponding to a two site unit cell. """
from .NtuEssentials import single_bond_local_update, ntu_machine

def ntu_update(Gamma, net, fid, Gates, Ds, step, truncation_mode, fix_bd):
    
    infos = []
   # local gate
    if Gates['loc'] is not None:
        Gamma = single_bond_local_update(Gamma, net, Gates['loc'])
   
    for iter in GB_list(net, Gates['nn']):
        GA, GB, bd = iter['gateA'], iter['gateB'], iter['bond'] 
        Gamma, info = ntu_machine(Gamma, net, fid, bd, GA, GB, Ds, truncation_mode, step, fix_bd)
        # show_leg_structure(net, Gamma)
        infos.append(info)

    # local gate
    if Gates['loc'] is not None:
        Gamma = single_bond_local_update(Gamma, net, Gates['loc'])

    if step=='svd-update':
        return Gamma, info 
    else: 
        info['ntu_error'] = [record['ntu_error'] for record in infos]
        info['optimal_cutoff'] = [record['optimal_cutoff'] for record in infos]
        info['svd_error'] = [record['svd_error'] for record in infos]
        return Gamma, info


def GB_list(net, nn_gates):
    # len(nn_gates) indicates the physical degrees of freedom; option to add more
    list_tuple = net.bonds(dirn='h') + net.bonds(dirn='v') + net.bonds(dirn='v', reverse=True) + net.bonds(dirn='h', reverse=True)

    coll = []
    if len(nn_gates) == 2:
        for m in list_tuple:
            coll.append({'gateA':nn_gates['GA'], 'gateB':nn_gates['GB'], 'bond':m})
    elif len(nn_gates) == 4:
        for m in list_tuple:
            coll.append({'gateA':nn_gates['GA_up'], 'gateB':nn_gates['GB_up'], 'bond':m})
            coll.append({'gateA':nn_gates['GA_dn'], 'gateB':nn_gates['GB_dn'], 'bond':m})

    return coll


def show_leg_structure(net, Gamma):
   for ms in net.sites():
        xs = Gamma._data[ms].unfuse_legs((0, 1))
        print("site ", str(ms), xs.get_shape()) 
  
