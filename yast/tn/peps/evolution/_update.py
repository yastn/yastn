""" Function performing NTU update on all four unique bonds corresponding to a two site unit cell. """
from ._routines import apply_local_gate_, ntu_machine
from typing import NamedTuple

class Gate_nn(NamedTuple):
    """ site_0 should be before site_1 in the fermionic order. """
    A : tuple = None
    B : tuple = None
    bond : tuple = None

class Gate_local(NamedTuple):
    """ site_0 should be before site_1 in the fermionic order. """
    A : tuple = None
    site : tuple = None

class Gates(NamedTuple):
    local : list = None   # list of Gate_local
    nn : list = None   # list of Gate_nn


# To be written
# def evolve_(gamma, Gates, .... )   # higher level routine; do many steps of the evolution
# here Gates can be a function generating gates based on something 
#    yield state

def evolution_step_(psi, gates, step, truncation_mode, env_type, opts_svd=None):  # perform a single step of evolution 
    """ 
    Apply a list of gates on peps; performing truncation; 
    it is a 2nd-order step in a sense that gates that gates contain half of the sweep,
    and the other half is applied in the reverse order
    """
    infos = []

    for gate in gates.local:
        psi = apply_local_gate_(psi, gate)

    for gate in gates.nn + gates.nn[::-1]:
        psi, info = ntu_machine(psi, gate, truncation_mode, step, env_type, opts_svd)

    for gate in gates.local[::-1]:
        psi = apply_local_gate_(psi, gate)

    if step=='svd-update':
        return psi, info 
    else: 
        info['ntu_error'] = [record['ntu_error'] for record in infos]
        info['optimal_cutoff'] = [record['optimal_cutoff'] for record in infos]
        info['svd_error'] = [record['svd_error'] for record in infos]
        return psi, info


"""def gates_homogeneous(psi, nn_gates, loc_gates):
    # len(nn_gates) indicates the physical degrees of freedom; option to add more
    bonds = psi.bonds(dirn='h') + psi.bonds(dirn='v')

    gates_nn = []   # nn_gates = [(GA, GB), (GA, GB)]   [(GA, GB, GA, GB)] 
    if len(nn_gates) == 2:
        for bd in bonds:
            gates_nn.append(Gate_nn(A=nn_gates['GA'], B=nn_gates['GB'], bond=bd))
    elif len(nn_gates) == 4:
        for bd in bonds:
            gates_nn.append(Gate_nn(A=nn_gates['GA_up'], B=nn_gates['GB_up'], bond=bd))
            gates_nn.append(Gate_nn(A=nn_gates['GA_dn'], B=nn_gates['GB_dn'], bond=bd))
    gates_loc = []
    for site in psi.sites():
        gates_loc.append(Gate_local(A=loc_gates, site=site))
    return Gates(local=gates_loc, nn=gates_nn)"""

def gates_homogeneous(psi, nn_gates, loc_gates):
    # len(nn_gates) indicates the physical degrees of freedom; option to add more
    bonds = psi.bonds(dirn='h') + psi.bonds(dirn='v')
    gates_nn = []   # nn_gates = [(GA, GB), (GA, GB)]   [(GA, GB, GA, GB)] 
    for bd in bonds:
        for i in range(len(nn_gates)):
            gates_nn.append(Gate_nn(A=nn_gates[i][0], B=nn_gates[i][1], bond=bd))
    gates_loc = []
    for site in psi.sites():
        gates_loc.append(Gate_local(A=loc_gates, site=site))
    return Gates(local=gates_loc, nn=gates_nn)


def show_leg_structure(psi):
   for ms in psi.sites():
        xs = psi[ms].unfuse_legs((0, 1))
        print("site ", str(ms), xs.get_shape()) 
  
