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

def evolution_step_(gamma, gates, Ds, step, truncation_mode, env_type):  # perform a single step of evolution 
    """ 
    Apply a list of gates on peps; performing truncation; 
    it is a 2nd-order step in a sense that gates that gates contain half of the sweep,
    and the orher half is applied in the reverse order
    """
    infos = []

    for gate in gates.local:
        gamma = apply_local_gate_(gamma, gate)

    for gate in gates.nn + gates.nn[::-1]:
        gamma, info = ntu_machine(gamma, gate, Ds, truncation_mode, step, env_type)

    for gate in gates.local[::-1]:
        gamma = apply_local_gate_(gamma, gate)

    if step=='svd-update':
        return gamma, info 
    else: 
        info['ntu_error'] = [record['ntu_error'] for record in infos]
        info['optimal_cutoff'] = [record['optimal_cutoff'] for record in infos]
        info['svd_error'] = [record['svd_error'] for record in infos]
        return gamma, info




def gates_homogeneous(gamma, nn_gates, loc_gates):
    # len(nn_gates) indicates the physical degrees of freedom; option to add more
    bonds = gamma.bonds(dirn='h') + gamma.bonds(dirn='v')

    gates_nn = []
    if len(nn_gates) == 2:
        for bd in bonds:
            gates_nn.append(Gate_nn(A=nn_gates['GA'], B=nn_gates['GB'], bond=bd))
    elif len(nn_gates) == 4:
        for bd in bonds:
            gates_nn.append(Gate_nn(A=nn_gates['GA_up'], B=nn_gates['GB_up'], bond=bd))
            gates_nn.append(Gate_nn(A=nn_gates['GA_dn'], B=nn_gates['GB_dn'], bond=bd))
    gates_loc = []
    for site in gamma.sites():
        gates_loc.append(Gate_local(A=loc_gates, site=site))
    return Gates(local=gates_loc, nn=gates_nn)


def show_leg_structure(gamma):
   for ms in gamma.sites():
        xs = gamma[ms].unfuse_legs((0, 1))
        print("site ", str(ms), xs.get_shape()) 
  
