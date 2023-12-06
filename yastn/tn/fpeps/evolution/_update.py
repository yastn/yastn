""" Function performing NTU update on all four unique bonds corresponding to a two site unit cell. """
from ._routines import apply_local_gate_, evol_machine
from typing import NamedTuple

class Gate_nn(NamedTuple):
    """ A should be before B in the fermionic order. """
    A : tuple = None
    B : tuple = None
    bond : tuple = None

class Gate_local(NamedTuple):
    A : tuple = None
    site : tuple = None

class Gates(NamedTuple):
    local : list = None   # list of Gate_local
    nn : list = None   # list of Gate_nn


def evolution_step_(env, gates, opts_evol, opts_svd=None):
    r"""
    Perform a single step of evolution on a PEPS by applying a list of gates,
    performing truncation and subsequent optimization.

    Parameters
    ----------

        peps: class Lattice
        gates: The gates to apply during the evolution. The `Gates` named tuple
                should contain a list of NamedTuples `Gate_local` and `Gate_nn`.
        step: str (Specifies the type of evolution step to perform. Can be either 'ntu-update' or 'svd-update'.
        truncation_mode: str (Specifies the truncation mode to use during the evolution.)
        env_type: str (Specifies the type of environment to use during the evolution.)
        opts_svd (dict, optional): Dictionary containing options for the SVD routine.

    Returns
    -------
        peps  : class Lattice
        info : dict (Dictionary containing information about the evolution.)

    """

    infos = []

    for gate in gates.local:
        env = apply_local_gate_(env, gate) # here psi will be update in place

    all_gates = gates.nn + gates.nn[::-1]

    for gate in all_gates:
        env, info = evol_machine(env, gate, opts_evol, opts_svd)
        infos.append(info)

    for gate in gates.local[::-1]:
        env = apply_local_gate_(env, gate)

    if env.depth == 'svd-update':
        return env, info

    if env.depth == 1 and not env.depth==0:
        info['ntu_error'] = [record['ntu_error'] for record in infos]
        info['optimal_cutoff'] = [record['optimal_cutoff'] for record in infos]
        info['svd_error'] = [record['svd_error'] for record in infos]
        return env, info


def gates_homogeneous(peps, nn_gates, loc_gates):

    """
    Generate a list of gates that is homogeneous over the lattice.

    Parameters
    ----------
    peps      : class Lattice
    nn_gates : list
              A list of two-tuples, each containing the tensors that form a two-site
              nearest-neighbor gate.
    loc_gates : A two-tuple containing the tensors that form the single-site gate.

    Returns
    -------
    Gates: The generated gates. The NamedTuple 'Gates` named tuple contains a list of
      local and nn gates along with info where they should be applied.
    """
    # len(nn_gates) indicates the physical degrees of freedom; option to add more
    bonds = peps.nn_bonds(dirn='h') + peps.nn_bonds(dirn='v')

    gates_nn = []   # nn_gates = [(GA, GB), (GA, GB)]   [(GA, GB, GA, GB)]
    for bd in bonds:
        for i in range(len(nn_gates)):
            gates_nn.append(Gate_nn(A=nn_gates[i][0], B=nn_gates[i][1], bond=bd))
    gates_loc = []
    if peps.lattice=='checkerboard':
        gates_loc.append(Gate_local(A=loc_gates, site=(0,0)))
        gates_loc.append(Gate_local(A=loc_gates, site=(0,1)))
    elif peps.lattice != 'checkerboard':
        for site in peps.sites():
            gates_loc.append(Gate_local(A=loc_gates, site=site))
    return Gates(local=gates_loc, nn=gates_nn)


def show_leg_structure(peps):
   """ Prints the leg structure of each site tensor in a PEPS """
   for ms in peps.sites():
        xs = peps[ms].unfuse_legs((0, 1))
        print("site ", str(ms), xs.get_shape())

