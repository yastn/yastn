from ._ctm_iteration_routines import check_consistency_tensors
from ._ctm_iteration_routines import fPEPS_2layers
from ._ctm_observable_routines import apply_TMO_left, EV2pt
from .... import tensordot

def nn_exp_dict(peps, env, op):

    r"""
    Returns two dictionaries 'obs_hor' and 'obs_ver' that store horizontal and vertical
    nearest-neighbor expectation values, respectively. Each such dictionary has as its keys
    labels of the input two-site operators which further stores a dictionary of NN sites as
    keys and the corresponding obervables as values.

    Parameters
    ----------
    peps : class Lattice
        class containing peps data along with the lattice structure data
    env: class CtmEnv
        class containing ctm environment tensors along with lattice structure data
    op: dictionary of two-site observables

    """

    peps = check_consistency_tensors(peps)
    obs_hor = {}
    obs_ver = {}

    for ms in op.keys():
        obs_hor[ms] = {}
        obs_ver[ms] = {}

        opt = op.get(ms)

        for bds_h in peps.bonds(dirn='h'):  # correlators on all horizontal bonds
            val_hor = EVnn(peps, env, bds_h.site0, bds_h.site1, opt)
            obs_hor[ms][(bds_h.site0, bds_h.site1)] = val_hor

        for bds_v in peps.bonds(dirn='v'):  # correlators on all vertical bonds
            val_ver = EVnn(peps, env, bds_v.site0, bds_v.site1, opt)
            obs_ver[ms][(bds_v.site0, bds_v.site1)] = val_ver

    return obs_hor, obs_ver

def EVnn(peps, env, site0, site1, op):

    r"""
    Returns two-point correlators given any two nearest-neighbor sites.

    Parameters
    ----------
    peps : class Lattice
    env: class CtmEnv
    site0, site1: sites where the two-point correlator is to be calculated
    op: observable whose two-point correlators need to be calculated

    """

    peps = check_consistency_tensors(peps) # to check if A has the desired fused form of legs i.e. t l b r [s a]
    norm_array = EV2pt(peps, env, site0, site1)
    op_array = EV2pt(peps, env, site0, site1, op)
    array_corr = op_array / norm_array

    return array_corr


