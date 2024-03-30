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

def one_site_dict(peps, env, op):
    r"""
    dictionary containing site coordinates as keys and their corresponding expectation values

    Parameters
    ----------
    peps : class Lattice
        class containing peps data along with the lattice structure data
    env: class CtmEnv
        class containing ctm environment tensors along with lattice structure data
    op: single site operator

    """

    site_exp_dict = {}  # Dictionary to store site-wise expectation values
    peps = check_consistency_tensors(peps)

    for ms in peps.sites():
        Am = peps[ms]
        val_op = measure_one_site_spin(Am, ms, env, op=op)
        val_norm = measure_one_site_spin(Am, ms, env, op=None)
        site_exp_value = val_op / val_norm  # expectation value of particular target site
        # Store the site and its expectation value in the dictionary
        site_exp_dict[ms] = site_exp_value

    return site_exp_dict

def measure_one_site_spin(A, ms, env, op=None):
    r"""
    Returns the overlap of bra and ket on a single site.

    Parameters
    ----------
    A : single peps tensor at site ms
    ms : site where we want to measure some observable
    env: class CtmEnv
        class containing ctm environmental tensors along with lattice structure data
    op: single site operator
    """

    if op is not None:
        AAb = fPEPS_2layers(A, op=op, dir='1s')
    elif op is None:
        AAb = fPEPS_2layers(A)
    vecl = tensordot(env[ms].l, env[ms].tl, axes=(2, 0))
    vecl = tensordot(env[ms].bl, vecl, axes=(1, 0))
    new_vecl = apply_TMO_left(vecl, env, ms, AAb)
    vecr = tensordot(env[ms].tr, env[ms].r, axes=(1, 0))
    vecr = tensordot(vecr, env[ms].br, axes=(2, 0))
    hor = tensordot(new_vecl, vecr, axes=((0, 1, 2), (2, 1, 0))).to_number()
    return hor


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


