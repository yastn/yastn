""" Functions performing many CTMRG steps until convergence and return of CTM environment tensors for mxn lattice. """
from typing import NamedTuple
import logging
from ._ctm_iteration_routines import CTM_it
from ._ctm_iteration_routines import fPEPS_2layers, fPEPS_fuse_layers, check_consistency_tensors
from ._ctm_env import CtmEnv, init_rand


#################################
#           ctmrg               #
#################################

class CTMRGout(NamedTuple):
    sweeps : int = 0
    env : dict = None
    proj_hor : dict = None
    proj_vert : dict = None


def ctmrg(psi, max_sweeps=1, iterator_step=None, AAb_mode=0, fix_signs=None, env=None, opts_svd=None):
    r"""
    Perform CTMRG sweeps until convergence, starting from PEPS and environmental corner and edge tensors :code:`psi`.

    The outer loop sweeps over PEPS updating only the environmental tensors through 2x2 windows of PEPS tensors.
    Convergence can be controlled based on observables or Schmidt values of projectors.
    The CTMRG algorithm sweeps through the lattice at most :code:`max_sweeps` times
    or until all convergence measures with provided tolerance change by less then the tolerance.

    Outputs generator if :code:`iterator_step` is given.
    It allows inspecting :code:`psi` outside of :code:`dmrg_` function after every :code:`iterator_step` sweeps.

    Parameters
    ----------
    psi: yaps.Peps
        peps tensors occupying all the lattice sites in 2D. Maybe obtained after real or imaginary time evolution.
        It is not updated during execution.

    env: yaps.CtmEnv
        Initial environmental tensors: maybe random or given by the user. It is updated during execution. 
        The virtual bonds aligning with the boundary can be of maximum bond dimension chi

    chi: maximal CTM bond dimension

    cutoff: controls removal of singular values smaller than cutoff during CTM projector construction

    prec: stop execution when 2 consecutive iterations give difference of a function used to evaluate conv smaller than prec

    max_sweeps: int
        Maximal number of sweeps.

    iterator_step: int
        If int, :code:`dmrg_` returns a generator that would yield output after every iterator_step sweeps.
        Default is None, in which case  :code:`dmrg_` sweeps are performed immidiatly.

    tcinit: symmetry sectors of initial corners legs

    Dcinit: dimensions of initial corner legs

    AAb = 0 (no double-peps tensors; 1 = double pepes tensors with identity; 2 = for all

    Returns
    -------
    out: CTMRGout(NamedTuple)
        Includes fields:
        :code:`sweeps` number of performed dmrg sweeps.
    """
    
    # if environment is not given, start with a random initialization
    pconfig =  psi[0,0].config
    if env is None:
        env = init_rand(psi, tc = ((0,) * pconfig.sym.NSYM,), Dc=(1,))  # initialization with random tensors 

    tmp = _ctmrg(psi, env, max_sweeps, iterator_step, AAb_mode, fix_signs, opts_svd)
    return tmp if iterator_step else next(tmp)


def _ctmrg(psi, env, max_sweeps, iterator_step, AAb_mode, fix_signs, opts_svd=None):

    """ Generator for ctmrg(). """
    psi = check_consistency_tensors(psi) # to check if A has the desired fused form of legs i.e. t l b r [s a]

    AAb = CtmEnv(psi)

    for ms in psi.sites():
        AAb[ms] = fPEPS_2layers(psi[ms])

    if AAb_mode >= 1:
        fPEPS_fuse_layers(AAb)
  
    for sweep in range(1, max_sweeps + 1):
        logging.info('CTM sweep: %2d', sweep)
        cheap_moves=False
        env, proj = CTM_it(env, AAb, cheap_moves, fix_signs, opts_svd)

        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield CTMRGout(sweep, env, proj)
    yield CTMRGout(sweep, env, proj)
