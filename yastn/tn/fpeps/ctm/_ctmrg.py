""" Functions performing many CTMRG steps until convergence and return of CTM environment tensors for mxn lattice. """
from typing import NamedTuple
import logging
import time
from ._ctm_iteration_routines import CTM_it
from ._ctm_iteration_routines import fPEPS_2layers, check_consistency_tensors
from ._ctm_env import CtmEnv, init_rand


#################################
#           ctmrg               #
#################################

class CTMRGout(NamedTuple):
    sweeps : int = 0
    env : dict = None
    proj : dict = None
    tt : float = None


def ctmrg(peps, max_sweeps=1, iterator_step=None, fix_signs=None, env=None, opts_svd=None):
    r"""
    Perform CTMRG sweeps until convergence, starting from PEPS and environmental corner and edge tensors :code:`peps`.

    The outer loop sweeps over PEPS updating only the environmental tensors through 2x2 windows of PEPS tensors.
    Convergence can be controlled based on observables or Schmidt values of projectors.
    The CTMRG algorithm sweeps through the lattice at most :code:`max_sweeps` times
    or through some externally set convergence criteria.

    Outputs iterator if :code:`iterator_step` is given.
    It allows inspecting :code:`CtmEnv` outside of :code:`ctmrg_` function after every :code:`iterator_step` sweeps.

    Parameters
    ----------
    peps: yastn.fPEPS.Peps
        peps tensors occupying all the lattice sites in 2D. Maybe obtained after real or imaginary time evolution.
        It is not updated during execution.

    env: yastn.fPEPS.CtmEnv
        Initial environmental tensors: maybe random or given by the user. It is updated during execution.
        The virtual bonds aligning with the boundary can be of maximum bond dimension chi

    chi: maximal CTM bond dimension

    cutoff: controls removal of singular values smaller than cutoff during CTM projector construction

    prec: stop execution when 2 consecutive iterations give difference of a function used to evaluate conv smaller than prec

    max_sweeps: int
        Maximal number of sweeps.

    iterator_step: int
        If int, :code:`ctmrg_` returns a generator that would yield output after every iterator_step sweeps.
        Default is None, in which case  :code:`ctmrg_` sweeps are performed immediately.

    tcinit: symmetry sectors of initial corners legs

    Dcinit: dimensions of initial corner legs

    AAb = 0 (no double-peps tensors; 1 = double peps tensors with identity; 2 = for all

    Returns
    -------
    out: CTMRGout(NamedTuple)
        Includes fields:
        :code:`sweeps` number of performed dmrg sweeps.
        :code:`env` environmental tensors.
        :code:`proj` projectors.
    """

    # if environment is not given, start with a random initialization
    pconfig =  peps[0,0].config
    if env is None:
        env = init_rand(peps, tc = ((0,) * pconfig.sym.NSYM,), Dc=(1,))  # initialization with random tensors

    tmp = _ctmrg(peps, env, max_sweeps, iterator_step, fix_signs, opts_svd)
    return tmp if iterator_step else next(tmp)


def _ctmrg(peps, env, max_sweeps, iterator_step, fix_signs, opts_svd=None):

    """ Generator for ctmrg(). """
    peps = check_consistency_tensors(peps) # to check if A has the desired fused form of legs i.e. t l b r [s a]

    AAb = CtmEnv(peps)

    for ms in peps.sites():
        AAb[ms] = fPEPS_2layers(peps[ms])

    for sweep in range(1, max_sweeps + 1):
        logging.info('CTM sweep: %2d', sweep)
        t_start = time.time()
        env, proj = CTM_it(env, AAb, fix_signs, opts_svd)
        t_end = time.time()
        tt = t_end - t_start
        logging.info('sweep time: %0.2f s.', tt)

        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield CTMRGout(sweep, env, proj, tt)
    yield CTMRGout(sweep, env, proj, tt)
