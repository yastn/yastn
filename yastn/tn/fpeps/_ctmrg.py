""" Functions performing many CTMRG steps until convergence and return of CTM environment tensors for mxn lattice. """
from typing import NamedTuple


class CTMRGout(NamedTuple):
    sweeps : int = 0
    env : dict = None


def ctmrg_(env, max_sweeps=1, iterator_step=1, fix_signs=None, opts_svd=None):
    r"""
    Generator for ctmrg().
    """

    for sweep in range(1, max_sweeps + 1):
        env.update_()

        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield CTMRGout(sweep, env)
    yield CTMRGout(sweep, env)
