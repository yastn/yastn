""" Functions performing many CTMRG steps until convergence and return of CTM environment tensors for mxn lattice. """
from typing import NamedTuple
import logging
from yastn.tensor.linalg import svd


logger = logging.Logger('ctmrg')


class CTMRGout(NamedTuple):
    sweeps : int = 0
    max_dsv : float = None


def calculate_corner_svd(env):
    corner_sv = {}
    for site in env.sites():
        corner_sv[site, 'tl'] = svd(env[site].tl, compute_uv=False)
        corner_sv[site, 'tr'] = svd(env[site].tr, compute_uv=False)
        corner_sv[site, 'bl'] = svd(env[site].bl, compute_uv=False)
        corner_sv[site, 'br'] = svd(env[site].br, compute_uv=False)
    return corner_sv


def ctmrg_(env, max_sweeps=1, iterator_step=1, method='2site', opts_svd=None, fix_signs=True, corner_tol=None):
    r"""
    Generator executing ctmrg().
    """
    # init
    max_dsv = None
    if corner_tol:
        old_corner_sv = calculate_corner_svd(env)

    for sweep in range(1, max_sweeps + 1):
        #
        env.update_(method=method, fix_signs=fix_signs, opts_svd=opts_svd)
        #
        if corner_tol:  # check convergence of corners singular values
            corner_sv = calculate_corner_svd(env)
            max_dsv = max((old_corner_sv[k] - v).norm().item() / v.norm().item() for k, v in corner_sv.items())
            old_corner_sv = corner_sv

        logger.info(f'Sweep = {sweep:03d}  max_d_corner_singular_values = {max_dsv}')

        if corner_tol and max_dsv < corner_tol:
            break

        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield CTMRGout(sweeps=sweep, max_dsv=max_dsv)
    yield CTMRGout(sweeps=sweep, max_dsv=max_dsv)
