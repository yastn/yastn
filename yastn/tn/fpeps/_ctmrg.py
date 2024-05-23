""" Functions performing many CTMRG steps until convergence and return of CTM environment tensors for mxn lattice. """
from typing import NamedTuple
from yastn.tensor.linalg import svd

class CTMRGout(NamedTuple):
    sweeps : int = 0
    env : dict = None
    fix_signs: bool = False

def check_corner_sv_error(corner_1, corner_2):
    err = 0
    for ii in range(max(len(corner_1), len(corner_2))):
        if ii >= len(corner_1):
            err = err + corner_2[ii] ** 2
        elif ii >= len(corner_2):
            err = err + corner_1[ii] ** 2
        else:
            err = err + (corner_1[ii] - corner_2[ii]) ** 2
    return err

def ctmrg_(env, max_sweeps=1, iterator_step=1, method='2site', fix_signs=True, opts_svd=None, corner_svd=False):
    r"""
    Generator for ctmrg().
    """

    # init
    if corner_svd:
        corner_sv_dict = dict()
        for site in env.sites():
            corner_sv_dict.update({site: [[0], [0], [0], [0]]})


    for sweep in range(1, max_sweeps + 1):

        env.update_(method=method, fix_signs=fix_signs, opts_svd=opts_svd)

        # Apply corner SV criteria
        if corner_svd:

            # Find SV of corner tensors
            new_corner_sv_dict = dict()
            for site in env.sites():
                s_tl = (svd(env[site].tl, sU=1, compute_uv=False)).compress_to_1d()[0].tolist()
                s_tr = (svd(env[site].tr, sU=1, compute_uv=False)).compress_to_1d()[0].tolist()
                s_bl = (svd(env[site].bl, sU=1, compute_uv=False)).compress_to_1d()[0].tolist()
                s_br = (svd(env[site].br, sU=1, compute_uv=False)).compress_to_1d()[0].tolist()
                new_corner_sv_dict.update({site: [s_tl, s_tr, s_bl, s_br]})

            # Check max error
            max_error = -1
            for site in env.sites():
                for ii in range(4):
                    # Check relative error
                    max_error = max(max_error, check_corner_sv_error(corner_sv_dict[site][ii], new_corner_sv_dict[site][ii]) \
                                               / max(sum(_ ** 2 for _ in corner_sv_dict[site][ii]), sum(_ ** 2 for _ in new_corner_sv_dict[site][ii])))

            if abs(max_error) < opts_svd['tol']:
                break

            # Update SV values
            for site in env.sites():
                for ii in range(4):
                    corner_sv_dict[site][ii] = new_corner_sv_dict[site][ii]


        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield CTMRGout(sweep, env, fix_signs=fix_signs)
    yield CTMRGout(sweep, env, fix_signs=fix_signs)
