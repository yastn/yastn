import logging
import numpy as np
from ._env_ctm import CTMRG_out, EnvCTM, update_storage_
from .._peps import Peps
from .._geometry import Site, SquareLattice
from ...._from_dict import from_dict
import ray
import psutil

def get_taskset_cpu_count():
    proc = psutil.Process()
    affinity = proc.cpu_affinity()
    return len(affinity)

import os
os.environ.update({
    'RAY_object_store_memory_lock': '0',
    'RAY_object_store_memory': '100000000000',
    'RAY_plasma_unlimited': '1',
    'RAY_enable_memory_plasma_logging': '1',
    'RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO': '0'})

def CreateCTMJobBundle(env:EnvCTM, cpus_per_task=4):

    n_tasks_per_batch = get_taskset_cpu_count() // cpus_per_task

    Lx = env.psi.geometry.Nx
    Ly = env.psi.geometry.Ny

    sites = env.psi.sites()

    nbundles_ver = min(max(n_tasks_per_batch // Ly, 1), Lx)
    ctm_jobs_ver = [[sites[ix + iy * Lx] for iy in range(Ly) for ix in range(ib * nbundles_ver, min((ib + 1) * nbundles_ver, Lx))] for ib in range(int(np.floor(Lx / nbundles_ver)))]

    nbundles_hor = min(max(n_tasks_per_batch // Lx, 1), Ly)
    ctm_jobs_hor = [[sites[ix + iy * Lx] for ix in range(Lx) for iy in range(ib * nbundles_hor, min((ib + 1) * nbundles_hor, Ly))] for ib in range(int(np.floor(Ly / nbundles_hor)))]

    return [ctm_jobs_hor, ctm_jobs_ver]

@ray.remote(num_cpus=4, num_gpus=0, num_returns=1)
def BuildProjector(site, move, env, opts_svd_ctm, cfg, method='2x2'):

    sites = [env.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))]
    if None in sites:
        return
    tl, tr, bl, br = sites
    if opts_svd_ctm is None:
        opts_svd_ctm = env.opts_svd

    env._update_projectors_(site, move, opts_svd_ctm, method=method)

    result_dict = {}

    if any(x in move for x in 'rh'):
        result_dict[(tr, 'hrb')] = env.proj[tr].hrb.to_dict(level=1)
        result_dict[(br, 'hrt')] = env.proj[br].hrt.to_dict(level=1)

    if any(x in move for x in 'lh'):
        result_dict[(tl, 'hlb')] = env.proj[tl].hlb.to_dict(level=1)
        result_dict[(bl, 'hlt')] = env.proj[bl].hlt.to_dict(level=1)

    if any(x in move for x in 'tv'):
        result_dict[(tl, 'vtr')] = env.proj[tl].vtr.to_dict(level=1)
        result_dict[(tr, 'vtl')] = env.proj[tr].vtl.to_dict(level=1)

    if any(x in move for x in 'bv'):
        result_dict[(bl, 'vbr')] = env.proj[bl].vbr.to_dict(level=1)
        result_dict[(br, 'vbl')] = env.proj[br].vbl.to_dict(level=1)

    return result_dict

@ray.remote(num_cpus=4, num_gpus=0, num_returns=6)
def UpdateSite(site, site_t_or_l, site_b_or_r, site_tl, site_tr, site_bl, site_br, env, cfg, move, proj_dict):

    env_tmp = EnvCTM(env.psi, init=None)

    if move in 'vtb':

        newt = None
        newb = None
        newtl = None
        newtr = None
        newbl = None
        newbr = None

        if move in 'vt':

            env._trivial_projectors_('t', sites=env.sites())

            temp_site = site_tl
            temp_site0 = canonical_site(env, env.nn_site(site, (-1, -1)))
            if temp_site0 is not None:
                env.proj[temp_site0].vtr = from_dict(config=cfg, d=proj_dict[(temp_site, 'vtr')])
                temp_site = site_t_or_l
                temp_site0 = canonical_site(env, env.nn_site(site, (-1, 0)))
                env.proj[temp_site0].vtl = from_dict(config=cfg, d=proj_dict[(temp_site, 'vtl')])


            temp_site = site_tr
            temp_site0 = canonical_site(env, env.nn_site(site, (-1, 1)))
            if temp_site0 is not None:
                env.proj[temp_site0].vtl = from_dict(config=cfg, d=proj_dict[(temp_site, 'vtl')])
                temp_site = site_t_or_l
                temp_site0 = canonical_site(env, env.nn_site(site, (-1, 0)))
                env.proj[temp_site0].vtr = from_dict(config=cfg, d=proj_dict[(temp_site, 'vtr')])

        if move in 'vb':

            env._trivial_projectors_('b', sites=env.sites())

            temp_site = site_bl
            temp_site0 = canonical_site(env, env.nn_site(site, (1, -1)))
            if temp_site0 is not None:
                env.proj[temp_site0].vbr = from_dict(config=cfg, d=proj_dict[(temp_site, 'vbr')])
                temp_site = site_b_or_r
                temp_site0 = canonical_site(env, env.nn_site(site, (1, 0)))
                env.proj[temp_site0].vbl = from_dict(config=cfg, d=proj_dict[(temp_site, 'vbl')])


            temp_site = site_br
            temp_site0 = canonical_site(env, env.nn_site(site, (1, 1)))
            if temp_site0 is not None:
                env.proj[temp_site0].vbl = from_dict(config=cfg, d=proj_dict[(temp_site, 'vbl')])
                temp_site = site_b_or_r
                temp_site0 = canonical_site(env, env.nn_site(site, (1, 0)))
                env.proj[temp_site0].vbr = from_dict(config=cfg, d=proj_dict[(temp_site, 'vbr')])

        env_tmp._update_env_(site, env, move=move)

        if move in 'vt':
            update_storage_(env, env_tmp)
            newt = env[site].t.to_dict(level=1)
            newtl = env[site].tl.to_dict(level=1)
            newtr = env[site].tr.to_dict(level=1)
        if move in 'vb':
            update_storage_(env, env_tmp)
            newb = env[site].b.to_dict(level=1)
            newbl = env[site].bl.to_dict(level=1)
            newbr = env[site].br.to_dict(level=1)

        return (newt, newb, newtl, newtr, newbl, newbr)

    if move in 'hlr':

        newl = None
        newr = None
        newtl = None
        newtr = None
        newbl = None
        newbr = None

        if move in 'hl':

            env._trivial_projectors_('l', sites=env.sites())

            temp_site = site_tl
            temp_site0 = canonical_site(env, env.nn_site(site, (-1, -1)))
            if temp_site0 is not None:
                env.proj[temp_site0].hlb = from_dict(config=cfg, d=proj_dict[(temp_site, 'hlb')])
                temp_site = site_t_or_l
                temp_site0 = canonical_site(env, env.nn_site(site, (0, -1)))
                env.proj[temp_site0].hlt = from_dict(config=cfg, d=proj_dict[(temp_site, 'hlt')])

            temp_site = site_bl
            temp_site0 = canonical_site(env, env.nn_site(site, (1, -1)))
            if temp_site0 is not None:
                env.proj[temp_site0].hlt = from_dict(config=cfg, d=proj_dict[(temp_site, 'hlt')])
                temp_site = site_t_or_l
                temp_site0 = canonical_site(env, env.nn_site(site, (0, -1)))
                env.proj[temp_site0].hlb = from_dict(config=cfg, d=proj_dict[(temp_site, 'hlb')])

        if move in 'hr':

            env._trivial_projectors_('r', sites=env.sites())

            temp_site = site_tr
            temp_site0 = canonical_site(env, env.nn_site(site, (-1, 1)))
            if temp_site0 is not None:
                env.proj[temp_site0].hrb = from_dict(config=cfg, d=proj_dict[(temp_site, 'hrb')])
                temp_site = site_b_or_r
                temp_site0 = canonical_site(env, env.nn_site(site, (0, 1)))
                env.proj[temp_site0].hrt = from_dict(config=cfg, d=proj_dict[(temp_site, 'hrt')])

            temp_site = site_br
            temp_site0 = canonical_site(env, env.nn_site(site, (1, 1)))
            if temp_site0 is not None:
                env.proj[temp_site0].hrt = from_dict(config=cfg, d=proj_dict[(temp_site, 'hrt')])
                temp_site = site_b_or_r
                temp_site0 = canonical_site(env, env.nn_site(site, (0, 1)))
                env.proj[temp_site0].hrb = from_dict(config=cfg, d=proj_dict[(temp_site, 'hrb')])

        env_tmp._update_env_(site, env, move=move)

        if move in 'hl':
            update_storage_(env, env_tmp)
            newl = env[site].l.to_dict(level=1)
            newtl = env[site].tl.to_dict(level=1)
            newbl = env[site].bl.to_dict(level=1)
        if move in 'hr':
            update_storage_(env, env_tmp)
            newr = env[site].r.to_dict(level=1)
            newtr = env[site].tr.to_dict(level=1)
            newbr = env[site].br.to_dict(level=1)

        return (newl, newr, newtl, newtr, newbl, newbr)

def SubWindow(env_psi: Peps | EnvCTM, site, top=1, left=1, bottom=1, right=1, only_site=False, site_load=None, env_load_dict=None):

    '''
    Find the window around the particular site. The boundary has been taken into consideration
    '''

    if site is None:
        return None, None, None, None

    nx0 = top
    ny0 = left
    Lx = top + bottom + 1
    Ly = left + right + 1

    if type(env_psi) is Peps:
        psi = env_psi
        ket = None
        env = None
    elif type(env_psi) is EnvCTM:
        env = env_psi
        if type(env.psi) is Peps:
            psi = env_psi.psi
            ket = None
        else:
            psi = env_psi.psi.bra
            ket = env_psi.psi._ket


    while psi.nn_site(site, (0, -left)) is None:
        left = left - 1
        Ly = Ly - 1
        ny0 = ny0 - 1

    while psi.nn_site(site, (0, right)) is None:
        right = right - 1
        Ly = Ly - 1

    while psi.nn_site(site, (-top, 0)) is None:
        top = top - 1
        Lx = Lx - 1
        nx0 = nx0 - 1

    while psi.nn_site(site, (bottom, 0)) is None:
        bottom = bottom - 1
        Lx = Lx - 1

    net_part = SquareLattice((Lx, Ly), 'obc')
    psi_part = Peps(net_part, psi[site])

    if ket is not None:
        ket_part = Peps(net_part, ket[site])
    else:
        ket_part = None


    site0 = Site(nx0, ny0)

    if only_site:
        return site0


    # site0 = Site(nx0, ny0)

    if site_load is None:
        for dx in range(-top, bottom + 1):
            for dy in range(-left, right + 1):
                d = (dx, dy)
                if psi.nn_site(site, d) is not None:
                    psi_part[psi_part.nn_site(site0, d)] = psi[psi.nn_site(site, d)]
                    if ket_part is not None:
                        ket_part[psi_part.nn_site(site0, d)] = ket[psi.nn_site(site, d)]
    else:
        for d in site_load:
            if psi.nn_site(site, d) is not None:
                psi_part[psi_part.nn_site(site0, d)] = psi[psi.nn_site(site, d)]
                if ket_part is not None:
                    ket_part[psi_part.nn_site(site0, d)] = ket[psi.nn_site(site, d)]

    if type(env_psi) is EnvCTM:
        env_part = EnvCTM(psi_part, init='eye', ket=ket_part)
        if env_load_dict is None:
            for dx in range(-top, bottom + 1):
                for dy in range(-left, right + 1):
                    d = (dx, dy)
                    if psi.nn_site(site, d) is not None:
                        for attr in ['l', 'r', 't', 'b', 'tl', 'bl', 'tr', 'br']:
                            setattr(env_part[psi_part.nn_site(site0, d)], attr, getattr(env[psi.nn_site(site, d)], attr))

        else:
            for d in list(env_load_dict.keys()):
                if psi.nn_site(site, d) is not None:
                    for attr in env_load_dict[d]:
                        setattr(env_part[psi_part.nn_site(site0, d)], attr, getattr(env[psi.nn_site(site, d)], attr))


        return env_part, site0, Lx, Ly

    else:
        return psi_part, site0, Lx, Ly




def canonical_site(env, site):
    if site is None:
        return None
    site_index = env.psi.geometry.site2index(site)
    if type(site_index) is tuple:
        return Site(*site_index)
    else:
        return env.psi.sites()[site_index]

def ParaUpdateCTM_(env:EnvCTM, sites, opts_svd_ctm, cfg, move='t', proj_dict=None, sites_to_be_updated=None, cpus_per_task=4, gpus_per_task=0):

    # Build projectors

    env_remote = ray.put(env)

    jobs = []
    for site in sites:

        if (move in 'hv') or (len(env.sites()) < env.Nx * env.Ny):
            site_ = site

        elif move in 't':
            site_ = canonical_site(env, env.nn_site(site, 't'))

        elif move in 'l':
            site_ = canonical_site(env, env.nn_site(site, 'l'))

        elif move in 'br':
            site_ = site

        if site_ is not None:
            jobs.append((site_, move))

    gathered_result_ = ray.get([BuildProjector.options(num_cpus=cpus_per_task, num_gpus=gpus_per_task).remote(*job, env_remote, opts_svd_ctm, cfg) for job in jobs])

    if proj_dict is None:
        proj_dict = {}

    for ii in range(len(jobs)):
        site = jobs[ii][0]

        if gathered_result_[ii] is not None:
            for key in gathered_result_[ii].keys():
                key0 = key[0]
                key1 = key[1]
                key0_new = canonical_site(env, Site(key0.nx, key0.ny))
                proj_dict[(key0_new, key1)] = gathered_result_[ii][key]

    proj_dict_remote = ray.put(proj_dict)

    if sites_to_be_updated is None:
        sites_to_be_updated=sites

    jobs.clear()

    for site in sites_to_be_updated:

        if move in 'vtb':
            jobs.append((site,
                         canonical_site(env, env.nn_site(site, (-1, 0))),
                         canonical_site(env, env.nn_site(site, (1, 0))),
                         canonical_site(env, env.nn_site(site, (-1, -1))),
                         canonical_site(env, env.nn_site(site, (-1, 1))),
                         canonical_site(env, env.nn_site(site, (1, -1))),
                         canonical_site(env, env.nn_site(site, (1, 1)))))
        elif move in 'hlr':
            jobs.append((site,
                         canonical_site(env, env.nn_site(site, (0, -1))),
                         canonical_site(env, env.nn_site(site, (0, 1))),
                         canonical_site(env, env.nn_site(site, (-1, -1))),
                         canonical_site(env, env.nn_site(site, (-1, 1))),
                         canonical_site(env, env.nn_site(site, (1, -1))),
                         canonical_site(env, env.nn_site(site, (1, 1)))))

    updated_ctm_tensors = []
    updated_ctm_tensors = ray.get([UpdateSite.options(num_cpus=cpus_per_task, num_gpus=gpus_per_task, num_returns=1).remote(*job, env_remote, cfg, move, proj_dict_remote) for job in jobs])

    proj_dict.clear()

    ii = 0
    for result in updated_ctm_tensors:
        site_ = sites_to_be_updated[ii]
        if move in "vt":
            env[site_].t = from_dict(config=cfg, d = result[0])
        if move in "vb":
            env[site_].b = from_dict(config=cfg, d = result[1])
        if move in "hl":
            env[site_].l = from_dict(config=cfg, d = result[0])
        if move in "hr":
            env[site_].r = from_dict(config=cfg, d = result[1])
        if move in 'hvtl':
            env[site_].tl = from_dict(config=cfg, d = result[2])
        if move in 'hvtr':
            env[site_].tr = from_dict(config=cfg, d = result[3])
        if move in 'hvbl':
            env[site_].bl = from_dict(config=cfg, d = result[4])
        if move in 'hvbr':
            env[site_].br = from_dict(config=cfg, d = result[5])

        ii = ii + 1


def _ctmrg_(env:EnvCTM, max_sweeps, iterator_step, corner_tol, opts_svd_ctm, ctm_jobs_hv=None, moves='hv', cpus_per_task=4, gpus_per_task=0):

    if ctm_jobs_hv is None:
        # ctm_jobs_hor, ctm_jobs_ver = CreateCTMJobBundle(env, cpus_per_task=cpus_per_task)
        ctm_jobs_hor, ctm_jobs_ver = env.psi.sites(), env.psi.sites()
    else:
        ctm_jobs_hor, ctm_jobs_ver = ctm_jobs_hv

    cfg = env.psi.config

    max_dsv, converged, history = None, False, []

    for sweep in range(1, max_sweeps + 1):

        for move in moves:

            if move in 'vtb':
                # if n_tasks_per_batch >= psi.geometry.Ny:
                for ctm_jobs in ctm_jobs_ver:
                    if ctm_jobs_hv is None:
                        ParaUpdateCTM_(env, ctm_jobs, opts_svd_ctm, cfg, move=move, cpus_per_task=cpus_per_task, gpus_per_task=gpus_per_task)
                    else:
                        ParaUpdateCTM_(env, ctm_jobs, opts_svd_ctm, cfg, move=move, sites_to_be_updated=ctm_jobs, cpus_per_task=cpus_per_task, gpus_per_task=gpus_per_task)

            if move in 'hlr':

                # if n_tasks_per_batch >= psi.geometry.Nx:
                for ctm_jobs in ctm_jobs_hor:
                    if ctm_jobs_hv is None:
                        ParaUpdateCTM_(env, ctm_jobs, opts_svd_ctm, cfg, move=move, cpus_per_task=cpus_per_task, gpus_per_task=gpus_per_task)
                    else:
                        ParaUpdateCTM_(env, ctm_jobs, opts_svd_ctm, cfg, move=move, sites_to_be_updated=ctm_jobs, cpus_per_task=cpus_per_task, gpus_per_task=gpus_per_task)

        if corner_tol is not None:
            # Evaluate convergence of CTM by computing the difference of environment corner spectra between consecutive CTM steps.
            corner_sv = env.calculate_corner_svd()
            max_dsv = max((corner_sv[k] - history[-1][k]).norm().item() for k in corner_sv) if history else float('Nan')
            history.append(corner_sv)
            converged = max_dsv < corner_tol
            logging.info(f'Sweep = {sweep:03d}; max_diff_corner_singular_values = {max_dsv:0.2e}')

            if converged:
                break

        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, max_D=env.max_D(), converged=converged)
    yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, max_D=env.max_D(), converged=converged)

@ray.remote(num_cpus=4, num_gpus=0)
def Measure1Site(site, env, op):
    return {site: env.measure_1site(op, site=site)}

@ray.remote(num_cpus=4, num_gpus=0)
def MeasureNN(bond, env, op0, op1):
    return {bond: env.measure_nn(op0, op1, bond)}

def ParaMeasure1Site(env, op, cpus_per_task=4, gpus_per_task=0):

    if not ray.is_initialized():
        ray.init(num_cpus=get_taskset_cpu_count(), ignore_reinit_error=True, namespace='Measure1Site')

    psi = env.psi
    env_remote = ray.put(env)
    list_of_dicts = ray.get([Measure1Site.options(num_cpus=cpus_per_task, num_gpus=gpus_per_task).remote(site, env_remote, op) for site in env.psi.sites()])

    result = {k: v for d in list_of_dicts for k, v in d.items()}
    return result

def ParaMeasureNN(env, op0, op1, cpus_per_task=4, gpus_per_task=0):

    if not ray.is_initialized():
        ray.init(num_cpus=get_taskset_cpu_count(), ignore_reinit_error=True, namespace='MeasureNN')

    psi = env.psi
    list_of_dicts = []

    env_remote = ray.put(env)

    list_of_dicts += ray.get([MeasureNN.options(num_cpus=cpus_per_task, num_gpus=gpus_per_task).remote(bond, env_remote, op0, op1) for bond in psi.bonds(dirn='h')])
    list_of_dicts += ray.get([MeasureNN.options(num_cpus=cpus_per_task, num_gpus=gpus_per_task).remote(bond, env_remote, op0, op1) for bond in psi.bonds(dirn='v')])

    result = {k: v for d in list_of_dicts for k, v in d.items()}
    return result


def PARActmrg_(env:EnvCTM, max_sweeps=50, iterator_step=1, opts_svd_ctm=None, corner_tol=None, ctm_jobs_hv=None, moves='hv', cpus_per_task=1, gpus_per_task=0):
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=get_taskset_cpu_count(), ignore_reinit_error=True, namespace='BuilidProjector')
    tmp = _ctmrg_(env, max_sweeps, iterator_step, corner_tol, opts_svd_ctm, cpus_per_task=cpus_per_task, gpus_per_task=gpus_per_task, ctm_jobs_hv=ctm_jobs_hv, moves=moves)
    return tmp if iterator_step else next(tmp)