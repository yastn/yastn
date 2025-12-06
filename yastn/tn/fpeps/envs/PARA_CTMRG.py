import yastn
import logging
import numpy as np

from joblib import Parallel, delayed
from ...fpeps import *
from ._env_ctm import CTMRG_out, EnvCTM, update_storage_
from .._peps import Peps
from ...fpeps._geometry import Site, Bond


def CreateCTMJobBundle(env:EnvCTM, n_cores=1):

    Lx = env.psi.geometry.Nx
    Ly = env.psi.geometry.Ny

    sites = env.psi.sites()

    # ctm_jobs_hor = [[sites[ix * Ly + iy]  for iy in range(Ly)] for ix in range(Lx)]
    # ctm_jobs_ver = []
    # nbundles_hor = max(n_cores // Ly, 1)
    nbundles_hor = max(n_cores // Ly, 1)
    ctm_jobs_hor = [[sites[ix + iy * Lx] for iy in range(Ly) for ix in range(ib * nbundles_hor, min((ib + 1) * nbundles_hor, Lx))] for ib in range(int(np.floor(Lx / nbundles_hor)))]

    nbundles_ver = max(n_cores // Lx, 1)
    ctm_jobs_ver = [[sites[ix + iy * Lx] for ix in range(Lx) for iy in range(ib * nbundles_ver, min((ib + 1) * nbundles_ver, Ly))] for ib in range(int(np.floor(Ly / nbundles_ver)))]

    return [ctm_jobs_ver, ctm_jobs_hor]


@delayed
def BuildProjector_(job, move, opts_svd_ctm, cfg, method='2site'):

    env_dict = job[0]
    site = job[1]
    env = load_from_dict(config=cfg, d=env_dict)

    sites = [env.nn_site(site, d=d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))]
    tl, tr, bl, br = sites
    if opts_svd_ctm is None:
        opts_svd_ctm = env.opts_svd
    # result = env.build_bond_projectors_(bond, opts_svd_ctm)
    env._update_projectors_(site, move, opts_svd_ctm, method=method)

    result_dict = {}

    if any(x in move for x in 'rh'):
        result_dict[(tr, 'hrb')] = yastn.save_to_dict(env.proj[tr].hrb)
        result_dict[(br, 'hrt')] = yastn.save_to_dict(env.proj[br].hrt)

    if any(x in move for x in 'lh'):
        result_dict[(tl, 'hlb')] = yastn.save_to_dict(env.proj[tl].hlb)
        result_dict[(bl, 'hlt')] = yastn.save_to_dict(env.proj[bl].hlt)

    if any(x in move for x in 'tv'):
        result_dict[(tl, 'vtr')] = yastn.save_to_dict(env.proj[tl].vtr)
        result_dict[(tr, 'vtl')] = yastn.save_to_dict(env.proj[tr].vtl)

    if any(x in move for x in 'bv'):
        result_dict[(bl, 'vbr')] = yastn.save_to_dict(env.proj[bl].vbr)
        result_dict[(br, 'vbl')] = yastn.save_to_dict(env.proj[br].vbl)

    return result_dict

def SubWindow(psi, site, fid, top=1, left=1, bottom=1, right=1, env=None, only_site=False, site_load=None, env_load_dict=None):

    '''
    Find the window around the particular site. The boundary has been taken into consideration
    '''

    nx0 = top
    ny0 = left
    Lx = top + bottom + 1
    Ly = left + right + 1

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
    psi_part = product_peps(net_part, fid)


    site0 = Site(nx0, ny0)

    if only_site:
        return site0


    # site0 = Site(nx0, ny0)

    if site_load is None:
        for dx in range(-top, bottom + 1):
            for dy in range(-left, right + 1):
                d = (dx, dy)
                # if psi.nn_site(site, d) is not None:
                psi_part[psi_part.nn_site(site0, d)] = psi[psi.nn_site(site, d)]
    else:
        for d in site_load:
            if psi.nn_site(site, d) is not None:
                psi_part[psi_part.nn_site(site0, d)] = psi[psi.nn_site(site, d)]

    if env is not None:

        env_part = EnvCTM(psi_part, init='eye')
        if env_load_dict is None:
            for dx in range(-top, bottom + 1):
                for dy in range(-left, right + 1):
                    d = (dx, dy)
                    # if psi.nn_site(site, d) is not None:
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


@delayed
def UpdateSite(job, cfg, dirn, proj_dict):

    env_: EnvCTM

    env_ = load_from_dict(config=cfg, d=job[0])
    site0 = job[1]


    if dirn in 'htb':

        newt = None
        newb = None
        newtl = None
        newtr = None
        newbl = None
        newbr = None

        if dirn in 'ht':

            env_._trivial_projectors_('t', sites=env_.sites())

            temp_site = job[5]
            temp_site0 = canonical_site(env_, env_.nn_site(site0, (-1, -1)))
            if temp_site0 is not None:
                env_.proj[temp_site0].vtr = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vtr')])
                temp_site = job[3]
                temp_site0 = canonical_site(env_, env_.nn_site(site0, (-1, 0)))
                env_.proj[temp_site0].vtl = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vtl')])


            temp_site = job[6]
            temp_site0 = canonical_site(env_, env_.nn_site(site0, (-1, 1)))
            if temp_site0 is not None:
                env_.proj[temp_site0].vtl = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vtl')])
                temp_site = job[3]
                temp_site0 = canonical_site(env_, env_.nn_site(site0, (-1, 0)))
                env_.proj[temp_site0].vtr = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vtr')])

            env_tmp = EnvCTM(env_.psi, init=None)
            env_tmp._update_env_(site0, env_, move='t')
            update_storage_(env_, env_tmp)

            newt = yastn.save_to_dict(env_[site0].t)
            newtl = yastn.save_to_dict(env_[site0].tl)
            newtr = yastn.save_to_dict(env_[site0].tr)


        if dirn in 'hb':

            env_._trivial_projectors_('b', sites=env_.sites())

            temp_site = job[7]
            temp_site0 = canonical_site(env_, env_.nn_site(site0, (1, -1)))
            if temp_site0 is not None:
                env_.proj[temp_site0].vbr = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vbr')])
                temp_site = job[4]
                temp_site0 = canonical_site(env_, env_.nn_site(site0, (1, 0)))
                env_.proj[temp_site0].vbl = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vbl')])


            temp_site = job[8]
            temp_site0 = canonical_site(env_, env_.nn_site(site0, (1, 1)))
            if temp_site0 is not None:
                env_.proj[temp_site0].vbl = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vbl')])
                temp_site = job[4]
                temp_site0 = canonical_site(env_, env_.nn_site(site0, (1, 0)))
                env_.proj[temp_site0].vbr = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vbr')])

            env_tmp = EnvCTM(env_.psi, init=None)
            env_tmp._update_env_(site0, env_, move='b')
            update_storage_(env_, env_tmp)

            newb = yastn.save_to_dict(env_[site0].b)
            newbl = yastn.save_to_dict(env_[site0].bl)
            newbr = yastn.save_to_dict(env_[site0].br)

        return [newt, newb, newtl, newtr, newbl, newbr]

    if dirn in 'vlr':

        newl = None
        newr = None
        newtl = None
        newtr = None
        newbl = None
        newbr = None

        if dirn in 'vl':

            env_._trivial_projectors_('l', sites=env_.sites())

            temp_site = job[5]
            temp_site0 = canonical_site(env_, env_.nn_site(site0, (-1, -1)))
            if temp_site0 is not None:
                env_.proj[temp_site0].hlb = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hlb')])
                temp_site = job[3]
                temp_site0 = canonical_site(env_, env_.nn_site(site0, (0, -1)))
                env_.proj[temp_site0].hlt = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hlt')])

            temp_site = job[7]
            temp_site0 = canonical_site(env_, env_.nn_site(site0, (1, -1)))
            if temp_site0 is not None:
                env_.proj[temp_site0].hlt = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hlt')])
                temp_site = job[3]
                temp_site0 = canonical_site(env_, env_.nn_site(site0, (0, -1)))
                env_.proj[temp_site0].hlb = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hlb')])

            env_tmp = EnvCTM(env_.psi, init=None)
            env_tmp._update_env_(site0, env_, move='l')
            update_storage_(env_, env_tmp)

            newl = yastn.save_to_dict(env_[site0].l)
            newtl = yastn.save_to_dict(env_[site0].tl)
            newbl = yastn.save_to_dict(env_[site0].bl)

        if dirn in 'vr':

            env_._trivial_projectors_('r', sites=env_.sites())

            temp_site = job[6]
            temp_site0 = canonical_site(env_, env_.nn_site(site0, (-1, 1)))
            if temp_site0 is not None:
                env_.proj[temp_site0].hrb = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hrb')])
                temp_site = job[4]
                temp_site0 = canonical_site(env_, env_.nn_site(site0, (0, 1)))
                env_.proj[temp_site0].hrt = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hrt')])

            temp_site = job[8]
            temp_site0 = canonical_site(env_, env_.nn_site(site0, (1, 1)))
            if temp_site0 is not None:
                env_.proj[temp_site0].hrt = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hrt')])
                temp_site = job[4]
                temp_site0 = canonical_site(env_, env_.nn_site(site0, (0, 1)))
                env_.proj[temp_site0].hrb = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hrb')])

            env_tmp = EnvCTM(env_.psi, init=None)
            env_tmp._update_env_(site0, env_, move='r')
            update_storage_(env_, env_tmp)

            newr = yastn.save_to_dict(env_[site0].r)
            newtr = yastn.save_to_dict(env_[site0].tr)
            newbr = yastn.save_to_dict(env_[site0].br)

        return [newl, newr, newtl, newtr, newbl, newbr]

def canonical_site(env, site):
    site_index = env.psi.geometry.site2index(site)
    if type(site_index) is tuple:
        return Site(*site_index)
    else:
        return env.psi.sites()[site_index]

def ParaUpdateCTM_(psi:Peps, env:EnvCTM, fid, sites, opts_svd_ctm, cfg, parallel_pool, move='t', proj_dict=None, sites_to_be_updated=None):

    # Build projectors
    # Only pick the needed peps tensor(s) and CTMRG tensors. We don't use 'h' and 'v' options to save memory, thus these two are not optimized. (But can be done anyway)

    jobs = []
    for site in sites:
        if move =='t':
            site_ = canonical_site(env, env.nn_site(site, 't'))
            env_part, site0, _, _ = SubWindow(psi, site_, fid, 0, 0, 2, 1, env,
                                              env_load_dict={(0, 0): ['l', 'tl', 't'], (0, 1): ['t', 'tr', 'r'],
                                                             (1, 0): ['l', 'bl', 'b'], (1, 1): ['b', 'br', 'r'],
                                                             (2, 0): ['b', 'bl', 'l'], (2, 1): ['r', 'br', 'r']})
        elif move =='l':
            site_ = canonical_site(env, env.nn_site(site, 'l'))
            env_part, site0, _, _ = SubWindow(psi, site_, fid, 0, 0, 1, 2, env,
                                              env_load_dict={(0, 0): ['l', 'tl', 't'], (0, 1): ['t', 'tr', 'r'], (0, 2): ['t', 'tr', 'r'],
                                                             (1, 0): ['l', 'bl', 'b'], (1, 1): ['b', 'br', 'r'], (1, 2): ['r', 'br', 'b']})
        elif move =='b':
            site_ = site
            env_part, site0, _, _ = SubWindow(psi, site_, fid, 1, 0, 1, 1, env,
                                              env_load_dict={(-1, 0):['l', 'tl', 't'], (-1, 1): ['t', 'tr', 'r'],
                                                             (0, 0): ['l', 'tl', 't'], (0, 1): ['t', 'tr', 'r'],
                                                             (1, 0): ['l', 'bl', 'b'], (1, 1): ['b', 'br', 'r']})
        elif move =='r':
            site_ = site
            env_part, site0, _, _ = SubWindow(psi, site_, fid, 0, 1, 1, 1, env,
                                              env_load_dict={(0, -1):['l', 'tl', 't'], (0, 0): ['l', 'tl', 't'], (0, 1):['t', 'tr', 'r'],
                                                             (1, -1):['b', 'bl', 'l'], (1, 0):['l', 'bl', 'b'], (1, 1):['b', 'br', 'r']})


        jobs.append([env_part.save_to_dict(), site0, site_])

    gathered_result_ = parallel_pool(BuildProjector_(job, move, opts_svd_ctm, cfg) for job in jobs)


    if proj_dict is None:
        proj_dict = {}

    for ii in range(len(jobs)):
        site0 = jobs[ii][1]
        site = jobs[ii][2]
        dx = -site0.nx + site.nx
        dy = -site0.ny + site.ny
        for key in gathered_result_[ii].keys():
            key0 = key[0]
            key1 = key[1]
            key0_new = canonical_site(env, Site(key0.nx + dx, key0.ny + dy))
            proj_dict[(key0_new, key1)] = gathered_result_[ii][key]

    if sites_to_be_updated is None:
        sites_to_be_updated=sites

    jobs.clear()

    for site in sites_to_be_updated:
        if move =='t':
            env_part, site0, _, _ = SubWindow(psi, site, fid, 1, 1, 0, 1, env, site_load = [(-1, 0)], env_load_dict={(-1, 0):['t', 'tl', 'tr', 'l', 'r']})
        elif move =='l':
            env_part, site0, _, _ = SubWindow(psi, site, fid, 1, 1, 1, 0, env, site_load = [(0, -1)], env_load_dict={(0, -1):['l', 'tl', 'bl', 't', 'b']})
        elif move =='b':
            env_part, site0, _, _ = SubWindow(psi, site, fid, 0, 1, 1, 1, env, site_load = [(1, 0)], env_load_dict={(1, 0):['b', 'bl', 'br', 'l', 'r']})
        elif move =='r':
            env_part, site0, _, _ = SubWindow(psi, site, fid, 1, 0, 1, 1, env, site_load = [(0, 1)], env_load_dict={(0, 1):['r', 'tr', 'br', 't', 'b']})

        if move in 'htb':
            jobs.append([env_part.save_to_dict(), site0, site,
                         canonical_site(env, env.nn_site(site, (-1, 0))),
                         canonical_site(env, env.nn_site(site, (1, 0))),
                         canonical_site(env, env.nn_site(site, (-1, -1))),
                         canonical_site(env, env.nn_site(site, (-1, 1))),
                         canonical_site(env, env.nn_site(site, (1, -1))),
                         canonical_site(env, env.nn_site(site, (1, 1)))])
        elif move in 'vlr':
            jobs.append([env_part.save_to_dict(), site0, site,
                         canonical_site(env, env.nn_site(site, (0, -1))),
                         canonical_site(env, env.nn_site(site, (0, 1))),
                         canonical_site(env, env.nn_site(site, (-1, -1))),
                         canonical_site(env, env.nn_site(site, (-1, 1))),
                         canonical_site(env, env.nn_site(site, (1, -1))),
                         canonical_site(env, env.nn_site(site, (1, 1)))])

    updated_ctm_tensors = parallel_pool(UpdateSite(job, cfg, move, proj_dict) for job in jobs)
    proj_dict.clear()

    ii = 0
    for result in updated_ctm_tensors:
        site_ = sites_to_be_updated[ii]
        if move in "ht":
            env[site_].t = yastn.load_from_dict(config=cfg, d = result[0])
        if move in "hb":
            env[site_].b = yastn.load_from_dict(config=cfg, d = result[1])
        if move in "vl":
            env[site_].l = yastn.load_from_dict(config=cfg, d = result[0])
        if move in "vr":
            env[site_].r = yastn.load_from_dict(config=cfg, d = result[1])
        if move in 'hvtl':
            env[site_].tl = yastn.load_from_dict(config=cfg, d = result[2])
        if move in 'hvtr':
            env[site_].tr = yastn.load_from_dict(config=cfg, d = result[3])
        if move in 'hvbl':
            env[site_].bl = yastn.load_from_dict(config=cfg, d = result[4])
        if move in 'hvbr':
            env[site_].br = yastn.load_from_dict(config=cfg, d = result[5])

        ii = ii + 1


def _ctmrg_(psi, env, fid, max_sweeps, iterator_step, corner_tol, opts_svd_ctm, cfg, n_cores=24, ctm_jobs_vh=None):

    if ctm_jobs_vh is None:
        ctm_jobs_ver, ctm_jobs_hor = CreateCTMJobBundle(env, n_cores)
    else:
        ctm_jobs_ver, ctm_jobs_hor = ctm_jobs_vh

    max_dsv, converged, history = None, False, []
    with Parallel(n_jobs = n_cores, verbose=0) as parallel_pool:
        for sweep in range(1, max_sweeps + 1):

            for move in 'tb':

                # if n_cores >= psi.geometry.Ny:
                for ctm_jobs in ctm_jobs_hor:
                    if ctm_jobs_vh is None:
                        ParaUpdateCTM_(psi, env, fid, ctm_jobs, opts_svd_ctm, cfg, parallel_pool=parallel_pool, move=move)
                    else:
                        ParaUpdateCTM_(psi, env, fid, ctm_jobs[0], opts_svd_ctm, cfg, parallel_pool=parallel_pool, move=move, sites_to_be_updated=ctm_jobs[1])

            for move in 'lr':

                # if n_cores >= psi.geometry.Nx:
                for ctm_jobs in ctm_jobs_ver:
                    if ctm_jobs_vh is None:
                        ParaUpdateCTM_(psi, env, fid, ctm_jobs, opts_svd_ctm, cfg, parallel_pool=parallel_pool, move=move)
                    else:
                        ParaUpdateCTM_(psi, env, fid, ctm_jobs[0], opts_svd_ctm, cfg, parallel_pool=parallel_pool, move=move, sites_to_be_updated=ctm_jobs[1])

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
    yield CTMRG_out(sweeps=sweep, max_dsv=max_dsv, converged=converged)

@delayed
def Measure1Site(job, op, cfg):
    env_dict, site0, site = job
    env = load_from_dict(cfg, env_dict)
    return {site: env.measure_1site(op, site=site0)}

@delayed
def MeasureNN(job, op0, op1, cfg):

    env_dict, bond0, bond = job
    env = load_from_dict(cfg, env_dict)
    return {bond: env.measure_nn(op0, op1, bond=bond0)}

def ParaMeasure1Site(psi, env, fid, op, cfg, n_cores = 24):

    list_of_dicts = []

    num_of_sites = len(psi.sites())
    with Parallel(n_jobs = n_cores) as parallel_pool:
        for ii in range(0, int(np.ceil(num_of_sites / n_cores))):
            jobs = []
            for site in psi.sites()[ii * n_cores:min((ii + 1) * n_cores, num_of_sites)]:
                env_part, site0, _, _ = SubWindow(psi, site, fid, 0, 0, 0, 0, env)
                jobs.append([env_part.save_to_dict(), site0, site])

            list_of_dicts += parallel_pool(Measure1Site(job, op, cfg) for job in jobs)
            jobs.clear()

    result = {k: v for d in list_of_dicts for k, v in d.items()}
    return result

def ParaMeasureNN(psi, env, fid, op0, op1, cfg, n_cores = 24):

    list_of_dicts = []

    num_of_hb = len(psi.bonds(dirn='h'))
    num_of_vb = len(psi.bonds(dirn='v'))

    with Parallel(n_jobs = n_cores) as parallel_pool:

        for ii in range(0, int(np.ceil(num_of_hb / n_cores))):
            jobs = []
            for bond in psi.bonds(dirn='h')[(ii * n_cores):(min((ii + 1) * n_cores, num_of_hb))]:
                # env_part, site0 = Window3x3(psi, env, bond.site0, fid)
                env_part, site0, _, _ = SubWindow(psi, bond.site0, fid, 0, 0, 0, 1, env, env_load_dict={(0, 0):['tl', 'bl', 'l', 't', 'b'], (0, 1):['tr', 'br', 'r', 't', 'b']})
                bond0 = Bond(site0, env_part.nn_site(site0, 'r'))
                jobs.append([env_part.save_to_dict(), bond0, bond])
            list_of_dicts += parallel_pool(MeasureNN(job, op0, op1, cfg) for job in jobs)
            jobs.clear()
        for ii in range(0, int(np.ceil(num_of_vb / n_cores))):
            jobs = []
            for bond in psi.bonds(dirn='v')[ii * n_cores:min((ii + 1) * n_cores, num_of_vb)]:
                env_part, site0, _, _ = SubWindow(psi, bond.site0, fid, 0, 0, 1, 0, env, env_load_dict={(0, 0):['tl', 'tr', 'l', 't', 'r'], (1, 0):['bl', 'br', 'r', 'l', 'b']})
                bond0 = Bond(site0, env_part.nn_site(site0, 'b'))
                jobs.append([env_part.save_to_dict(), bond0, bond])
            list_of_dicts += parallel_pool(MeasureNN(job, op0, op1, cfg) for job in jobs)
            jobs.clear()

    result = {k: v for d in list_of_dicts for k, v in d.items()}
    return result


def PARActmrg_(psi:Peps, env:EnvCTM, fid, cfg, max_sweeps=50, iterator_step=1, opts_svd_ctm=None, corner_tol=None, n_cores=1, ctm_jobs_vh=None):
    tmp = _ctmrg_(psi, env, fid, max_sweeps, iterator_step, corner_tol, opts_svd_ctm, cfg, n_cores=n_cores, ctm_jobs_vh=ctm_jobs_vh)
    return tmp if iterator_step else next(tmp)