import yastn
import logging
import numpy as np

from joblib import Parallel, delayed
from ...fpeps import *
from ._env_ctm import CTMRG_out, EnvCTM_projectors, update_env_, EnvCTM, trivial_projectors_, update_storage_
from .._peps import Peps
from ...fpeps._geometry import Site, Bond


def CreateCTMJobBundle(env, n_cores=1):

    Lx = env.psi.geometry.Nx
    Ly = env.psi.geometry.Ny

    # ctm_jobs_hor = [[], []]
    ctm_jobs_hor = []
    njobs_hor  = len(env.psi.bonds(dirn="h")) // Lx
    num_of_hor = max(int(np.floor(n_cores / Ly)), 1)
    for n_bundle in range(0, int(np.ceil(Lx / num_of_hor))):
        ctm_jobs_hor.append([])
        for nrow in range(n_bundle * num_of_hor, min(Lx, (n_bundle + 1) * num_of_hor)):
            ctm_jobs_hor[len(ctm_jobs_hor) - 1] += [env.psi.bonds(dirn="h")[nrow + Lx * _ ] for _ in range(0, njobs_hor)]
            # ctm_jobs_hor.append([env.psi.bonds(dirn="h")[nrow + _ * njobs_hor] for _ in range(1, njobs_hor, 2)])

    ctm_jobs_ver = []
    njobs_ver  = len(env.psi.bonds(dirn="v")) // Ly
    num_of_ver = max(int(np.floor(n_cores / Lx)), 1)
    for n_bundle in range(0, int(np.ceil(Ly / num_of_ver))):
        ctm_jobs_ver.append([])
        for ncol in range(n_bundle * num_of_ver, min(Ly, (n_bundle + 1) * num_of_ver)):
            ctm_jobs_ver[len(ctm_jobs_ver) - 1] += [env.psi.bonds(dirn="v")[ncol * njobs_ver + _] for _ in range(0, njobs_ver)]
        # ctm_jobs_ver.append([env.psi.bonds(dirn="v")[ncol * njobs_ver + _] for _ in range(1, njobs_ver, 2)])

    return [ctm_jobs_ver, ctm_jobs_hor]

@delayed
def BuildProjector_(job, opts_svd_ctm, cfg):

    env_dict = job[0]
    bond = job[2]

    env = load_from_dict(config=cfg, d=env_dict)
    if opts_svd_ctm is None:
        opts_svd_ctm = env.opts_svd
    result = env.build_bond_projectors_(bond, opts_svd_ctm)
    return result

def Window3x3(psi, env, site, fid, only_site=False):

    '''
    Find the 3x3 window around the particular site. The boundary has been taken into consideration
    '''

    ds = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),  (0, 0),  (0, 1),
            (1, -1),  (1, 0),  (1, 1), ]
    flags = {}
    for d in ds:
        if psi.nn_site(site, d) is not None:
            flags[d] = True
        else:
            flags[d] = False

    nx0 = 1
    ny0 = 1
    Lx = 3
    Ly = 3
    if not flags[(0, -1)]:
        Ly = Ly - 1
        ny0 = ny0 - 1
    if not flags[(0, 1)]:
        Ly = Ly - 1
    if not flags[(-1, 0)]:
        Lx = Lx - 1
        nx0 = nx0 - 1
    if not flags[(1, 0)]:
        Lx = Lx - 1

    site0 = Site(nx0, ny0)

    if only_site:
        return site0

    net_part = SquareLattice((Lx, Ly), 'obc')
    psi_part = product_peps(net_part, fid)
    # site0 = Site(nx0, ny0)

    for d in ds:
        if psi.nn_site(site, d) is not None:
            psi_part[psi_part.nn_site(site0, d)] = psi[psi.nn_site(site, d)]

    env_part = EnvCTM(psi_part, init="eye")
    for d in ds:
        if psi.nn_site(site, d) is not None:
            env_part[psi_part.nn_site(site0, d)].l = env[psi.nn_site(site, d)].l
            env_part[psi_part.nn_site(site0, d)].r = env[psi.nn_site(site, d)].r
            env_part[psi_part.nn_site(site0, d)].t = env[psi.nn_site(site, d)].t
            env_part[psi_part.nn_site(site0, d)].b = env[psi.nn_site(site, d)].b

            env_part[psi_part.nn_site(site0, d)].tl = env[psi.nn_site(site, d)].tl
            env_part[psi_part.nn_site(site0, d)].bl = env[psi.nn_site(site, d)].bl
            env_part[psi_part.nn_site(site0, d)].tr = env[psi.nn_site(site, d)].tr
            env_part[psi_part.nn_site(site0, d)].br = env[psi.nn_site(site, d)].br


    return env_part, site0


def SubWindow(psi, site, fid, top=1, left=1, bottom=1, right=1, env=None, only_site=False):

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

    for dx in range(-top, bottom + 1):
        for dy in range(-left, right + 1):
            d = (dx, dy)
            # if psi.nn_site(site, d) is not None:
            psi_part[psi_part.nn_site(site0, d)] = psi[psi.nn_site(site, d)]

    if env is not None:

        env_part = EnvCTM(psi_part, init="eye")
        for dx in range(-top, bottom + 1):
            for dy in range(-left, right + 1):
                d = (dx, dy)
                # if psi.nn_site(site, d) is not None:
                env_part[psi_part.nn_site(site0, d)].l = env[psi.nn_site(site, d)].l
                env_part[psi_part.nn_site(site0, d)].r = env[psi.nn_site(site, d)].r
                env_part[psi_part.nn_site(site0, d)].t = env[psi.nn_site(site, d)].t
                env_part[psi_part.nn_site(site0, d)].b = env[psi.nn_site(site, d)].b

                env_part[psi_part.nn_site(site0, d)].tl = env[psi.nn_site(site, d)].tl
                env_part[psi_part.nn_site(site0, d)].bl = env[psi.nn_site(site, d)].bl
                env_part[psi_part.nn_site(site0, d)].tr = env[psi.nn_site(site, d)].tr
                env_part[psi_part.nn_site(site0, d)].br = env[psi.nn_site(site, d)].br


        return env_part, site0, Lx, Ly

    else:
        return psi_part, site0, Lx, Ly


@delayed
def UpdateSite(job, cfg, dirn, gathered_result, ):
    env_ = load_from_dict(config=cfg, d=job[0])
    site0 = job[1]
    proj = Peps(env_.geometry)
    for site_ in proj.sites():
        proj[site_] = EnvCTM_projectors()

    if dirn == 'h':

        trivial_projectors_(proj, 'v', env_, sites=env_.sites())

        temp_site = job[5]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (-1, -1)))
        if temp_site0 is not None:
            proj[temp_site0].vtr = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'vtr')])
            temp_site = job[3]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (-1, 0)))
            proj[temp_site0].vtl = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'vtl')])


        temp_site = job[6]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (-1, 1)))
        if temp_site0 is not None:
            proj[temp_site0].vtl = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'vtl')])
            temp_site = job[3]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (-1, 0)))
            proj[temp_site0].vtr = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'vtr')])

        temp_site = job[7]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, -1)))
        if temp_site0 is not None:
            proj[temp_site0].vbr = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'vbr')])
            temp_site = job[4]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, 0)))
            proj[temp_site0].vbl = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'vbl')])

        temp_site = job[8]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, 1)))
        if temp_site0 is not None:
            proj[temp_site0].vbl = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'vbl')])
            temp_site = job[4]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, 0)))
            proj[temp_site0].vbr = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'vbr')])

        env_tmp = EnvCTM(env_.psi, init=None)
        update_env_(env_tmp, site0, env_, proj, move='v')
        update_storage_(env_, env_tmp)

        return [yastn.save_to_dict(env_[site0].t),
                yastn.save_to_dict(env_[site0].b),
                yastn.save_to_dict(env_[site0].tl),
                yastn.save_to_dict(env_[site0].tr),
                yastn.save_to_dict(env_[site0].bl),
                yastn.save_to_dict(env_[site0].br)]

    if dirn == 'v':

        trivial_projectors_(proj, 'h', env_, sites=env_.sites())

        temp_site = job[5]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (-1, -1)))
        if temp_site0 is not None:
            proj[temp_site0].hlb = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'hlb')])
            temp_site = job[3]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (0, -1)))
            proj[temp_site0].hlt = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'hlt')])

        temp_site = job[6]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (-1, 1)))
        if temp_site0 is not None:
            proj[temp_site0].hrb = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'hrb')])
            temp_site = job[4]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (0, 1)))
            proj[temp_site0].hrt = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'hrt')])

        temp_site = job[7]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, -1)))
        if temp_site0 is not None:
            proj[temp_site0].hlt = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'hlt')])
            temp_site = job[3]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (0, -1)))
            proj[temp_site0].hlb = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'hlb')])

        temp_site = job[8]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, 1)))
        if temp_site0 is not None:
            proj[temp_site0].hrt = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'hrt')])
            temp_site = job[4]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (0, 1)))
            proj[temp_site0].hrb = yastn.load_from_dict(config=cfg, d=gathered_result[(temp_site, 'hrb')])

        env_tmp = EnvCTM(env_.psi, init=None)
        update_env_(env_tmp, site0, env_, proj, move='h')
        update_storage_(env_, env_tmp)

        return [yastn.save_to_dict(env_[site0].l),
                yastn.save_to_dict(env_[site0].r),
                yastn.save_to_dict(env_[site0].tl),
                yastn.save_to_dict(env_[site0].tr),
                yastn.save_to_dict(env_[site0].bl),
                yastn.save_to_dict(env_[site0].br)]

def ParaUpdateCTM_(psi, env, fid, bonds, opts_svd_ctm, cfg, n_cores, parallel_pool, dirn='h'):

    # Build projectors

    jobs = []
    for bond in bonds:
        env_part, site0 = Window3x3(psi, env, bond.site0, fid)
        if dirn == 'h':
            bond0 = Bond(site0, env_part.nn_site(site0, 'r'))
        if dirn == 'v':
            bond0 = Bond(site0, env_part.nn_site(site0, 'b'))
        jobs.append([env_part.save_to_dict(), site0, bond0, bond])

    gathered_result_ = []

    for ii in range(0, int(np.ceil(len(jobs) / n_cores))):
        gathered_result_ += parallel_pool(BuildProjector_(job, opts_svd_ctm, cfg) for job in jobs[(ii * n_cores):min(len(jobs), (ii + 1) * n_cores)])
    # gathered_result_ = [BuildProjector_(job, opts_svd_ctm, cfg) for job in jobs]

    gathered_result = {}
    for ii in range(len(jobs)):
        site0 = jobs[ii][1]
        bond = jobs[ii][3]
        dx = -site0.nx + bond.site0.nx
        dy = -site0.ny + bond.site0.ny
        for key in gathered_result_[ii].keys():
            key0 = key[0]
            key1 = key[1]
            key0_new = env.canonical_site(Site(key0.nx + dx, key0.ny + dy))
            gathered_result[(key0_new, key1)] = gathered_result_[ii][key]

    # Update CTMRG

    if dirn == "h":
        sites_to_be_updated = [Site(nx_, ny_) for ny_ in range(env.psi.Ny) for nx_ in range(bonds[0].site0.nx, bonds[-1].site0.nx + 1)]
    else:
        sites_to_be_updated = [Site(nx_, ny_) for nx_ in range(env.psi.Nx)  for ny_ in range(bonds[0].site0.ny, bonds[-1].site0.ny + 1)]

    jobs.clear()
    for site in sites_to_be_updated:
        env_part, site0 = Window3x3(psi, env, site, fid)
        if dirn == 'h':
            jobs.append([env_part.save_to_dict(), site0, site,
                         env.canonical_site(env.nn_site(site, (-1, 0))),
                         env.canonical_site(env.nn_site(site, (1, 0))),
                         env.canonical_site(env.nn_site(site, (-1, -1))),
                         env.canonical_site(env.nn_site(site, (-1, 1))),
                         env.canonical_site(env.nn_site(site, (1, -1))),
                         env.canonical_site(env.nn_site(site, (1, 1)))])
        elif dirn == 'v':
            jobs.append([env_part.save_to_dict(), site0, site,
                         env.canonical_site(env.nn_site(site, (0, -1))),
                         env.canonical_site(env.nn_site(site, (0, 1))),
                         env.canonical_site(env.nn_site(site, (-1, -1))),
                         env.canonical_site(env.nn_site(site, (-1, 1))),
                         env.canonical_site(env.nn_site(site, (1, -1))),
                         env.canonical_site(env.nn_site(site, (1, 1)))])

    gathered_result = parallel_pool(UpdateSite(job, cfg, dirn, gathered_result) for job in jobs)
    # gathered_result = [UpdateSite(job, cfg, dirn, gathered_result) for job in jobs]

    ii = 0
    for result in gathered_result:
        site_ = sites_to_be_updated[ii]
        if dirn == "h":
            env[site_].t = yastn.load_from_dict(config=cfg, d = result[0])
            env[site_].b = yastn.load_from_dict(config=cfg, d = result[1])
        if dirn == "v":
            env[site_].l = yastn.load_from_dict(config=cfg, d = result[0])
            env[site_].r = yastn.load_from_dict(config=cfg, d = result[1])
        env[site_].tl = yastn.load_from_dict(config=cfg, d = result[2])
        env[site_].tr = yastn.load_from_dict(config=cfg, d = result[3])
        env[site_].bl = yastn.load_from_dict(config=cfg, d = result[4])
        env[site_].br = yastn.load_from_dict(config=cfg, d = result[5])

        ii = ii + 1


def _ctmrg_(psi, env, fid, max_sweeps, iterator_step, corner_tol, opts_svd_ctm, cfg, n_cores=24):

    ctm_jobs_ver, ctm_jobs_hor = CreateCTMJobBundle(env, n_cores)
    max_dsv, converged, history = None, False, []
    with Parallel(n_jobs = n_cores, verbose=0) as parallel_pool:
        for sweep in range(1, max_sweeps + 1):
            for ctm_jobs in ctm_jobs_hor:
                ParaUpdateCTM_(psi, env, fid, ctm_jobs, opts_svd_ctm, cfg, n_cores=n_cores, parallel_pool=parallel_pool, dirn='h')
            for ctm_jobs in ctm_jobs_ver:
                ParaUpdateCTM_(psi, env, fid, ctm_jobs, opts_svd_ctm, cfg, n_cores=n_cores, parallel_pool=parallel_pool, dirn='v')


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

    jobs = []
    for site in psi.sites():
        env_part, site0 = Window3x3(psi, env, site, fid)
        jobs.append([env_part.save_to_dict(), site0, site])

    list_of_dicts =  Parallel(n_jobs = n_cores)(Measure1Site(job, op, cfg) for job in jobs)
    result = {k: v for d in list_of_dicts for k, v in d.items()}
    return result

def ParaMeasureNN(psi, env, fid, op0, op1, cfg, n_cores = 24):

    jobs = []
    for bond in psi.bonds(dirn='h'):
        env_part, site0 = Window3x3(psi, env, bond.site0, fid)
        bond0 = Bond(site0, env_part.nn_site(site0, 'r'))
        jobs.append([env_part.save_to_dict(), bond0, bond])
    list_of_dicts = Parallel(n_jobs = n_cores)(MeasureNN(job, op0, op1, cfg) for job in jobs)

    jobs.clear()
    for bond in psi.bonds(dirn='v'):
        env_part, site0 = Window3x3(psi, env, bond.site0, fid)
        bond0 = Bond(site0, env_part.nn_site(site0, 'b'))
        jobs.append([env_part.save_to_dict(), bond0, bond])
    list_of_dicts += Parallel(n_jobs = n_cores)(MeasureNN(job, op0, op1, cfg) for job in jobs)

    result = {k: v for d in list_of_dicts for k, v in d.items()}
    return result


def PARActmrg_(psi, env, fid, cfg, max_sweeps=50, iterator_step=1, opts_svd_ctm=None, corner_tol=None, n_cores=1):
    tmp = _ctmrg_(psi, env, fid, max_sweeps, iterator_step, corner_tol, opts_svd_ctm, cfg, n_cores=n_cores)
    return tmp if iterator_step else next(tmp)
    # ParaUpdate(env, ctm_jobs_ver, ctm_jobs_hor, opts_svd_ctm, cfg)
