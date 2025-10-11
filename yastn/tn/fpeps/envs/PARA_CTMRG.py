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
    if n_cores >= Ly:
        njobs_hor  = len(env.psi.bonds(dirn="h")) // Lx
        num_of_hor = max(int(np.floor(n_cores / Ly)), 1)
        for n_bundle in range(0, int(np.ceil(Lx / num_of_hor))):
            ctm_jobs_hor.append([])
            for nrow in range(n_bundle * num_of_hor, min(Lx, (n_bundle + 1) * num_of_hor)):
                ctm_jobs_hor[len(ctm_jobs_hor) - 1] += [env.psi.bonds(dirn="h")[nrow + Lx * _ ] for _ in range(0, njobs_hor)]
                # ctm_jobs_hor.append([env.psi.bonds(dirn="h")[nrow + _ * njobs_hor] for _ in range(1, njobs_hor, 2)])
    else:
        njobs_hor  = len(env.psi.bonds(dirn="h")) // Lx
        num_of_hor = int(np.ceil(njobs_hor / n_cores))
        for nrow in range(Lx):
            for jj in range(num_of_hor):
                ctm_jobs_hor.append([])
                ctm_jobs_hor[len(ctm_jobs_hor) - 1] += [env.psi.bonds(dirn="h")[nrow + Lx * _ ]
                                                        for _ in range(jj * n_cores, min(njobs_hor, (jj + 1) * n_cores))]
                # ctm_jobs_hor.append([env.psi.bonds(dirn="h")[nrow + _ * njobs_hor] for _ in range(1, njobs_hor, 2)])


    ctm_jobs_ver = []
    if n_cores >= Lx:
        njobs_ver  = len(env.psi.bonds(dirn="v")) // Ly
        num_of_ver = max(int(np.floor(n_cores / Lx)), 1)
        for n_bundle in range(0, int(np.ceil(Ly / num_of_ver))):
            ctm_jobs_ver.append([])
            for ncol in range(n_bundle * num_of_ver, min(Ly, (n_bundle + 1) * num_of_ver)):
                ctm_jobs_ver[len(ctm_jobs_ver) - 1] += [env.psi.bonds(dirn="v")[ncol * njobs_ver + _] for _ in range(0, njobs_ver)]
            # ctm_jobs_ver.append([env.psi.bonds(dirn="v")[ncol * njobs_ver + _] for _ in range(1, njobs_ver, 2)])
    else:
        njobs_ver = len(env.psi.bonds(dirn="v")) // Ly
        num_of_ver = int(np.ceil(njobs_ver / n_cores))
        for ncol in range(Ly):
            for jj in range(num_of_ver):
                ctm_jobs_ver.append([])
                ctm_jobs_ver[len(ctm_jobs_ver) - 1] += [env.psi.bonds(dirn="v")[ncol * njobs_ver + _]
                                                        for _ in range(jj * n_cores, min(njobs_ver, (jj + 1) * n_cores))]

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
def UpdateSite(job, cfg, dirn, proj_dict):
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
            proj[temp_site0].vtr = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vtr')])
            temp_site = job[3]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (-1, 0)))
            proj[temp_site0].vtl = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vtl')])


        temp_site = job[6]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (-1, 1)))
        if temp_site0 is not None:
            proj[temp_site0].vtl = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vtl')])
            temp_site = job[3]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (-1, 0)))
            proj[temp_site0].vtr = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vtr')])

        temp_site = job[7]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, -1)))
        if temp_site0 is not None:
            proj[temp_site0].vbr = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vbr')])
            temp_site = job[4]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, 0)))
            proj[temp_site0].vbl = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vbl')])

        temp_site = job[8]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, 1)))
        if temp_site0 is not None:
            proj[temp_site0].vbl = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vbl')])
            temp_site = job[4]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, 0)))
            proj[temp_site0].vbr = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'vbr')])

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
            proj[temp_site0].hlb = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hlb')])
            temp_site = job[3]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (0, -1)))
            proj[temp_site0].hlt = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hlt')])

        temp_site = job[6]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (-1, 1)))
        if temp_site0 is not None:
            proj[temp_site0].hrb = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hrb')])
            temp_site = job[4]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (0, 1)))
            proj[temp_site0].hrt = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hrt')])

        temp_site = job[7]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, -1)))
        if temp_site0 is not None:
            proj[temp_site0].hlt = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hlt')])
            temp_site = job[3]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (0, -1)))
            proj[temp_site0].hlb = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hlb')])

        temp_site = job[8]
        temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, 1)))
        if temp_site0 is not None:
            proj[temp_site0].hrt = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hrt')])
            temp_site = job[4]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (0, 1)))
            proj[temp_site0].hrb = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hrb')])

        env_tmp = EnvCTM(env_.psi, init=None)
        update_env_(env_tmp, site0, env_, proj, move='h')
        update_storage_(env_, env_tmp)

        return [yastn.save_to_dict(env_[site0].l),
                yastn.save_to_dict(env_[site0].r),
                yastn.save_to_dict(env_[site0].tl),
                yastn.save_to_dict(env_[site0].tr),
                yastn.save_to_dict(env_[site0].bl),
                yastn.save_to_dict(env_[site0].br)]

def ParaUpdateCTM_(psi, env, fid, bonds, opts_svd_ctm, cfg, n_cores, parallel_pool, dirn='h', proj_dict=None, sites_to_be_updated=None):

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

    if proj_dict is None:
        proj_dict = {}

    for ii in range(len(jobs)):
        site0 = jobs[ii][1]
        bond = jobs[ii][3]
        dx = -site0.nx + bond.site0.nx
        dy = -site0.ny + bond.site0.ny
        for key in gathered_result_[ii].keys():
            key0 = key[0]
            key1 = key[1]
            key0_new = env.canonical_site(Site(key0.nx + dx, key0.ny + dy))
            proj_dict[(key0_new, key1)] = gathered_result_[ii][key]

    # Update CTMRG

    if sites_to_be_updated is None:
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

    gathered_result = parallel_pool(UpdateSite(job, cfg, dirn, proj_dict) for job in jobs)

    if dirn == 'h':
        for site in sites_to_be_updated[1:-1]:
            if env.nn_site(site, (-1, 1)) is not None:
                proj_dict.pop((env.canonical_site(env.nn_site(site, (-1, 0))), 'vtr'))
            if env.nn_site(site, (-1, -1)) is not None:
                proj_dict.pop((env.canonical_site(env.nn_site(site, (-1, 0))), 'vtl'))
            if env.nn_site(site, (1, 1)) is not None:
                proj_dict.pop((env.canonical_site(env.nn_site(site, (1, 0))), 'vbr'))
            if env.nn_site(site, (1, -1)) is not None:
                proj_dict.pop((env.canonical_site(env.nn_site(site, (1, 0))), 'vbl'))
    if dirn == 'v':
        for site in sites_to_be_updated[1:-1]:
            if env.nn_site(site, (-1, 1)) is not None:
                proj_dict.pop((env.canonical_site(env.nn_site(site, (0, 1))), 'hrt'))
            if env.nn_site(site, (1, 1)) is not None:
                proj_dict.pop((env.canonical_site(env.nn_site(site, (0, 1))), 'hrb'))
            if env.nn_site(site, (-1, -1)) is not None:
                proj_dict.pop((env.canonical_site(env.nn_site(site, (0, -1))), 'hlt'))
            if env.nn_site(site, (1, -1)) is not None:
                proj_dict.pop((env.canonical_site(env.nn_site(site, (0, -1))), 'hlb'))

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

            if n_cores >= psi.geometry.Ny:
                for ctm_jobs in ctm_jobs_hor:
                    ParaUpdateCTM_(psi, env, fid, ctm_jobs, opts_svd_ctm, cfg, n_cores=n_cores, parallel_pool=parallel_pool, dirn='h')
            else:
                for nrows in range(psi.geometry.Nx):
                    proj_dict = {}
                    updated_flag = [False for _ in range(psi.geometry.Ny)]
                    for irow in range(len(ctm_jobs_hor) // psi.geometry.Nx):
                        ctm_jobs = ctm_jobs_hor[nrows * (len(ctm_jobs_hor) // psi.geometry.Nx) + irow]

                        sites_to_be_updated = []
                        if len((proj_dict)) == 0:
                            for ny_ in range(ctm_jobs[0].site1.ny, ctm_jobs[-1].site0.ny + 1):
                                sites_to_be_updated.append(Site(ctm_jobs[0].site0.nx, ny_))
                                updated_flag[ny_] = True
                        else:
                            for ny_ in range(ctm_jobs[0].site0.ny, ctm_jobs[-1].site0.ny + 1):
                                sites_to_be_updated.append(Site(ctm_jobs[0].site0.nx, ny_))
                                updated_flag[ny_] = True

                        # the first and the last

                        if irow == (len(ctm_jobs_hor) // psi.geometry.Nx) - 1:
                            if not updated_flag[0]:
                                sites_to_be_updated.append(Site(nrows, 0))
                            if not updated_flag[-1]:
                                sites_to_be_updated.append(Site(nrows, psi.geometry.Ny - 1))
                        ParaUpdateCTM_(psi, env, fid, ctm_jobs, opts_svd_ctm, cfg, n_cores=n_cores, parallel_pool=parallel_pool, dirn='h', proj_dict=proj_dict, sites_to_be_updated=sites_to_be_updated)

            if n_cores >= psi.geometry.Ny:
                for ctm_jobs in ctm_jobs_ver:
                    ParaUpdateCTM_(psi, env, fid, ctm_jobs, opts_svd_ctm, cfg, n_cores=n_cores, parallel_pool=parallel_pool, dirn='v')
            else:
                for ncols in range(psi.geometry.Ny):
                    proj_dict = {}
                    updated_flag = [False for _ in range(psi.geometry.Nx)]
                    for icol in range(len(ctm_jobs_ver) // psi.geometry.Ny):
                        ctm_jobs = ctm_jobs_ver[ncols * (len(ctm_jobs_ver) // psi.geometry.Ny) + icol]

                        sites_to_be_updated = []
                        if len((proj_dict)) == 0:
                            for nx_ in range(ctm_jobs[0].site1.nx, ctm_jobs[-1].site0.nx + 1):
                                sites_to_be_updated.append(Site(nx_, ctm_jobs[0].site0.ny))
                                updated_flag[nx_] = True
                        else:
                            for nx_ in range(ctm_jobs[0].site0.nx, ctm_jobs[-1].site0.nx + 1):
                                sites_to_be_updated.append(Site(nx_, ctm_jobs[0].site0.ny))
                                updated_flag[nx_] = True

                        # the first and the last

                        if icol == (len(ctm_jobs_ver) // psi.geometry.Ny) - 1:
                            if not updated_flag[0]:
                                sites_to_be_updated.append(Site(0, ncols))
                            if not updated_flag[-1]:
                                sites_to_be_updated.append(Site(psi.geometry.Nx - 1, ncols))
                        ParaUpdateCTM_(psi, env, fid, ctm_jobs, opts_svd_ctm, cfg, n_cores=n_cores, parallel_pool=parallel_pool, dirn='v', proj_dict=proj_dict, sites_to_be_updated=sites_to_be_updated)



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
                env_part, site0, _, _ = SubWindow(psi, bond.site0, fid, 0, 0, 0, 1, env)
                bond0 = Bond(site0, env_part.nn_site(site0, 'r'))
                jobs.append([env_part.save_to_dict(), bond0, bond])
            list_of_dicts += parallel_pool(MeasureNN(job, op0, op1, cfg) for job in jobs)
            jobs.clear()
        for ii in range(0, int(np.ceil(num_of_vb / n_cores))):
            jobs = []
            for bond in psi.bonds(dirn='v')[ii * n_cores:min((ii + 1) * n_cores, num_of_vb)]:
                env_part, site0, _, _ = SubWindow(psi, bond.site0, fid, 0, 0, 1, 0, env)
                bond0 = Bond(site0, env_part.nn_site(site0, 'b'))
                jobs.append([env_part.save_to_dict(), bond0, bond])
            list_of_dicts += parallel_pool(MeasureNN(job, op0, op1, cfg) for job in jobs)
            jobs.clear()

    result = {k: v for d in list_of_dicts for k, v in d.items()}
    return result


def PARActmrg_(psi, env, fid, cfg, max_sweeps=50, iterator_step=1, opts_svd_ctm=None, corner_tol=None, n_cores=1):
    tmp = _ctmrg_(psi, env, fid, max_sweeps, iterator_step, corner_tol, opts_svd_ctm, cfg, n_cores=n_cores)
    return tmp if iterator_step else next(tmp)
    # ParaUpdate(env, ctm_jobs_ver, ctm_jobs_hor, opts_svd_ctm, cfg)
