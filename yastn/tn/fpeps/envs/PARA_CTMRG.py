import yastn
import logging
import numpy as np

from joblib import Parallel, delayed
from ...fpeps import *
from ._env_ctm import CTMRG_out, EnvCTM_projectors, update_env_, EnvCTM, trivial_projectors_, update_storage_
from .._peps import Peps
from ...fpeps._geometry import Site, Bond


def CreateCTMJobBundle(env:EnvCTM, n_cores=1):

    Lx = env.psi.geometry.Nx
    Ly = env.psi.geometry.Ny

    # ctm_jobs_hor = [[], []]
    ctm_jobs_hor = []
    bonds_h = [Bond(env.canonical_site(bond.site0), env.canonical_site(bond.site1))for bond in env.psi.bonds(dirn="h")]
    if Bond(env.canonical_site(Site(0, Ly - 1)), env.canonical_site(Site(0, 0))) in bonds_h:
        njobs_hor  = Ly
    else:
        njobs_hor = Ly - 1
    Lx_ = int(min(Lx, len(bonds_h) / njobs_hor))
    if n_cores >= Ly:
        num_of_hor = max(int(np.floor(n_cores / Ly)), 1)
        for n_bundle in range(0, int(np.ceil(Lx_ / num_of_hor))):
            ctm_jobs_hor.append([])
            for nrow in range(n_bundle * num_of_hor, min(Lx_, (n_bundle + 1) * num_of_hor)):
                ctm_jobs_hor[len(ctm_jobs_hor) - 1] += [bonds_h[nrow + Lx_ * _ ] for _ in range(0, njobs_hor)]
                # ctm_jobs_hor.append([env.psi.bonds(dirn="h")[nrow + _ * njobs_hor] for _ in range(1, njobs_hor, 2)])
    else:
        num_of_hor = int(np.ceil(njobs_hor / n_cores))
        for nrow in range(Lx_):
            for jj in range(num_of_hor):
                ctm_jobs_hor.append([])
                ctm_jobs_hor[len(ctm_jobs_hor) - 1] += [bonds_h[nrow + Lx_ * _ ]
                                                        for _ in range(jj * n_cores, min(njobs_hor, (jj + 1) * n_cores))]
                # ctm_jobs_hor.append([env.psi.bonds(dirn="h")[nrow + _ * njobs_hor] for _ in range(1, njobs_hor, 2)])


    ctm_jobs_ver = []
    bonds_v = [Bond(env.canonical_site(bond.site0), env.canonical_site(bond.site1)) for bond in env.psi.bonds(dirn="v")]
    if Bond(env.canonical_site(Site(Lx - 1, 0)), env.canonical_site(Site(0, 0))) in bonds_v:
        njobs_ver  = Lx
    else:
        njobs_ver  = Lx - 1
    Ly_ = int(min(Ly, len(bonds_v) / njobs_ver))
    if n_cores >= Lx:
        num_of_ver = max(int(np.floor(n_cores / Lx)), 1)
        for n_bundle in range(0, int(np.ceil(Ly_ / num_of_ver))):
            ctm_jobs_ver.append([])
            for ncol in range(n_bundle * num_of_ver, min(Ly_, (n_bundle + 1) * num_of_ver)):
                ctm_jobs_ver[len(ctm_jobs_ver) - 1] += [bonds_v[ncol * njobs_ver + _] for _ in range(0, njobs_ver)]
            # ctm_jobs_ver.append([env.psi.bonds(dirn="v")[ncol * njobs_ver + _] for _ in range(1, njobs_ver, 2)])
    else:
        num_of_ver = int(np.ceil(njobs_ver / n_cores))
        for ncol in range(Ly_):
            for jj in range(num_of_ver):
                ctm_jobs_ver.append([])
                ctm_jobs_ver[len(ctm_jobs_ver) - 1] += [bonds_v[ncol * njobs_ver + _]
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
    env_ = load_from_dict(config=cfg, d=job[0])
    site0 = job[1]
    proj = Peps(env_.geometry)
    for site_ in proj.sites():
        proj[site_] = EnvCTM_projectors()

    if dirn in 'htb':

        newt = None
        newb = None
        newtl = None
        newtr = None
        newbl = None
        newbr = None

        if dirn in 'ht':

            trivial_projectors_(proj, 't', env_, sites=env_.sites())

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

            env_tmp = EnvCTM(env_.psi, init=None)
            update_env_(env_tmp, site0, env_, proj, move='t')
            update_storage_(env_, env_tmp)

            newt = yastn.save_to_dict(env_[site0].t)
            newtl = yastn.save_to_dict(env_[site0].tl)
            newtr = yastn.save_to_dict(env_[site0].tr)


        if dirn in 'hb':

            trivial_projectors_(proj, 'b', env_, sites=env_.sites())

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
            update_env_(env_tmp, site0, env_, proj, move='b')
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

            trivial_projectors_(proj, 'l', env_, sites=env_.sites())

            temp_site = job[5]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (-1, -1)))
            if temp_site0 is not None:
                proj[temp_site0].hlb = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hlb')])
                temp_site = job[3]
                temp_site0 = env_.canonical_site(env_.nn_site(site0, (0, -1)))
                proj[temp_site0].hlt = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hlt')])

            temp_site = job[7]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, -1)))
            if temp_site0 is not None:
                proj[temp_site0].hlt = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hlt')])
                temp_site = job[3]
                temp_site0 = env_.canonical_site(env_.nn_site(site0, (0, -1)))
                proj[temp_site0].hlb = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hlb')])

            env_tmp = EnvCTM(env_.psi, init=None)
            update_env_(env_tmp, site0, env_, proj, move='l')
            update_storage_(env_, env_tmp)

            newl = yastn.save_to_dict(env_[site0].l)
            newtl = yastn.save_to_dict(env_[site0].tl)
            newbl = yastn.save_to_dict(env_[site0].bl)

        if dirn in 'vr':

            trivial_projectors_(proj, 'r', env_, sites=env_.sites())

            temp_site = job[6]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (-1, 1)))
            if temp_site0 is not None:
                proj[temp_site0].hrb = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hrb')])
                temp_site = job[4]
                temp_site0 = env_.canonical_site(env_.nn_site(site0, (0, 1)))
                proj[temp_site0].hrt = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hrt')])

            temp_site = job[8]
            temp_site0 = env_.canonical_site(env_.nn_site(site0, (1, 1)))
            if temp_site0 is not None:
                proj[temp_site0].hrt = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hrt')])
                temp_site = job[4]
                temp_site0 = env_.canonical_site(env_.nn_site(site0, (0, 1)))
                proj[temp_site0].hrb = yastn.load_from_dict(config=cfg, d=proj_dict[(temp_site, 'hrb')])

            env_tmp = EnvCTM(env_.psi, init=None)
            update_env_(env_tmp, site0, env_, proj, move='r')
            update_storage_(env_, env_tmp)

            newr = yastn.save_to_dict(env_[site0].r)
            newtr = yastn.save_to_dict(env_[site0].tr)
            newbr = yastn.save_to_dict(env_[site0].br)

        return [newl, newr, newtl, newtr, newbl, newbr]

def ParaUpdateCTM_(psi, env, fid, bonds, opts_svd_ctm, cfg, n_cores, parallel_pool, dirn='h', proj_dict=None, sites_to_be_updated=None):

    # Build projectors
    # Only pick the needed peps tensor(s) and CTMRG tensors. We don't use 'h' and 'v' options to save memory, thus these two are not optimized. (But can be done anyway)

    jobs = []
    for bond in bonds:
        if dirn in 'h':
            env_part, site0, _, _ = SubWindow(psi, bond.site0, fid, 1, 0, 1, 1, env)
        elif dirn in 'v':
            env_part, site0, _, _ = SubWindow(psi, bond.site0, fid, 0, 1, 1, 1, env)
        elif dirn =='t':
            env_part, site0, _, _ = SubWindow(psi, bond.site0, fid, 1, 0, 0, 1, env, env_load_dict={(0, 0):['bl', 'b', 'l'], (0, 1):['br', 'b', 'r'], (-1, 0):['tl', 't', 'l'], (-1, 1):['tr', 't', 'r']})
        elif dirn =='l':
            env_part, site0, _, _ = SubWindow(psi, bond.site0, fid, 0, 1, 1, 0, env, env_load_dict={(0, 0):['tr', 't', 'r'], (0, -1):['tl', 't', 'l'], (1, 0):['br', 'b', 'r'], (1, -1):['bl', 'b', 'l']})
        elif dirn =='b':
            env_part, site0, _, _ = SubWindow(psi, bond.site0, fid, 0, 0, 1, 1, env, env_load_dict={(0, 0):['tl', 't', 'l'], (0, 1):['tr', 't', 'r'], (1, 0):['bl', 'b', 'l'], (1, 1):['br', 'b', 'r']})
        elif dirn =='r':
            env_part, site0, _, _ = SubWindow(psi, bond.site0, fid, 0, 0, 1, 1, env, env_load_dict={(0, 0):['tl', 't', 'l'], (0, 1):['tr', 't', 'r'], (1, 0):['bl', 'b', 'l'], (1, 1):['br', 'b', 'r']})

        if dirn in 'htb':
            bond0 = Bond(site0, env_part.nn_site(site0, 'r'))
        if dirn in 'vlr':
            bond0 = Bond(site0, env_part.nn_site(site0, 'b'))
        jobs.append([env_part.save_to_dict(), site0, bond0, bond])

    gathered_result_ = []

    for ii in range(0, int(np.ceil(len(jobs) / n_cores))):
        gathered_result_ += parallel_pool(BuildProjector_(job, opts_svd_ctm, cfg) for job in jobs[(ii * n_cores):min(len(jobs), (ii + 1) * n_cores)])

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

    if sites_to_be_updated is None:
        sites_to_be_updated = []
        for bond in bonds:
            if not bond.site0 in sites_to_be_updated:
                sites_to_be_updated.append(bond.site0)
            if not bond.site1 in sites_to_be_updated:
                sites_to_be_updated.append(bond.site1)

    jobs.clear()

    for site in sites_to_be_updated:
        # Only pick the needed peps tensor(s) and CTMRG tensors
        if dirn in 'h':
            env_part, site0, _, _ = SubWindow(psi, site, fid, 1, 1, 1, 1, env, site_load=[(-1, 0), (1, 0)], env_load_dict={(-1, 0):['t', 'tl', 'tr', 'l', 'r'], (1, 0):['b', 'bl', 'br', 'l', 'r']})
        if dirn in 'v':
            env_part, site0, _, _ = SubWindow(psi, site, fid, 1, 1, 1, 1, env, site_load=[(0, -1), (0, -1)], env_load_dict={(0, -1):['l', 'tl', 'bl', 't', 'b'], (0, 1):['r', 'tr', 'br', 't', 'b']})
        elif dirn =='t':
            env_part, site0, _, _ = SubWindow(psi, site, fid, 1, 1, 0, 1, env, site_load = [(-1, 0)], env_load_dict={(-1, 0):['t', 'tl', 'tr', 'l', 'r']})
        elif dirn =='l':
            env_part, site0, _, _ = SubWindow(psi, site, fid, 1, 1, 1, 0, env, site_load = [(0, -1)], env_load_dict={(0, -1):['l', 'tl', 'bl', 't', 'b']})
        elif dirn =='b':
            env_part, site0, _, _ = SubWindow(psi, site, fid, 0, 1, 1, 1, env, site_load = [(1, 0)], env_load_dict={(1, 0):['b', 'bl', 'br', 'l', 'r']})
        elif dirn =='r':
            env_part, site0, _, _ = SubWindow(psi, site, fid, 1, 0, 1, 1, env, site_load = [(0, 1)], env_load_dict={(0, 1):['r', 'tr', 'br', 't', 'b']})

        if dirn in 'htb':
            jobs.append([env_part.save_to_dict(), site0, site,
                         env.canonical_site(env.nn_site(site, (-1, 0))),
                         env.canonical_site(env.nn_site(site, (1, 0))),
                         env.canonical_site(env.nn_site(site, (-1, -1))),
                         env.canonical_site(env.nn_site(site, (-1, 1))),
                         env.canonical_site(env.nn_site(site, (1, -1))),
                         env.canonical_site(env.nn_site(site, (1, 1)))])
        elif dirn in 'vlr':
            jobs.append([env_part.save_to_dict(), site0, site,
                         env.canonical_site(env.nn_site(site, (0, -1))),
                         env.canonical_site(env.nn_site(site, (0, 1))),
                         env.canonical_site(env.nn_site(site, (-1, -1))),
                         env.canonical_site(env.nn_site(site, (-1, 1))),
                         env.canonical_site(env.nn_site(site, (1, -1))),
                         env.canonical_site(env.nn_site(site, (1, 1)))])

    gathered_result = parallel_pool(UpdateSite(job, cfg, dirn, proj_dict) for job in jobs)

    if dirn in 'htb':
        for site in sites_to_be_updated[1:-1]:
            if dirn in 'ht':
                if env.nn_site(site, (-1, 1)) is not None:
                    proj_dict.pop((env.canonical_site(env.nn_site(site, (-1, 0))), 'vtr'))
                if env.nn_site(site, (-1, -1)) is not None:
                    proj_dict.pop((env.canonical_site(env.nn_site(site, (-1, 0))), 'vtl'))
            if dirn in 'hb':
                if env.nn_site(site, (1, 1)) is not None:
                    proj_dict.pop((env.canonical_site(env.nn_site(site, (1, 0))), 'vbr'))
                if env.nn_site(site, (1, -1)) is not None:
                    proj_dict.pop((env.canonical_site(env.nn_site(site, (1, 0))), 'vbl'))
    if dirn in 'vlr':
        for site in sites_to_be_updated[1:-1]:
            if dirn in 'vr':
                if env.nn_site(site, (-1, 1)) is not None:
                    proj_dict.pop((env.canonical_site(env.nn_site(site, (0, 1))), 'hrt'))
                if env.nn_site(site, (1, 1)) is not None:
                    proj_dict.pop((env.canonical_site(env.nn_site(site, (0, 1))), 'hrb'))
            if dirn in 'vl':
                if env.nn_site(site, (-1, -1)) is not None:
                    proj_dict.pop((env.canonical_site(env.nn_site(site, (0, -1))), 'hlt'))
                if env.nn_site(site, (1, -1)) is not None:
                    proj_dict.pop((env.canonical_site(env.nn_site(site, (0, -1))), 'hlb'))

    ii = 0
    for result in gathered_result:
        site_ = sites_to_be_updated[ii]
        if dirn in "ht":
            env[site_].t = yastn.load_from_dict(config=cfg, d = result[0])
        if dirn in "hb":
            env[site_].b = yastn.load_from_dict(config=cfg, d = result[1])
        if dirn in "vl":
            env[site_].l = yastn.load_from_dict(config=cfg, d = result[0])
        if dirn in "vr":
            env[site_].r = yastn.load_from_dict(config=cfg, d = result[1])
        if dirn in 'hvtl':
            env[site_].tl = yastn.load_from_dict(config=cfg, d = result[2])
        if dirn in 'hvtr':
            env[site_].tr = yastn.load_from_dict(config=cfg, d = result[3])
        if dirn in 'hvbl':
            env[site_].bl = yastn.load_from_dict(config=cfg, d = result[4])
        if dirn in 'hvbr':
            env[site_].br = yastn.load_from_dict(config=cfg, d = result[5])

        ii = ii + 1


def _ctmrg_(psi, env, fid, max_sweeps, iterator_step, corner_tol, opts_svd_ctm, cfg, n_cores=24):

    ctm_jobs_ver, ctm_jobs_hor = CreateCTMJobBundle(env, n_cores)
    max_dsv, converged, history = None, False, []
    with Parallel(n_jobs = n_cores, verbose=0) as parallel_pool:
        for sweep in range(1, max_sweeps + 1):

            for dirn in 'tb':

                if n_cores >= psi.geometry.Ny:
                    for ctm_jobs in ctm_jobs_hor:
                        ParaUpdateCTM_(psi, env, fid, ctm_jobs, opts_svd_ctm, cfg, n_cores=n_cores, parallel_pool=parallel_pool, dirn=dirn)
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
                            ParaUpdateCTM_(psi, env, fid, ctm_jobs, opts_svd_ctm, cfg, n_cores=n_cores, parallel_pool=parallel_pool, dirn=dirn, proj_dict=proj_dict, sites_to_be_updated=sites_to_be_updated)

            for dirn in 'lr':

                if n_cores >= psi.geometry.Nx:
                    for ctm_jobs in ctm_jobs_ver:
                        ParaUpdateCTM_(psi, env, fid, ctm_jobs, opts_svd_ctm, cfg, n_cores=n_cores, parallel_pool=parallel_pool, dirn=dirn)
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
                            ParaUpdateCTM_(psi, env, fid, ctm_jobs, opts_svd_ctm, cfg, n_cores=n_cores, parallel_pool=parallel_pool, dirn=dirn, proj_dict=proj_dict, sites_to_be_updated=sites_to_be_updated)



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


def PARActmrg_(psi, env, fid, cfg, max_sweeps=50, iterator_step=1, opts_svd_ctm=None, corner_tol=None, n_cores=1):
    tmp = _ctmrg_(psi, env, fid, max_sweeps, iterator_step, corner_tol, opts_svd_ctm, cfg, n_cores=n_cores)
    return tmp if iterator_step else next(tmp)