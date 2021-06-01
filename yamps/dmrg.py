"""
various variants of the DMRG algorithm for mps
"""

from yast import linalg
import logging
import numpy as np
from ._env3 import Env3


logger = logging.getLogger('yast.mps.dmrg')


class FatalError(Exception):
    pass

#################################
#           dmrg                #
#################################


def dmrg_OBC(psi, H, env=None, version='1site', cutoff_sweep=1, cutoff_dE=-1, hermitian=True, k=4, eigs_tol=1e-14, opts_svd=None, SV_min=None, D_totals=None, tol_svds=None, versions=('1site', '2site'), algorithm='arnoldi'):
    r"""
    Perform dmrg on system with open boundary conditions, updating initial state psi.  Assume input psi is right canonical.

    Parameters
    ----------
    psi: Mps, nr_phys=1
        Initial state.
    H: Mps, nr_phys=2
        Operator given in MPO decomposition.
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]
    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set up with respect to the last site.
    nor: Env2
        default = None
        nor - dummy - is not used in DMRG
        Initial overlap <psi|psi>
        Initial environments must be set up with respect to the last site.
    measure_O: list of lists, each of format [n, operator, list_of_ids]
        default = None
        n: int
            1(for single site operator) or 2(for two-site operator)
        operator: Tensor or list of Tensors
            if n==1: Tensor
            if n==2: list of format [Tensor, Tensor]
        list_of_ids: list
            List  of sites which you want to measure.
        e.g.  [2, [OL, OR], [1, 2, 3]], measure expectation value of 2-site operator OL-OR on sites (1, 2), (2, 3), (3, 4)
    version: string
        default = '1site'
        Version of dmrg to use. Options: 0site, 1site, 2site, 2site_group.
    cutoff_sweep: int
        default=20
        Upper bound for number of sweeps.
    cutoff_dE: double
        default=1e-9
        Target convergence on the H expectation value.
    dtype: str
        default='complex128'
        Type of Tensor.
    hermitian: bool
        default=True
        Is MPO hermitian
    k: int
        default=4
        Dimension of Krylov subspace for linalg.eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for linalg.eigs(.)
    opts_svd: dict
        default=None
        options for truncation

    Returns
    -------
    env: Env3
        Overlap <psi| H |psi> as Env3.
    E: double
        Final expectation value of H
    dE: double
        Final variation on <H>
    out: list
        list of measured expectation values. Set of operators provided by measure_O
    """
    E0 = 0
    dE = cutoff_dE + 1
    sweep = 0
    while sweep < cutoff_sweep and dE > cutoff_dE:
        if version == '0site':
            env = dmrg_sweep_0site(psi, H=H, env=env, k=k,
                                   hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd)
        elif version == '1site':
            env = dmrg_sweep_1site(psi, H=H, env=env, k=k,
                                   hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd)
        elif version == '2site':
            env = dmrg_sweep_2site(psi, H=H, env=env, k=k,
                                   hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd)
        elif version == '2site_group':
            env = dmrg_sweep_2site_group(
                psi, H=H, env=env, k=k, hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd)
        else:  # mix
            env = dmrg_sweep_mix(psi=psi, SV_min=SV_min, versions=versions, H=H, env=env, hermitian=hermitian, k=k,
                                 eigs_tol=eigs_tol, D_totals=D_totals, tol_svds=tol_svds, opts_svd=opts_svd)
        E = env.measure()
        dE = abs(E - E0)
        print('Iteration: ', sweep, ' energy: ', E,
              ' dE: ', dE, ' D: ', max(psi.get_D()))
        E0 = E
        sweep += 1
    return env, E, dE


def dmrg_sweep_0site(psi, H, env=None, hermitian=True, k=4, eigs_tol=1e-14, opts_svd={}, algorithm='arnoldi'):
    r"""
    Perform sweep with 0site-DMRG where update is made on the central site.
    Assume input psi is right canonical.
    Sweep consists of iterative updates from last site to first and back to the first one.

    Parameters
    ----------
    psi: Mps, nr_phys=1
        Initial state.
    H: Mps, nr_phys=2
        Operator given in MPO decomposition.
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]
    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set up with respect to the last site.
    dtype: str
        default='complex128'
        Type of Tensor.
    hermitian: bool
        default=True
        Is MPO hermitian
    k: int
        default=4
        Dimension of Krylov subspace for linalg.eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for linalg.eigs(.)
    opts_svd: dict
        default=None
        options for truncation

    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    psi: Mps
        Is self updated.
    """
    if opts_svd:
        logger.warning("dmrg_sweep_0site: Truncation not implemeted.")
    if not env:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')

    for n in psi.sweep(to='last'):  # sweep from fist to last
        psi.orthogonalize_site(n, to='last')
        env.clear_site(n)
        env.update(n, to='last')
        if n != psi.sweep(to='last')[-1]:
            init = psi.A[psi.pC]
            # update site n using eigs
            Av = lambda v: env.Heff0(v, psi.pC)
            _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
            psi.A[psi.pC] = vec[0]
        psi.absorb_central(to='last')

    for n in psi.sweep(to='first'):
        psi.orthogonalize_site(n, to='first')
        env.clear_site(n)
        env.update(n, to='first')
        if n != psi.sweep(to='first')[-1]:
            init = psi.A[psi.pC]
            # update site n using eigs
            Av = lambda v: env.Heff0(v, psi.pC)
            _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
            # canonize and save
            psi.A[psi.pC] = vec[0]
        psi.absorb_central(to='first')
    return env


def dmrg_sweep_1site(psi, H, env=None, hermitian=True, k=4, eigs_tol=1e-14, opts_svd={}, algorithm='arnoldi'):
    r"""
    Perform sweep with 1ite-DMRG.
    Assume input psi is right canonical.
    Sweep consists of iterative updates from last site to first and back to the first one.

    Parameters
    ----------
    psi: Mps, nr_phys=1
        Initial state.
    H: Mps, nr_phys=2
        Operator given in MPO decomposition.
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]
    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set up with respect to the last site.
    dtype: str
        default='complex128'
        Type of Tensor.
    hermitian: bool
        default=True
        Is MPO hermitian
    k: int
        default=4
        Dimension of Krylov subspace for linalg.eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for linalg.eigs(.)
    opts_svd: dict
        default=None
        options for truncation

    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    psi: Mps
        Is self updated.
    """
    if opts_svd:
        logger.warning("dmrg_sweep_1site: Truncation not implemeted.")
    if not env:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')

    for n in psi.sweep(to='last'):  # sweep from fist to last
        psi.absorb_central(to='last')
        init = psi.A[n]
        Av = lambda v: env.Heff1(v, n)
        _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
        # canonize and save
        psi.A[n] = vec[0]
        psi.orthogonalize_site(n, to='last')
        env.clear_site(n)
        env.update(n, to='last')

    for n in psi.sweep(to='first'):
        psi.absorb_central(to='first')
        init = psi.A[n]
        Av = lambda v: env.Heff1(v, n)
        _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
        # canonize and save
        psi.A[n] = vec[0]
        psi.orthogonalize_site(n, to='first')
        env.clear_site(n)
        env.update(n, to='first')

    return env


def dmrg_sweep_2site(psi, H, env=None, hermitian=True, k=4, eigs_tol=1e-14, opts_svd={}, algorithm='arnoldi'):
    r"""
    Perform sweep with 2site-DMRG.
    Assume input psi is right canonical.
    Sweep consists of iterative updates from last site to first and back to the first one.

    Parameters
    ----------
    psi: Mps, nr_phys=1
        Initial state.
    H: Mps, nr_phys=2
        Operator given in MPO decomposition.
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]
    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set up with respect to the last site.
    dtype: str
        default='complex128'
        Type of Tensor.
    hermitian: bool
        default=True
        Is MPO hermitian
    k: int
        default=4
        Dimension of Krylov subspace for linalg.eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for linalg.eigs(.)
    opts_svd: dict
        default=None
        options for truncation

    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    psi: Mps
        Is self updated.
    """

    if not env:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')

    for n in psi.sweep(to='last', dl=1):
        n1, _, _ = psi.g.from_site(n, to='last')
        init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
        # update site n using eigs
        Av = lambda v: env.Heff2(v, n)
        _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
        # split and save
        x, S, y = linalg.svd(vec[0], axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
        psi.A[n] = x
        psi.A[n1] = S.tensordot(y, axes=(1, 0))
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n, to='last')

    for n in psi.sweep(to='first', dl=1):
        n1, _, _ = psi.g.from_site(n, to='last')
        init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
        # update site n using eigs
        Av = lambda v: env.Heff2(v, n)
        _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
        # split and save
        x, S, y = linalg.svd(vec[0], axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
        psi.A[n] = x.tensordot(S, axes=(2, 0))
        psi.A[n1] = y
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n1, to='first')
    env.update(n, to='first')

    return env  # can be used in the next sweep


def dmrg_sweep_2site_group(psi, H, env=None, hermitian=True, k=4, eigs_tol=1e-14, opts_svd={}, algorithm='arnoldi'):
    r"""
    Perform sweep of two-site DMRG with groupping neigbouring sites.
    Assume input psi is right canonical.
    Sweep consists of iterative updates from last site to first and back to the first one.

    Parameters
    ----------
    psi: Mps, nr_phys=1
        Initial state.
    H: Mps, nr_phys=2
        Operator given in MPO decomposition.
        Legs are [left-virtual, ket-physical, bra-physical, right-virtual]
    env: Env3
        default = None
        Initial overlap <psi| H |psi>
        Initial environments must be set up with respect to the last site.
    dtype: str
        default='complex128'
        Type of Tensor.
    hermitian: bool
        default=True
        Is MPO hermitian
    k: int
        default=4
        Dimension of Krylov subspace for linalg.eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for linalg.eigs(.)
    opts_svd: dict
        default=None
        options for truncation

    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    psi: Mps
        Is self updated.
    """

    if not env:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')

    for n in psi.sweep(to='last', dl=1):
        n1, _, _ = psi.g.from_site(n, to='last')
        init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
        init.fuse_legs(axes=(0, (1, 2), 3), inplace=True)
        # init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
        # update site n using eigs
        Av = lambda v: env.Heff2_group(v, n)
        _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
        init = vec[0].unfuse_legs(axes=1, inplace=True)
        # init = vec[0].ungroup_leg(axis=1, leg_order=leg_order)
        # split and save
        x, S, y = linalg.svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
        psi.A[n] = x
        psi.A[n1] = S.tensordot(y, axes=(1, 0))
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n, to='last')

    for n in psi.sweep(to='first', dl=1):
        n1, _, _ = psi.g.from_site(n, to='last')
        init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
        init.fuse_legs(axes=(0, (1, 2), 3), inplace=True)
        #init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
        # update site n using eigs
        Av = lambda v: env.Heff2_group(v, n)
        _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
        init = vec[0].unfuse_legs(axes=1, inplace=True)
        # init = vec[0].ungroup_leg(axis=1, leg_order=leg_order)
        # split and save
        x, S, y = linalg.svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
        psi.A[n] = x.tensordot(S, axes=(2, 0))
        psi.A[n1] = y
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n1, to='first')
    env.update(n, to='first')

    return env  # can be used in the next sweep


def dmrg_sweep_mix(psi, SV_min, versions, H, env=None, hermitian=True, k=4, eigs_tol=1e-14, bi_orth=True, NA=None, D_totals=None, tol_svds=None, opts_svd=None, algorithm='arnoldi'):
    r"""
    Perform mixed 1site-2site sweep of DMRG basing on SV_min (smallest Schmidt value on the bond).
    Assume input psi is right canonical.
    Sweep consists of iterative updates from last site to first and back to the first one.

    Parameters
    ----------
    psi: Mps, nr_phys=1
        initial state.
    H: Mps, nr_phys=2
        operator given in MPO decomposition.
        legs are [left-virtual, ket-physical, bra-physical, right-virtual]
    env: Env3
        default = None
        initial overlap <psi| H |psi>
        initial environments must be set up with respect to the last site.
    SV_min: list
        list of minimal Schmidt values on each bond
    D_totals: list
        list of upper bound on bond dimension
    tol_svds: list
        list of lower bound on Schmidt values.
    versions: 2-element tuple
        version = (else, version to increase bond_dimension)
        tuple with what algorithm to use during a sweep. Algorithm is chosen basing on Schmidt values.
    hermitian: bool
        default = True
        is MPO hermitian
    k: int
        default = 4
        Dimension of Krylov subspace for linalg.eigs(.)
    eigs_tol: float
        default = 1e-14
        Cutoff for krylov subspace for linalg.eigs(.)
    bi_orth: bool
        default = True
        Option for exponentiation = exp(). For True and non-Hermitian cases will bi-orthogonalize set of generated vectors.
    NA: bool
        default = None
        The cost of matrix-vector multiplication used to optimize Krylov subspace and time intervals.
        Option for exponentiation = exp().
    opts_svd: dict
        default=None
        options for truncation on virtual d.o.f.

    Returns
    -------
    env: Env3
     Overlap <psi| H |psi> as Env3.
    psi: Mps
        Is self updated.
    """
    Ds = psi.get_D()
    if not D_totals:
        D_totals = [None]*(psi.N+1)
        max_vdim = 1
        for n in range(psi.N):
            D_totals[n] = min([max_vdim, opts_svd['D_total']])
            max_vdim = D_totals[n] * np.prod(psi.A[n].get_shape(psi.phys))
        max_vdim = 1
        for n in range(psi.N-1,-1,-1):
            max_vdim *= np.prod(psi.A[n].get_shape(psi.phys))
            D_totals[n] = min([D_totals[n], max_vdim, opts_svd['D_total']])
            max_vdim = D_totals[n]
        D_totals[-1] = 1
    if not tol_svds:
        tol_svds = [opts_svd['tol'] for n in Ds]
    #
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')
    for n in psi.sweep(to='last'):
        opts_svd['D_total'] = D_totals[n]
        opts_svd['tol'] = tol_svds[n]
        #
        if (SV_min[n] > tol_svds[n] and Ds[n] < D_totals[n]) or (Ds[n] > D_totals[n]):  # choose 2site
            version = versions[1]
        else:  # choose 1site
            version = versions[0]
        #
        #print(n, version, SV_min[n], tol_svds[n], Ds[n], D_totals[n])
        if version == '0site':
            psi.orthogonalize_site(n, to='last')
            env.clear_site(n)
            env.update(n, to='last')
            if n != psi.sweep(to='last')[-1]:
                init = psi.A[psi.pC]
                # update site n using eigs
                Av = lambda v: env.Heff0(v, psi.pC)
                _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
                # canonize and save
                psi.A[psi.pC] = vec[0]
            psi.absorb_central(to='last')
        elif version == '1site':
            init = psi.A[n]
            Av = lambda v: env.Heff1(v, n)
            _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
            # canonize and save
            psi.A[n] = vec[0]
            psi.orthogonalize_site(n, to='last')
            env.clear_site(n)
            env.update(n, to='last')
            psi.absorb_central(to='last')
        elif version == '2site':
            if n == psi.sweep(to='last')[-1]:
                env.clear_site(n)
                env.update(n, to='first')
            else:
                n1, _, _ = psi.g.from_site(n, to='last')
                init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
                # update site n using eigs
                Av = lambda v: env.Heff2(v, n)
                _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
                # split and save
                x, S, y = linalg.svd(vec[0], axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                psi.A[n] = x
                psi.A[n1] = S.tensordot(y, axes=(1, 0))
                env.clear_site(n)
                env.clear_site(n1)
                env.update(n, to='last')
        elif version == '2site_group':
            if n == psi.sweep(to='last')[-1]:
                env.clear_site(n)
                env.update(n, to='first')
            else:
                n1, _, _ = psi.g.from_site(n, to='last')
                init = psi.A[n].tensordot(psi.A[n1], axes=(psi.right, psi.left))
                init.fuse_legs(axes=(0, (1, 2), 3), inplace=True)
                # init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
                # update site n using eigs
                Av = lambda v: env.Heff2_group(v, n)
                _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
                init = vec[0].unfuse_legs(axes=1, inplace=True)
                # init = vec[0].ungroup_leg(axis=1, leg_order=leg_order)
                # split and save
                x, S, y = linalg.svd(init, axes=(psi.left + psi.phys, tuple(
                    a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                psi.A[n] = x
                psi.A[n1] = S.tensordot(y, axes=(1, 0))
                env.clear_site(n)
                env.clear_site(n1)
                env.update(n, to='last')
    Ds = psi.get_D()
    for n in psi.sweep(to='first'):
        opts_svd['D_total'] = D_totals[n]
        opts_svd['tol'] = tol_svds[n]
        #
        if (SV_min[n] > tol_svds[n] and Ds[n] < D_totals[n]) or (Ds[n] > D_totals[n]):  # choose 2site
            version = versions[1]
        else:  # choose 1site
            version = versions[0]
        #print(n, version, SV_min[n], tol_svds[n], Ds[n], D_totals[n])
        #
        if version == '0site':
            psi.orthogonalize_site(n, to='first')
            env.clear_site(n)
            env.update(n, to='first')
            if n != psi.sweep(to='first')[-1]:
                init = psi.A[psi.pC]
                # update site n using eigs
                Av = lambda v: env.Heff0(v, psi.pC)
                _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
                # canonize and save
                psi.A[psi.pC] = vec[0]
            psi.absorb_central(to='first')
        elif version == '1site':
            init = psi.A[n]
            Av = lambda v: env.Heff1(v, n)
            _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
            # canonize and save
            psi.A[n] = vec[0]
            psi.orthogonalize_site(n, to='first')
            env.clear_site(n)
            env.update(n, to='first')
            psi.absorb_central(to='first')
        elif version == '2site':
            if n == psi.sweep(to='first')[-1]:
                env.clear_site(n)
                env.update(n, to='first')
            else:
                n1, _, _ = psi.g.from_site(n, to='first')
                init = psi.A[n1].tensordot(psi.A[n], axes=(psi.right, psi.left))
                # update site n using eigs
                Av = lambda v: env.Heff2(v, n)
                _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
                # split and save
                x, S, y = linalg.svd(vec[0], axes=(psi.left + psi.phys, tuple(
                    a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                psi.A[n1] = x.tensordot(S, axes=(2, 0))
                psi.A[n] = y
                env.clear_site(n)
                env.update(n, to='first')
                env.clear_site(n1)
        elif version == '2site_group':
            if n == psi.sweep(to='first')[-1]:
                env.clear_site(n)
                env.update(n, to='first')
            else:
                n1, _, _ = psi.g.from_site(n, to='first')
                init = psi.A[n1].tensordot(psi.A[n], axes=(psi.right, psi.left))
                init.fuse_legs(axes=(0, (1, 2), 3), inplace=True)
                # init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
                # update site n using eigs
                Av = lambda v: env.Heff2_group(v, n)
                _, vec = linalg.eigs(f=Av, v0=init, hermitian=hermitian, k=1, ncv=k, which='SR', tol=eigs_tol)
                init = vec[0].unfuse_legs(axes=1, inplace=True)
                # init = vec[0].ungroup_leg(axis=1, leg_order=leg_order)
                # split and save
                x, S, y = linalg.svd(init, axes=(psi.left + psi.phys, tuple(a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                psi.A[n1] = x.tensordot(S, axes=(2, 0))
                psi.A[n] = y
                env.clear_site(n)
                env.update(n, to='first')
                env.clear_site(n1)
    return env
