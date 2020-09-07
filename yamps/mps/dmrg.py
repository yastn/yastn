import numpy as np
import logging
from yamps.mps import Env3
from yamps.tensor import eigs


class FatalError(Exception):
    pass


logger = logging.getLogger('yamps.mps.dmrg')


#################################
#           dmrg                #
#################################


def dmrg_OBC(psi, H, env=None, version='1site', cutoff_sweep=1, cutoff_dE=-1, hermitian=True, k=4, eigs_tol=1e-14, opts_svd=None, SV_min=None, D_totals=None, tol_svds=None, versions=('1site', '2site'), algorithm='arnoldi'):
    r"""
    Perform dmrg on system with open boundary conditions. The version of dmrg update p[rovoded by version.
    Assume input psi is right canonical.

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
        e.g.  [2,[OL, OR], [1,2,3]], measure expectation value of 2-site operator OL-OR on sites (1,2), (2,3), (3,4)
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
        Dimension of Krylov subspace for eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
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
                                   hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd, algorithm=algorithm)
        elif version == '1site':
            env = dmrg_sweep_1site(psi, H=H, env=env, k=k,
                                   hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd, algorithm=algorithm)
        elif version == '2site':
            env = dmrg_sweep_2site(psi, H=H, env=env, k=k,
                                   hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd, algorithm=algorithm)
        elif version == '2site_group':
            env = dmrg_sweep_2site_group(
                psi, H=H, env=env, k=k, hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd, algorithm=algorithm)
        else:  # mix
            env = dmrg_sweep_mix(psi=psi, SV_min=SV_min, versions=versions, H=H, env=env, hermitian=hermitian, k=k,
                                 eigs_tol=eigs_tol, D_totals=D_totals, tol_svds=tol_svds, opts_svd=opts_svd, algorithm=algorithm)
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
        Dimension of Krylov subspace for eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
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
        env.setup_to_first()

    for n in psi.g.sweep(to='last'):  # sweep from fist to last
        psi.orthogonalize_site(n, towards=psi.g.last)
        env.clear_site(n)
        env.update(n, towards=psi.g.last)
        if n != psi.g.sweep(to='last')[-1]:
            init = psi.A[psi.pC]
            # update site n using eigs
            def Av(v): return env.Heff0(v, psi.pC)
            if algorithm == 'lanczos' and not hermitian:
                def Bv(v): return env.Heff0(v, psi.pC, conj=True)
            else:
                Bv = None
            _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[init], hermitian=hermitian, k=1,
                             sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
            # canonize and save
            psi.A[psi.pC] = vec[0]
        psi.absorb_central(towards=psi.g.last)

    for n in psi.g.sweep(to='first'):
        psi.orthogonalize_site(n, towards=psi.g.first)
        env.clear_site(n)
        env.update(n, towards=psi.g.first)
        if n != psi.g.sweep(to='first')[-1]:
            init = psi.A[psi.pC]
            # update site n using eigs
            def Av(v): return env.Heff0(v, psi.pC)
            if algorithm == 'lanczos' and not hermitian:
                def Bv(v): return env.Heff0(v, psi.pC, conj=True)
            else:
                Bv = None
            _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[init], hermitian=hermitian, k=1,
                             sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
            # canonize and save
            psi.A[psi.pC] = vec[0]
        psi.absorb_central(towards=psi.g.first)
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
        Dimension of Krylov subspace for eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
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
        env.setup_to_first()

    for n in psi.g.sweep(to='last'):  # sweep from fist to last
        psi.absorb_central(towards=psi.g.last)
        init = psi.A[n]
        def Av(v): return env.Heff1(v, n)
        if algorithm == 'lanczos' and not hermitian:
            def Bv(v): return env.Heff1(v, n, conj=True)
        else:
            Bv = None
        _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[init], hermitian=hermitian, k=1,
                         sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
        # canonize and save
        psi.A[n] = vec[0]
        psi.orthogonalize_site(n, towards=psi.g.last)
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

    for n in psi.g.sweep(to='first'):
        psi.absorb_central(towards=psi.g.first)
        init = psi.A[n]
        def Av(v): return env.Heff1(v, n)
        if algorithm == 'lanczos' and not hermitian:
            def Bv(v): return env.Heff1(v, n, conj=True)
        else:
            Bv = None
        _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[init], hermitian=hermitian, k=1,
                         sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
        # canonize and save
        psi.A[n] = vec[0]
        psi.orthogonalize_site(n, towards=psi.g.first)
        env.clear_site(n)
        env.update(n, towards=psi.g.first)

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
        Dimension of Krylov subspace for eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
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
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        # update site n using eigs
        def Av(v): return env.Heff2(v, n)
        if algorithm == 'lanczos' and not hermitian:
            def Bv(v): return env.Heff2(v, n, conj=True)
        else:
            Bv = None
        _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[init], hermitian=hermitian, k=1,
                         sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
        # split and save
        x, S, y = vec[0].split_svd(axes=(psi.left + psi.phys, tuple(
            a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
        psi.A[n] = x
        psi.A[n1] = y.dot_diag(S, axis=0)
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n, towards=psi.g.last)

    for n in psi.g.sweep(to='first', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        # update site n using eigs
        def Av(v): return env.Heff2(v, n)
        if algorithm == 'lanczos' and not hermitian:
            def Bv(v): return env.Heff2(v, n, conj=True)
        else:
            Bv = None
        _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[init], hermitian=hermitian, k=1,
                         sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
        # split and save
        x, S, y = vec[0].split_svd(axes=(psi.left + psi.phys, tuple(
            a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
        psi.A[n] = x.dot_diag(S, axis=2)
        psi.A[n1] = y
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n1, towards=psi.g.first)
    env.update(n, towards=psi.g.first)

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
        Dimension of Krylov subspace for eigs(.)
    eigs_tol: float
        default=1e-14
        Cutoff for krylov subspace for eigs(.)
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
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
        # update site n using eigs
        def Av(v): return env.Heff2_group(v, n)
        if algorithm == 'lanczos' and not hermitian:
            def Bv(v): return env.Heff2_group(v, n, conj=True)
        else:
            Bv = None
        _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[init], hermitian=hermitian, k=1,
                         sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
        init = vec[0].ungroup_leg(axis=1, leg_order=leg_order)
        # split and save
        x, S, y = init.split_svd(axes=(psi.left + psi.phys, tuple(
            a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
        psi.A[n] = x
        psi.A[n1] = y.dot_diag(S, axis=0)
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n, towards=psi.g.last)

    for n in psi.g.sweep(to='first', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
        # update site n using eigs
        def Av(v): return env.Heff2_group(v, n)
        if algorithm == 'lanczos' and not hermitian:
            def Bv(v): return env.Heff2_group(v, n, conj=True)
        else:
            Bv = None
        _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[init], hermitian=hermitian, k=1,
                         sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
        init = vec[0].ungroup_leg(axis=1, leg_order=leg_order)
        # split and save
        x, S, y = init.split_svd(axes=(psi.left + psi.phys, tuple(
            a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
        psi.A[n] = x.dot_diag(S, axis=2)
        psi.A[n1] = y
        env.clear_site(n)
        env.clear_site(n1)
        env.update(n1, towards=psi.g.first)
    env.update(n, towards=psi.g.first)

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
        Dimension of Krylov subspace for eigs(.)
    eigs_tol: float
        default = 1e-14
        Cutoff for krylov subspace for eigs(.)
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
            max_vdim = D_totals[n]
            _, lDs = psi.A[n].get_tD()
            leg_dim = [sum(xx) for xx in lDs]
            max_vdim *= np.prod([leg_dim[x] for x in psi.phys])
        max_vdim = 1
        for n in range(psi.N-1, -1, -1):
            _, lDs = psi.A[n].get_tD()
            leg_dim = [sum(xx) for xx in lDs]
            max_vdim *= np.prod([leg_dim[x] for x in psi.phys])
            D_totals[n] = min([D_totals[n], max_vdim, opts_svd['D_total']])
            max_vdim = D_totals[n]
        D_totals[-1] = 1
    if not tol_svds:
        tol_svds = [opts_svd['tol'] for n in Ds]
    #
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()
    for n in psi.g.sweep(to='last'):
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
            psi.orthogonalize_site(n, towards=psi.g.last)
            env.clear_site(n)
            env.update(n, towards=psi.g.last)
            if n != psi.g.sweep(to='last')[-1]:
                init = psi.A[psi.pC]
                # update site n using eigs
                def Av(v): return env.Heff0(v, psi.pC)
                if algorithm == 'lanczos' and not hermitian:
                    def Bv(v): return env.Heff0(v, psi.pC, conj=True)
                else:
                    Bv = None
                _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[
                                 init], hermitian=hermitian, k=1, sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
                # canonize and save
                psi.A[psi.pC] = vec[0]
            psi.absorb_central(towards=psi.g.last)
        elif version == '1site':
            init = psi.A[n]
            def Av(v): return env.Heff1(v, n)
            if algorithm == 'lanczos' and not hermitian:
                def Bv(v): return env.Heff1(v, n, conj=True)
            else:
                Bv = None
            _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[init], hermitian=hermitian, k=1,
                             sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
            # canonize and save
            psi.A[n] = vec[0]
            psi.orthogonalize_site(n, towards=psi.g.last)
            env.clear_site(n)
            env.update(n, towards=psi.g.last)
            psi.absorb_central(towards=psi.g.last)
        elif version == '2site':
            if n == psi.g.sweep(to='last')[-1]:
                env.clear_site(n)
                env.update(n, towards=psi.g.first)
            else:
                n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
                init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
                # update site n using eigs
                def Av(v): return env.Heff2(v, n)
                if algorithm == 'lanczos' and not hermitian:
                    def Bv(v): return env.Heff2(v, n, conj=True)
                else:
                    Bv = None
                _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[
                                 init], hermitian=hermitian, k=1, sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
                # split and save
                x, S, y = vec[0].split_svd(axes=(psi.left + psi.phys, tuple(
                    a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                psi.A[n] = x
                psi.A[n1] = y.dot_diag(S, axis=0)
                env.clear_site(n)
                env.clear_site(n1)
                env.update(n, towards=psi.g.last)
        elif version == '2site_group':
            if n == psi.g.sweep(to='last')[-1]:
                env.clear_site(n)
                env.update(n, towards=psi.g.first)
            else:
                n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
                init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
                init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
                # update site n using eigs
                def Av(v): return env.Heff2_group(v, n)
                if algorithm == 'lanczos' and not hermitian:
                    def Bv(v): return env.Heff2_group(v, n, conj=True)
                else:
                    Bv = None
                _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[
                                 init], hermitian=hermitian, k=1, sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
                init = vec[0].ungroup_leg(axis=1, leg_order=leg_order)
                # split and save
                x, S, y = init.split_svd(axes=(psi.left + psi.phys, tuple(
                    a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                psi.A[n] = x
                psi.A[n1] = y.dot_diag(S, axis=0)
                env.clear_site(n)
                env.clear_site(n1)
                env.update(n, towards=psi.g.last)
    Ds = psi.get_D()
    for n in psi.g.sweep(to='first'):
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
            psi.orthogonalize_site(n, towards=psi.g.first)
            env.clear_site(n)
            env.update(n, towards=psi.g.first)
            if n != psi.g.sweep(to='first')[-1]:
                init = psi.A[psi.pC]
                # update site n using eigs
                def Av(v): return env.Heff0(v, psi.pC)
                if algorithm == 'lanczos' and not hermitian:
                    def Bv(v): return env.Heff0(v, psi.pC, conj=True)
                else:
                    Bv = None
                _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[
                                 init], hermitian=hermitian, k=1, sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
                # canonize and save
                psi.A[psi.pC] = vec[0]
            psi.absorb_central(towards=psi.g.first)
        elif version == '1site':
            init = psi.A[n]
            def Av(v): return env.Heff1(v, n)
            if algorithm == 'lanczos' and not hermitian:
                def Bv(v): return env.Heff1(v, n, conj=True)
            else:
                Bv = None
            _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[init], hermitian=hermitian, k=1,
                             sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
            # canonize and save
            psi.A[n] = vec[0]
            psi.orthogonalize_site(n, towards=psi.g.first)
            env.clear_site(n)
            env.update(n, towards=psi.g.first)
            psi.absorb_central(towards=psi.g.first)
        elif version == '2site':
            if n == psi.g.sweep(to='first')[-1]:
                env.clear_site(n)
                env.update(n, towards=psi.g.first)
            else:
                n1, _, _ = psi.g.from_site(n, towards=psi.g.first)
                init = psi.A[n1].dot(psi.A[n], axes=(psi.right, psi.left))
                # update site n using eigs
                def Av(v): return env.Heff2(v, n1)
                if algorithm == 'lanczos' and not hermitian:
                    def Bv(v): return env.Heff2(v, n1, conj=True)
                else:
                    Bv = None
                _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[
                                 init], hermitian=hermitian, k=1, sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
                # split and save
                x, S, y = vec[0].split_svd(axes=(psi.left + psi.phys, tuple(
                    a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                psi.A[n1] = x.dot_diag(S, axis=2)
                psi.A[n] = y
                env.clear_site(n)
                env.update(n, towards=psi.g.first)
                env.clear_site(n1)
        elif version == '2site_group':
            if n == psi.g.sweep(to='first')[-1]:
                env.clear_site(n)
                env.update(n, towards=psi.g.first)
            else:
                n1, _, _ = psi.g.from_site(n, towards=psi.g.first)
                init = psi.A[n1].dot(psi.A[n], axes=(psi.right, psi.left))
                init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
                # update site n using eigs
                def Av(v): return env.Heff2_group(v, n1)
                if algorithm == 'lanczos' and not hermitian:
                    def Bv(v): return env.Heff2_group(v, n1, conj=True)
                else:
                    Bv = None
                _, vec, _ = eigs(Av=Av, Bv=Bv, v0=[
                                 init], hermitian=hermitian, k=1, sigma=None, ncv=k, which=None, tol=eigs_tol, algorithm=algorithm)
                init = vec[0].ungroup_leg(axis=1, leg_order=leg_order)
                # split and save
                x, S, y = init.split_svd(axes=(psi.left + psi.phys, tuple(
                    a + psi.right[0] - 1 for a in psi.phys + psi.right)), sU=-1, **opts_svd)
                psi.A[n1] = x.dot_diag(S, axis=2)
                psi.A[n] = y
                env.clear_site(n)
                env.update(n, towards=psi.g.first)
                env.clear_site(n1)

    return env
