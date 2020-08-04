from yamps.mps import Env3
from yamps.tensor import eigs

#################################
#           dmrg                #
#################################


def dmrg_OBC(psi, H, env=None, nor=None, measure_O=None, version='1site', cutoff_sweep=20, cutoff_dE=1e-9, dtype='complex128', hermitian=True, k=4, eigs_tol=1e-14, opts_svd=None):
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
            env = dmrg_sweep_0site(psi, H=H, env=env, dtype=dtype, k=k,
                                   hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd)
        elif version == '2site':
            env = dmrg_sweep_2site(psi, H=H, env=env, dtype=dtype, k=k,
                                   hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd)
        elif version == '2site_group':
            env = dmrg_sweep_2site_group(
                psi, H=H, env=env, dtype=dtype, k=k, hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd)
        else:
            env = dmrg_sweep_1site(psi, H=H, env=env, dtype=dtype, k=k,
                                   hermitian=hermitian, eigs_tol=eigs_tol, opts_svd=opts_svd)

        E = env.measure()/psi.norma
        dE = abs(E - E0)
        print('Iteration: ', sweep, ' energy: ', E, ' dE: ', dE,
              ' norma:', psi.norma, ' D: ', max(psi.get_D()))
        E0 = E
        sweep += 1
        out = ()
    return env, E, dE, out


def dmrg_sweep_0site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, eigs_tol=1e-14, opts_svd=None):
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
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last'):  # sweep from fist to last
        psi.orthogonalize_site(n, towards=psi.g.last)
        env.clear_site(n)
        env.update(n, towards=psi.g.last)
        if n != psi.g.sweep(to='last')[-1]:
            init = psi.A[psi.pC]
            # update site n using eigs
            if not hermitian:
                val, vec, _ = eigs(Av=lambda v: env.Heff0(v, psi.pC), Bv=lambda v: env.Heff0(
                    v, psi.pC, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
            else:
                val, vec, _ = eigs(Av=lambda v: env.Heff0(v, psi.pC), init=[
                                   init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
            init = vec[list(val).index(min(list(val)))]
            # canonize and save
            if opts_svd:
                U, S, V = init.split_svd(axes=((0), (1)), sU=-1, **opts_svd)
                psi.A[psi.pC] = U.dot(
                    S.dot(V, axes=((1), (0))), axes=((1), (0)))
            else:
                psi.A[psi.pC] = init
        psi.absorb_central(towards=psi.g.last)

    for n in psi.g.sweep(to='first'):
        psi.orthogonalize_site(n, towards=psi.g.first)
        env.clear_site(n)
        env.update(n, towards=psi.g.first)
        if n != psi.g.sweep(to='first')[-1]:
            init = psi.A[psi.pC]
            # update site n using eigs
            if not hermitian:
                val, vec, _ = eigs(Av=lambda v: env.Heff0(v, psi.pC), Bv=lambda v: env.Heff0(
                    v, psi.pC, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
            else:
                val, vec, _ = eigs(Av=lambda v: env.Heff0(v, psi.pC), init=[
                                   init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
            init = vec[list(val).index(min(list(val)))]
            # canonize and save
            if opts_svd:
                U, S, V = init.split_svd(axes=((0), (1)), sU=-1, **opts_svd)
                psi.A[psi.pC] = U.dot(
                    S.dot(V, axes=((1), (0))), axes=((1), (0)))
            else:
                psi.A[psi.pC] = init
        psi.absorb_central(towards=psi.g.first)
    return env


def dmrg_sweep_1site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, eigs_tol=1e-14, opts_svd=None):
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

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last'):  # sweep from fist to last
        psi.absorb_central(towards=psi.g.last)
        init = psi.A[n]
        # update site n using eigs
        if not hermitian:
            val, vec, _ = eigs(Av=lambda v: env.Heff1(v, n), Bv=lambda v: env.Heff1(
                v, n, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=lambda v: env.Heff1(v, n), init=[
                               init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        init = vec[list(val).index(min(list(val)))]
        # canonize and save
        if opts_svd:
            U, S, V = init.split_svd(
                axes=(psi.left + psi.phys, psi.right), sU=-1, **opts_svd)
            psi.A[n] = U
            psi.pC = (n, n+1)
            psi.A[psi.pC] = S.dot(V, axes=((1), (0)))
        else:
            psi.A[n] = init
            psi.orthogonalize_site(n, towards=psi.g.last)
        env.clear_site(n)
        env.update(n, towards=psi.g.last)

    for n in psi.g.sweep(to='first'):
        psi.absorb_central(towards=psi.g.first)
        init = psi.A[n]
        if not hermitian:
            val, vec, _ = eigs(Av=lambda v: env.Heff1(v, n), Bv=lambda v: env.Heff1(
                v, n, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=lambda v: env.Heff1(v, n), init=[
                               init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        init = vec[list(val).index(min(list(val)))]
        # canonize and save
        if opts_svd:
            U, S, V = init.split_svd(
                axes=(psi.left, psi.phys + psi.right), sU=-1, **opts_svd)
            psi.A[n] = V
            psi.pC = (n - 1, n)
            psi.A[psi.pC] = U.dot(S, axes=((1), (0)))
        else:
            psi.A[n] = init
            psi.orthogonalize_site(n, towards=psi.g.first)
        env.clear_site(n)
        env.update(n, towards=psi.g.first)

    return env


def dmrg_sweep_2site(psi, H, env=None, dtype='complex128', hermitian=True, k=4, eigs_tol=1e-14, opts_svd={}):
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

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        # update site n using eigs
        if not hermitian:
            val, vec, _ = eigs(Av=lambda v: env.Heff2(v, n), Bv=lambda v: env.Heff2(
                v, n, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=lambda v: env.Heff2(v, n), init=[
                               init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        init = vec[list(val).index(min(list(val)))]
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
        # update site n using eigs
        if not hermitian:
            val, vec, _ = eigs(Av=lambda v: env.Heff2(v, n), Bv=lambda v: env.Heff2(
                v, n, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=lambda v: env.Heff2(v, n), init=[
                               init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        init = vec[list(val).index(min(list(val)))]
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


def dmrg_sweep_2site_group(psi, H, env=None, dtype='complex128', hermitian=True, k=4, eigs_tol=1e-14, opts_svd={}):
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

    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_to_first()

    for n in psi.g.sweep(to='last', dl=1):
        n1, _, _ = psi.g.from_site(n, towards=psi.g.last)
        init = psi.A[n].dot(psi.A[n1], axes=(psi.right, psi.left))
        init, leg_order = init.group_legs(axes=(1, 2), new_s=1)
        # update site n using eigs
        if not hermitian:
            val, vec, _ = eigs(Av=lambda v: env.Heff2_group(v, n), Bv=lambda v: env.Heff2_group(
                v, n, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=lambda v: env.Heff2_group(v, n), init=[
                               init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        init = vec[list(val).index(min(list(val)))]
        init = init.ungroup_leg(axis=1, leg_order=leg_order)
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
        if not hermitian:
            val, vec, _ = eigs(Av=lambda v: env.Heff2_group(v, n), Bv=lambda v: env.Heff2_group(
                v, n, conj=True), init=[init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        else:
            val, vec, _ = eigs(Av=lambda v: env.Heff2_group(v, n), init=[
                               init], tol=eigs_tol, k=k, hermitian=True, dtype=dtype)
        init = vec[list(val).index(min(list(val)))]
        init = init.ungroup_leg(axis=1, leg_order=leg_order)
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
