""" Various variants of the TDVP algorithm for mps."""
from ._env import Env3
from ._mps import YampsError

#################################
#           tdvp                #
#################################


def tdvp_sweep_1site(psi, H=False, dt=0.1, env=None, opts_expmv=None):
    r"""
    Perform sweep with 1-site TDVP, calculating the update psi(dt) = exp( dt * H ) @ psi(0).

    Assume input psi is canonized to first site.

    Parameters
    ----------
    psi: Mps
        initial state. It is updated during the execution

    H: Mps, nr_phys=2
        evolution generator given in the form of mpo.

    dt: double
        time step

    env: Env3
        Can provide environment <psi|H|psi> from the previous sweep.
        It is initialized if set to None

    opts_expmv: dict
        options passed to expmv;
        If there is information from previous excecutions stored in env,
        overrid the initial guess of the size of krylov space opts_expmv['ncv'] will be overriden.

    Returns
    -------
    env: Env3
        Environment of the <psi|H|psi> ready for the next iteration.
    """
    opts = {} if opts_expmv is None else opts_expmv.copy()
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')
    if not (env.bra is psi and env.ket is psi):
        raise YampsError('Require environment env where ket is bra is psi')

    for n in psi.sweep(to='last'):
        env.update_A(n, 0.5 * dt, opts)
        psi.orthogonalize_site(n, to='last')
        env.clear_site(n)
        env.update(n, to='last')
        env.update_C(-0.5 * dt, opts)
        psi.absorb_central(to='last')

    for n in psi.sweep(to='first'):
        env.update_A(n, 0.5 * dt, opts)
        psi.orthogonalize_site(n, to='first')
        env.clear_site(n)
        env.update(n, to='first')
        env.update_C(-0.5 * dt, opts)
        psi.absorb_central(to='first')

    env.update(0, to='first')
    return env


def tdvp_sweep_2site(psi, H=False, dt=0.1, env=None, opts_expmv=None, opts_svd=None):
    r"""
    Perform sweep with 2-site TDVP, calculating the update psi(dt) = exp( dt * H ) @ psi(0).

    Assume input psi is canonized to first site.

    Parameters
    ----------
    psi: Mps
        initial state. It is updated during the execution

    H: Mps, nr_phys=2
        evolution generator given in the form of mpo.

    dt: double
        time step

    env: Env3
        Can provide environment <psi|H|psi> from the previous sweep.
        It is initialized if set to None

    opts_expmv: dict
        options passed to expmv;
        If there is information from previous excecutions stored in env,
        overrid the initial guess of the size of krylov space opts_expmv['ncv'] will be overriden.

    opts_svd: dict
        options passed to svd to truncate virtual bond dimensions when unmerging two merged sites.

    Returns
    -------
    env: Env3
        Environment of the <psi|H|psi> ready for the next iteration.
    """
    opts = {} if opts_expmv is None else opts_expmv.copy()
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')
    if not (env.bra is psi and env.ket is psi):
        raise YampsError('Require environment env where ket is bra is psi')

    for n in psi.sweep(to='last', dl=1):
        env.update_AA((n, n + 1), 0.5 * dt, opts, opts_svd)
        psi.absorb_central(to='last')
        env.clear_site(n, n + 1)
        env.update(n, to='last')
        if n + 1 != psi.last:
            env.update_A(n + 1, -0.5 * dt, opts)

    for n in psi.sweep(to='first', dl=1):
        env.update_AA((n, n + 1), 0.5 * dt, opts, opts_svd)
        psi.absorb_central(to='first')
        env.clear_site(n, n + 1)
        env.update(n + 1, to='first')
        if n != psi.first:
            env.update_A(n, -0.5 * dt, opts)

    env.clear_site(0)
    env.update(0, to='first')
    return env


def tdvp_sweep_mix(psi, H=False, dt=1., env=None, opts_expmv=None, opts_svd=None):
    r"""
    Perform mixed 1site-2site sweep of TDVP basing on SV_min (smallest Schmidt value on the bond).
    Procedure performs exponantiation: psi(dt) = exp( dt * H )*psi(0).
    Assume input psi is right canonical.
    Sweep consists of iterative updates from last site to first and back to the first one.

    Parameters
    ----------
    psi: Mps
        initial state.

    Returns
    -------
    env: Env3
        Overlap <psi| H |psi> as Env3.
    """
    opts = {} if opts_expmv is None else opts_expmv.copy()
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')

    update_two = False
    for n in psi.sweep(to='last'):
        if not update_two:
            if env.enlarge_bond[(n, n + 1)]:
                update_two = True
            else:
                env.update_A(n, 0.5 * dt, opts)
                psi.orthogonalize_site(n, to='last')
                env.clear_site(n)
                env.update(n, to='last')
                env.update_C(-0.5 * dt, opts)
                psi.absorb_central(to='last')
        else:
            env.update_AA((n - 1, n), 0.5 * dt, opts, opts_svd)
            psi.absorb_central(to='last')
            env.clear_site(n - 1, n)
            env.update(n - 1, to='last')
            if env.enlarge_bond[(n, n + 1)]:
                if n + 1 != psi.last:
                    env.update_A(n + 1, -0.5 * dt, opts)
            else:
                psi.ortogonalize_site(n, to='last')
                env.update(n, to='last')
                env.update_C(-0.5 * dt, opts)
                psi.absorb_central(to='last')

    for n in psi.sweep(to='first'):
        if not update_two:
            if env.enlarge_bond[(n - 1, n)]:
                update_two = True
            else:
                env.update_A(n, 0.5 * dt, opts)
                psi.orthogonalize_site(n, to='first')
                env.clear_site(n)
                env.update_C(-0.5 * dt, opts)
                psi.absorb_central(to='last')
        else:
            env.update_AA((n, n + 1), 0.5 * dt, opts, opts_svd)
            psi.absorb_central(to='first')
            env.clear_site(n, n + 1)
            env.update(n + 1, to='first')
            if env.enlarge_bond[(n - 1, n)]:
                if n != psi.first:
                    env.update_A(n, -0.5 * dt, opts)
            else:
                psi.ortogonalize_site(n, to='first')
                env.update_C(-0.5 * dt, opts)
                psi.absorb_central(to='last')

    env.clear_site(0)
    env.update(0, to='first')
    return env
