""" Various variants of the TDVP algorithm for mps."""
from ._env import Env3
from ._mps import YampsError, MpsMpo
import numpy as np

#################################
#           tdvp                #
#################################


def tdvp(psi, H, time=(0, 0.125), dt=0.125, env=None, version='1site', order='2nd', opts_expmv=None, opts_svd=None, normalize=True):
    r"""
    Perform TDVP sweep, calculating the update psi(dt) = exp( dt * H ) @ psi(0).

    Assume input psi is canonized to the first site.

    Parameters
    ----------
    psi: Mps
        initial state. It is updated during the execution.

    H: Mps, nr_phys=2
        evolution generator given in the form of mpo (time-independent H),
        or a function outputting mpo (time-dependent H).

    time: float64 or tuple(float64)
        initial and final time; can also provide intermidiate times to reached.
        Can provide only final time, in which case initial time is set to 0.

    dt: double
        time step.
        It is adjusted down to have an integer number of time-steps covering total evolution time.

    env: Env3
        For time-independent H can provide environment <psi|H|psi> from the previous sweep.
        It is initialized if None.

    version: str
        which tdvp procedure to use in ('1site', '2site', 'mix')

    order: str
        order of Suzuki-Trotter decomposition in ('2nd', '4th')
        4th order is composed from 5 sweeps with 2nd order.

    opts_expmv: dict
        options passed to :meth:`yast.expmv`
        If there is information from previous excecutions stored in env,
        overrid the initial guess of the size of krylov space opts_expmv['ncv'] will be overriden.

    opts_svd: dict
        options passed to :meth:`yast.linalg.svd` to truncate virtual bond dimensions when unmerging two merged sites.

    Returns
    -------
    env: Env3
        Environment of the <psi|H|psi> ready for the next iteration.
        Can contain temporary objects to reuse from previous sweeps.
    """
    time_independent = isinstance(H, MpsMpo)
    if dt <= 0:
        raise YampsError('dt should be positive.')
    if not isinstance(time, tuple):
        time = (0, time)
    if any(t1 - t0 <= 0 for t1, t0 in zip(time[1:], time[:-1])):
        raise YampsError('Time should be an ascending tuple.')

    if version == '1site' and time_independent:
        routine = lambda t, dt0, env: tdvp_sweep_1site(psi, H, dt0, env, opts_expmv, normalize)
    elif version == '2site' and time_independent:
        routine = lambda t, dt0, env: tdvp_sweep_2site(psi, H, dt0, env, opts_expmv, opts_svd, normalize)
    elif version == '12site' and time_independent:
        routine = lambda t, dt0, env: tdvp_sweep_12site(psi, H, dt0, env, opts_expmv, opts_svd, normalize)
    elif version == '1site' and not time_independent:
        routine = lambda t, dt0, env: tdvp_sweep_1site(psi, H(t), dt0, None, opts_expmv, normalize)
    elif version == '2site' and not time_independent:
        routine = lambda t, dt0, env: tdvp_sweep_2site(psi, H(t), dt0, None, opts_expmv, opts_svd, normalize)
    elif version == '12site' and not time_independent:
        routine = lambda t, dt0, env: tdvp_sweep_12site(psi, H(t), dt0, None, opts_expmv, opts_svd, normalize)
    else:
        raise YampsError('tdvp version %s not recognized' % version, order)

    # perform time-steps
    for t0, t1 in zip(time[:-1], time[1:]):
        steps = np.ceil((t1 - t0) / dt)
        t, dt1 = t0, (t1 - t0) / steps

        if order == '2nd':
            env = routine(t + dt1/2, dt1, env)
        elif order == '4th':
            s2 = 0.41449077179437573714
            env = routine(t + 0.5 * s2 * dt1, dt1 * s2, env)
            env = routine(t + 1.5 * s2 * dt1, dt1 * s2, env)
            env = routine(t + 0.5 * dt1, dt1 * (1 - 4 * s2), env)
            env = routine(t + (1 - 1.5 * s2) * dt1, dt1 * s2, env)
            env = routine(t + (1 - 0.5 * s2) * dt1, dt1 * s2, env)
        else:
            raise YampsError("order should be in ('2nd', '4th')")
        t = t + dt1
        # measurment here
    return env


def tdvp_sweep_1site(psi, H, dt=0.1, env=None, opts_expmv=None, normalize=True):
    r""" Perform sweep with 1-site TDVP update, see :meth:`tdvp` for description. """
    
    env, opts = _init_tdvp(psi, H, env, opts_expmv)

    for to in ('last', 'first'):
        for n in psi.sweep(to=to):
            env.update_A(n, 0.5 * dt, opts, normalize=normalize)
            psi.orthogonalize_site(n, to=to, normalize=normalize)
            env.clear_site(n)
            env.update_env(n, to=to)
            env.update_C(-0.5 * dt, opts, normalize=normalize)
            psi.absorb_central(to=to)

    env.update_env(psi.first, to='first')
    return env


def tdvp_sweep_2site(psi, H, dt=0.1, env=None, opts_expmv=None, opts_svd=None, normalize=True):
    r""" Perform sweep with 2-site TDVP update, see :meth:`tdvp` for description. """

    env, opts = _init_tdvp(psi, H, env, opts_expmv)

    for to, dn in (('last', 1), ('first', 0)):
        for n in psi.sweep(to=to, dl=1):
            env.update_AA((n, n + 1), 0.5 * dt, opts, opts_svd, normalize=normalize)
            psi.absorb_central(to=to)
            env.clear_site(n, n + 1)
            env.update_env(n + 1 - dn, to=to)
            if n + dn != getattr(psi, to):
                env.update_A(n + dn, -0.5 * dt, opts, normalize=normalize)

    env.clear_site(psi.first)
    env.update_env(psi.first, to='first')
    return env


def tdvp_sweep_12site(psi, H=False, dt=1., env=None, opts_expmv=None, opts_svd=None, normalize=True):
    r"""
    Perform sweep with mixed TDVP update, see :meth:`tdvp` for description.

    This mixes 1-site and 2-site updates based on smallest Schmidt value and maximal bond dimension

    NOT FINISHED
    """

    env, opts = _init_tdvp(psi, H, env, opts_expmv)

    for to, dn in (('last', 1), ('first', 0)):
        update_two = False
        for n in psi.sweep(to=to):
            if not update_two:
                if env.enlarge_bond((n - 1 + dn, n + dn), opts_svd):
                    update_two = True
                else:
                    env.update_A(n, 0.5 * dt, opts, normalize=normalize)
                    psi.orthogonalize_site(n, to=to, normalize=normalize)
                    env.clear_site(n)
                    env.update_env(n, to=to)
                    env.update_C(-0.5 * dt, opts, normalize=normalize)
                    psi.absorb_central(to=to)
            else:
                env.update_AA((n - dn , n - dn + 1), 0.5 * dt, opts, opts_svd, normalize=normalize)
                psi.absorb_central(to=to)
                env.clear_site(n - dn, n - dn + 1)
                env.update_env(n + 1 - 2 * dn, to=to)
                if env.enlarge_bond((n - 1 + dn, n + dn), opts_svd):
                    env.update_A(n, -0.5 * dt, opts, normalize=normalize)
                else:
                    psi.orthogonalize_site(n, to=to, normalize=normalize)
                    env.update_env(n, to=to)
                    env.update_C(-0.5 * dt, opts, normalize=normalize)
                    psi.absorb_central(to=to)
                    update_two = False

    env.clear_site(psi.first)
    env.update_env(psi.first, to='first')
    return env


def _init_tdvp(psi, H, env, opts_expmv):
    """ tests and initializations for all tdvp methods. """
    opts = {} if opts_expmv is None else opts_expmv.copy()
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup(to='first')
    if not (env.bra is psi and env.ket is psi):
        raise YampsError('Require environment env where ket == bra == psi')
    return env, opts
