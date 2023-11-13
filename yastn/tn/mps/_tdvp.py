""" Various variants of the TDVP algorithm for mps."""
from typing import NamedTuple
from ._env import Env3
from ._mps import MpsMpo
from ... import YastnError

#################################
#           tdvp                #
#################################

class TDVP_out(NamedTuple):
    ti: float = 0.
    tf: float = 0.
    time_independent: bool = None
    dt: float = 0.
    steps: int = 0


def tdvp_(psi, H, times=(0, 0.1), dt=0.1, u=1j, method='1site', order='2nd', opts_expmv=None, opts_svd=None, normalize=True):
    r"""
    Iterator performing TDVP sweeps to solve :math:`\frac{d}{dt} |\psi(t)\rangle = -uH|\psi(t)\rangle`,

    Parameters
    ----------
    psi: Mps
        Initial state. It is updated during execution.
        It is first canonized to the first site, if not provided in such a form.
        Resulting state is also canonized to the first site.

    H: Mps, nr_phys=2
        Evolution generator given either as MPO for time-independent problem
        or as a function returning MPO for time-dependent problem, i.e. ``Callable[[float], Mpo]``.

    time: float64 or tuple(float64)
        Initial and final times; can also provide intermediate times for snapshots returned
        by the iterator. If only the final time is provided, initial time is set to 0.

    dt: double
        Time step.
        It is adjusted down to have an integer number of time-steps to reach the next snapshot.

    u: number
        '1j' for real time evolution, 1 for imaginary time evolution.
        Default is 1j.

    method: str
        Algorithm to use in ('1site', '2site', '12site')

    order: str
        Order of Suzuki-Trotter decomposition in ('2nd', '4th').
        4th order step is composed of five 2nd order steps.

    opts_expmv: dict
        Options passed to :meth:`yastn.expmv`
        If there is information from previous time-steps stored under the hood,
        the initial guess of the size of krylov space opts_expmv['ncv'] is overriden.

    opts_svd: dict
        Options passed to :meth:`yastn.linalg.svd` used to truncate virtual spaces in :code:`method='2site'` and :code:`'12site'`.
        If None, use default {'tol': 1e-13}.

    Returns
    -------
    TDVP_out(NamedTuple)
        NamedTuple with fields:

            * :code:`ti` initial time of the time-interval.
            * :code:`tf` current time.
            * :code:`time_independent` if the Hamiltonian is time-independent.
            * :code:`dt` time-step used.
            * :code:`steps` number of time-steps in the last time-interval.
    """
    time_independent = isinstance(H, MpsMpo)
    if dt <= 0:
        raise YastnError('TDVP: dt should be positive.')
    if not hasattr(times, '__iter__'):
        times = (0, times)
    if any(t1 - t0 <= 0 for t0, t1 in zip(times[:-1], times[1:])):
        raise YastnError('TDVP: times should be an ascending tuple.')

    if method == '1site' and time_independent:
        routine = lambda t, dt0, env: _tdvp_sweep_1site_(psi, H, dt0, u, env, opts_expmv, normalize)
    elif method == '2site' and time_independent:
        routine = lambda t, dt0, env: _tdvp_sweep_2site_(psi, H, dt0, u, env, opts_expmv, opts_svd, normalize)
    elif method == '12site' and time_independent:
        routine = lambda t, dt0, env: _tdvp_sweep_12site_(psi, H, dt0, u, env, opts_expmv, opts_svd, normalize)
    elif method == '1site' and not time_independent:
        routine = lambda t, dt0, env: _tdvp_sweep_1site_(psi, H(t), dt0, u, None, opts_expmv, normalize)
    elif method == '2site' and not time_independent:
        routine = lambda t, dt0, env: _tdvp_sweep_2site_(psi, H(t), dt0, u, None, opts_expmv, opts_svd, normalize)
    elif method == '12site' and not time_independent:
        routine = lambda t, dt0, env: _tdvp_sweep_12site_(psi, H(t), dt0, u, None, opts_expmv, opts_svd, normalize)
    else:
        raise YastnError('TDVP: tdvp method %s not recognized' % method)

    env = None
    # perform time-steps
    for t0, t1 in zip(times[:-1], times[1:]):
        steps = int((t1 - t0 - 1e-12) // dt) + 1
        t, ds = t0, (t1 - t0) / steps
        for _ in range(steps):
            if order == '2nd':
                env = routine(t + ds/2, ds, env)
            elif order == '4th':
                s2 = 0.41449077179437573714
                env = routine(t + 0.5 * s2 * ds, ds * s2, env)
                env = routine(t + 1.5 * s2 * ds, ds * s2, env)
                env = routine(t + 0.5 * ds, ds * (1 - 4 * s2), env)
                env = routine(t + (1 - 1.5 * s2) * ds, ds * s2, env)
                env = routine(t + (1 - 0.5 * s2) * ds, ds * s2, env)
            else:
                raise YastnError("TDVP: order should be in ('2nd', '4th')")
            t = t + ds
        yield TDVP_out(t0, t, time_independent, ds, steps)


def _tdvp_sweep_1site_(psi, H, dt=0.1, u=1j, env=None, opts_expmv=None, normalize=True):
    r""" Perform sweep with 1-site TDVP update, see :meth:`tdvp` for description. """

    env, opts = _init_tdvp(psi, H, env, opts_expmv)

    for to in ('last', 'first'):
        for n in psi.sweep(to=to):
            env.update_A(n, -u * 0.5 * dt, opts, normalize=normalize)
            psi.orthogonalize_site_(n, to=to, normalize=normalize)
            env.clear_site_(n)
            env.update_env_(n, to=to)
            env.update_C(u * 0.5 * dt, opts, normalize=normalize)
            psi.absorb_central_(to=to)

    env.update_env_(psi.first, to='first')
    return env


def _tdvp_sweep_2site_(psi, H, dt=0.1, u=1j, env=None, opts_expmv=None, opts_svd=None, normalize=True):
    r""" Perform sweep with 2-site TDVP update, see :meth:`tdvp` for description. """

    env, opts = _init_tdvp(psi, H, env, opts_expmv)

    for to, dn in (('last', 1), ('first', 0)):
        for n in psi.sweep(to=to, dl=1):
            env.update_AA((n, n + 1), -u * 0.5 * dt, opts, opts_svd, normalize=normalize)
            psi.absorb_central_(to=to)
            env.clear_site_(n, n + 1)
            env.update_env_(n + 1 - dn, to=to)
            if n + dn != getattr(psi, to):
                env.update_A(n + dn, u * 0.5 * dt, opts, normalize=normalize)

    env.clear_site_(psi.first)
    env.update_env_(psi.first, to='first')
    return env


def _tdvp_sweep_12site_(psi, H, dt=0.1, u=1j, env=None, opts_expmv=None, opts_svd=None, normalize=True):
    r"""
    Perform sweep with mixed TDVP update, see :meth:`tdvp` for description.

    This mixes 1-site and 2-site updates based on smallest Schmidt value and maximal bond dimension
    """

    env, opts = _init_tdvp(psi, H, env, opts_expmv)

    for to, dn in (('last', 1), ('first', 0)):
        update_two = False
        for n in psi.sweep(to=to):
            if not update_two:
                if env.enlarge_bond((n - 1 + dn, n + dn), opts_svd):
                    update_two = True
                else:
                    env.update_A(n, -u * 0.5 * dt, opts, normalize=normalize)
                    psi.orthogonalize_site_(n, to=to, normalize=normalize)
                    env.clear_site_(n)
                    env.update_env_(n, to=to)
                    env.update_C(u * 0.5 * dt, opts, normalize=normalize)
                    psi.absorb_central_(to=to)
            else:
                env.update_AA((n - dn , n - dn + 1), - u * 0.5 * dt, opts, opts_svd, normalize=normalize)
                psi.absorb_central_(to=to)
                env.clear_site_(n - dn, n - dn + 1)
                env.update_env_(n + 1 - 2 * dn, to=to)
                if env.enlarge_bond((n - 1 + dn, n + dn), opts_svd):
                    env.update_A(n, u * 0.5 * dt, opts, normalize=normalize)
                else:
                    psi.orthogonalize_site_(n, to=to, normalize=normalize)
                    env.update_env_(n, to=to)
                    env.update_C(u * 0.5 * dt, opts, normalize=normalize)
                    psi.absorb_central_(to=to)
                    update_two = False

    env.clear_site_(psi.first)
    env.update_env_(psi.first, to='first')
    return env


def _init_tdvp(psi, H, env, opts_expmv):
    """ tests and initializations for all tdvp methods. """
    opts = {} if opts_expmv is None else opts_expmv.copy()
    if env is None:
        env = Env3(bra=psi, op=H, ket=psi)
        env.setup_(to='first')
    return env, opts
