# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Various variants of the TDVP algorithm for Mps."""
from __future__ import annotations
from typing import NamedTuple
from tqdm import tqdm
from ._measure import Env
from ... import YastnError, expmv

#################################
#           tdvp                #
#################################

class TDVP_out(NamedTuple):
    ti: float = 0.
    tf: float = 0.
    time_independent: bool = None
    dt: float = 0.
    steps: int = 0


def tdvp_(psi, H,
          times=(0, 0.1), dt=0.1, u=1j, method='1site', order='2nd',
          opts_expmv=None, opts_svd=None, normalize=True,
          progressbar=False, yield_initial=False):
    r"""
    Iterator performing TDVP sweeps to solve :math:`\frac{d}{dt} |\psi(t)\rangle = -uH|\psi(t)\rangle`,

    Parameters
    ----------
    psi: Mps
        Initial state. It is updated during execution.
        It is first canonized to the first site, if not provided in such a form.
        Resulting state is also canonized to the first site.

    H: yastn.tn.mps.MpsMpoOBC | Sequence | Callable
        Evolution generator given either as (sum of) MPO for time-independent problem
        or as a function returning (sum of) MPO for time-dependent problem, i.e. :code:`Callable[[float], Mpo]`
        or :code:`Callable[[float], Sequence[Mpo]]`, see :meth:`Env()<yastn.tn.mps.Env>`.

    time: float64 or tuple(float64)
        Initial and final times; can also provide intermediate times for snapshots returned
        by the iterator. If only the final time is provided, initial time is set to 0.

    dt: double
        Time step.
        It is adjusted down to have an integer number of time-steps to reach the next snapshot.

    u: number
        for real time evolution ``u=1j``, for imaginary time evolution ``u=1``.
        The default is 1j.

    method: str
        Algorithm to use among '1site', '2site', '12site'.

    order: str
        Order of Suzuki-Trotter decomposition, '2nd' or '4th'.
        4th order step is composed of five 2nd order steps.

    opts_expmv: dict
        Options passed to :meth:`yastn.expmv`, in particular whether `H` is hermitian.
        Unspecified options use default :meth:`yastn.expmv` settings.
        If there is information from previous time-steps stored under the hood,
        the initial guess of the size of krylov space opts_expmv['ncv'] is overriden.

    opts_svd: dict
        Options passed to :meth:`yastn.linalg.svd` used to truncate virtual spaces in :code:`method='2site'` and :code:`'12site'`.

    progressbar: bool
        Whether to show the progress bar toward the next snapshot. The default is False.

    yield_initial: bool
        Whether to yield the initial state before performing evolution. The default is False.

    Returns
    -------
    TDVP_out(NamedTuple)
        NamedTuple with fields:

            * :code:`ti` initial time of the time-interval.
            * :code:`tf` current time.
            * :code:`time_independent` whether the Hamiltonian is time-independent.
            * :code:`dt` time-step used.
            * :code:`steps` number of time-steps in the last time-interval.
    """
    if opts_svd is None and method in ('2site', '12site'):
        raise YastnError("TDVP: provide opts_svd for %s method." % method)

    if dt <= 0:
        raise YastnError('TDVP: dt should be positive.')
    if not hasattr(times, '__iter__'):
        times = (0, times)
    if any(t1 - t0 <= 0 for t0, t1 in zip(times[:-1], times[1:])):
        raise YastnError('TDVP: times should be an ascending tuple.')

    time_independent = not callable(H)
    if time_independent:
        Ht = lambda t: H
        et = lambda e: e
    else:
        Ht = H
        et = lambda e: None

    if method == '1site':
        routine = lambda t, dt0, env: _tdvp_sweep_1site_(psi, Ht(t), dt0, u, et(env), opts_expmv, normalize)
    elif method == '2site':
        routine = lambda t, dt0, env: _tdvp_sweep_2site_(psi, Ht(t), dt0, u, et(env), opts_expmv, opts_svd, normalize)
    elif method == '12site':
        routine = lambda t, dt0, env: _tdvp_sweep_12site_(psi, Ht(t), dt0, u, et(env), opts_expmv, opts_svd, normalize)
    else:
        raise YastnError('TDVP: tdvp method %s not recognized' % method)

    env = None
    if yield_initial:
        yield TDVP_out(times[0], times[0], time_independent, dt, 0)
    # perform time-steps
    for t0, t1 in zip(times[:-1], times[1:]):
        steps = int((t1 - t0 - 1e-12) // dt) + 1
        t, ds = t0, (t1 - t0) / steps
        rsteps = tqdm(range(steps), desc="TDVP...", disable=not progressbar)
        for _ in rsteps:
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
            _update_A(env, n, -u * 0.5 * dt, opts, normalize=normalize)
            psi.orthogonalize_site_(n, to=to, normalize=normalize)
            env.clear_site_(n)
            env.update_env_(n, to=to)
            _update_C(env, u * 0.5 * dt, opts, normalize=normalize)
            psi.absorb_central_(to=to)

    env.update_env_(psi.first, to='first')
    return env


def _tdvp_sweep_2site_(psi, H, dt=0.1, u=1j, env=None, opts_expmv=None, opts_svd=None, normalize=True):
    r""" Perform sweep with 2-site TDVP update, see :meth:`tdvp` for description. """

    env, opts = _init_tdvp(psi, H, env, opts_expmv)

    for to, dn in (('last', 1), ('first', 0)):
        for n in psi.sweep(to=to, dl=1):
            _update_AA(env, (n, n + 1), -u * 0.5 * dt, opts, opts_svd, normalize=normalize)
            psi.absorb_central_(to=to)
            env.clear_site_(n, n + 1)
            env.update_env_(n + 1 - dn, to=to)
            if n + dn != getattr(psi, to):
                _update_A(env, n + dn, u * 0.5 * dt, opts, normalize=normalize)

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
                    _update_A(env, n, -u * 0.5 * dt, opts, normalize=normalize)
                    psi.orthogonalize_site_(n, to=to, normalize=normalize)
                    env.clear_site_(n)
                    env.update_env_(n, to=to)
                    _update_C(env, u * 0.5 * dt, opts, normalize=normalize)
                    psi.absorb_central_(to=to)
            else:
                _update_AA(env, (n - dn , n - dn + 1), - u * 0.5 * dt, opts, opts_svd, normalize=normalize)
                psi.absorb_central_(to=to)
                env.clear_site_(n - dn, n - dn + 1)
                env.update_env_(n + 1 - 2 * dn, to=to)
                if env.enlarge_bond((n - 1 + dn, n + dn), opts_svd):
                    _update_A(env, n, u * 0.5 * dt, opts, normalize=normalize)
                else:
                    psi.orthogonalize_site_(n, to=to, normalize=normalize)
                    env.update_env_(n, to=to)
                    _update_C(env, u * 0.5 * dt, opts, normalize=normalize)
                    psi.absorb_central_(to=to)
                    update_two = False

    env.clear_site_(psi.first)
    env.update_env_(psi.first, to='first')
    return env


def _init_tdvp(psi, H, env, opts_expmv):
    """ tests and initializations for all tdvp methods. """
    opts = {} if opts_expmv is None else opts_expmv.copy()
    if env is None:
        env = Env(psi, [H, psi]).setup_(to='first')
        env._temp = {'expmv_ncv': {}}
    return env, opts


def _update_A(env, n, du, opts, normalize=True):
    """ Updates env.bra[n] by exp(du Heff1). """
    if n in env._temp['expmv_ncv']:
        opts['ncv'] = env._temp['expmv_ncv'][n]
    f = lambda x: env.Heff1(x, n)
    env.bra[n], info = expmv(f, env.bra[n], du, **opts, normalize=normalize, return_info=True)
    env._temp['expmv_ncv'][n] = info['ncv']


def _update_C(env, du, opts, normalize=True):
    """ Updates env.bra[bd] by exp(du Heff0). """
    bd = env.bra.pC
    if bd[0] != -1 and bd[1] != env.N:  # do not update central block outsite of the chain
        if bd in env._temp['expmv_ncv']:
            opts['ncv'] = env._temp['expmv_ncv'][bd]
        f = lambda x: env.Heff0(x, bd)
        env.bra.A[bd], info = expmv(f, env.bra[bd], du, **opts, normalize=normalize, return_info=True)
        env._temp['expmv_ncv'][bd] = info['ncv']


def _update_AA(env, bd, du, opts, opts_svd, normalize=True):
    """ Merge two sites given in bd into AA, updates AA by exp(du Heff2) and unmerge the sites. """
    ibd = bd[::-1]
    if ibd in env._temp['expmv_ncv']:
        opts['ncv'] = env._temp['expmv_ncv'][ibd]
    AA = env.bra.merge_two_sites(bd)
    f = lambda v: env.Heff2(v, bd)
    AA, info = expmv(f, AA, du, **opts, normalize=normalize, return_info=True)
    env._temp['expmv_ncv'][ibd] = info['ncv']
    env.bra.unmerge_two_sites_(AA, bd, opts_svd)
