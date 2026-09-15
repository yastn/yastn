Fixed-point CTMRG with implicit differentiation
===============================================

Module :mod:`yastn.tn.fpeps.envs.fixed_pt` provides
:func:`yastn.tn.fpeps.envs.fixed_pt.fp_ctmrg`, a CTMRG driver whose output
environment is differentiable with respect to the PEPS tensors through the
fixed-point condition rather than through the unrolled CTMRG iterations.
It requires the ``torch`` backend.

Given a converged CTM environment :math:`C^\ast` of the state :math:`A`, one
CTMRG sweep :math:`f` leaves it invariant, :math:`C^\ast = f(C^\ast, A)`.
The implicit-function theorem then gives the derivative of any observable
:math:`\mathcal{L}(C^\ast, A)` without storing the forward sweeps,

.. math::

    \frac{d\mathcal{L}}{dA}
    = \frac{\partial \mathcal{L}}{\partial A}
    + \frac{\partial \mathcal{L}}{\partial C}
      \Big(1 - \frac{\partial f}{\partial C}\Big)^{-1}
      \frac{\partial f}{\partial A},

where the inverse is expanded as a Neumann series
:math:`\sum_{n \ge 0} (\partial f / \partial C)^n`.  This works only if the
environment converges *element-wise*, which CTMRG alone does not guarantee:
each sweep is free to change the gauge of the environment bonds (a unitary on
each bond and a phase on each tensor).  The gauge is therefore fixed
explicitly before the fixed-point condition is used.

Forward pass
------------

:func:`yastn.tn.fpeps.envs.fixed_pt.fp_ctmrg` wraps the custom autograd
function :class:`yastn.tn.fpeps.envs.fixed_pt.FixedPoint`, whose forward pass
does the following.

1. **Converge.**  :meth:`FixedPoint.get_converged_env` runs
   :meth:`yastn.tn.fpeps.EnvCTM.ctmrg_` until the corner spectra stop changing
   (``corner_tol``) or ``max_sweeps`` is reached.  An optional stall detector
   (``stuck_block``) declares a solve non-convergent when the running minimum
   of the corner change stops improving across blocks of sweeps.
2. **One more sweep.**  A single CTM step with the options in ``ctm_opts_fp``
   (full-rank SVD, so that the step has a well-defined derivative) produces
   ``env_new`` from the converged ``env_old``.
3. **Gauge fixing.**  :func:`find_gauge_multi_sites` finds, for every site and
   every edge direction, the bond gauge ``sigma`` relating the old and new
   edge tensors ``T`` along a full row or column of the unit cell
   (:func:`fast_env_T_gauge_multi_sites`, the leading-eigenvector method of
   `arXiv:2311.11894 <https://arxiv.org/abs/2311.11894>`_).  The gauges are
   collected in an :class:`EnvGauge`, applied to edges and corners by
   :func:`apply_sigma`, and the residual per-tensor U(1) phase is extracted by
   :func:`U1_phase` and applied with :func:`apply_U1_`.  The transformed
   ``env_new`` then equals ``env_old`` element-wise.
4. **Save.**  The converged environment and the gauge are stored for the
   backward pass; the environment tensors are returned as one flat
   ``torch`` tensor so that autograd tracks them.

If any step fails, a :class:`NoFixedPointError` is raised (CTM does not
converge, the symmetry sectors of the edge tensors change between sweeps, or
no gauge is found).

Backward pass
-------------

:meth:`FixedPoint.backward` builds the gauge-fixed sweep
:meth:`FixedPoint.fixed_point_iter`, i.e. one CTM step followed by the saved
``sigma`` and phase transformations, as a function of the environment data
and the PEPS data, and takes its vector-Jacobian products.  The gauge transformation part
is detached from the computation graph.
The Neumann series is summed until the increment falls below ``corner_tol``, or it has
not decreased for ``neumann_patience`` consecutive steps (the best estimate so
far is then kept), or until ``max_sweeps`` steps.  The result is the gradient
with respect to the raw PEPS tensors in the order of ``site2index``.

Multi-device runs (``devices`` with more than one entry) route the forward
convergence, the extra CTM step and the Neumann iterations through the
distributed CTM workers of :mod:`yastn.tn.fpeps.envs._env_ctm_dist_mp_AD`.

Example
-------

A gradient-based optimization with the CTM workers on several GPUs.  The
state, the gauge fixing and the main-side autograd graph live on the
``default_device`` of the yastn config (the *home* device); the CTMRG
stages of the forward convergence, of the fixed-point step and of the
Neumann backward are dispatched to one spawned worker process per entry of
``devices``.  Keeping the home device out of ``devices`` leaves the
:math:`O(\chi^2)` backward graph off the worker GPUs.  The worker pool is
created on the first call and reused by every later ``fp_ctmrg`` call with
the same devices and config, so the spawn cost is paid once per run; it is
shut down at interpreter exit.  Because the workers are spawned, the script
needs the usual ``if __name__ == '__main__'`` guard::

    import torch
    import yastn
    import yastn.tn.fpeps as fpeps
    from yastn.tn.fpeps.envs.fixed_pt import fp_ctmrg, NoFixedPointError

    def main():
        home = 'cuda:0'                                   # state + autograd graph
        devices = ['cuda:1', 'cuda:2', 'cuda:3']          # CTM worker pool
        config = yastn.make_config(backend='torch', sym='U1', fermionic=True,
                                   default_device=home, default_dtype='complex128')

        psi = fpeps.Peps(geometry, tensors=...)           # tensors on `home`
        params = [psi[s]._data.requires_grad_(True) for s in psi.sites()]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=1, history_size=10)

        chi = 64
        env_leg = yastn.Leg(config, s=1, t=(0,), D=(chi,))
        ctm_opts_fwd = {'method': '2x2', 'corner_tol': 1e-8, 'max_sweeps': 200,
                        'opts_svd': {'D_total': chi, 'tol': 1e-10,
                                     'eps_multiplet': 1e-8, 'truncate_multiplets': True},
                        'use_qr': False, 'stuck_block': 10, 'verbosity': 0}
        ctm_opts_fp = {'opts_svd': {'policy': 'fullrank'}, 'corner_tol': 1e-8,
                       'max_sweeps': 100, 'neumann_patience': 10}

        env = fpeps.EnvCTM(psi, init='eye', leg=env_leg)

        def closure():
            nonlocal env
            opt.zero_grad()
            try:
                # reuse the previous environment as the starting point
                env = fp_ctmrg(env, ctm_opts_fwd=ctm_opts_fwd,
                               ctm_opts_fp=ctm_opts_fp, devices=devices)
            except NoFixedPointError as e:
                raise                                     # or perturb psi and retry
            loss = energy_per_site(psi, env)              # any observable of env
            loss.backward()                               # IFT gradient into `params`
            return loss

        for step in range(100):
            loss = opt.step(closure)
            torch.cuda.empty_cache()                      # home device; the pool
            print(step, loss.item())                      # frees its own cache

    if __name__ == '__main__':
        main()

With ``devices=[home]`` (or ``devices=None``) the same script runs serially
in the main process without spawning workers.

API
---

.. autofunction:: yastn.tn.fpeps.envs.fixed_pt.fp_ctmrg

.. autoclass:: yastn.tn.fpeps.envs.fixed_pt.FixedPoint
    :members: forward, backward, get_converged_env, fixed_point_iter

.. autoclass:: yastn.tn.fpeps.envs.fixed_pt.NoFixedPointError

.. autoclass:: yastn.tn.fpeps.envs.fixed_pt.EnvGauge
    :members: to_dict, from_dict

.. autofunction:: yastn.tn.fpeps.envs.fixed_pt.find_gauge_multi_sites
.. autofunction:: yastn.tn.fpeps.envs.fixed_pt.fast_env_T_gauge_multi_sites
.. autofunction:: yastn.tn.fpeps.envs.fixed_pt.apply_sigma
.. autofunction:: yastn.tn.fpeps.envs.fixed_pt.U1_phase
.. autofunction:: yastn.tn.fpeps.envs.fixed_pt.apply_U1_
