Fixed-point CTM
===============

Differentiating a converged CTM environment by unrolling every sweep is expensive: the autograd
graph grows with the number of iterations, and the memory it holds scales with it. The fixed-point
formulation avoids that. Once CTMRG has converged, the environment satisfies
:math:`C = f(C, A)` for the CTM step :math:`f` and the PEPS tensors :math:`A`, so the gradient
follows from the implicit function theorem and needs only *one* CTM step to be differentiated, not
the whole history.

:func:`yastn.tn.fpeps.envs.fixed_pt.fp_ctmrg` implements this in three stages:

1. **Converge** the environment with ordinary CTMRG.
2. **Fix the gauge.** A converged CTM environment is defined only up to a gauge on its virtual
   legs, so consecutive sweeps need not agree element-wise even when the spectra have converged.
   That residual disagreement is what ``max_de`` reports during ordinary CTMRG -- the largest
   change in environment element *moduli* between consecutive sweeps -- and it can stay finite
   while ``max_dsv``, which compares corner spectra, has already reached ``corner_tol``. Both are
   fields of the ``CTMRG_out`` returned by :meth:`yastn.tn.fpeps.EnvCTM.iterate_`.
   One further CTM step is taken and the gauge relating it
   to the converged environment is extracted; applying it makes the fixed-point condition hold
   element-wise, which is what the derivative is taken about.
3. **Differentiate** through that single step, summing the
   `Neumann series <https://en.wikipedia.org/wiki/Neumann_series>`_
   :math:`\sum_k (\partial f/\partial C)^k` to invert :math:`(1 - \partial f/\partial C)`.

The series converges only when the spectral radius of :math:`\partial f/\partial C` is below one.
The implementation monitors the increment of the running gradient estimate, keeps the best estimate
seen, and stops once it is no longer improving -- see ``neumann_patience`` below.

If the forward CTMRG does not converge, or no gauge can be found, ``fp_ctmrg`` raises
``NoFixedPointError`` rather than returning a silently wrong gradient.


Forward options carry over to the backward
------------------------------------------

``fp_ctmrg`` takes two option sets. **The fixed-point step inherits from the forward one.**

``ctm_opts_fp`` is not an independent configuration. It starts as a copy of ``ctm_opts_fwd`` and
overrides it selectively, with ``opts_svd`` merged key by key rather than replaced. So::

    env = fp_ctmrg(env,
                   ctm_opts_fwd={'opts_svd': {'D_total': 64, 'tol': 1e-8},
                                 'corner_tol': 1e-8,
                                 'max_sweeps': 100,
                                 'method': '2x2',
                                 'use_qr': False},
                   ctm_opts_fp={'opts_svd': {'policy': 'fullrank'}})

gives a fixed-point step that keeps the forward's ``method``, ``use_qr``, ``max_sweeps`` and
``corner_tol``, and whose ``opts_svd`` is ``{'D_total': 64, 'tol': 1e-8, 'policy': 'fullrank'}`` --
the forward truncation with the full-rank SVD policy merged in. Only what you name in
``ctm_opts_fp`` changes.

That inheritance extends to the backward pass, because ``FixedPoint`` reverses the CTM step by hand:
the Neumann loop takes its iteration budget from the fixed-point step's ``max_sweeps`` and its
gradient tolerance from its ``corner_tol``. Raising the forward sweep budget therefore also allows
the backward series more terms, which is usually what you want; setting them apart is done by naming
them in ``ctm_opts_fp``.

One backward control has no forward counterpart and so is its own option:

``neumann_patience``
    How many Neumann iterations may pass without the gradient estimate improving before the series
    is judged non-contracting and stopped, returning the best estimate rather than the diverged
    tail. The default is 10.

The resolved form of this is :class:`yastn.tn.fpeps.envs.FixedPointOpts`, which can be passed
directly as ``opts=`` instead of the two dicts. See :doc:`ctm_options`.


C4v-symmetric variant
---------------------

:func:`yastn.tn.fpeps.envs.fixed_pt_c4v.fp_ctmrg_c4v` is the counterpart for C4v-symmetric
single-site iPEPS, used with :class:`yastn.tn.fpeps.EnvCTM_c4v`. It takes the same two option dicts
with the same inheritance.

It is a **separate implementation**, not a thin wrapper: the gauge-fixing works on the single
C and T tensor rather than a full unit cell, and its Neumann loop uses a different stopping
heuristic. Do not assume a change to one applies to the other.


Running on several devices
--------------------------

Passing ``devices=[...]`` with more than one entry routes the forward convergence, the fixed-point
CTM step and the Neumann backward through an AD-aware distributed dispatch over a persistent worker
pool. With a single device -- or ``None``, which falls back to the environment's own device --
everything runs serially.

Forward settings such as ``method``, ``moves`` and ``use_qr`` are honoured on the distributed path
exactly as on the serial one.

The workers are separate processes, so the options travelling to them must be picklable; the option
dataclasses are. Note that ``devices=['cpu', 'cpu']`` is a legitimate configuration and exercises
the full multiprocess path without needing several GPUs, which is how the distributed code is
tested.


API
---

.. autofunction:: yastn.tn.fpeps.envs.fixed_pt.fp_ctmrg

.. autofunction:: yastn.tn.fpeps.envs.fixed_pt_c4v.fp_ctmrg_c4v

.. seealso::

    :doc:`environment_ctm` for the CTMRG iteration being differentiated, and
    :doc:`ctm_options` for the option objects.
