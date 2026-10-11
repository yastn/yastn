Environment CTM
===============

Corner transfer matrix renormalization group (CTMRG) associates a local CTM environment with each lattice site
and the corresponding rank-4 PEPS tensors **a** (potentially, coming from a :ref:`double-layer<fpeps/initialization:Double PEPS Tensor>` contraction of bra and ket PEPSs).
We show it below with the convention for ordering the indices in the CTMRG environment tensors:

::

    ┌──────┐           ┌─────┐           ┌──────┐
    | C_tl ├── 1   0 ──┤ T_t ├── 2   0 ──┤ C_tr |
    └──┬───┘           └──┬──┘           └───┬──┘
       |                  |                  |
       0                  1                  1

       2                  0                  0
       │                  |                  |
    ┌──┴──┐            ┌──┴──┐            ┌──┴──┐
    | T_l ├── 1    1 ──┤  a  ├── 3    1 ──┤ T_r |
    └──┬──┘            └──┬──┘            └──┬──┘
       |                  |                  |
       0                  2                  2

       1                  1                  0
       |                  |                  |
    ┌──┴───┐           ┌──┴──┐           ┌───┴──┐
    | C_bl ├── 0   2 ──┤ T_b ├── 0   1 ──┤ C_br |
    └──────┘           └─────┘           └──────┘

Operations on the CTM environment are supported by :class:`yastn.tn.fpeps.EnvCTM`,
where each local CTM environment :class:`yastn.tn.fpeps.EnvCTM_local` can be accessed specifying site coordinates in :code:`[]`
and PEPS network of rank-4 tensors is available via attribute ``psi``.
The CTM environment class supports CTMRG updates for converging the environment, expectation value calculations,
bond metric, sampling, etc.

A single iteration of the CTMRG update, consisting of horizontal and vertical moves,
is performed with :meth:`yastn.tn.fpeps.EnvCTM.update_`.
Performing multiple updates is automatized in :meth:`yastn.tn.fpeps.EnvCTM.iterate_` (or equivalently :meth:`yastn.tn.fpeps.EnvCTM.ctmrg_`).
One can stop the CTM after a fixed number of iterations or, e.g., convergence of corner singular values.
Stopping criteria can also be set based on the convergence of one or more observables, e.g., total energy.
Once the CTMRG converges, it is straightforward to obtain one-site :meth:`yastn.tn.fpeps.EnvCTM.measure_1site` and
two-site nearest-neighbor observables :meth:`yastn.tn.fpeps.EnvCTM.measure_nn`, or other expectation values of interests.
Products of operators on a rectangular window of sites are contracted exactly by :meth:`yastn.tn.fpeps.EnvCTM.measure_nsite_exact`
and, with contraction-path optimization, bond unrolling and MPO-valued operators, by
:meth:`yastn.tn.fpeps.EnvCTM.measure_nsite_exact_oe`; see :doc:`measurement_oe`.


The CTMRG iteration
-------------------

One :meth:`yastn.tn.fpeps.EnvCTM.update_` performs a single CTMRG iteration. Each *move* in it
builds projectors from enlarged corners and then absorbs a row or column of the network into the
environment tensors:

* ``moves`` is the sequence of moves making up one iteration. ``'hv'`` (the default) updates all
  sites simultaneously in a horizontal and then a vertical move; ``'lrtb'`` executes left, right,
  top and bottom causally, row after row or column after column.
* ``method`` selects how projectors are built. ``'2x2'`` uses enlarged 2x2 corners forming a 4x4
  patch and can *grow* the environment bond dimension; ``'1x2'`` uses smaller 1x2 corners, which is
  considerably faster but less stable and cannot grow the bond dimension.
* ``opts_svd`` fixes the environment bond dimension, and is passed on to
  :meth:`yastn.linalg.svd_with_truncation`.

:meth:`yastn.tn.fpeps.EnvCTM.iterate_` (equivalently :meth:`yastn.tn.fpeps.EnvCTM.ctmrg_`) repeats
the iteration up to ``max_sweeps`` times, stopping early when ``corner_tol`` is met by the change of
corner singular values, or when a custom ``conv_check`` says so. With ``iterator=True`` it returns a
generator yielding after each sweep, so observables can be inspected as the environment converges.

Every option above, and the rest, is described in :doc:`ctm_options`.


Subspace-iteration mode
-----------------------

By default each projector pair comes from a truncated SVD of the enlarged corners, computed from
scratch on every sweep. In **subspace-iteration (SI) mode**
(`arXiv:2607.15158 <https://arxiv.org/abs/2607.15158>`_) the pair is instead obtained from a pair
of recycled range-finder bases :math:`X, Y`, refreshed by a few power iterations and carried over to
the next sweep. Successive CTMRG environments differ little once the iteration settles, so the bases
from the previous sweep are already a good starting guess, and the full decomposition can be
replaced by a much smaller one.

The mode is switched on per call::

    env.ctmrg_(opts_svd={'D_total': chi}, max_sweeps=100, corner_tol=1e-8,
               opts_si={'enabled': True, 'oversampling': 4, 'niter': 2})

Alongside the options, each projector pair accumulates its own *state* -- how many times it has been
updated (``age``), the subspace error its bases last reported, and where those bases came from.
That is :class:`yastn.tn.fpeps.envs.SI_state`, and it is what makes the schedule below depend on the
history of the run rather than on the options alone.

The chart below traces a single projector-pair update, naming the :class:`yastn.tn.fpeps.envs.SIOpts`
field that governs each branch.

.. graphviz::
    :caption: How SI options enter one projector-pair update. Diamonds are decisions; the
              italicised names are the options that control them.
    :align: center

    digraph si_update {
        rankdir=TB;
        node [fontname="sans-serif", fontsize=10];
        edge [fontname="sans-serif", fontsize=9];

        start   [shape=oval, label="projector pair\nfor this move"];
        enabled [shape=diamond, style=filled, fillcolor="#eef3fa", label="enabled?"];
        plain   [shape=box, style="rounded,filled", fillcolor="#f3f3f3",
                 label="proj_corners\ntruncated SVD of the corners"];

        due     [shape=diamond, style=filled, fillcolor="#eef3fa",
                 label="inside the\nredistribution window?\nage ≤ warmup, or every\nredistribute_frequency"];
        skip    [shape=diamond, style=filled, fillcolor="#eef3fa",
                 label="skip_SI_update\nand error < tol ?"];
        niter0  [shape=box, style="rounded,filled", fillcolor="#fdf6e3",
                 label="niter = 0\nreuse the bases as they are"];
        niterN  [shape=box, style="rounded,filled", fillcolor="#fdf6e3",
                 label="niter passes requested"];

        rank    [shape=box, style="rounded,filled", fillcolor="#f3f3f3",
                 label="rank = D_total + oversampling\ncapped by shared sector capacity"];
        compat  [shape=diamond, style=filled, fillcolor="#eef3fa",
                 label="recycled bases still\nfit the corners?"];
        rebase  [shape=diamond, style=filled, fillcolor="#eef3fa", label="rebase?"];
        carried [shape=box, style="rounded,filled", fillcolor="#e8f5e9", label="bases = 'rebased'"];
        redrawn [shape=box, style="rounded,filled", fillcolor="#fde8e8", label="bases = 'reinitialized'"];
        reused  [shape=box, style="rounded,filled", fillcolor="#e8f5e9", label="bases = 'reused'"];

        refine  [shape=box, style="rounded,filled", fillcolor="#f3f3f3",
                 label="si_refinement\nredistribute rank between sectors\n(refinement, adaptive_spectrum_iterations)"];
        power   [shape=box, style="rounded,filled", fillcolor="#f3f3f3",
                 label="subspace iteration\nup to niter passes,\nstop once error < tol"];
        trunc   [shape=box, style="rounded,filled", fillcolor="#f3f3f3",
                 label="truncate\n(tol, D_total, D_block, eps_multiplet, …)"];
        grad    [shape=diamond, style=filled, fillcolor="#eef3fa", label="recycle_grad?"];
        keep    [shape=box, style="rounded,filled", fillcolor="#f3f3f3",
                 label="store X, Y for the next sweep\nattached to / detached from the graph"];
        done    [shape=oval, label="projectors p0, p1\nSI_state updated"];

        start -> enabled;
        enabled -> plain    [label="  no  (default)"];
        enabled -> due      [label="  yes"];

        due -> skip         [label="  no"];
        due -> niterN       [label="  yes"];
        skip -> niter0      [label="  yes"];
        skip -> niterN      [label="  no"];

        niter0 -> rank;
        niterN -> rank;
        rank -> compat;

        compat -> reused    [label="  yes"];
        compat -> rebase    [label="  no"];
        rebase -> carried   [label="  yes, and it fits"];
        rebase -> redrawn   [label="  no, or it does not fit"];

        reused  -> refine;
        carried -> refine;
        redrawn -> refine;

        refine -> power     [label="  redistribute_sectors\l  and window open"];
        refine -> power     [label="  otherwise: skipped", style=dashed];
        power -> trunc -> grad -> keep -> done;
        plain -> done;
    }

Two points the chart makes concrete. ``skip_SI_update`` does not skip the projector -- it skips only
the *power iterations*, reusing the incoming bases, and it is deliberately disabled while the
redistribution window is open so a pair is never frozen before its sectors have settled. And
``rebase`` matters whenever the environment bond dimension is still growing: without it, every
change of the corner legs throws the accumulated bases away and redraws them from noise.

Note that the SI path truncates on a narrower set of ``opts_svd`` keys than the default path; the
set is named once, as ``SI_TRUNCATION_KEYS`` in :mod:`yastn.tn.fpeps.envs`.


API
---

.. autoclass:: yastn.tn.fpeps.EnvCTM
    :members: to_dict, from_dict, reset_, bond_metric, update_, update_bond_, iterate_, ctmrg_, measure_1site, measure_nn, 
              sample, measure_2x2, measure_line, measure_nsite, measure_nsite_exact, measure_2site, transfer_matrix_spectrum

.. autoclass:: yastn.tn.fpeps.envs.EnvCTM_local

.. autoclass:: yastn.tn.fpeps.envs.SI_state

.. seealso::

    :doc:`ctm_options` for every option accepted above, and
    :doc:`fixed_point` for the differentiable fixed-point variant.
