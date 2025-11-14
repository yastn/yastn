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

.. autoclass:: yastn.tn.fpeps.EnvCTM
    :members: to_dict, from_dict, reset_, bond_metric, update_, iterate_, measure_1site, measure_nn, sample, measure_2x2, measure_line, measure_nsite

.. autoclass:: yastn.tn.fpeps.envs.EnvCTM_local
