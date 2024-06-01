Time evolution
==============

Time evolution in YASTN operates via Trotterization of evolution operator
to a set of nearet-neighbor and local gates organized as

.. autoclass:: yastn.tn.fpeps.Gates

.. autoclass:: yastn.tn.fpeps.Gate_nn

.. autoclass:: yastn.tn.fpeps.Gate_local

Some predefined gates can be found in :code:`yastn.tn.fpeps.gates`

A single time-step is performed with

.. autofunction:: yastn.tn.fpeps.evolution_step_

It takes as he first argument an environment, which containes the PEPS state (updated in place), and
a method to calculate `bond_metric`. A primary class in this context is :class:`yastn.tn.fpeps.EnvNTU`,
however :class:`yastn.tn.fpeps.EnvCTM` can be used for a (fast) full update scheme, and
:class:`yastn.tn.fpeps.EnvApproximae` for claster updates that require approximate contraction.

