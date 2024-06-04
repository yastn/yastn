Time evolution
==============

Evolution step
--------------

Time evolution in YASTN operates via the Trotterization of
evolution operator to a set of nearest-neighbor and local gates.
A single evolution step consists of the application of trotterized gates,
together with truncation of the bond dimension after each application of a nearest-neighbor gate.
It is performed by :meth:`yastn.tn.fpeps.evolution_step_`, which takes as the first argument
an environment, which contains the PEPS state (updated in place) and a method to calculate `bond_metric`.
A primary class in this context is :class:`yastn.tn.fpeps.EnvNTU`.
:class:`yastn.tn.fpeps.EnvCTM` can be used for a (fast) full update scheme, and
:class:`yastn.tn.fpeps.EnvApproximate` for cluster updates with approximate contraction.

.. autofunction:: yastn.tn.fpeps.evolution_step_

Auxiliary function helps to accumulate truncation errors stored in output informaion of`` many evolution steps.

.. autofunction:: yastn.tn.fpeps.accumulated_truncation_error


Gates
-----

Gates classes are organized as

.. autoclass:: yastn.tn.fpeps.Gates
.. autoclass:: yastn.tn.fpeps.Gate_nn
.. autoclass:: yastn.tn.fpeps.Gate_local

Some predefined gates can be found in :code:`yastn.tn.fpeps.gates`, including

.. autofunction:: yastn.tn.fpeps.gates.gate_nn_hopping
.. autofunction:: yastn.tn.fpeps.gates.gate_local_Coulomb
.. autofunction:: yastn.tn.fpeps.gates.gate_local_occupation

An auxiliary function :meth:`yastn.tn.fpeps.gates.distribute`
distribute a set of gates homogeneously over the entire lattice.

.. autofunction:: yastn.tn.fpeps.gates.distribute
