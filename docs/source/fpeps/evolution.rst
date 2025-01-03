Time evolution
==============

Evolution step
--------------

Time evolution in YASTN operates via the Trotterization of
the evolution operator to a set of nearest-neighbor and local gates.
A single evolution step consists of the application of trotterized gates,
together with truncation of the bond dimension after each application of a nearest-neighbor gate.
For a detailed explanation, please refer to the :ref:`Basic concepts/Time evolution <theory/fpeps/basics:Time evolution>`.

The main method for performing a time evolution step is :meth:`yastn.tn.fpeps.evolution_step_`. This
function requires an environment containing the PEPS state (updated in place) and provides control
over bond truncation via different ways to calculate **bond metric**. Three classes which support this are:

- :class:`yastn.tn.fpeps.EnvNTU`: Employ small local clusters, which can be contracted numerically exactly, resulting in a stable and positively defined bond metric.
- :class:`yastn.tn.fpeps.EnvCTM`: Employ CTMRG environment to calculate the bond metric, with information from the entire network, allowing for a fast full update approach.
- :class:`yastn.tn.fpeps.EnvApproximate`: Employ local clusters of sizes beyond exact contraction, utilizing approximate boundary MPS to calculate the bond metric.


.. autofunction:: yastn.tn.fpeps.evolution_step_

Auxiliary functions, such as :meth:`yastn.tn.fpeps.accumulated_truncation_error`, assist with tracking cumulative
truncation error across multiple evolution steps, facilitating error analysis in time evolution simulations.

.. autofunction:: yastn.tn.fpeps.accumulated_truncation_error


Gates
-----

Gates classes used in :meth:`yastn.tn.fpeps.evolution_step_` are organized as

.. autoclass:: yastn.tn.fpeps.Gates
.. autoclass:: yastn.tn.fpeps.Gate_nn
.. autoclass:: yastn.tn.fpeps.Gate_local

An auxiliary function :meth:`yastn.tn.fpeps.gates.distribute`
distribute a set of gates homogeneously over the entire lattice.

.. autofunction:: yastn.tn.fpeps.gates.distribute


Predefined gates
----------------

Some predefined gates can be found in :code:`yastn.tn.fpeps.gates`, including

.. autofunction:: yastn.tn.fpeps.gates.gate_nn_hopping
.. autofunction:: yastn.tn.fpeps.gates.gate_nn_Ising
.. autofunction:: yastn.tn.fpeps.gates.gate_local_Coulomb
.. autofunction:: yastn.tn.fpeps.gates.gate_local_occupation
.. autofunction:: yastn.tn.fpeps.gates.gate_local_field
