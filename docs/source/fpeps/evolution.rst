Time evolution
==============

Evolution step
--------------

Time evolution in YASTN operates via the Trotterization of
the evolution operator to a set of gates. Each gate acts on a path of sites,
single site for a local gate, nearest-neighbor sites for a two-site gate, etc.
A single evolution step consists of the application of trotterized gates,
together with truncation of the bond dimension after each application of a non-local gate.
For a detailed explanation, please refer to the :ref:`Basic concepts/Time evolution <theory/fpeps/basics:Time evolution>`.

The main method for performing a time evolution step is :meth:`yastn.tn.fpeps.evolution_step_`.
This function requires an environment containing the PEPS state (updated in place) and provides control
over bond truncation via different ways to calculate **bond metric**. Classes which support this include:

- :class:`yastn.tn.fpeps.EnvNTU`: Employ small local clusters, which can be contracted numerically exactly, resulting in a stable and positively defined bond metric.
- :class:`yastn.tn.fpeps.EnvBP`: Employ belief propagation, either to define a bipartite bond metric (equivalent to simple update), or to gauge NTU-like clusters.
- :class:`yastn.tn.fpeps.EnvCTM`: Employ CTMRG environment to calculate the bond metric, with information from the entire network, allowing for a full update approach.
- :class:`yastn.tn.fpeps.EnvApproximate`: Employ local clusters of sizes beyond exact contraction, utilizing approximate boundary MPS to calculate the bond metric.

.. autofunction:: yastn.tn.fpeps.evolution_step_

Auxiliary functions, such as :meth:`yastn.tn.fpeps.accumulated_truncation_error`, assist with tracking cumulative
truncation error across multiple evolution steps, facilitating error analysis in time evolution simulations.

.. autofunction:: yastn.tn.fpeps.accumulated_truncation_error

To apply a gate to a PEPS, without truncating the bond dimension, one can use :meth:`yastn.tn.fpeps.Peps.apply_gate_`.
Truncation of the bond dimension can be done later, e.g., via :meth:`yastn.tn.fpeps.truncate_`.
Both operations modify the PEPS in place.

.. automethod:: yastn.tn.fpeps.Peps.apply_gate_
.. automethod:: yastn.tn.fpeps.truncate_


Gates
-----

:meth:`yastn.tn.fpeps.evolution_step_` takes a list of gates to be applied on PEPS

.. autoclass:: yastn.tn.fpeps.Gate

An auxiliary function :meth:`yastn.tn.fpeps.gates.distribute`
distribute a set of gates homogeneously over the entire lattice.
By default, it complements gates with their adjoint (gates repeated in reverse order),
forming a 2nd order approximation for a small time step -- this assumes that individual gates take half of the time step.

.. autofunction:: yastn.tn.fpeps.gates.distribute


Predefined gates
----------------

Some predefined gates can be found in :code:`yastn.tn.fpeps.gates`, including

.. autofunction:: yastn.tn.fpeps.gates.gate_nn_hopping
.. autofunction:: yastn.tn.fpeps.gates.gate_nn_Ising
.. autofunction:: yastn.tn.fpeps.gates.gate_nn_Heisenberg
.. autofunction:: yastn.tn.fpeps.gates.gate_local_Coulomb
.. autofunction:: yastn.tn.fpeps.gates.gate_local_occupation
.. autofunction:: yastn.tn.fpeps.gates.gate_local_field
.. autofunction:: yastn.tn.fpeps.gates.gate_nn_tJ


Custom gates
------------

Auxiliary functions help generate gates by exponentiating hermitian Hamiltonians,
generate two-site Hamiltonian ensuring proper fermionic order, etc.

.. autofunction:: yastn.tn.fpeps.fkron
.. autofunction:: yastn.tn.fpeps.gates.gate_nn_exp
.. autofunction:: yastn.tn.fpeps.gates.gate_local_exp
.. autofunction:: yastn.tn.fpeps.gates.decompose_nn_gate
