Time evolution
==============

Evolution step
--------------

Time evolution in YASTN operates via the Trotterization of
evolution operator to a set of nearest-neighbor and local gates.
A single evolution step consists of the application of trotterized gates,
together with truncation of the bond dimension after each application of a nearest-neighbor gate.
For a detailed explanation, please refer to the `Time Evolution <../theory/fpeps/basics.html>`_ section .

The main method for performing a time evolution step is :meth:`yastn.tn.fpeps.evolution_step_`. This 
function requires an environment containing the PEPS state (which is updated in place) and provides control 
over bond truncation via different ways to calculate `bond_metric`. Three classes which support this are:

- **:class:`yastn.tn.fpeps.EnvNTU`**: Suitable for local truncation with faster simulation.
- **:class:`yastn.tn.fpeps.EnvCTM`**: Used for fast full update schemes with high accuracy.
- **:class:`yastn.tn.fpeps.EnvApproximate`**: A lower-cost alternative for approximate updates on large clusters (Marek ?).


.. autofunction:: yastn.tn.fpeps.evolution_step_

Auxiliary functions, such as :meth:`yastn.tn.fpeps.accumulated_truncation_error`, assist with tracking cumulative 
truncation error across multiple evolution steps, facilitating error analysis in time evolution simulations.

.. autofunction:: yastn.tn.fpeps.accumulated_truncation_error


Gates
-----

Gates classes are organized as

.. autoclass:: yastn.tn.fpeps.Gates
.. autoclass:: yastn.tn.fpeps.Gate_nn
.. autoclass:: yastn.tn.fpeps.Gate_local

Some predefined gates can be found in :code:`yastn.tn.fpeps.gates`, including

.. autofunction:: yastn.tn.fpeps.gates.gate_nn_hopping
.. autofunction:: yastn.tn.fpeps.gates.gate_nn_Ising
.. autofunction:: yastn.tn.fpeps.gates.gate_local_Coulomb
.. autofunction:: yastn.tn.fpeps.gates.gate_local_occupation
.. autofunction:: yastn.tn.fpeps.gates.gate_local_field

An auxiliary function :meth:`yastn.tn.fpeps.gates.distribute`
distribute a set of gates homogeneously over the entire lattice.

.. autofunction:: yastn.tn.fpeps.gates.distribute
