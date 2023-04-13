Building MPO from LaTex
=======================

The MPO can be constructed automatically using dedicated :class:`yastn.tn.mps.Generator` supplied with the set of local operators.
Automatic generator creates MPO with symmetries inherited from local operators.

Hamiltonian for nearest-neighbor hopping/XX model
-------------------------------------------------

.. literalinclude:: /../../tests/mps/test_generator.py
        :pyobject: mpo_nn_hopping_latex

Spinless fermions with hopping at arbitrary range
-------------------------------------------------

.. literalinclude:: /../../tests/mps/test_generator.py
        :pyobject: mpo_hopping_latex
