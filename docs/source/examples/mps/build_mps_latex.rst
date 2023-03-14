Building MPS/MPO from LaTex
===========================

Hopping spinless fermions
-------------------------

The MPO can be constructed automatically using dedicated :class:`yast.tn.mps.Generator` supplied with set of local operators.
Automatic generator creates MPO with symmetries inherited from local operators.

.. todo::
    
    Split example and test
        

.. literalinclude:: /../../tests/mps/test_generator.py
        :pyobject: test_generator_mpo


Radnom hopping spinless fermions
--------------------------------

.. literalinclude:: /../../tests/mps/test_generator.py
        :pyobject: test_mpo_from_latex