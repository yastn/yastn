Test of NTU algorithm
=======================

NTU
----

In order to execute the neighborhood tensor update algorithm :ref:`NTU<fpeps/Optimization_algorithms:Neighborhood tensor update (NTU) algorithm>` for calulating thermal states, 
we need to initialize the PEPS at each site as a maximaly entangled state of its physical and ancilliary degrees of freedom :ref:`Purification<theory/fpeps/purification:Purification>`.
Then we need to perform imaginary time evolution with NTU and subsequently calculate expectation values using 
Corner Transfer Matrix Renormalization Algorithm :ref:`CTMRG<fpeps/expectation_values:Corner transfer matrix renormalization group (CTMRG) algorithm>`.

Thermal expectation value of spinless fermi sea
-----------------------------------------------

.. literalinclude:: /../../tests/peps/test_NTU_CTM_purification_spinless_fermions.py
        :pyobject: test_NTU_spinless_finite

.. literalinclude:: /../../tests/peps/test_NTU_CTM_purification_spinless_fermions.py
        :pyobject: test_NTU_spinless_infinite

Thermal expectation value of spinful fermi sea
----------------------------------------------

.. literalinclude:: /../../tests/peps/test_NTU_CTM_purification_spinful_fermions.py
        :pyobject: test_NTU_spinful_finite

.. literalinclude:: /../../tests/peps/test_NTU_CTM_purification_spinful_fermions.py
        :pyobject: test_NTU_spinful_infinite

