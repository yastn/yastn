Testing for NTU algorithms
==========================

NTU
----

In order to execute the NTU algorithm :ref:`NTU<peps/algorithms_NTU:Neighborhood tensor update (NTU) algorithm>` for calulating thermal states, 
we need to initialize the PEPS at each site as a maximaly entangled state of its physical and ancilliary degrees of freedom :ref:`Purification<theory/peps/purifcation:Purification>`.
Then we need to perform imaginary time evolution with NTU and subsequntly calculate expectation values using 
CTMRG :ref:`CTMRG<peps/algorithms_CTMRG:Corner transfer matrix renormalization group (CTMRG) algorithm>`.

Here is a simple example for how to obtain thermal states of 2D spinless fermi sea for finite and infinite peps at high tempeartures respectively:

.. literalinclude:: /../../tests/peps/test_NTU_CTM_purification_spinless_fermions.py
        :pyobject: test_NTU_spinless_finite
        :pyobject: test_NTU_spinless_infinite


Here is a simple example for how to obtain thermal states of spinful fermi sea for finite and infinite peps at high temperatures respectively:

.. literalinclude:: /../../tests/peps/test_NTU_CTM_purification_spinful_fermions.py
        :pyobject: test_NTU_spinful_finite
        :pyobject: test_NTU_spinful_infinite

