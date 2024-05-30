Test of NTU algorithm
=====================


In order to execute the neighborhood tensor update algorithm :ref:`NTU<theory/fpeps/basics:Neighborhood tensor update (NTU)>` for calulating thermal states,
we need to initialize the PEPS at each site as a maximaly entangled state of its physical and ancilliary degrees of freedom :ref:`Purification<theory/fpeps/basics:Purification>`.
Then we need to perform imaginary time evolution with NTU and subsequently calculate expectation values using
Corner Transfer Matrix Renormalization Algorithm :ref:`CTMRG<fpeps/environments:Corner transfer matrix renormalization group (CTMRG) algorithm>`.


Thermal expectation value of spinful fermi sea
----------------------------------------------

.. literalinclude:: /../../tests/peps/test_purification_Hubbard.py
        :pyobject: test_NTU_spinful_finite

.. literalinclude:: /../../tests/peps/test_purification_Hubbard.py
        :pyobject: test_NTU_spinful_infinite

