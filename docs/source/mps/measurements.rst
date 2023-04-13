Overlap of two MPSs
===================

:ref:`Measurement<theory/mps/basics:Measurements>` of the overlap between two MPSs is calculated by contracting a network formed by MPS and conjugate of another or the same MPS. 

.. autofunction:: yastn.tn.mps.measure_overlap


While the above allows calculating the norm of MPS or MPO, a dedicated method extracts the norm through canonization.
This can be more precise for norms close to zero.

.. autofunction:: yastn.tn.mps.MpsMpo.norm

Expectation value of MPO
========================

:ref:`Measurement<theory/mps/basics:Measurements>` of MPO's expectation value 
is calculated by contracting a network formed by MPS, MPO, and a conjugate 
of the same or different MPS.

.. autofunction:: yastn.tn.mps.measure_mpo


Shorthand notation
==================

.. autofunction:: yastn.tn.mps.vdot

