Overlap of two MPSs
===================

:ref:`Measurement<theory/mps/basics:Measurements>` of the overlap between two MPSs is calculated by contracting a network formed by MPS and conjugate of another or the same MPS. In the latter case, :code:`yastn.tn.mps.measure_overlap(psi,psi)`, 
returns the norm of MPS :code:`psi`.

.. autofunction:: yastn.tn.mps.measure_overlap


Expectation value of MPO
========================

:ref:`Measurement<theory/mps/basics:Measurements>` of MPO's expectation value 
is calculated by contracting a network formed by MPS, MPO, and a conjugate 
of the same or different MPS.

.. autofunction:: yastn.tn.mps.measure_mpo


Shorthand notation
==================

.. autofunction:: yastn.tn.mps.vdot