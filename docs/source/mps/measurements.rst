Ovelap between two MPSs
=========================

:ref:`theory/mps/basics:Measurements` of the overlap between two MPSs is calculated by contracting a MPS and a conjugation of another (or the same) MPS. The ovelap :code:`yamps.measure_overlap(psi,psi)` returns a norm of :code:`psi`.

.. autofunction:: yamps.measure_overlap


Expectation value of MPO
=========================

:ref:`theory/mps/basics:Measurements` of the expectation value of the operator written as MPO is equivalent to a contraction of MPS, MPO and conjucation of (the same and another) MPS. 
The ovelap :code:`yamps.measure_mpo(psi,op,psi)` returns an expectation value of operator :code:`op` for :code:`psi` state.

.. autofunction:: yamps.measure_mpo

