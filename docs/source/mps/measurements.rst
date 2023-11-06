
Expectation values
==================

Examples can be found :ref:`here <examples/mps/measurement:Expectation values>`.

Overlaps and MPO expectation values
-----------------------------------

:ref:`Measurement<theory/mps/basics:Measurements>` of the overlap between two MPSs is calculated
by contracting a network formed by MPS and conjugate of another or the same MPS. Works also for two MPOs.

.. autofunction:: yastn.tn.mps.measure_overlap

While the above allows calculating the norm of MPS or MPO, a dedicated method extracts the norm through canonization.
This can be more precise for norms close to zero.

.. autofunction:: yastn.tn.mps.MpsMpo.norm

:ref:`Measurement<theory/mps/basics:Measurements>` of MPO's expectation value
is calculated by contracting a network formed by MPS, MPO, and a conjugate
of the same or different MPS. Works also for three MPOs.

.. autofunction:: yastn.tn.mps.measure_mpo

Shorthand notation
------------------

The above function can be also executed by

.. autofunction:: yastn.tn.mps.vdot

One- and two-point expectation values
-------------------------------------

.. autofunction:: yastn.tn.mps.measure_1site

.. autofunction:: yastn.tn.mps.measure_2site


Schmidt values and entropy profile
----------------------------------

The Schmidt values are computed by performing bipartition of the MPS/MPO across
each of the bonds. This amounts to SVD decomposition with respect to a bond,
where all sites to the left are in :ref:`left-canonical form<theory/mps/basics:canonical form>` and all sites to the right
are in :ref:`right-canonical form<theory/mps/basics:canonical form>`.

.. autofunction:: yastn.tn.mps.MpsMpo.get_Schmidt_values

.. autofunction:: yastn.tn.mps.MpsMpo.get_entropy
