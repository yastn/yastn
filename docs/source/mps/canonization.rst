Canonizing MPS/MPO
==================

MPS/MPO can be put into :ref:`canonical form<theory/mps/basics:Canonical form>` to reveal most advantageous truncation or as a part of the setup for
:ref:`DMRG<mps/algorithms_dmrg:Density matrix renormalization group (DMRG)>` or
:ref:`TDVP<mps/algorithms_tdvp:Time-dependent variational principle (TDVP)>` algorithms.

The canonical form obtained by QR decomposition is fast, but does not allow for truncation
of the virtual spaces of MPS/MPO.

.. automethod:: yastn.tn.mps.MpsMpo.canonize_

See examples: :ref:`examples/mps/algebra:canonical form by qr`.

The canonization by SVD allows truncating virtual dimension/spaces
with the lowest weight (smallest singular values).

.. automethod:: yastn.tn.mps.MpsMpo.truncate_

See examples: :ref:`examples/mps/algebra:canonical form by svd`.
