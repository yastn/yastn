Canonizing MPS/MPO
==================

MPS/MPO can be put into :ref:`canonical form<theory/mps/basics:Canonical form>` to reveal the most advantageous truncation
or as a part of the setup for many algorithms, including
:ref:`DMRG<mps/algorithms_dmrg:Density matrix renormalization group (DMRG)>` and
:ref:`TDVP<mps/algorithms_tdvp:Time-dependent variational principle (TDVP)>`.

The canonical form obtained by QR decomposition is fast, but does not allow for truncation
of the virtual spaces of MPS/MPO.

.. automethod:: yastn.tn.mps.MpsMpoOBC.canonize_

See examples: :ref:`examples/mps/algebra:canonical form by qr`.

The canonization by SVD allows truncating virtual dimension/spaces
with the lowest weights (smallest singular values).

.. automethod:: yastn.tn.mps.MpsMpoOBC.truncate_

See examples: :ref:`examples/mps/algebra:canonical form by svd`.

The above routines operate on entire MPS or MPO.
They are built from functions operating on individual tensors of MPS/MPO:

.. automethod:: yastn.tn.mps.MpsMpoOBC.orthogonalize_site_
.. automethod:: yastn.tn.mps.MpsMpoOBC.absorb_central_
.. automethod:: yastn.tn.mps.MpsMpoOBC.diagonalize_central_
