Canonizing MPS/MPO
==================

MPS/MPO can be put into :ref:`theory/mps/basics:Canonical form` to reveal most advantageous truncation or as a part of the setup for
:ref:`DMRG<mps/algorithms_dmrg:density matrix renormalization group (dmrg) algorithm>` or
:ref:`TDVP<mps/algorithms_tdvp:time-dependent variational principle (tdvp) algorithm>` algorithms.

The canonical form obtained by QR decomposition is fast, but does not allow for truncation
of the virtual spaces of MPS/MPO.

.. autofunction:: yastn.tn.mps.MpsMpo.canonize_

See examples: :ref:`examples/mps/algebra:canonical form by qr`.

Restoring canonical form locally: For example, while performing DMRG sweeps,
the tensors getting updated will not be in canonical form after the update.
It is necessary to restore their canonical form during sweeping.

.. autofunction:: yastn.tn.mps.MpsMpo.orthogonalize_site

The canonization by `singular value decomposition` (SVD) allows
to truncate virtual dimension/spaces with the lowest weight
(lowest singular values).

.. autofunction:: yastn.tn.mps.MpsMpo.truncate_

See examples: :ref:`examples/mps/algebra:canonical form by svd`.
