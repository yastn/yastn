SU(2) symmetric tensors
=======================

``yastn.Tensor`` is intentionally an Abelian block-sparse tensor.  SU(2)
fusion branches, for example ``1/2 x 1/2 = 0 + 1``, and cannot be represented
correctly by that one-output-per-charge data model.  SU(2) support therefore
lives in the separate :mod:`yastn.su2` API and does not change any existing
Tensor input/output format.

Irreps use the exact integer label ``two_j = 2j``.  A leg's ``D`` is the
degeneracy dimension, excluding the magnetic dimension ``two_j + 1``.

.. code-block:: python

    import numpy as np
    import yastn

    spin_half = yastn.SU2Leg(t=(1,), D=(1,))
    singlet = yastn.SU2Tensor(
        (spin_half, spin_half),
        {((1, 1), ()): np.array(1.0)})

    dense = singlet.to_dense()
    restored = yastn.SU2Tensor.from_dense(dense, (spin_half, spin_half))

Each reduced block key is ``(irreps, path)``.  ``path`` gives the intermediate
``two_j`` values of a left-associated Clebsch--Gordan tree and explicitly
keeps independent fusion channels.  For four spin halves the two singlets use
paths ``(0, 1)`` and ``(2, 1)``.

``SU2Tensor.tensordot`` contracts conjugate legs (opposite signatures) with
the SU(2) invariant pairing, then projects the result back to reduced blocks.
It raises if the result contains a non-invariant component.  Dictionaries
produced by ``SU2Tensor.to_dict`` have ``type='SU2Tensor'`` and can also be
loaded through ``yastn.from_dict``; old ``Tensor`` dictionaries retain their
original version and representation.
