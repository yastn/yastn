Krylov methods
==============

Implemented in yastn
--------------------

We provide a high-level, backend-agnostic implementation of some Krylov-based algorithms used in :class:`yastn.tn.mps`.
They assume a linear operation acting on a generalized vector, with the vector being an instance of a class that includes methods
``norm``, ``vdot``, ``linear_combination``, ``expand_krylov_space``, among others.
Examples of such a vector include :class:`yastn.Tensor` (see :ref:`methods<tensor-aux>`).

.. autofunction:: yastn.expmv
.. autofunction:: yastn.eigs
.. autofunction:: yastn.lin_solve

Other libraries
---------------
With *NumPy* backend, it is possible to link to algorithms in
`sparse.sparse.linalg <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_,
employing :meth:`yastn.Tensor.compress_to_1d` and :meth:`yastn.decompress_from_1d`.
See, an :ref:`example<examples/tensor/decomposition:combining with scipy.sparse.linalg.eigs>`.
