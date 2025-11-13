Krylov methods
==============

Implemented in yastn
--------------------

We provide a high-level, backend-agnostic implementation of some Krylov-based algorithms used in :class:`yastn.tn.mps`.
They assume a linear operation acting on a generalized vector, with the vector being an instance of a class that includes methods
``norm``, ``vdot``, ``add``, ``expand_krylov_space``, among others.
Examples of such a vector include :class:`yastn.Tensor` (see :ref:`methods<tensor-aux>`).

.. autofunction:: yastn.expmv
.. autofunction:: yastn.eigs
.. autofunction:: yastn.lin_solver

Other libraries
---------------
With *NumPy* backend, it is possible to link to algorithms in
`sparse.sparse.linalg <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_,
employing :meth:`yastn.Tensor.to_dict` and :meth:`yastn.Tensor.from_dict`, combined
with :meth:`yastn.split_data_and_meta` and :meth:`yastn.combine_data_and_meta`.
See, an :ref:`example<examples/tensor/decomposition:combining with scipy.sparse.linalg.eigs>`.
