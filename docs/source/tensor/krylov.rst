Krylov methods
==============

Implemented in yast
-------------------

We provide a high-level, backend-agnostic implementation of some Krylov-based algorithms used in :class:`yast.tn.mps`.
They assume a linear operation acting on a generalized vector, with the vector being an instance of a class that includes methods
``norm``, ``vdot``, ``linear_combination``, ``expand_krylov_space``, among others.
Examples of such a vector include :class:`yast.Tensor` (see :ref:`methods<tensor-aux>`). 

.. autofunction:: yast.expmv
.. autofunction:: yast.eigs

Other libraries
---------------
With numpy backend, it is possible to link to algorithms in 
`sparse.sparse.linalg <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_, 
employing :meth:`yast.compress_to_1d` and :meth:`yast.decompress_from_1d`. 
See, an :ref:`example<examples/tensor/decomposition:scipy.sparse.linalg.eigs>`.


