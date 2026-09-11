Linear algebra with symmetric tensors
=====================================

.. code-block:: python

   import yastn
   import pytest
   config_kwargs = {"backend": "np"}

Basic algebra operations
------------------------

The example below demonstrates basic algebraic operations on symmetric tensors.

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: test_syntax_basic_algebra


Tensor contractions
-------------------

Basic contractions with :meth:`yastn.tensordot`, the matrix-multiplication operator ``@``, and tracing with :meth:`yastn.trace`.

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: test_syntax_contraction

The higher-level interface ``ncon`` (or equivalently ``einsum``) composes simple contractions.

.. literalinclude:: /../../tests/tensor/test_ncon_einsum.py
   :pyobject: test_ncon_einsum_syntax


Transposition
-------------

.. literalinclude:: /../../tests/tensor/test_transpose.py
   :pyobject: test_transpose_syntax


Fusion (reshaping)
------------------

The following example showcases fusion, in particular its ``'hard'`` mode (the default).
In this case, the tensor data is reshuffled and resized in memory.

.. literalinclude:: /../../tests/tensor/test_fuse_hard.py
   :pyobject: test_fuse_hard


Conjugation of symmetric tensors
--------------------------------

.. literalinclude:: /../../tests/tensor/test_conj.py
   :pyobject: test_conj_Z2xU1
