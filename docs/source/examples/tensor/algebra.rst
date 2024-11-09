Linear algebra with symmetric tensors
=====================================

Basic algebra operations
------------------------

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: test_syntax_basic_algebra


Tensor contractions
-------------------

Basic contractions with :meth:`yastn.tensordot`,  matrix-multiplication operator ``@``, tracing with :meth:`yastn.trace`

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: test_syntax_contraction

Higher-level interface ``ncon`` (or equivalently ``einsum``) composing simple contractions

.. literalinclude:: /../../tests/tensor/test_ncon_einsum.py
   :pyobject: test_ncon_einsum_syntax


Transposition
-------------

.. literalinclude:: /../../tests/tensor/test_transpose.py
   :pyobject: test_transpose_syntax


Fusion (reshaping)
------------------

Following example showcases fusion, in particular its `hard` mode.
In this case, the tensor data is reshuffled/resized in memory.

.. literalinclude:: /../../tests/tensor/test_fuse_hard.py
   :pyobject: test_fuse_hard


Conjugation of symmetric tensors
--------------------------------

.. literalinclude:: /../../tests/tensor/test_conj.py
   :pyobject: test_conj_Z2xU1
