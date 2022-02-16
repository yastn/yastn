Linear algebra with symmetric tensors
=====================================

Basic algebra operations
------------------------

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: TestSyntaxBasicAlgebra.test_syntax_basic_algebra


Tensor contractions
-------------------

Basic contractions with :meth:`yast.tensordot`,  matrix-multiplication operator ``@``, tracing with :meth:`yast.trace` 

.. literalinclude:: /../../tests/tensor/test_syntax.py
   :pyobject: TestSyntaxContractions.test_syntax_contraction

Higher-level interface ``ncon`` composing simple contractions

.. literalinclude:: /../../tests/tensor/test_ncon.py
   :pyobject: test_ncon_syntax


Transposition
-------------

.. literalinclude:: /../../tests/tensor/test_transpose.py
   :pyobject: TestSyntaxTranspose.test_transpose_syntax


Fusion (reshaping)
------------------

Following example showcases fusion, in particular its `hard` mode.
In this case, the tensor data is reshuffled/resized in memory.

.. literalinclude:: /../../tests/tensor/test_fuse_hard.py
   :pyobject: FusionSyntax.test_fuse_hard


Conjugation of symmetric tensors
--------------------------------

.. literalinclude:: /../../tests/tensor/test_conj.py
   :pyobject: TestConj_Z2xU1.test_conj_Z2xU1