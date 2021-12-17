Basics concepts
===============

Tensors
-------

In general, tensors are multilinear maps from products of several vector spaces

.. math::

    T = \sum_{abc...ijk...} T^{abc...}_{ijk...} e^ie^je^k...e_ae_be_c...

In some contexts, its often useful to distinguish underlying spaces as co- or contra-variant
with respect to transformations acting on these spaces. Often such distinction is encoded
through position of the indices - subscript or superscript.

In quantum mechanics, it is useful to distinguish between :math:`\langle bra |` 
and :math:`|ket \rangle` spaces, due to different action of symmetry transformations on these spaces 

.. math::

    T = \sum_{abc...ijk...} T^{abc...}_{ijk...} |i \rangle|j \rangle|k \rangle ... 
    \langle a |\langle b |\langle c |...

In YAST, similar to other implementations [cite ITensor, TenPy], the distinction between
:math:`\langle bra |` and :math:`|ket \rangle` spaces, or Hilbert space :math:`\mathcal{H}` and its dual :math:`\mathcal{H}^*`, is encoded through `signature`.

.. note::
    `signature`, :attribute:`yast.Tensor.s`, is a tuple/list/1-D array of signs :math:`\pm 1`


Conjugation
-----------