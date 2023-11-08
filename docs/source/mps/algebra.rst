Algebra
=======

Multiplication by a scalar
---------------------------

MPS/MPO can be multiplied by a scalar using regular :code:`*` and :code:`/` operators,
i.e.,:code:`B = a * A`, :code:`B = A * a` or :code:`B = A / a`.

Addition of MPS/MPO
-------------------

Two MPS's or two MPO's can be added up, provided that their length, physical dimensions, and symmetry agree.
The sum of two such objects :code:`A` and :code:`B` results in new MPS/MPO :code:`C = A + B`, with tensor of :code:`C` given by the direct sum of :code:`A`'s
and :code:`B`'s tensors along virtual dimension.

::

    # a product of two MPS's
               ___     ___    ___    ___      ___      ___    ___    ___
      A       |___|-D-|___|--|___|--|___|    |   |    |   |  |   |  |   |
 C =  +  =  __ d|   ___ |  ___ |  ___ |   =  |   |-2D-|   |--|   |--|   |
      B    |___|:D-|___|:-|___|:-|___|:      |___|    |___|  |___|  |___|
            d|  |    |  |   |  |   |  |       d|        |      |      |
              \/      \/     \/     \/

To add many MPS/MPOs :math:`A_0, A_1,\dots` at once, use :code:`yastn.tn.mps.add(A_0, A_1,...)`.

.. autofunction:: yastn.tn.mps.add

Following example show an addition of two MPSs:

::

    import yastn
    import yastn.tn.mps as mps

.. literalinclude:: /../../tests/mps/test_algebra.py
        :pyobject: addition_example


Products of MPO and MPS
-----------------------

API supports *product* ``@`` of

i) MPO with MPS resulting in a new MPS in analogy with
:math:`\hat{O}|\psi\rangle = |\phi\rangle` (i.e. matrix-vector multiplication).

::

 # a product of MPO O and MPS A

              ___        ___         ___
             |___|--D'--|___|--...--|___|      ___        ___         ___
 C = O @ A =  _|_        _|_         _|_   =  |___|-DxD'-|___|--...--|___|
             |___|--D --|___|--...--|___|       |          |           |
               |          |           |

ii) two MPOs resulting in a new MPO corresponding to usual
operator product :math:`\hat{O}\hat{P} = \hat{C}` (matrix-matrix multiplication).

::

 # a product of two MPO's O and P

              _|d_       _|_         _|_
             |___|--D---|___|--...--|___|    _|d          _|_         _|_
 C = O @ P =  _|_        _|_         _|_  = |___|--DxD'--|___|--...--|___|
             |___|--D'--|___|--...--|___|     |d           |           |
               |d         |           |

One can either use call :code:`C = A @ B` or in a more verbose form
:code:`C = mps.multiply(A, B)`. See examples here: :ref:`examples/mps/algebra:Multiplication`.

.. autofunction:: yastn.tn.mps.multiply


Multiplication with truncation
------------------------------

A fast procedure to multiply MPO by MPO/MPS while performing truncation is a `zipper`.
The result can be subsequently fine-tuned using :ref:`variational optimization<mps/algorithms_overlap:Variational overlap maximalization>`.

.. autofunction:: yastn.tn.mps.zipper
