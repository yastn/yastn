Algebra
=======

Multiplication by a scalar
---------------------------

MPS/MPO can be multiplied by a scalar using regular :code:`*` and :code:`/` operators,
i.e., :code:`B = a * A`, :code:`B = A * a` or :code:`B = A / a`.


Conjugation and transpose
-------------------------

.. automethod:: yastn.tn.mps.MpsMpoOBC.conj
.. automethod:: yastn.tn.mps.MpsMpoOBC.transpose
.. autoproperty:: yastn.tn.mps.MpsMpoOBC.T
.. automethod:: yastn.tn.mps.MpsMpoOBC.conjugate_transpose
.. autoproperty:: yastn.tn.mps.MpsMpoOBC.H
.. automethod:: yastn.tn.mps.MpsMpoOBC.reverse_sites


Addition of MPS/MPO
-------------------

Two MPS's or two MPO's can be added when their lengths, physical dimensions, and symmetries agree.
The sum of two such objects, :code:`A` and :code:`B`, results in a new MPS/MPO, :code:`C = A + B`,
with tensors forming :code:`C` given by a direct sum of :code:`A`'s and :code:`B`'s tensors along a virtual dimension.

::

    # a sum of two MPS's
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
        :pyobject: test_addition_example


Products of MPO and MPS
-----------------------

See examples at :ref:`examples/mps/algebra:Multiplication`.

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
:code:`C = mps.multiply(A, B)`.

.. autofunction:: yastn.tn.mps.multiply


Multiplication with truncation
------------------------------

See examples at :ref:`examples/mps/algebra:Multiplication`.

A fast procedure to multiply MPO by MPO/MPS while performing truncation is a **zipper**.
The result can be subsequently fine-tuned using :ref:`variational optimization<mps/algorithms_overlap:Variational overlap maximization>`.

.. autofunction:: yastn.tn.mps.zipper
