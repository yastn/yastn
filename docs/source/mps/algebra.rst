Algebra
=======

Multiplication by a scalar
---------------------------

MPS/MPO can be multiplied from both left and right by a scalar using regular `*` operator.
For example, :code:`B = a * A` or :code:`B = A * a` results in a new MPS/MPO :code:`B`
with first tensor multiplied by the phase of the number `a`, and modulus of `a` included in
:code:`B.factor`.

..
    .. autofunction:: yamps.MpsMpo.__mul__
    .. autofunction:: yamps.MpsMpo.__rmul__

Addition of MPS/MPO
-------------------

Two MPS's or two MPO's can be added up, provided that their length, physical dimensions, and symmetry agree.
The sum of two such objects :code:`A` and :code:`B` results in new MPS/MPO :code:`C = A + B`, with tensor of :code:`C` given by the direct sum of :code:`A`'s
and :code:`B`'s tensors along virtual dimension.

::

    # a product of two MPS's
               ___     ___    ___    ___    ___    ___     ___      ___    ___    ___    ___    ___
      A       |___|-D-|___|--|___|--|___|--|___|--|___|   |   |    |   |  |   |  |   |  |   |  |   |
 C =  +  =  __ d|   ___ |  ___ |  ___ |  ___ |  ___ |   = |   |-2D-|   |--|   |--|   |--|   |--|   |
      B    |___|:D-|___|:-|___|:-|___|:-|___|:-|___||     |___|    |___|  |___|  |___|  |___|  |___|
            d|  |    |  |   |  |   |  |   |  |   |  |      d|        |      |      |      |      |
              \/      \/     \/     \/     \/     \/

..
    .. autofunction:: yamps.MpsMpo.__add__

To make a sum of many MPS/MPOs :math:`\{A_0, A_1,\dots\}` at once use :code:`yamps.add(A_0, A_1,...)`.

.. autofunction:: yastn.tn.mps.add

Following example show an addition of two MPSs:

.. literalinclude:: /../../tests/mps/test_algebra.py
        :pyobject: test_addition_basic


Products of MPS/MPO
-------------------

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
 C= O @ P =  _|_        _|_         _|_  = |___|--DxD'--|___|--...--|___|
            |___|--D'--|___|--...--|___|     |d           |           |
              |d         |           |

One can either use product operator :code:`C = A @ B` or more verbose
:code:`C = yastn.tn.mps.multiply(A, B)`.

See examples here: :ref:`examples/mps/algebra:Multiplication`.

.. autofunction:: yastn.tn.mps.multiply
