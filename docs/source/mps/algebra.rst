Algebra
=======

Creating a copy of MPS/MPO
--------------------------

To create an independent copy or clone of `YAMPS` MPS/MPO :code:`A` call :code:`A.copy()`
or :code:`A.clone()` respectively.

.. autofunction:: yast.tn.mps.MpsMpo.copy

.. autofunction:: yast.tn.mps.MpsMpo.clone

Multiplication by a scalar
---------------------------

`YAMPS` MPS/MPO can be multiplied from both left and right by a scalar using regular `*` operator. 
For example, :code:`B = a * A` or :code:`B = A * a` results in a new MPS/MPO :code:`B`
with first tensor multiplied by a number `a`.

..
    .. autofunction:: yamps.MpsMpo.__mul__
    .. autofunction:: yamps.MpsMpo.__rmul__

Addition of MPS/MPO
-------------------

Two MPS's or two MPO's can be added up provided that their length, physical dimensions, and symmetry agree. The sum of two such objects :code:`A` and :code:`B` results in new MPS/MPO :code:`C = A + B`, with tensor of :code:`C` given by the direct sum of :code:`A`'s 
and :code:`B`'s tensors along virtual dimension.

::

    # a product of two MPS's
               ___     ___    ___    ___    ___    ___     ___      ___    ___    ___    ___    ___
      A       |___|-D-|___|--|___|--|___|--|___|--|___|   |   |-  -|   |--|   |--|   |--|   |--|   |
 C =  +  =  __ d|   ___ |  ___ |  ___ |  ___ |  ___ |   = |   | 2D |   |  |   |  |   |  |   |  |   |
      B    |___|:D-|___|:-|___|:-|___|:-|___|:-|___||     |___|-  -|___|--|___|--|___|--|___|--|___|
            d|  |    |  |   |  |   |  |   |  |   |  |      d|        |      |      |      |      |
              \/      \/     \/     \/     \/     \/

..
    .. autofunction:: yamps.MpsMpo.__add__

To make a sum of many MPS/MPOs :math:`\{A_0,A_1,\dots\}` at once use :code:`yamps.add(A_0,A_1,...)`.

.. autofunction:: yast.tn.mps.add

Following example show an addition of two MPSs:

.. literalinclude:: /../../tests/mps/test_algebra.py
        :pyobject: test_addition_basic


Products of MPS/MPO
-------------------

`YAMPS` supports *product* ``@`` of 
    
i) MPO with MPS resulting in a new MPS in analogy with 
:math:`\hat{O}|\psi\rangle = |\phi\rangle` (i.e. matrix-vector multiplication).

::

 # a product of MPO O and MPS A
              
             _|d        _|_         _|_ 
            |___|--D---|___|--...--|___|     _|_        _|_         _|_
 C= O @ A =  _|_        _|_         _|_   = |___|-DxD'-|___|--...--|___|
            |___|--D'--|___|--...--|___|    

ii) two MPOs resulting in a new MPO corresponding to usual 
operator product :math:`\hat{O}\hat{P} = \hat{C}` (matrix-matrix multiplication).

::
 
 # a product of two MPO's O and P

             _|d_       _|_         _|_
            |___|--D---|___|--...-:|___|    _|d          _|_         _|_
 C= O @ P =  _|_        _|_         _|_  = |___|--DxD'--|___|--...--|___|
            |___|--D'--|___|--...--|___|     |d           |           |
              |d         |           |

One can either use product operator :code:`C=A@B` or more verbose 
:code:`C=yast.tn.mps.multiply(A,B)`. Note that for MPO-MPS product, the *@*
is commutative, i.e., :code:`O@A` and :code:`A@O` are equivalent.

See examples here: :ref:`examples/mps/mps:Multiplication`.

.. autofunction:: yast.tn.mps.multiply


Canonizing MPS/MPO
------------------

MPS/MPO can be put into :ref:`theory/mps/basics:Canonical form` to reveal most advantageous truncation or as a part of the setup for 
:ref:`DMRG<mps/algorithms:density matrix renormalisation group (dmrg) algorithm>` or 
:ref:`TDVP<mps/algorithms:time-dependent variational principle (tdvp) algorithm>` algorithms. 

The canonical form obtained by QR decomposition is fast, but does not allow for truncation 
of the virtual spaces of MPS/MPO. 

.. autofunction:: yast.tn.mps.MpsMpo.canonize_sweep

See examples: :ref:`examples/mps/mps:Canonical form by QR`.

Restoring canonical form locally: For example, while performing DMRG sweeps, 
the tensors getting updated will not be in canonical form after the update. 
It is necessary to restore their canonical form in course of sweeping. 

.. autofunction:: yast.tn.mps.MpsMpo.orthogonalize_site

The canonisation by `singular value decomposition` (SVD) allows 
to truncate virtual dimension/spaces with the lowest weight 
(lowest singular values).

.. autofunction:: yast.tn.mps.MpsMpo.truncate_sweep

See examples: :ref:`examples/mps/mps:Canonical form by SVD`.
