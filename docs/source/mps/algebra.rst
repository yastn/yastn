Algebra
=========


Copying an object
---------------------------------

To create an independent `YAMPS` object use :code:`mp_new = mp_old.copy()`

.. automodule:: yamps.MpsMpo
	:noindex:
	:members: copy

See examples here :ref:`examples/mps/mps:Copying`.


Multiplication by a number
---------------------------

`YAMPS` object can be multiplied by a number using `*`-sign. For example, :code:`y = a * x` produces  a new `YAMPS` object `y` which is old tensor `x` multiplied by a number `a`. `YAMPS` treats :code:`y = a * x` and :code:`y = x * a` the same.

.. autofunction:: yamps.MpsMpo.__mul__
.. autofunction:: yamps.MpsMpo.__rmul__


Addition
----------

Addition of two Mps's or two Mpo's can be made if their symmetry, length and physical dimensions agree. The sum of two objects  :code:`A` and :code:`B` is :code:`C = A + B` mekes a direct sum along virtual dimension.

.. autofunction:: yamps.MpsMpo.__add__

::

    # a product of two MPS's
              ___     ___    ___    ___    ___    ___     ___       ___    ___    ___    ___    ___
    A   =    |___|-D-|___|--|___|--|___|--|___|--|___|   |   |- - -|   |--|   |--|   |--|   |--|   |
 C = +     __ d|   ___ |  ___ |  ___ |  ___ |  ___ |   = |___|-2xD-|___|--|___|--|___|--|___|--|___|
      B = |___|:D-|___|:-|___|:-|___|:-|___|:-|___||      d|         |      |      |      |      |
           d|  |    |  |   |  |   |  |   |  |   |  |
             \/      \/     \/     \/     \/     \/


To make a sum of many objects :math:`\{A0,A1,\dots\}` at ones use :code:`C = yamps.add(A0,A1,...)`. The function allows to add amplitudes for each element of the sum, such that to calculate :math:`C=\sum_{j=0}^{L-1} h_j A_j` you can use 
:code:`C = yamps.add(A_0,A_1,.., amplitudes=[h_0,h_1,..])` with amplitudes given by a list of numbers.

.. autofunction:: yamps.add

See examples here :ref:`examples/mps/mps:Addition`.


Multiplication
--------------

`YAMPS` objects can be combined by multiplication. For example, :code:`z = x @ y` (or :code:`z=yamps.multiply(x,y)` ) produces a new `YAMPS` object `z` which is a product of `x` and `y` `YAMPS` objects.
In direct product we take two `YAMPS` object, e.g. `Mps`'s and collect allong physical dimensions making direct product along virtual dimension.

::

    # a product of two MPO's

             /\        /\     /\     /\     /\     /\        d
           d| _|d_    | _|_  | _|_  | _|_  | _|_  | _|_     _|_       _|_    _|_    _|_    _|_    _|_
      y =   ||___|--D-:|___|-:|___|-:|___|-:|___|-:|___|   |   |- - -|   |--|   |--|   |--|   |--|   |
 z=  @     _|_ |     _|_ |  _|_ |  _|_ |  _|_ |  _|_ |   = |___|-D^2-|___|--|___|--|___|--|___|--|___|
    x   = |___|:-D--|___|:-|___|:-|___|:-|___|:-|___||       |         |      |      |      |      |    
           d|  |d     |  |   |  |   |  |   |  |   |  |       d
             \/        \/     \/     \/     \/     \/ 
              

.. autofunction:: yamps.MpsMpo.__matmul__

.. autofunction:: yamps.multiply


See examples here :ref:`examples/mps/mps:Multiplication`.


Canonical form
---------------------------------


:ref:`theory/mps/basics:Canonical form` of the MPS/MPO can be imposed to bring it to the most advantageous truncation or to prepare it for :ref:`theory/mps/algorithms:DMRG` or :ref:`theory/mps/algorithms:TDVP` algorithms. 
The object can be brought to left canonical by setting :code:`to='first'` or to right canonical by setting :code:`to='last'` (we assume that tensors building are enumerated increasing from index `0`=:code:`first` to `N-1`=:code:`last`).
The canonical version can be perfomed with QR decomposition or SVD decomposition. As default the canonisation normalizes `YAMPS` object to `1`. You can omitt the normalisation by setting a flag :code:`normalize=False`.

The canonical form obtained from `QR decomposition` doesn't truncate any information of matrix product. The canonisation is relatively cheep. 

.. autofunction:: yamps.MpsMpo.canonize_sweep

See examples: :ref:`examples/mps/mps:Canonical form by QR decomposition`.

The canonisation by `singular value decomposition` (SVD) allows to truncate components of the lowest applitude according to instructions provided in :code:`opts`. As default the object is not truncated. 
The truncation happens on each site of the object after persorming SVD of the tensor.

.. autofunction:: yamps.MpsMpo.truncate_sweep

See examples: :ref:`examples/mps/mps:Canonical form by SVD decomposition`.
