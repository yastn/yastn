Creating symmetric YAST tensors
===============================

YAST configuration
------------------

TODO

Backend, symmetry, ...


Initializing symmetric tensors from scratch
-------------------------------------------

Symmetric tensor can be initialized as "blank" or more precisely empty.
In this case only its rank (specified through signature) and symmetry
needs to known when initializing such empty symmetric tensors. 
The data, in form of non-zero blocks, can be added at later time.

See examples: :ref:`examples/init:create empty tensor and fill it block by block`.

.. autoclass:: yast.Tensor
	:members: __init__
	:exclude-members: __new__


Basic creation operations
-------------------------

Basic creation operations such as random tensors,
tensors filled with zeros, or diagonal identity tensors.

See examples: :ref:`examples/init:create tensors from scratch`.

.. automodule:: yast
   :members: rand, randR, randC, zeros, ones, eye
   :show-inheritance:


Copying and cloning with autograd
---------------------------------

.. autoclass:: yast.Tensor
	:noindex:
	:exclude-members: __init__, __new__
	:members: copy, clone, detach


Moving tensors between devices
------------------------------

Support for different compute devices, i.e. `CPU`, `GPU`, or others,
depends on the selected backend. For example 
    
    * `NumPy` backend supports only `CPU`
    * `PyTorch` backend supports also `GPU` (and other devices)

Tensors can be moved between devices

.. automethod:: yast.Tensor.to


Importing YAST tensor from different formats
--------------------------------------------

These utility operations can re-create tensors from
different formats. For example, 1D representation or dictionary.
Their export counterparts are
	
	* yast.save_to_dict 
	* yast.compress_to_1d 

See examples: :ref:`examples/init:serialization of symmetric tensors`.

.. automodule:: yast
   :members: load_from_dict, decompress_from_1d
   :noindex:
   :show-inheritance: