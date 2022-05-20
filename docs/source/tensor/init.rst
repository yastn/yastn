Creating symmetric YAST tensors
===============================

Initializing symmetric tensors from scratch
-------------------------------------------

Symmetric tensor can be initialized as "blank" or more precisely empty.
In this case only its rank (specified through signature) and symmetry
needs to known when initializing such empty symmetric tensors. 
The data, in form of non-zero blocks, can be added at later time.

See examples: :ref:`examples/tensor/init:create empty tensor and fill it block by block`.

.. autoclass:: yast.Tensor
	:members: __init__
	:exclude-members: __new__


Basic creation operations
-------------------------

Basic creation operations such as random tensors,
tensors filled with zeros, or diagonal identity tensors.

See examples: :ref:`examples/tensor/init:create tensors from scratch`.

.. automodule:: yast
   :members: rand, randR, randC, zeros, ones, eye
   :show-inheritance:


Copying and cloning with autograd
---------------------------------

TODO add link

YAST follows the semantics of PyTorch with regards to creating
differentiable `clones` or non-differentiable `copies` of symmetric
tensors.

See examples: :ref:`examples/tensor/init:clone, detach or copy tensors`.

.. autoclass:: yast.Tensor
	:noindex:
	:exclude-members: __init__, __new__
	:members: copy, clone, detach


Changing tensor's device or dtype
---------------------------------

Support for different compute devices, i.e. `CPU`, `GPU`, or others,
depends on the selected backend. For example 
    
    * `NumPy` backend supports only `CPU`
    * `PyTorch` backend supports also `GPU` (and other devices)

Tensors can be moved between devices and/or their `dtype` changed 

.. automethod:: yast.Tensor.to


Importing YAST tensor from different formats
--------------------------------------------

These utility operations can re-create tensors from
different formats. For example, 1D representation or dictionary.
Their export counterparts are
	
	* :meth:`yast.save_to_dict`
	* :meth:`yast.save_to_hdf5` 
	* :meth:`yast.compress_to_1d` 

See examples: :ref:`examples/tensor/init:serialization of symmetric tensors`.

.. automodule:: yast
   :members: load_from_dict, load_from_hdf5, decompress_from_1d
   :noindex:
   :show-inheritance: