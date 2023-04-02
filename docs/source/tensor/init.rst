Creating symmetric YAST tensors
===============================

Initializing symmetric tensors from scratch
-------------------------------------------

Symmetric tensor can be initialized as "blank" or more precisely empty.
In this case only its rank (specified through signature) and symmetry
needs to be known when initializing such empty symmetric tensors.
The data, in form of non-zero blocks, can be added at later time.

See examples: :ref:`examples/tensor/init:create empty tensor and fill it block by block`.

.. autoclass:: yastn.Tensor

.. automethod:: yastn.Tensor.set_block

Basic creation operations
-------------------------

Basic creation operations such as random tensors,
tensors filled with zeros, or diagonal identity tensors.

The symmetry structure of the tensor can be given either
by directly listing all charge sectors for each leg and dimensions
of these sectors `or` by passing a list of legs.

See examples: :ref:`examples/tensor/init:create tensors from scratch`.

.. automodule:: yastn
   :members: rand, randR, randC, zeros, ones, eye
   :show-inheritance:


Copying and cloning with autograd
---------------------------------

YAST follows the semantics of PyTorch with regards to creating
differentiable `clones` or non-differentiable `copies` of symmetric
tensors. See 
`clone <https://pytorch.org/docs/stable/generated/torch.clone.html#torch.clone>`_ 
and 
`detach <https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html?highlight=detach#torch.Tensor.detach>`_ for PyTorch.

See YAST examples: :ref:`examples/tensor/init:clone, detach or copy tensors`.

.. automethod:: yastn.Tensor.copy
.. automethod:: yastn.Tensor.clone
.. automethod:: yastn.Tensor.detach


Changing tensor's device or dtype
---------------------------------

Support for different compute devices, i.e. `CPU`, `GPU`, or others,
depends on the selected backend. For example

    * `NumPy` backend supports only `CPU`
    * `PyTorch` backend supports also `GPU` (and other devices)

Tensors can be moved between devices and/or their `dtype` changed

.. automethod:: yastn.Tensor.to


Import/Export of YAST tensors from/to different formats
-------------------------------------------------------

These utility operations can re-create tensors from
different formats. For example, 1D representation or dictionary.
Their export counterparts are

	* :meth:`yastn.save_to_dict`
	* :meth:`yastn.save_to_hdf5`
	* :meth:`yastn.compress_to_1d`

See examples: :ref:`examples/tensor/init:serialization of symmetric tensors`.

.. automodule:: yastn
   :members: load_from_dict, load_from_hdf5, decompress_from_1d,
             save_to_dict, save_to_hdf5, compress_to_1d
   :show-inheritance:
