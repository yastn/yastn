Creating symmetric YASTN tensors
================================

Initializing symmetric tensors from scratch
-------------------------------------------

Symmetric tensor can be initialized as blank or, more precisely, empty.
In this case only its rank (specified through tuple of signatures) and symmetry
needs to be known when initializing such empty symmetric tensors.
The data, in a form of non-zero blocks, can be added at later time.

See examples at :ref:`examples/tensor/init:create empty tensor and fill it block by block`.

.. autoclass:: yastn.Tensor

.. automethod:: yastn.Tensor.set_block

Basic creation operations
-------------------------

See examples at :ref:`examples/tensor/init:create tensors from scratch`.

Basic creation operations includes operations such as creating random tensors,
tensors filled with zeros, or diagonal identity tensors.

The symmetry structure of the tensor can be given either
by directly listing all charge sectors for each leg and dimensions
of these sectors or by passing a list of legs.

.. automodule:: yastn
   :noindex:
   :members: rand, randR, randC, zeros, ones, eye
   :show-inheritance:

Copying and cloning with autograd
---------------------------------

See examples at :ref:`examples/tensor/init:clone, detach or copy tensors`.

YASTN follows the semantics of PyTorch with regards to creating
differentiable *clones* or non-differentiable *copies* of symmetric
tensors. See
`clone <https://pytorch.org/docs/stable/generated/torch.clone.html#torch.clone>`_
and
`detach <https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html?highlight=detach#torch.Tensor.detach>`_ for PyTorch.

.. automethod:: yastn.Tensor.copy
.. automethod:: yastn.Tensor.clone
.. automethod:: yastn.Tensor.detach


Changing tensor's device or dtype
---------------------------------

Support for different compute devices, i.e. `CPU`, `GPU`, or others,
depends on the selected backend. For example

    * `NumPy` backend supports only `CPU`,
    * `PyTorch` backend supports also `GPU` (and other devices).

Tensors can be moved between devices and/or their `dtype` may be changed

.. automethod:: yastn.Tensor.to


Import/Export of YASTN tensors from/to different formats
--------------------------------------------------------

See examples at :ref:`examples/tensor/init:serialization of symmetric tensors`.

These utility operations can export and then import tensors from
different formats. For example, exporting and importing tensor or MPS to and from a file.

.. autofunction:: yastn.Tensor.to_dict
.. autofunction:: yastn.Tensor.from_dict
.. autofunction:: yastn.Tensor.save_to_hdf5
.. autofunction:: yastn.load_from_hdf5
.. autofunction:: yastn.Tensor.save_to_dict
.. autofunction:: yastn.load_from_dict

Dictionaries generated ny `to_dict` methods can be further split into tuple of data arrays and remaining metadata identifying the tensor structure
(or structure of more complex objects, e.g., Mps or Peps) using :meth:`yastn.split_data_and_meta` and later combined back with :meth:`yastn.combine_data_and_meta`.

.. autofunction:: yastn.split_data_and_meta
.. autofunction:: yastn.combine_data_and_meta
