Creating symmetric YAST tensors
===============================

Initializing symmetric tensors from scratch
-------------------------------------------

.. autoclass:: yast.Tensor
	:members: __init__

Basic creation operations 
-------------------------

Basic creation operations such as random tensors,
tensors filled with zeros, or diagonal identity tensors

.. automodule:: yast
   :members: rand, randR, randC, zeros, ones, eye
   :show-inheritance:


Importing YAST tensor from different formats
--------------------------------------------

These utility operations can re-create tensors from
different formats. For example, 1D representation or dictionary.
Their export counterparts are
	
	* yast.tensor.export_to_dict 
	* yast.tensor.compress_to_1d

.. automodule:: yast
   :members: import_from_dict, decompress_from_1d
   :noindex:
   :show-inheritance: