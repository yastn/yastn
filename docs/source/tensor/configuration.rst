YASTN configuration
===================

All YASTN tensors have to be provided with **configuration**, which defines:

   #. linear algebra backend,

   #. :doc:`abelian symmetry group</tensor/symmetry>`,

   #. how to handle fusion (reshaping) of tensors,

   #. default data type (``float64``, ``complex128``) and device (provided it is supported by backend)
   of tensors.

The configuration can be provided as a Python module,
`types.SimpleNamespace <https://docs.python.org/3/library/types.html#types.SimpleNamespace>`_,
`typing.NamedTuple <https://docs.python.org/3/library/typing.html#typing.NamedTuple>`_ or similar which defines following members

* required: ``backend``, ``sym``,
* optional: ``default_device``, ``default_dtype``, ``default_fusion``, ``fermionic``, ``force_fusion``.

The configuration can be conveninently generated using

.. autofunction:: yastn.make_config

Below is an example of configuration defined as a plain Python module,
using NumPy backend and :math:`U(1)` symmetry.

.. code-block:: python

   import yastn.backend.backend_np as backend
   from yastn.sym import sym_U1 as sym

   default_device: str = 'cpu'
   default_dtype: str = 'float64'
   fermionic = False
   default_fusion: str = 'hard'
   force_fusion: str = None
