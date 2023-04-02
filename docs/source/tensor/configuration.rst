YASTN configuration
===================

All YASTN tensors have to be provided with `configuration`, which defines

#. Linear algebra backend
#. :doc:`abelian symmetry group</tensor/symmetry>`
#. how to handle fusion (reshaping) of tensors
#. default dtype (``float64``, ``complex128``) and device (provided it is supported by backend)
   of tensors

The configuration can be a Python module, 
`types.SimpleNamespace <https://docs.python.org/3/library/types.html#types.SimpleNamespace>`_,
`typing.NamedTuple <https://docs.python.org/3/library/typing.html#typing.NamedTuple>`_ or similar which defines following members

* required: ``backend``, ``sym``
* optional: ``default_device``, ``default_dtype``, ``default_fusion``, ``fermionic``, ``force_fusion``

For easy way to generate `configurations`, a convenience function is provided

.. autofunction:: yastn.make_config

Below is an example of `configuration` defined as a plain Python module, 
using NumPy backend and U(1) symmetry

.. literalinclude::  /../../tests/tensor/configs/config_U1.py
