YASTN configuration
===================

All YASTN tensors have to be provided with `configuration`, which contains information about

#. backend for linear algebra,
#. :doc:`abelian symmetry group</tensor/symmetry>`,
#. how to handle fusion (reshaping) of tensors,
#. default dtype (``float64``, ``complex128``) and device (provided it is supported by backend)
   of tensors.

The configuration can be a Python module, 
`types.SimpleNamespace <https://docs.python.org/3/library/types.html#types.SimpleNamespace>`_,
`typing.NamedTuple <https://docs.python.org/3/library/typing.html#typing.NamedTuple>`_ or similar which defines following members

* required: ``backend``, ``sym``
* optional: ``default_device``, ``default_dtype``, ``default_fusion``, ``fermionic``, ``force_fusion``

The configuration information has to be provide in an appropriate form which can be constructed using

.. autofunction:: yastn.make_config

For example, the `configuration` defined as a plain Python module, with a NumPy backend, and U(1) symmetry is:

.. literalinclude::  /../../tests/tensor/configs/config_U1.py

