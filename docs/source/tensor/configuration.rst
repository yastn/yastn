YAST configuration
==================

All YAST tensors have to be provided with `configuration`, which defines

* Linear algebra backend
* abelian symmetry group
* how to handle fusion (reshaping) of tensors
* default dtype (``float64``, ``complex128``) and device (provided its supported by backend)
  of tensors

The configuration can be a Python module, :class:`types.SimpleNamespace`, :class:`typing.NamedTuple` or similar which defines following members

* required: ``backend``, ``sym``
* optional: ``default_device``, ``default_dtype``, ``default_fusion``, ``default_tensordot``,
  ``fermionic``, ``force_fusion``, ``force_tensordot``

For easy way to generate `configurations`, a convenience function is provided

.. autofunction:: yast.make_config

See example module with `configuration`, using NumPy backend and U(1) symmetry,

.. literalinclude::  /../../tests/tensor/configs/config_U1.py
