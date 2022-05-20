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

See example of such `configuration`, with description of individual options

.. literalinclude::  /../../tests/tensor/configs/config_U1.py