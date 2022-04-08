Input/Output the MPS/MPO
=========================

Entropy, and bond dimension
---------------------------------

See examples: :ref:`examples/init:clone, detach or copy tensors`.

.. autoclass:: yast.Mps
	:noindex:
	:exclude-members: __init__, __new__
	:members: get_bond_dimensions, get_bond_charges_dimensions, get_entropy

Get structure
-------------

This should be about the bond dimension, entropy and Schmidt values.

.. autoclass:: yamps.Mps
	:noindex:
	:members: get_entropy, get_bond_charges_dimensions
	
ADD get_Schmidt_values !!!


Export the MPS/MPO object
---------------------------------

# export this object to the dictionary and export it to hdf5 + export-test-file

See examples: :ref:`examples/init:clone, detach or copy tensors`.

.. autoclass:: yamps.Mps
	:noindex:
	:members: save_to_hdf5


Import the MPS/MPO object
---------------------------------

# import this object from the dictionary and export it to hdf5 + export-test-file
# what aboot the configuration files, I shuld export them as well. That will be more convenient

See examples: :ref:`examples/init:clone, detach or copy tensors`.

.. autoclass:: yamps
	:noindex:
	:members: load_from_hdf5
