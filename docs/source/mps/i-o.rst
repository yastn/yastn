I-O for the matrix products
=========================


Entropy, and bond dimension
---------------------------------

The properties of virtual bond dimensions for the matrix product can be extracted by calling `get_*` functions on `yamps.Mps` object. 
Those information can be used to quantify the entanaglemnt encoded by the virtual dimension as well as the structure of singlets encoding the correlations. 
This can be used both for matrix-producs state and matrix product operator.

.. autoclass:: yamps.Mps
	:noindex:
	:exclude-members: __init__, __new__
	:members: get_bond_dimensions, get_bond_charges_dimensions, get_entropy

Full information on the structure of virtual dimension is given by Schmidt values which is obtained by SVD. If the symmetries are present the Schmidt values are split to blocks.

.. autoclass:: yamps.Mps
	:noindex:
	:exclude-members: __init__, __new__
	:members: get_Schmidt_values


Saving and loading `yamps.Mps`
-------------------------------------------

The Mps object can be convenienty saved as a dictionary or to a HDF5 file.

.. autoclass:: yamps.Mps
	:noindex:
	:exclude-members: __init__, __new__
	:members: save_to_dict, save_to_hdf5

It can be later loaded from a dictionary or to a HDF5 file into `yamps.Mps`.

.. automodule:: yamps
	:members: load_from_dict, load_from_hdf5

.. todo:: can I simplify the export smh, what about the configuration files, nr_phys etc, I shuld export them as well.

Examples for saving and loading matrix products can be found in :ref:`examples/mps/mps:save and load`.
