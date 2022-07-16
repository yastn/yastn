Import and export
===================================


Saving and loading `yamps.Mps`
-------------------------------------------

The Mps object can be convenienty saved as a dictionary or to a HDF5 file.

.. autoclass:: yamps.Mps
	:noindex:
	:exclude-members: __init__, __new__
	:members: save_to_dict, save_to_hdf5

It can be later loaded from a dictionary or to a HDF5 file into `yamps.Mps`.

.. automodule:: yamps
	:noindex:
	:members: load_from_dict, load_from_hdf5

.. todo:: can I simplify the export smh, what about the configuration files, nr_phys etc, I shuld export them as well.

Examples for saving and loading matrix products can be found in :ref:`examples/mps/mps:save and load`.
