Import and export
===================================

`YAMPS` can save MPS/MPO as Python :code:`dict` or HDF5 file. 
The MPS/MPO previously serialized by :meth:`yamps.MpsMpo.save_to_dict` or :meth:`yamps.MpsMpo.save_to_hdf5` can be again deserialized into *YAMPS* MPS/MPO.

Examples of exporting and loading MPS/MPO can be found in 
:ref:`examples/mps/mps:save and load mps/mpo`.

Export/save
------------



.. autofunction:: yamps.MpsMpo.save_to_dict

.. autofunction:: yamps.MpsMpo.save_to_hdf5


Import/load
-----------

.. autofunction:: yamps.load_from_dict

.. 
	If the information are saved in HDF5 format :code:`file` under an address  :code:`my_address` then encoding is made by :code:`A_new = yamps.load_from_hdf5(file, './my_address/')`.

.. autofunction:: yamps.load_from_hdf5