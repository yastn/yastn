Import and export
===================================


Export/save
------------

`YAMPS` allows to save MPS/MPO in the form of python dicrionary or HDF5 file. This allows to have the object in a form of a describtion which can be decoded back to create a `YAMPS` object. 
The fist method writes MPS/MPO :code:`A` in a form of a dictionary :code:`d = A.save_to_dict()`. 

.. autofunction:: yamps.MpsMpo.save_to_dict

The second one redirects all information to HDF5 :code:`file` to a group which has the path :code:`my_mps_address` such that running :code:`A.save_to_hdf5(file, './my_mps_address/')`. Keep the adress as you will need to have it to encode the object back to `YAMPS`.

.. autofunction:: yamps.MpsMpo.save_to_hdf5

Using custom programms the HDF5 can be viewed and anylised by user. 


Import/load
------------

After :ref:`mps/i-o:Export/save` the instruction about the `YAMPS` object can be used to create a new one. The information can be encoded from a dictionary producing a new object `A_new = yamps.load_from_dict()`.

.. autofunction:: yamps.load_from_dict

If the information are saved in HDF5 format :code:`file` under an address  :code:`my_mps_address` then encoding is made by :code:`A_new = yamps.load_from_hdf5(file, './my_mps_address/')`.

.. autofunction:: yamps.load_from_hdf5

Examples for import and expoer can be found in :ref:`examples/mps/mps:save and load`.

