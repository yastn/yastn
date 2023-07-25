Structure
=========

Geometry
---------

The _geometry.py module contains the classes and structures that are used to manage and represent the geometry of Projected Entangled Pair States (PEPS).

The classes defined in the module include:

- **Bond**: A Named Tuple that represents a bond between two lattice sites. The sites are arranged in the fermionic order.
Each bond has a directionality, captured by the "dirn" property. In the context of PEPS, a bond represents the entangled 
pair of quantum states. 
.. autofunction:: yastn.tn.fpeps._geometry.Bond

- **Lattice**: A Class that represents the geometric information about a 2D lattice. The Lattice class holds information
about the geometry of the lattice on which the PEPS is defined. It can handle different lattice types, like 'checkerboard' or 'rectangle', 
and different boundary conditions, 'finite' or 'infinite'. The Lattice class also provides methods to navigate
this geometry, for instance, by providing the neighbouring sites or bonds. It thus provides the backbone for 
the PEPS by defining its spatial structure. In the context of strongly correlated systems, the lattice and its properties can drastically
affect the system's behavior.
.. autofunction:: yastn.tn.fpeps._geometry.Lattice

- **Peps**: The Peps class extends the Lattice class, holding the PEPS data itself,
and provides additional functionalities specifically related to PEPS. The methods included in the Peps class, such 
as mpo and boundary_mps, allow for efficient manipulation and transformation of the PEPS, which are key tasks
in many numerical algorithms used to study these systems.
.. autofunction:: yastn.tn.fpeps._geometry.Lattice

.. literalinclude:: /../../tests/peps/test_geometry.py
        :pyobject: test_Lattice
        :pyobject: test_Pes_get_set
        :pyobject: test_NtuEnv



Double Peps Tensor
------------------

.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor

The class `DoublePepsTensor`, deals with double-layer Projected Entangled Pair States (PEPS). An instance contains a 
top tensor, a bottom tensor, and a rotation. The attribute rotation is optional and defaults to 0.
The properties and methods `ndim`, `get_shape`, and `get_legs` return details about the tensor such as its number of dimensions, 
its shape along specified axes, and the tensor's legs along the specified axes.

The key functions (`append_a_bl`, `append_a_tr`, `append_a_tl`, and `append_a_br`) append the top and botton 
PEPS tensor to a given four legged tensor at a specific position (top-right, top-left, bottom-right, or bottom-left)
respectively. These functions are useful for contrusting the metric tensor in time evolution algorithms as well
as during CTMRG procedures. In the following we provide the index ordering conventions of each of these functions. Let 
tt be the four legged tensor to which the DoublePepsTensor instance is appended to.
Let the output tensor after the contraction be tt'.

- **append_a_bl**:
::                                                             
                  3                                             2    3
                  |                                             |    |
                __|___                                         _|____|_
               |      |  2                                    |        |
               |      |  |                                    |        |--- 1
    tt =       |      |__|___                     tt' =       |        |______ 
               |             |----- 2                         |               |
               |_____________|----- 1                         |_______________|----- 0


.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor.append_a_bl


- **append_a_tr**:
::                                                             
                                                           
                                                        
                      _______________                                               _______________
              0 -----|               |                                      0 -----|               |
              1 -----|________       |                                      1 -----|_______        |
    tt =                  |   |      |                       tt' =                  3 -----|       |
                          |   |______|                                              2 -----|_______|
                          2       |                                                               
                                  |
                                  3

.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor.append_a_tr


- **append_a_tl**:
::                                                             
                                                           
                 _____________                                        _____________
                |             |--- 3                                 |             |--- 2
                |     ________|                                      |             |
    tt =        |    |       |                       tt' =           |             |--- 3                      
                |    |       |                                       |             |
                |____|--- 1  2                                       |_____________|         
                   |                                                     |    |
                   |                                                     |    |
                   0                                                     0    1


.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor.append_a_tl



- **append_a_br**: 
::
                             1   0                                            1      0
                             |   |                                            |      |
                            _|___|_                                         __|______|___
                     3  2  |       |                                       |             |
                     |  |  |       |                                  3 ---|             |
    tt =            _|__|__|       |                       tt' =           |             |                      
                   |               |                                  2 ---|             |
                   |_______________|                                       |_____________|         
                                                    
                                                                               
                                                                       


.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor.append_a_br


The functions (`_attach_01` and `_attach_23`) are similar to the append methods but they attach the 
tensor to the top or bottom left if rotation = 0, and to the top or bottom right if rotation = 90.

.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor._attach_01
.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor._attach_23


- **fPEPS_fuse_layers**: This method fuses the top and bottom layers of a PEPS tensor network for a particular
instance of DoublePepsTensor. It can be used when it is convienient to work with contracted double tensors rather
than keeping them separate. It is generally avoided due to higher computational complexity.
