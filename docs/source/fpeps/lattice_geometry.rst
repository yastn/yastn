Lattice Geometry
==================

Geometry
--------

The module :code:`yastn.tn.fpeps._geometry.py` contains classes to represent the lattice geometry and also the infomration contents
of the Projected Entangled Pair States (PEPS). The classes defined in the module include:

**Bond**: A Named Tuple that represents a bond between two lattice sites. The sites are arranged in the fermionic order. Each bond has a directionality, 
captured by the "dirn" property. In the context of PEPS, a bond represents the entangled pair of quantum states. 

.. autofunction:: yastn.tn.fpeps._geometry.Bond

**Lattice**: A Class that represents the geometric information about a 2D lattice. The Lattice class holds information about the geometry of the lattice on which the PEPS is defined. It can handle different lattice types, like 'checkerboard' or 'square', 
and different boundary conditions, 'obc' (open boundary conditions) or 'infinite'. The Lattice class also provides methods to navigate this geometry, for instance, by providing the neighbouring sites or bonds. It thus provides the backbone for the PEPS by 
defining its spatial structure. In the context of strongly correlated systems, the lattice and its properties can drastically affect the system's behavior.

.. autofunction:: yastn.tn.fpeps._geometry.Lattice



Double Peps Tensor
------------------

The class `DoublePepsTensor`, deals with double-layer Projected Entangled Pair States (PEPS). An instance contains a
top tensor, a bottom tensor, and a rotation. The attribute rotation is optional and defaults to 0.
The properties and methods `ndim`, `get_shape`, and `get_legs` return details about the tensor such as its number of dimensions,
its shape along specified axes, and the tensor's legs along the specified axes.

The key functions (`append_a_bl`, `append_a_tr`, `append_a_tl`, and `append_a_br`) append the top and bottom
PEPS tensor to a given four-legged tensor at a specific position (top-right, top-left, bottom-right, or bottom-left)
respectively. These functions are useful for constructing the metric tensor in time evolution algorithms as well
as during CTMRG procedures. In the following, we provide the index ordering conventions of each of these functions.
Let `tt`` be the four-legged tensor to which the DoublePepsTensor instance is appended.
Let the output tensor after the contraction be `tt'`.


.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor


- **append_a_bl**:

::

            3                    2   3
           _|_                  _|___|_
          |   |-2              |       |--1
    tt =  |   |  1      tt' =  |       |
          |   |__|_            |       |
          |________|--0        |_______|--0


.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor.append_a_bl


- **append_a_tr**:

::

           ________             ________
       0--|____    |        0--|        |
           |   |   |           |        |
    tt =   1   |   |   tt' =   |        |
             2-|___|        1--|________|
                 |               |    |
                 3               3    2

.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor.append_a_tr


- **append_a_tl**:

::

           ________              ________
          |    ____|--3         |        |--2
          |   |   |             |        |
    tt =  |   |   2      tt' =  |        |
          |___|-1               |________|--3
            |                     |    |
            0                     0    1


.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor.append_a_tl

- **append_a_br**:

::

                 0                 1    0
                _|_               _|____|_
             1-|   |          3--|        |
    tt =    2  |   |    tt' =    |        |
           _|__|   |             |        |
       3--|________|          2--|________|


