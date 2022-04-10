Setting up matrix product state and operators
===============================

General information on Mps class
--------------------------------

.. todo:: can we change the name Mps to e.g. Mp (or MP or TP for tensor product) to make it more general between MPS and MPO objects

The class `yamps.Mps` allows to create an object which represents a product operator. It consists of the numbered list of YAST tensors (members of class `yast.Tensor`, together with 
properties definying waht do the matrix product represents.

The `yamps` module supports one-dimensional structures with open boundary conditions. It allows to manipulate matrix product states and matrix product operators and use algorithms such as density matrix renormalisation group (:ref:`mps/algorithms:DMRG`) and time-dependend variational principle (:ref:`mps/algorithms:TDVP`) algorithms.


Creating `yamps.Mps` matrix product
---------------------------------------------

The object of class `yamps.Mps` are designed to represent one-dimensional matrix products with open-boundary 
condition. The bond dimension on the edges is 1 by default.

.. autoclass:: yamps.Mps
	:noindex:
	:members: __init__
	:exclude-members: __new__

Empty `yamps.Mps` matrix product state of 10 tensors is created using :code:`mps = yamps.Mps(N=10, nr_phys=1)`
while the matrix producs operator :code:`mpo = yamps.Mps(N=10, nr_phys=2)`.

The symmetry of `yamps.Mps` is inherited by its building blocks. Its building blocks can be assigned directly to `yamps.Mps` by :ref:`examples/mps/mps:Filling matrix product with tensors`.

The matrix product can be generated automatically basing on the structure defined by a user

.. automodule:: yamps
	:noindex:
	:members: automatic_Mps

For examples for the function see the code in :ref:`examples/mps/mps:Automatically generated matrix product`.

.. todo:: make automatic_Mps better and erase *Mij version for automatically generated MP