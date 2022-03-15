Setting up matrix product state and operators
===============================

Properties of MPS objects
------------------

The product operator is a numered list of tensors created using YAST with additional parameters which define their properties as a collection of elements. 
For now, the object support one-dimensional structures with open boundary conditions.

# main elements, how is the Tensor saved there, nr_phys, how does the ends are being handled, 


Configuration of symmetries
-------------------------------------------

#configuration of the symmetries, see test/configs, comment on the default flags /is that dictionary/, 

See examples: :ref:`examples/init:create empty tensor and fill it block by block`.

.. autoclass:: yast.Tensor
	:members: __init__
	:exclude-members: __new__

Configuration of backend
-------------------------------------------

#backed is should be added in the configuration, 

See examples: :ref:`examples/init:create empty tensor and fill it block by block`.

.. autoclass:: yast.Tensor
	:members: __init__
	:exclude-members: __new__


Creating MPS/MPO from scratch
-------------------------

MPS/MPO can be initialized as random. It is initialized from scratch with a symmetric given by configuration.
#make separate test for initialization
# random and automatic 

See examples: :ref:`examples/init:create tensors from scratch`.

.. automodule:: yast
   :members: rand, randR, randC, zeros, ones, eye
   :show-inheritance:


Creating MPO using Tensor-s
-------------------------

MPS/MPO can be composed using Tensor-s. 

# push a tensor to each site in MPS/MPO. No checkes for the validity of your choice.
# clock tensor into a big tensor, cite the block for for the prodcut state, cite example for Hamiltonian generation, sepatate example file.

See examples: :ref:`examples/init:create tensors from scratch`.

.. automodule:: yast
   :members: generate_Mij, automatic_Mps
