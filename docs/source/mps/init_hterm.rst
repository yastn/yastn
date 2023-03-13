Setting MPS/MPO tensors with Hterm
==================================

:class:`yast.tn.mps.Hterm` is a basic building block of operators. Each ``Hterm`` represents
a product of local (on-site) operators. 

.. autoclass:: yast.tn.mps.Hterm

.. note::
	The :code:`Hterm` has operators :code:`Hterm.operators` without virtual legs, i.e., rank-1 for MPS and rank-2 for MPO.

A list of such ``Hterms`` can be used to create a sum of products. In order to generate full MPO use :code:`exMPO = mps.generate_mpo(I, man_input)`, 
where :code:`man_input` is a list of ``Hterm``-s and :code:`I` is the identity operator for your basis.

.. autofunction:: yast.tn.mps.generate_mpo

In order to generate MPS use :code:`exMPS = mps.generate_mps(I, man_input)`, where `I` is identity matrix for your basis.

.. autofunction:: yast.tn.mps.generate_mps

.. note::
	To create MPS you need to provide a vector for each site `0` through `N-1`.

An example for MPO using this method can be found :ref:`here<examples/mps/mps:Create MPS or MPO based on templete object>`.