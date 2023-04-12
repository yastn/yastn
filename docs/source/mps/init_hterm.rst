Setting MPS/MPO tensors with Hterm
==================================

:class:`yastn.tn.mps.Hterm` is a basic building block of operators. Each ``Hterm`` represents
a product of local (on-site) operators. 

.. autoclass:: yastn.tn.mps.Hterm

.. note::
	The :code:`Hterm` has operators :code:`Hterm.operators` without virtual legs, i.e., rank-1 for MPS and rank-2 for MPO.

A list of such ``Hterms`` can be used to create a sum of products. In order to generate full MPO use :code:`exMPO = mps.generate_mpo(I, man_input)`, 
where :code:`man_input` is a list of ``Hterm``-s and :code:`I` is the identity operator for your basis.

.. autofunction:: yastn.tn.mps.generate_mpo

An example using this method can be found :ref:`here<examples/mps/build_mps_hterm:building mpo using hterm>`.