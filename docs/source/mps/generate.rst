Creating MPS/MPO tensors with Generator
=======================================

:class:`yast.tn.mps.Generator` automatizes creation of MPS and MPO. 
Given a set of local (on-site) operators, i.e. :class:`yast.operators.Spin12` 
or :class:`yast.operators.SpinlessFermions`,
one can build both the states and operators. The MPS/MPO can be given as a LaTeX-like expression. 

.. autoclass:: yast.tn.mps.Generator
	:members: random_seed, I, random_mps, random_mpo

MPS/MPOs from LaTex 
-------------------
In examples below you can find a detailed description how to build MPS or MPO using LaTeX-like instructions

	* :ref:`Generate MPS from LaTex<examples/mps/mps:Generate MPS from LaTex-like instruction>`
	* :ref:`Generate MPO from LaTex<examples/mps/mps:Generate MPO from LaTex-like instruction>`