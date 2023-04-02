MPS
===

Abelian-symmetric matrix product states/operators and related algorithms.
Abelian-symmetric tensor algebra is provided by :doc:`YASTN<index>`.

.. automodule:: yastn.tn.mps
   :members:
   :undoc-members:
   :show-inheritance:

Basic concepts
---------------------------------------
   
.. toctree::
   :glob:

   theory/mps/basics


API: MPS and MPO
---------------------------------------

.. toctree::
   :glob:
   :maxdepth: 2

   mps/convention
   mps/properties
   mps/algebra
   mps/i-o


API: Creating MPS and MPO
-------------------------

.. toctree::
   :glob:
   :maxdepth: 2

   mps/init
   mps/init_hterm
   mps/generate


API: Expectation values
---------------------------

.. toctree::
   :glob:
   :maxdepth: 2

   mps/measurements


API: Algorithms
---------------------

.. toctree::
   :glob:
   :maxdepth: 2

   mps/algorithms_dmrg
   mps/algorithms_tdvp
   mps/algorithms_overlap


Examples
---------------------------

.. toctree::
   :glob:
   :maxdepth: 2

   examples/mps/build_mps_manually
   examples/mps/build_mps_Hterm
   examples/mps/build_mps_latex
   examples/mps/build_mps_random
   examples/mps/mps
