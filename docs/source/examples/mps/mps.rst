.. QR and SVD

Canonizing MPS/MPO
==================

There are different algorithms, which can bring MPS/MPO into a canonical form.

Canonical form by QR
--------------------

This is the cheapest way to bring MPS/MPO into left or right canonical form. 

.. note::

        Both :ref:`DMRG<mps/algorithms_dmrg:Density matrix renormalization group (DMRG) algorithm>` 
        and :ref:`TDVP<mps/algorithms_tdvp:Time-dependent variational principle (TDVP) algorithm>` 
        algorithms expect initial MPS to be canonized towards the first site.

Bring MPS/MPO into canonical form by QR decomposition

::

        # Generate random MPS with no symmetry
        psi = yamps.random_dense_mps(N=16, D=15, d=2)

        # rigth canonical form
        #
        # --A*--    --
        #   |   | =   |Identity
        # --A---    --
        #
        psi.canonize_(to='first')

        # left canonical form
        #
        #  --A*--             --
        # |  |     = Identity|
        #  --A---             --
        #
        psi.canonize_(to='last')

Check if MPS/MPO is in left/right canonical form by verifying 
if each tensor forms an isometry after appropriate contraction
with its conjugate. For either left or right canonical form

.. literalinclude:: /../../tests/mps/test_canonize.py
        :pyobject: check_canonize

Canonical form by SVD
---------------------

Bringing MPS/MPO into canonical form through SVD decomposition 
is computationally more expensive than QR, but allows for truncation.
Truncation is governed by options passed as :code:`opts_dict`
to :code:`yastn.linalg.truncation_mask` that is applied after each SVD during the sweep through MPS/MPO.

.. note::
        Faithfull SVD truncation requires MPS/MPO to be in the canonical form
        of the opposite direction to the direction of the truncation sweep.

::

        # There are different options which we can pass, see yastn.linalg.svd. 
        # Defaults are assumed for options not explictly specified.
        #
        opts_svd = {
                'D_total': 4,      # total number of singular values to keep
                'D_block': 2,      # maximal number of singular values to keep in a single block
                'tol': 1e-6,       # relative tolerance of singular values below which to 
                                   # truncate across all blocks
                'tol_blocks': 1e-6 # relative tolerance of singular values below which to
                                   # truncate within individual blocks
        }

        # Generate random MPS with no symmetry
        #
        psi = yamps.random_dense_mps(N=16, D=15, d=2)

        # Bring MPS to canonical form and truncate (here, right canonical form).
        # For MPS we usually normalize the state.
        #
        psi.canonize_(to='last')
        psi.truncate_(to='first', opts_svd=opts_svd)

        # Generate random MPO with no symmetry
        #
        H= generate_random.mpo_random(config_dense, N=16, Dmax=25, d=2, d_out=2)

        # Bring MPO to canonical form and truncate (here, left canonical form).
        # Note: for MPO we do not want to change overall scale, thus no normalization. 
        #
        H.canonize_(to='first', normalize=False)
        H.truncate_(to='last', opts_svd=opts_svd, normalize=False)


.. algorithms

DMRG
====

In order to execute :ref:`DMRG<mps/algorithms_dmrg:Density matrix renormalization group (DMRG) algorithm>` we need the hermitian operator (typically a Hamiltonian) written as MPO, and an initial guess for MPS.

Here is a simple example for DMRG used to obtain a ground state for a quadratic Hamiltonian:

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: test_dense_dmrg

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: run_dmrg


The same can be done for any symmetry:

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: test_Z2_dmrg

.. literalinclude:: /../../tests/mps/test_dmrg.py
        :pyobject: test_U1_dmrg

Also see the test in examples for :ref:`examples/mps/mps:Multiplication`.


TDVP
====

Test :ref:`TDVP<mps/algorithms_tdvp:time-dependent variational principle (tdvp) algorithm>` simulating time evolution after a sudden quench in a free-fermionic model.

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: test_tdvp_hermitian

Slow quench across a quantum critical point in a transverse Ising chain.

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: test_tdvp_time_dependent


.. FOR ALGEBRA

Multiplication
==============

We can test the multiplication of MPO and MPS using an example of a ground state obtained with :ref:`DMRG<mps/algorithms_dmrg:Density matrix renormalization group (DMRG) algorithm>`.

.. literalinclude:: /../../tests/mps/test_algebra.py
        :pyobject: test_multiplication


Save and load MPS/MPO
=====================

MPS/MPO can be saved/loaded either to/from a dictionary or an HDF5 file. 

.. note::
        :ref:`YASTN configuration<tensor/configuration:yastn configuration>` 
        of on-site tensors of MPS/MPO must be provided
        when loading either from dictionary or HDF5 file.

Using Python's dictionary
-------------------------

.. literalinclude:: /../../tests/mps/test_save_load.py
        :pyobject: test_basic_dict

Using HDF5 format
-----------------

.. literalinclude:: /../../tests/mps/test_save_load.py
        :pyobject: test_basic_hdf5