Building MPS/MPO manually
=========================

The tensor making up MPS/MPO can be assigned manualy by setting them one by one.

.. note::
        The virtual dimensions/spaces of the neighbouring MPS/MPO tensors have to remain consistent.

Ground state of Spin-1 AKLT model
---------------------------------

Here, we show an example of such code for the most well-known exact MPS: The ground state of  
`Affleck-Kennedy-Lieb-Tasaki (AKLT) model <https://en.wikipedia.org/wiki/AKLT_model>`_.

.. literalinclude:: /../../tests/mps/test_initialization.py
        :pyobject: test_assign_block


MPO for hopping model (no symmetry)
-----------------------------------

The same can be done for MPOs. Here, we show construction of a simple 
nearest-neighbour hopping Hamiltonian with hopping amplitude `t` 
and on-site energy :math:`\mu` with different 
realizations of explicit symmetry.

.. literalinclude:: /../../tests/mps/generate_by_hand.py
        :pyobject: mpo_XX_model_dense

MPO for hopping model with :math:`\mathbb{Z}_2` symmetry
--------------------------------------------------------

.. literalinclude:: /../../tests/mps/generate_by_hand.py
        :pyobject: mpo_XX_model_Z2

MPO for hopping model with U(1) symmetry
----------------------------------------

.. literalinclude:: /../../tests/mps/generate_by_hand.py
        :pyobject: mpo_XX_model_U1


Generating MPS/MPO automatically
================================

The MPO can be constructed automatically using dedicated generator supplied with the Hamiltonian. 
This can be shown for a simple nearest-neightbour hopping Hamiltonian 
with hopping amplitude `t` and on-site energy `mu`.

Automatic generator creates MPO with symmetries defies by building operators. 

.. literalinclude:: /../../tests/mps/generate_automatic.py
        :pyobject: mpo_XX_model

Generation of more complex Hamiltonian can be drown to Ising model with decaying coupling strength. 

.. literalinclude:: /../../tests/mps/generate_automatic.py
        :pyobject: mpo_Ising_model


.. FOR ALGEBRA 

Multiplication
=====================================

We can test the multiplication of MPO and MPS using a practical example for a ground state obtained with :ref:`theory/mps/algorithms:DMRG`.

.. literalinclude:: /../../tests/mps/test_algebra.py
        :pyobject: test_multiplication


.. QR and SVD

Canonizing MPS/MPO
==================

There are different algorithms, which can bring MPS/MPO into
canonical form.

Canonical form by QR
--------------------

This is the cheapest way to bring MPS/MPO into left or right canonical form. 

.. note::

        Both :ref:`DMRG<mps/algorithms:density matrix renormalisation group (dmrg) algorithm>` 
        and :ref:`TDVP<mps/algorithms:time-dependent variational principle (tdvp) algorithm>` 
        algorithms expect initial MPS to be the right canonical form.

Bring MPS/MPO into canonical form by QR decomposition

::

        # Generate random MPS with no symmetry
        psi = yamps.random_dense_mps(N=16, D=15, d=2)

        # rigth canonical form
        #
        psi.canonize_sweep(to='first')

        # left canonical form
        #
        psi.canonize_sweep(to='last')

Check if MPS/MPO is in left/right canonical form by verifying 
if each tensor forms an isometry after appropriate contraction
with its conjugate. For left canonical form

.. literalinclude:: /../../tests/mps/test_mps.py
        :pyobject: is_left_canonical

And similarily for right canonical form

.. literalinclude:: /../../tests/mps/test_mps.py
        :pyobject: is_right_canonical


Canonical form by SVD
---------------------

Bringing MPS/MPO into canonical form through SVD decomposition 
is computationally more expensive than with QR, but allows for truncation.
Truncation is governed by options passed as :code:`opts_dict` (internally to SVD).

::

        # There are different options which we can pass, see yast.linalg.svd. 
        # Defaults are assumed for options not explictly specified.
        #
        opts_dict = {
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
        psi.truncate_sweep(to='first', opts=opts_dict)
        
        # Generate random MPO with no symmetry
        #
        H= generate_random.mpo_random(config_dense, N=16, Dmax=25, d=2, d_out=2)

        # Bring MPO to canonical form and truncate (here, left canonical form).
        # Note: for MPO we do not want to change overall scale, thus no normalization. 
        #
        H.truncate_sweep(to='last', opts=opts_dict, normalize=False)


Save and load MPS/MPO
=====================

YAMPS MPS/MPO can be saved/loaded either to/from a dictionary or an HDF5 file. 

.. note::
        :ref:`YAST configuration<tensor/configuration:yast configuration>` 
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


.. algorithms
DMRG
=====================================

In order to perform :ref:`theory/mps/algorithms:DMRG` we need initial guess for MPS the hermitian operator (typically a Hamiltonian) written as MPO. 
We should start with :ref:`mps/init:initialisation` of the MPS and MPO which we push to DMRG.

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

Also see the test in examples for  :ref:`examples/mps/mps:Multiplication`.


TDVP
=====================================

In order to perform :ref:`theory/mps/algorithms:TDVP` we need initial MPS the operator written as MPO.
We should start with :ref:`mps/init:initialisation` of the MPS and MPO which we push to DMRG.

Here is a simple example for TDVP used to obtain a ground state for a quadratic Hamiltonian 
through imaginary time evolution:

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: test_dense_tdvp

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: run_tdvp_imag


The same can be done for any symmetry:

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: test_Z2_tdvp

.. literalinclude:: /../../tests/mps/test_tdvp.py
        :pyobject: test_U1_tdvp
