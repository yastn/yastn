Building YAMPS object manually
=====================================

The content of MPS/MPO can be assigned manualy by setting known tensors one by one.
In order to do that you should prepare :meth:`yast.Tensor` which fits the structure of physical and virtual legs according to :ref:`mps/properties:YAMPS properties`. 
After initialising `YAMPS` object we assign tensor of index :code:`j` with :code:`psi.A[3] = <example tensor>`.
We have to make sure that assined tensor fit along virtual dimension.

The most well known exact MPS construction is the ground state for Affleck-Kennedy-Lieb-Tasaki (AKLT) model.

.. literalinclude:: /../../tests/mps/test_initialization.py
        :pyobject: test_assign_block

The same can be done for MPO. Depending on symmetry of the tensors we will have diffrent definition of them.  
This can be shown for a simple nearest-neightbour hopping Hamiltonian with hopping amplitude `t` and on-site energy `mu`.

.. literalinclude:: /../../tests/mps/generate_by_hand.py
        :pyobject: mpo_XX_model_dense

.. literalinclude:: /../../tests/mps/generate_by_hand.py
        :pyobject: mpo_XX_model_Z2

.. literalinclude:: /../../tests/mps/generate_by_hand.py
        :pyobject: mpo_XX_model_U1


Building YAMPS object automatically
=====================================

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

Canonical form by QR
=====================================

The right/left canonical form of a Tensor in MPS can be tested 
by checking if the right/left overlaps are identities. 

.. literalinclude:: /../../tests/mps/test_mps.py
        :pyobject: is_right_canonical

.. literalinclude:: /../../tests/mps/test_mps.py
        :pyobject: is_left_canonical

The canonical form for a full state can be obtained by QR decomposition:

::

        # rigth canonical form
        #
        psi.canonize_sweep(to='first')
        #
        # left canonical form
        #
        psi.canonize_sweep(to='last')

This is the cheapest way to get canonical. Preparing a state in right canonical form is necessary when we prepare MPS for 
:ref:`theory/mps/algorithms:DMRG` and :ref:`theory/mps/algorithms:TDVP` algorithm. E.g. see the example in :ref:`examples/mps/mps:Multiplication`.

The canonical for is locally restored by applying QR on a Tensor.

::

        # If we have m > n in right canonical we right-orthogonalize n-th Tensor
        # and obtain all m >= n in right canonical form.
        #
        psi.orthogonalize_site(n, to='first', normalize=True)
        #
        # For canonical form the normalization normalize=True is default 
        # but you can impose normalize=False if needed (same for full canonize_sweep).


Canonical form by SVD
=====================================

The SVD decomposition can be used as a method for obtaining effective truncation of MPS or MPO.
To perform the truncation we should set an conditions:

::

        # the instructions are written in a form of dictionary. 
        #
        opts_dict = {}
        #
        # There are different options which we can pass. 
        # If not all are defined the truncation uses defaults.
        #
        opts_dict['D_total'] = 4     # largest total number of singular values to keep
        opts_dict['D_block'] = 2     # largest number of singular values to keep in a single block
        opts_dict['tol'] = 1e-6     # relative tolerance of singular values below which to truncate across all blocks
        opts_dict['tol_blocks'] = 1e-6     # relative tolerance of singular values below which to truncate within individual blocks

With desined options :code:`opts_dict` we can sweep SVD through MPS/MPO. At the end we obtain truncated object in left/right 
canonical form.

::

        # for MPS we usually normalize the state. Here, resulting in right canonical form.
        #
        psi.truncate_sweep(to='first', opts=opts_dict)
        #
        # while for MP0 we want to truncate it but we don't want to loose an applitude for an operator. 
        # Here, resulting in right canonical form.
        #
        H.truncate_sweep(to='first', opts=opts_dict, normalize=False)

Custom test for truncation:

.. literalinclude:: /../../tests/mps/test_truncate_svd.py


.. outside world
Save and load
=====================================

Any MPS/MPO can be saved/loaded (exported/imported) to/from a dictionary or a HDF5 file. In order to import 
an object we have to provide configuration for :meth:`yast.Tensor`'s.

For a dictionary:

.. literalinclude:: /../../tests/mps/test_save_load.py
        :pyobject: test_full_dict

For a HDF5 file:

.. literalinclude:: /../../tests/mps/test_save_load.py
        :pyobject: test_full_hdf5


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
