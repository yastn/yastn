Basics concepts
===============


Matrix product state
--------------------

The Hilbert space of the quantum system grows exponentially with system size. If the system is composed of `N` elements of the local Hilbert space `d` then total size is `d^N`. 
The matrix representation soon becomes very expensive to represent on classical computer. The separation of variablem allows to write the matrix representation of size `d^N` into a simpler form where the matrix is decomposed into a product of tensors.
The tensor product representation writes the system into the ordered set of interconnected tensors. 
The separation of variables can be performed using e.g. singular value decomposition (SVD) NOTE:cite SVD.
If the SVD decomposition keeps all Schmidt vactors with non-zero aplitude then the decomposition to tensor product is exact. Otherwise the quality of MPS is quantified by the virtual dimensions `D_{i,j}`

::

    #individual tensor in MPS
                ___
    D_{j-1,j}--|___|--D_{j,j+1}
                 |
                d_j


.. math::

    \Psi_0 \otimes \Psi_1 \otimes \dots \otimes \Psi_{N-1} \rightarrow \sum_{j_0,j_1\dots j_{N-1}} \sum_{\sigma_0,\sigma_1\dots \sigma_{N-1}} \, \Theta_{j_0,j_1\dots j_{N-1}}^{\sigma_0,\sigma_1\dots \sigma_{N-1}} \, A^{\sigma_0}_{,j_0} A^{\sigma_1}_{j_0,j_1} \dots A^{\sigma_{N-2}}_{j_{N-2},j_{N-1}} A^{\sigma_{N-1}}_{j_{N-1},}


::

        #matrix product state for the open boundary conditions 
           ___           ___           ___           ___           ___           ___  
        1-|___|-D_{0,1}-|___|-D_{1,2}-|___|-D_{2,3}-|___|-D_{3,4}-|___|-D_{4,5}-|___|-1
            |             |             |             |             |             |   
           d_0           d_1           d_2           d_3           d_4           d_5


.. note::
        The symmetry of MPS is a consequence of symmetry of the `Tensor`s building it. Note that common vitual dimension has to be matching for two nwighbouring tensors. 


Matrix product operator
-----------------------

::

        individual tensor in MPO
                d^j
                _|_
    D_{j-1,j}--|___|--D_{j,j+1}
                 |
                d_j


Canonical form 
---------------

Algorithms
----------

Overlaps
----------
