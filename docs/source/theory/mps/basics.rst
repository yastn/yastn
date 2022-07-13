Basics concepts
===============


Matrix product state
--------------------

The numerical simulation of quantum system has been proven to be hard due to exponentially growing size of matrix representation if the system. In particular, if the local Hilbert space for `j`-th particle is :math: `\mathcal{H}_j`of dimension `d` 
(e.g. `d=2` for qubit and `d=3` for qutrit) then for `N` particles it will be :math: `\mathcal{H}_0 \otimes \mathcal{H}_1 \cdots \otimes \mathcal{H}_{N-1}` which is `d^N`. 
Tensor Network representation introduce a concept of efficient separation of variables which will iteratively split `j`-th particle from others by performing :ref:`tensor/algebra:Spectral decompositions` e.g. singular value decomposition (SVD). 
The SVD operation at first isolates `0`-th from `1`-to-`N`-th and then `1`-th from `2`-to-`N`-th until the state is decomposed to `N` tensors forming  `matrix product state` (MPS). 

.. math::

    \Psi \in \mathcal{H}_0 \otimes \mathcal{H}_1 \cdots \otimes \mathcal{H}_{N-1} \xrightarrow{SVD}{\sum_{j_0,j_1\dots j_{N-1}} \sum_{\sigma_0,\sigma_1\dots \sigma_{N-1}} \, \Theta_{j_0,j_1\dots j_{N-1}}^{\sigma_0,\sigma_1\dots \sigma_{N-1}} \, A^{\sigma_0}_{,j_0} A^{\sigma_1}_{j_0,j_1} \dots A^{\sigma_{N-2}}_{j_{N-2},j_{N-1}} A^{\sigma_{N-1}}_{j_{N-1},}}

A single tensor `A_j` is an object of the size :math:`D_{j-1,j} \times d_j \times D_{j,j+1}`. 

::
    
    # individual tensor in MPS
                      ___
    A_j = D_{j-1,j}--|___|--D_{j,j+1}
                       |
                      d_j


The MPS forms one-dimensional structure with each tensor having a physical dimension `d` (`d_j` for general case when particles/quidict are different) and virtual dimensions `D_{i,j}` conecting `i`-th particle with `j`-th particle. `YAMPS` allows to perform computation on one dimensional MPS with open boundary conditions. 
The schematic picture for general MPS is visible below.

::

        # matrix product state for the open boundary conditions N=6
           ___           ___           ___           ___           ___           ___  
        1-|___|-D_{0,1}-|___|-D_{1,2}-|___|-D_{2,3}-|___|-D_{3,4}-|___|-D_{4,5}-|___|-1
            |             |             |             |             |             |   
           d_0           d_1           d_2           d_3           d_4           d_5

The above presents top-to-bottom construction however one can also think about MPS as an ansatz for calculation. 

The virtul dimension (also known as the bond dimension) can be interprete




.. note::
        The symmetry of MPS is a consequence of symmetry of the `Tensor`s building it. Note that common vitual dimension has to be matching for two nwighbouring tensors. 


Matrix product operator
-----------------------

::

        #individual tensor in MPO

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

References & Related works
--------------------------

1. "Tensor Network Contractions: Methods and Applications to Quantum Many-Body Systems" Shi-Ju Ran, Emanuele Tirrito, Cheng Peng, Xi Chen, Luca Tagliacozzo, Gang Su, Maciej Lewenstein `Lecture Notes in Physics (LNP, volume 964), (2020) <https://link.springer.com/book/10.1007/978-3-030-34489-4>`_

