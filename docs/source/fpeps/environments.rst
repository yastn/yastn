Environments
============

Contractions of PEPS lattice are supported by environment classes. Those include:
    * :class:`yastn.tn.fpeps.EnvNTU` supports a family of NTU approximations of the bond metric for time evolution.
    * :class:`yastn.tn.fpeps.EnvCTM` CTMRG for finite or infinite lattices. Supports local expectation values, bond metric, etc.
    * :class:`yastn.tn.fpeps.EnvBoundaryMps` BoundaryMPS for contracting finite lattice. It supports expectation values, including long-range correlations, sampling, etc.
    * :class:`yastn.tn.fpeps.EnvApproximate` Supports calculation of bond metric for larger clusters that are contracted approximately.


Neighberhood tensor update environments
---------------------------------------

.. autoclass:: yastn.tn.fpeps.EnvNTU
    :members: bond_metric


Corner transfer matrix renormalization group (CTMRG)
----------------------------------------------------

:class:`yastn.tn.fpeps.EnvCTM` associates with each lattice site
a local CTM environment, that can be accessed via :code:`[]`, and
contracted PEPS of rank-4 tensors available via atribute :code:`psi`.
The convention for ordering the indices in the CTMRG environment tensors are:

::

     _______             _______             _______
    |       |           |       |           |       |
    |  C_nw |--1     0--|  T_n  |--2     0--|  C_ne |
    |_______|           |_______|           |_______|
        |                   |                   |
        0                   1                   1

        2                   0                   0
     ___|___             ___|___             ___|___
    |       |           |       |           |       |
    |  T_w  |--1     1--|   O   |--3     1--|  T_e  |
    |_______|           |_______|           |_______|
        |                   |                   |
        0                   2                   2

        1                   1                   0
     ___|___             ___|___             ___|___
    |       |           |       |           |       |
    |  C_sw |--0     2--|  T_s  |--0     1--|  C_se |
    |_______|           |_______|           |_______|


A single iteration of the CTMRG update, consisting of horizontal and vertical moves,
is performed with :meth:`yastn.tn.fpeps.EnvCTM.update_`.
One can stop the CTM after a fixed number of iterations, or e.g., convergence of corner singular values.
Stopping criteria can also be set based on the convergence of one or more observables,
for example, total energy. Once the CTMRG converges, it is straightforward to obtain
one-site :meth:`yastn.tn.fpeps.EnvCTM.measure_1site` and
two-site nearest-neighbor observables :meth:`yastn.tn.fpeps.EnvCTM.measure_nn`.

.. autoclass:: yastn.tn.fpeps.EnvCTM
    :members: save_to_dict, reset_, bond_metric, update_, ctmrg_, measure_1site, measure_nn, sample

.. autoclass:: yastn.tn.fpeps.EnvCTM_local


Boundary MPS
------------

.. autoclass:: yastn.tn.fpeps.EnvBoundaryMps
    :members: measure_1site,  measure_2site, sample, sample_MC_

Approximate cluster update
--------------------------

.. autoclass:: yastn.tn.fpeps.EnvApproximate
    :members: bond_metric
