Expectation values
==================

Expectation values in PEPS require contraction of the lattice. This can be approximately done using CTMRG.

.. autofunction:: yastn.tn.fpeps.ctm._ctmrg

One can stop the CTM after a fixed number of iterations. Stopping criteria can also be set based on
the convergence of one or more observables or by comparing the singular values of the projectors.
Once the CTMRG environment tensors are found, it is straightforward to obtain one-site and two-site
observables using the following functions.

The expectation value of one-site observables for all sites can be calculated using the functionalities

.. autofunction:: yastn.tn.fpeps.ctm._ctm_observables.one_site_avg

and average of two-point correlators can be calculated using the function

.. autofunction:: yastn.tn.fpeps.ctm._ctm_observables.nn_avg


Double Peps Tensor
------------------

The class `DoublePepsTensor`, deals with double-layer Projected Entangled Pair States (PEPS). An instance contains a
top tensor, a bottom tensor, and a rotation. The attribute rotation is optional and defaults to 0.
The properties and methods `ndim`, `get_shape`, and `get_legs` return details about the tensor such as its number of dimensions,
its shape along specified axes, and the tensor's legs along the specified axes.

The key functions (`append_a_bl`, `append_a_tr`, `append_a_tl`, and `append_a_br`) append the top and bottom
PEPS tensor to a given four-legged tensor at a specific position (top-right, top-left, bottom-right, or bottom-left)
respectively. These functions are useful for constructing the metric tensor in time evolution algorithms as well
as during CTMRG procedures. In the following, we provide the index ordering conventions of each of these functions.
Let `tt`` be the four-legged tensor to which the DoublePepsTensor instance is appended.
Let the output tensor after the contraction be `tt'`.


.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor


- **append_a_bl**:

::

            3                    2   3
           _|_                  _|___|_
          |   |-2              |       |--1
    tt =  |   |  1      tt' =  |       |
          |   |__|_            |       |
          |________|--0        |_______|--0


.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor.append_a_bl


- **append_a_tr**:

::

           ________             ________
       0--|____    |        0--|        |
           |   |   |           |        |
    tt =   1   |   |   tt' =   |        |
             2-|___|        1--|________|
                 |               |    |
                 3               3    2

.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor.append_a_tr


- **append_a_tl**:

::

           ________              ________
          |    ____|--3         |        |--2
          |   |   |             |        |
    tt =  |   |   2      tt' =  |        |
          |___|-1               |________|--3
            |                     |    |
            0                     0    1


.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor.append_a_tl


- **append_a_br**:

::

                 0                 1    0
                _|_               _|____|_
             1-|   |          3--|        |
    tt =    2  |   |    tt' =    |        |
           _|__|   |             |        |
       3--|________|          2--|________|


.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor.append_a_br


The functions (`_attach_01` and `_attach_23`) are similar to the append methods but they attach the
tensor to the top or bottom left if rotation = 0, and to the top or bottom right if rotation = 90.

.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor._attach_01
.. autofunction:: yastn.tn.fpeps._doublePepsTensor.DoublePepsTensor._attach_23


- **fPEPS_fuse_layers**:

This method fuses the top and bottom layers of a PEPS tensor network for a particular
instance of DoublePepsTensor. It can be used when it is convenient to work with contracted double tensors rather
than keeping them separate. It is generally avoided due to higher computational complexity.
