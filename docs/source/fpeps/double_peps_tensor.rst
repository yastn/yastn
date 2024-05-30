Double Peps Tensor
==================

The class `DoublePepsTensor`, deals with double-layer Projected Entangled Pair States (PEPS). An instance contains a
top tensor, a bottom tensor, and a rotation. The attribute rotation is optional and defaults to 0.
The properties and methods `ndim`, `get_shape`, and `get_legs` return details about the tensor such as its number of dimensions,
its shape along specified axes, and the tensor's legs along the specified axes.

The key functions (`_attach_01`, `_attach_12`, `_attach_23`, `_attach_30`) append the top and bottom
PEPS tensor to a given four-legged tensor at a specific position (top-right, top-left, bottom-right, or bottom-left)
respectively. These functions are useful for constructing the metric tensor in time evolution algorithms as well
as during CTMRG procedures. In the following, we provide the index ordering conventions of each of these functions.
Let `tt`` be the four-legged tensor to which the DoublePepsTensor instance is appended.
Let the output tensor after the contraction be `tt'`.


.. autoclass:: yastn.tn.fpeps.DoublePepsTensor

