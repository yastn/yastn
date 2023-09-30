Basic concepts
--------------

Tensors
^^^^^^^

In general, tensors are multilinear maps from products of several vector spaces

.. math::

    T:\quad V^i\otimes V^j\otimes V^k\otimes...\otimes V_a\otimes V_b\otimes V_c\otimes... \rightarrow scalar,

where `T` expressed in bases and components is

.. math::
    T = \sum_{abc...ijk...} T^{abc...}_{ijk...} e^ie^je^k...e_ae_be_c...

YASTN refers to individual vector spaces :math:`V` as `legs` representing designated index of a tensor. In graphical notation
the shapes represent tensors, while lines protruding from these shapes are `legs` corresponding to individual spaces.  

::

    (generic) tensor,    matrix,             vector
          ___
    V^i--|   |--V_a            ___                 _
    V^j--| T |--V_b      V^i--|_M_|--V_a     V^i--|W|
    V^k--|   |--V_c
      ...|___|...

.. note::
        YASTN defines a vector space and its abelian symmetry structure through :class:`yastn.Leg`.

In some contexts, it is useful to distinguish underlying spaces as co- or contra-variant with respect to transformations acting on them.
Often such a distinction is indicated in writing as a subscript or superscript of the tensor.

An example from quantum mechanics is a different between :math:`\langle bra |`
and :math:`|ket \rangle` spaces to separate co- and contra-variant spaces we write

.. math::

    T = \sum_{abc...ijk...} T^{abc...}_{ijk...} |i \rangle|j \rangle|k \rangle ...
    \langle a |\langle b |\langle c |...

In YASTN, similar to other implementations (:ref:`see below <refs_basics>`), the convension for spaces of different meaning is through a `signature`. 
The binary choice between :math:`\langle bra |` and :math:`|ket \rangle` spaces, or Hilbert space :math:`\mathcal{H}` and its dual :math:`\mathcal{H}^*`, is given by a sign for a given leg of the tensor.

.. note::
    `signature`, :attr:`yastn.Tensor.s`, is a tuple of signs :math:`\pm 1`

Action of abelian symmetry
^^^^^^^^^^^^^^^^^^^^^^^^^^

For any element `g` of abelian group G, its action on tensor elements :math:`T^{ab...}_{ij...}`
in a proper basis can be represented by `diagonal` matrices `U(g)` acting on each of the vector spaces

.. math::

    (gT)^{ab...}_{ij...} = \sum_{a'b'...i'j'...} T^{a'b'...}_{i'j'...} [U(g)^*]^{a}_{a'} [U(g)^*]^{b}_{b'} ... {U(g)}^{i'}_{i} {U(g)}^{j'}_{j}...,

where the elements of `U(g)` are complex phases defined by **charges** :math:`t_i`,
in YASTN always taken to be integers :math:`\mathbb{Z}` or their subset, as

.. math::

    U(g)^j_k=\exp(-i\theta_g t_j)\delta_{jk}

with angle :math:`\theta_g \in [0,2\pi)`, which depends on :math:`g \in G`, and :math:`\delta_{jk}` being
Kronecker delta.

This structure gives a simple selection rule that all symmetric tensors must obey.
Taking group element :math:`g \in G` for **all non-zero** elements of `T`, it must hold

.. math::

    (gT)^{ab...}_{ij...} = T^{ab...}_{ij...}exp[i\theta_g(t_a+t_b+...-t_i-t_j-...)].

.. _symmetry selection rule:

The selection rule can be equivalently expressed as charge conservation

.. math::
    t_a+t_b+...-t_i-t_j-... = N

with total charge of the tensor `N` being independent of tensor elements :math:`T^{ab...}_{ij...}`. In the case of :math:`N=0`, such a tensor is invariant (unchanged) under the action of the symmetry transformation.
Otherwise, it transforms covariantly because as all its elements are altered by the same complex phase :math:`\exp(i\theta_gN)`.

The charges :math:`t_i,\ N` and the addition :math:`+` operation depends on tensor abelian group. 

.. note::
    * Total charge `N` of YASTN tensor can be accessed by :attr:`yastn.Tensor.n`
    * To inspect what charge sectors :math:`t_i` exist on legs of a tensor
      use :meth:`yastn.Tensor.get_legs`.


Examples for selected groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **U(1)**: The charges are integer numbers :math:`t_i \in \mathbb{Z}` with normal integer addition.
  With :math:`\theta_g` being an angle :math:`\theta_g \in [0,2\pi)`.
* **Z(2)**: The charges are from a subset of integers :math:`t_i \in \{0,1\}` with addition modeulo :math:`2`. 
In a similar way, two elements of group :math:`Z_2` are mapped to the angles :math:`\{0,1\}\xrightarrow{\theta} \{0,\pi\}`. 
* direct product of symmetries :math:`\mathbf{Z_2xU(1)}`: The charges of individual groups are accummulated in a vector :math:`t_i \in \{0,1\}\otimes \mathbb{Z}`. The addition operation matches the symmetry such that 

.. math::

    t_i+t'_i := \begin{pmatrix} t_{i,0} \\ t_{i,1} \end{pmatrix} + \begin{pmatrix} t'_{i,0} \\ t'_{i,1} \end{pmatrix} = \begin{pmatrix} t_{i,0} + t'_{i,0}\ \textrm{mod}\ 2\\ t'_{i,1} + t'_{i,1} \end{pmatrix}

.. note::
    To learn more on how YASTN defines symmetries with examples visit :ref:`API docs<tensor/symmetry:specifying symmetry>`.

Conjugation
^^^^^^^^^^^

Conjugation of a tensor complex-conjugates tensor elements, flips tensor signature :attr:`yastn.Tensor.s` by
replacing :math:`\pm 1 \to \mp 1`, as well as the total charge :math:`N \to -N`.
In the latter, charge negation :math:`-` depends on the abelian group.

YAST enable to control the signature, i.e., interpretation, of each individual leg. The signature flip changes the interpretation of the leg and negates the charges for a flipped leg. 

.. note::
    For various types of conjugation see :ref:`API docs<tensor/algebra:Conjugation of symmetric tensors>`.

.. _refs_basics:

References & Related works
^^^^^^^^^^^^^^^^^^^^^^^^^^

* `ITensor <https://itensor.org/>`_
* `TenPy <https://github.com/tenpy/tenpy>`_
* `TensorNetwork <https://github.com/google/TensorNetwork>`_

1. "From density-matrix renormalization group to matrix product states" Ian P McCulloch, `J. Stat. Mech., (2007) P10014 <https://iopscience.iop.org/article/10.1088/1742-5468/2007/10/P10014>`_
2. "Tensor network states and algorithms in the presence of a global U(1) symmetry" Sukhwinder Singh, Robert N. C. Pfeifer, Guifre Vidal, `Phys. Rev. B 83, 115125 (2011) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.83.115125>`_ or arXiv version `arXiv:1008.4774 <https://arxiv.org/abs/1008.4774>`_
