Basic concepts
==============

Tensors
-------

Tensors are multilinear maps from products of several vector spaces

.. math::

    T:\quad V^i\otimes V^j\otimes V^k\otimes...\otimes V_a\otimes V_b\otimes V_c\otimes... \rightarrow scalar,

where :math:`V^j` and :math:`V_j` refers to vector space that is either covariant or contravariant with respect to transformations acting on these spaces.
YASTN refers to individual spaces :math:`V` as **legs**.
The tensor :math:`T` expressed in bases and components is

.. math::
    T = \sum_{abc...ijk...} T^{abc...}_{ijk...} e^ie^je^k...e_ae_be_c...

For tensors we introduce graphical notation where shapes represent tensors and lines
protruding from the shape (legs) correspond to individual vector spaces

::

    (generic) tensor,    matrix,             vector
          ___
    V^i--|   |--V_a            ___                 _
    V^j--| T |--V_b      V^i--|_M_|--V_a     V^i--|W|
    V^k--|   |--V_c
      ...|___|...

.. note::
        YASTN defines a vector space and its abelian symmetry structure through :class:`yastn.Leg`.

In quantum mechanics, we introduce an operator

.. math::

    T = \sum_{abc...ijk...} T^{abc...}_{ijk...} |i \rangle|j \rangle|k \rangle ...
    \langle a |\langle b |\langle c |...

where due to different actions of symmetry transformations vector spaces are split between :math:`\langle bra |` and :math:`|ket \rangle` spaces, or Hilbert space :math:`\mathcal{H}` and its dual :math:`\mathcal{H}^*`.
In YASTN, similar to other implementations (:ref:`see below <refs_basics>`), the distinction between
:math:`\langle bra |` and :math:`|ket \rangle` spaces, is encoded through **signature** atribute of :class:`yastn.Leg` assigned to a tensor.

.. note::
    Signature of the tensor, i.e, :attr:`yastn.Tensor.s`, is a tuple of signs :math:`\pm 1` matching signatures of individual legs.

Action of abelian symmetry
--------------------------

For any element :math:`g` of abelian group :math:`G`, its action on tensor elements :math:`T^{ab...}_{ij...}`
in a proper basis can be represented by diagonal matrices :math:`U(g)` acting on each of the vector spaces

.. math::

    (gT)^{ab...}_{ij...} = \sum_{a'b'...i'j'...} T^{a'b'...}_{i'j'...} [U(g)^*]^{a}_{a'} [U(g)^*]^{b}_{b'} ... {U(g)}^{i'}_{i} {U(g)}^{j'}_{j}...,

where the elements of :math:`U(g)` are complex phases defined by **charges** :math:`t_i`.
In YASTN the charges are integers :math:`t_i\in\mathbb{Z}` or their subset---or tuples of integers for direct product of multiple symmetric groups.
They are related to symmetry transformation

.. math::

    U(g)^j_k=\exp(-i\theta_g t_j)\delta_{jk}

where :math:`\delta_{jk}` is a Kronecker delta and the angle :math:`\theta_g \in [0,2\pi)` depends on :math:`g \in G`.
The structure gives a simple selection rule that all symmetric tensors must obey.

Taking group element :math:`g \in G` for **all non-zero** elements of :math:`T`, it must hold that

.. math::

    (gT)^{ab...}_{ij...} = T^{ab...}_{ij...}exp[i\theta_g(t_a+t_b+...-t_i-t_j-...)].

.. _symmetry selection rule:

The selection rule can be equivalently expressed as charge conservation

.. math::

    \sum_j s_{j} t_{j} = n

where :math:`s_j` is the signature and :math:`t_j` is the change of corresponding sectors.
For the tensor :math:`T` in the examples above

.. math::

    t_a+t_b+...-t_i-t_j-... = n

with total charge of the tensor :math:`n` being independent of tensor elements :math:`T^{ab...}_{ij...}`.
For :math:`n=0`, a tensor is invariant (unchanged) under the action of the symmetry.
Otherwise, it transforms covariantly as all its elements are altered by the same complex phase :math:`\exp(i\theta_g n)`.

The charges :math:`t_i,\ n` and precise form of their addition :math:`+` depends on the abelian group
considered.

.. note::
    * Total charge :math:`n` of YASTN tensor is accessed by :attr:`yastn.Tensor.n`.
    * To inspect what charge sectors :math:`t_i` exist on legs of a tensor
      use :meth:`yastn.Tensor.get_legs`.


Examples for selected groups
----------------------------

* :math:`U(1)`: allowed charges are integers :math:`t_i \in \mathbb{Z}` with usual integer addition and :math:`\theta_g` is usual angle :math:`\theta_g \in [0,2\pi)`.
* :math:`Z_2`: allowed charges are a subset of integers :math:`t_i \in \{0,1\}` with addition :math:`\textrm{mod 2}`. Two elements of the group map to angles :math:`\{0,1\}\xrightarrow{\theta} \{0,\pi\}`.
* :math:`Z_2 \times U(1)`: direct product of two symmetries lead to allowed charges that are individual group charges accummulated in a vector :math:`t_i \in \{0,1\} \otimes \mathbb{Z}`. The addition is distributed, i.e.,

.. math::

    t_i+t'_i := \begin{pmatrix} t_{i,0} \\ t_{i,1} \end{pmatrix} + \begin{pmatrix} t'_{i,0} \\ t'_{i,1} \end{pmatrix} = \begin{pmatrix} t_{i,0} + t'_{i,0}\ \textrm{mod}\ 2\\ t'_{i,1} + t'_{i,1} \end{pmatrix}

See :ref:`API docs<tensor/symmetry:specifying symmetry>` on how YASTN defines symmetries.

Conjugation
-----------

Conjugation of a tensor acts such as all tensor elements are complex-conjugated, tensor leg signature is flipped by
replacing :math:`\pm 1 \to \mp 1` in leg signature :attr:`yastn.Tensor.s`, and, similarly, the total charge is flipped :math:`n \to -n`.
In the latter, the change of a sign by minus depends on the abelian group.

Individual flip of the signature of a specific leg is also possible and is accompanied by negation of charges on that leg.

See :ref:`API docs<tensor/algebra:Conjugation of symmetric tensors>` for various types of conjugation.

.. _refs_basics:

References & Related works
--------------------------

* `ITensor <https://itensor.org/>`_
* `TenPy <https://github.com/tenpy/tenpy>`_
* `TensorNetwork <https://github.com/google/TensorNetwork>`_

1. "From density-matrix renormalization group to matrix product states" Ian P McCulloch, `J. Stat. Mech., (2007) P10014 <https://iopscience.iop.org/article/10.1088/1742-5468/2007/10/P10014>`_
2. "Tensor network states and algorithms in the presence of a global U(1) symmetry" Sukhwinder Singh, Robert N. C. Pfeifer, Guifre Vidal, `Phys. Rev. B 83, 115125 (2011) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.83.115125>`_ or arXiv version `arXiv:1008.4774 <https://arxiv.org/abs/1008.4774>`_
