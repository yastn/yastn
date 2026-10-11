Specifying symmetry
===================

YASTN specifies symmetry through any object that is a plain Python module or class, which defines

    #. ``SYM_ID`` string label specifying the symmetry

    #. ``NSYM`` number of elements in the charge vector. For example, NSYM=1
       for :math:`U(1)` or :math:`Z_2` group. For product groups such
       as :math:`U(1)\times U(1)`, NSYM=2

    #. ``fuse`` function that specifies the addition of charges

.. automodule:: yastn.sym.sym_abelian
    :members: sym_abelian

Example symmetries defined in YASTN
-----------------------------------

* :math:`U(1)` symmetry

.. literalinclude:: /../../yastn/sym/sym_U1.py
    :pyobject: sym_U1

* :math:`Z_2` symmetry

.. literalinclude:: /../../yastn/sym/sym_Z2.py
    :pyobject: sym_Z2

* :math:`Z_2\times U(1)`

.. literalinclude:: /../../yastn/sym/sym_Z2xU1.py
    :pyobject: sym_Z2xU1

Non-Abelian symmetries
---------------------

The reduced SU2/SU2xU1 implementation supports invariant tensors and PEPS
purification. Independent two-site exact-diagonalization tests cover Hubbard,
t-J, and spin-1/2 and spin-1 Heisenberg evolution, including complex time steps,
NTU truncation without discarded states, and CTM observables. Checkerboard
quickstarts compare high-temperature scalar observables with Abelian simulations;
run them with ``--quickstart``. These small-system/high-temperature checks do not
establish convergence at low temperature or at large bond dimension.

``sym='SU2'`` labels irreducible representations by the non-negative integer
``2*j``. ``sym='SU2xU1'`` uses pairs ``(2*j, q)``, where ``q`` is an integer
particle-number charge. Conjugation preserves ``2*j`` and reverses ``q``.
For example, two spin-1/2 representations fuse into spin zero and spin one::

    import yastn

    config = yastn.make_config(sym='SU2')
    assert config.sym.fusion_outcomes((1,), (1,)) == ((0,), (2,))
    leg = yastn.Leg(config, s=1, t=(0, 1), D=(2, 3))

Here ``D`` counts copies of each irrep: the example has two singlets and three
doublets, giving a physical dimension of eight. Stored blocks contain reduced
multiplicity data. In particular, ``to_numpy()`` does not expand the magnetic
states of each irrep. Hard fusion retains intermediate fusion channels, and
``unfuse_legs()`` restores the specified leg order. Truncation with ``D_total``
counts physical irrep dimensions and keeps whole multiplets.

Unlike Abelian charge addition, fusion can have several outcomes. Use
``fusion_outcomes``, ``fusion_paths``, and ``can_fuse`` to inspect them;
``add_charges`` only accepts products with a unique outcome. The symmetry
classes also provide Clebsch--Gordan fusion isometries and F/R coefficients.

Scalar operators and evolution gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Spin12(sym='SU2')`` and ``Spin1(sym='SU2')`` provide the identity on a spin
doublet and triplet, respectively. Individual spin
components are not invariant scalar operators and raise ``YastnError``.
``SpinfulFermions(sym='SU2')`` and ``SpinfulFermions(sym='SU2xU1')`` provide ``I()``, ``n_total()``, ``d()``
(double occupancy), and ``h()`` (empty-site projector). Its fermionic grading
is ``(False, True)`` for SU2xU1, determined by particle-number parity. For pure
SU2 it is ``True``: ``2*j mod 2`` is the fermion parity. The pure SU2 Hubbard
space has two spin-zero copies (empty, doubly occupied) and one spin doublet.
``SpinfulFermions_tJ`` supports both symmetries with double occupancy excluded.

The PEPS gate constructors ``gate_nn_Heisenberg_SU2``,
``gate_nn_Hubbard_SU2``, ``gate_nn_Hubbard_SU2xU1``, ``gate_nn_tJ_SU2``, and
``gate_nn_tJ_SU2xU1`` build invariant two-site
gates. They accept real ``step`` for imaginary-time evolution or imaginary
``step`` for real-time evolution. For example::

    from yastn.tn import fpeps

    ops = yastn.operators.SpinfulFermions(sym='SU2xU1')
    gate = fpeps.gates.gate_nn_Hubbard_SU2xU1(
        t=1, U=4, mu=2, step=0.01j, I=ops.I())

Hard-fused operator blocks are ordinary matrices in an orthonormal coupled
basis. Fusion/unfusion accounts for CG normalization and duality phases;
``gate_nn_exp`` exponentiates those matrices, and a zero time step reproduces
``fkron(I, I)``. Traces and Frobenius norms of rank-two maps include the irrep
dimension. Higher-rank invariant tensors use normalized CG trees.

The implementation separates structural CG data from multiplicity coefficients.
CG contractions use small vertex networks, and SU2xU1 shares numerical spin
coefficients across particle-number sectors. This avoids expanding whole
magnetic tensors for contractions. The fixed-basis/variable-coefficient separation
can also be seen in `peps-torch IPEPS_LC
<https://github.com/jurajHasik/peps-torch/blob/master/ipeps/ipeps_lc.py>`_; its
`groups/su2.py <https://github.com/jurajHasik/peps-torch/blob/master/groups/su2.py>`_
constructs explicit magnetic spin matrices, whereas YASTN stores reduced blocks.
