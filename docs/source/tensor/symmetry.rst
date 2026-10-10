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

The reduced SU2/SU2xU1 implementation is experimental. Tensor and gate
regressions are available for NumPy and Torch, but PEPS evolution and
measurement still have a numerical consistency gap: the untruncated two-site
Hubbard comparison against exact diagonalization does not yet agree at
machine precision. This is tracked by
``tests/quickstart/test_Hubbard_SU2.py::test_two_site_hubbard_SU2_against_ED``
(run with ``--quickstart``). The checkerboard Hubbard quickstart also fails
its comparison with U1xU1, with substantial errors in double occupancy and
nearest-neighbor density correlations. Passing the tensor and gate tests
does not yet establish the correctness of non-Abelian PEPS simulations.

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

``Spin12(sym='SU2')`` provides the identity on a spin doublet. Individual spin
components are not invariant scalar operators and raise ``YastnError``.
``SpinfulFermions(sym='SU2xU1')`` provides ``I()``, ``n_total()``, ``d()``
(double occupancy), and ``h()`` (empty-site projector). Its fermionic grading
is ``(False, True)``, determined by particle-number parity.

The PEPS gate constructors ``gate_nn_Heisenberg_SU2``,
``gate_nn_Hubbard_SU2xU1``, and ``gate_nn_tJ_SU2xU1`` build invariant two-site
gates. They accept real ``step`` for imaginary-time evolution or imaginary
``step`` for real-time evolution. For example::

    from yastn.tn import fpeps

    ops = yastn.operators.SpinfulFermions(sym='SU2xU1')
    gate = fpeps.gates.gate_nn_Hubbard_SU2xU1(
        t=1, U=4, mu=2, step=0.01j, I=ops.I())

Reduced fused operator blocks include CG normalization and duality phases;
their identity blocks need not be unit matrices or positive definite.
``gate_nn_exp`` accounts for this convention when exponentiating an invariant
Hamiltonian, and a zero time step reproduces ``fkron(I, I)``.
