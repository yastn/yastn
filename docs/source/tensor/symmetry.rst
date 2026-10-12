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
NTU truncation without discarded states, and CTM observables. Finite-patch RDM
tests also compare single-site, nearest-neighbor, diagonal, and full 2x2 density
matrices with independent magnetic/Fock-basis references. The checkerboard
Hubbard benchmark reports energy, double occupancy, and nearest-neighbor spin
correlations against the METTS cylinder data in ``test_Hubbard.py`` at beta=2
(also available at beta=4 and 6). Heisenberg and t-J quickstarts compare with
Abelian simulations at beta=0.5. All three use a purification time step of 0.05;
run them with ``--quickstart``. Bond dimension, time step, CTM convergence, and
the difference between cylinder and infinite-plane geometries must be assessed
separately from the METTS statistical errors.

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

The ``sym_nonabelian`` base class declares the group-specific methods
``canonical_charge``, ``conj_charge``, ``fusion_outcomes``, ``irrep_dimension``,
``fusion_isometry``, ``f_symbol``, and ``braiding_phase`` as abstract interfaces.
It supplies generic left-associated ``fusion_paths``, signed fusion helpers,
multiplicity and selection checks, and unique-outcome ``add_charges``/``fuse``.
Signatures must contain one entry per input charge, with values +1 or -1.
The empty product is the identity irrep returned by ``zero()``.

Generic paths contain canonical intermediate charge tuples and preserve
repeated channels, including multiplicity at the final fusion vertex.
SU2 and SU2xU1 retain their optimized path enumerators and existing label
formats. Implementing selection rules alone does not supply the CG and F/R
data needed by tensor contractions.

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
