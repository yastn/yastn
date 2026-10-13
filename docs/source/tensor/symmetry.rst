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

Scalar operators, multiplets, and evolution gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Spin12(sym='SU2')`` and ``Spin1(sym='SU2')`` provide the identity on a spin
doublet and triplet, respectively. Individual spin
components are not invariant scalar operators and raise ``YastnError``.
``SpinfulFermions(sym='SU2')`` and ``SpinfulFermions(sym='SU2xU1')`` provide ``I()``, ``n_total()``, ``d()``
(double occupancy), and ``h()`` (empty-site projector). Its fermionic grading
is ``(False, True)`` for SU2xU1, determined by particle-number parity. For pure
SU2 it is ``True``: ``2*j mod 2`` is the fermion parity. The pure SU2 Hubbard
space has two spin-zero copies (empty, doubly occupied) and one spin doublet.
``SpinfulFermions_tJ`` supports both symmetries with double occupancy excluded.

All four operator families provide ``S()``, an invariant rank-three tensor
with legs ``(out, in, irrep)`` and signatures ``(+1, -1, -1)``. The third leg
has charge ``(2,)`` for SU2 or ``(2, 0)`` for SU2xU1. Its magnetic components
are ``(Sm/sqrt(2), Sz, -Sp/sqrt(2))``.

The fermionic families also provide ``cp_spinor()`` with signatures
``(+1, -1, -1)`` and its dual ``c_spinor()`` with ``(+1, -1, +1)``.
Their third legs carry charge ``(1,)`` or ``(1, +1)``; the opposite signatures
encode creation versus annihilation. All multiplets are ordinary ``Tensor``
objects with total charge zero. The third leg has multiplicity one, but its
SU2 representation has dimension two for spinors and three for ``S()``.
Pair contractions retain this representation as well as its fixed parity.

``mps.generate_mpo`` accepts consecutive pairs of multiplets in each ``Hterm``.
``(S(), S())`` means the spin dot product, and
``(cp_spinor(), c_spinor())`` means spin-summed hopping, without factors of
``1/3`` or ``1/2``. Pairs follow operator order, including reversed or repeated
sites. Scalar factors and multiple pairs may appear in the same term::

    from yastn.tn import mps

    ops = yastn.operators.SpinfulFermions(sym='SU2xU1')
    H = mps.generate_mpo(ops.I(), [
        mps.Hterm(-1, (0, 1), (ops.cp_spinor(), ops.c_spinor())),
        mps.Hterm(-1, (1, 0), (ops.cp_spinor(), ops.c_spinor())),
    ], N=2)

The MPS ``measure_2site`` and ``measure_nsite`` interfaces accept the same
pairs, as do the CTM ``measure_nn``, ``measure_line``, ``measure_2site``, and
``measure_nsite`` interfaces. CTM ``measure_nsite`` accepts ``opts_svd`` and
``opts_var`` to control boundary-MPS compression. ``measure_line``,
``measure_2x2`` and ``measure_nsite_exact`` retain their respective exact
contractions, without boundary-MPS truncation. ``measure_2site`` propagates
the fixed-irrep wire within its row/column sweep and reuses source environments. Boundary-MPS and belief-propagation environments also
support nearest-neighbor multiplet pairs.

``gate_nn_exp(step, I, H)`` accepts a two-site Hamiltonian MPO, including the
example above. ``gate_nn_Heisenberg(J, step, I, S)`` accepts a spin multiplet,
and ``gate_nn_hopping(t, step, I, c, cdag)`` accepts annihilation and creation
spinors to build spin-summed hopping gates.

The same gate constructors work with SU2 and SU2xU1 multiplets. Real ``step``
gives imaginary-time evolution and imaginary ``step`` gives real-time evolution.
For a complete Hubbard bond, add the scalar on-site interactions before
exponentiating::

    from yastn.tn import fpeps

    t, U, mu = 1, 4, 2
    c, cp = ops.c_spinor(), ops.cp_spinor()
    local = U * ops.d() - mu * ops.n_total()
    H = mps.generate_mpo(ops.I(), [
        mps.Hterm(-t, (0, 1), (cp, c)),
        mps.Hterm(-t, (1, 0), (cp, c)),
        mps.Hterm(1, (0,), (local,)),
        mps.Hterm(1, (1,), (local,)),
    ], N=2)
    gate = fpeps.gates.gate_nn_exp(0.01j, ops.I(), H)

For t-J, use ``SpinfulFermions_tJ`` and replace the Hubbard interaction with
``Hterm(J, (0, 1), (S, S))`` and ``Hterm(-J/4, (0, 1), (n, n))``, where
``S = ops.S()`` and ``n = ops.n_total()``; include the desired on-site chemical
potentials as scalar terms. This uses the same ``gate_nn_exp`` interface.

Observables use the multiplets directly::

    hopping = env.measure_nn(ops.cp_spinor(), ops.c_spinor())
    spin_dot = env.measure_nn(ops.S(), ops.S())

For a spin-rotation-invariant state, ``SzSz`` is ``spin_dot / 3`` on each bond.
The reverse ordering ``measure_nn(c, cp)`` gives the negative of the reverse
hopping on distinct sites, due to fermionic anticommutation.

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
